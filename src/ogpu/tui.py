"""`ogpu top`: a small, colorful live view of each machine — GPU telemetry, who holds the GPU, who is queued, spend."""
from __future__ import annotations

import json
import select
import sys
import termios
import threading
import time
import tty

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from . import agents, ledger, ssh
from .config import Config, load_state
from .providers import SERVERLESS
from .runner import JOBRUN, fmt_dur
from .telemetry import SMI, Telemetry, sparkline

STATUS_STYLE = {"RUNNING": "bold black on green", "EXITED": "bold black on yellow", "TERMINATED": "bold white on red"}


def bar(frac: float, width: int, hi: str = "green") -> Text:
    frac = max(0.0, min(1.0, frac))
    n = round(frac * width)
    style = "red" if frac > 0.9 else "yellow" if frac > 0.7 else hi
    return Text("█" * n, style=style) + Text("░" * (width - n), style="grey35")


def agent_text(name: str, track: str | None = None) -> Text:
    t = Text() + Text(name, style=f"bold {agents.color(name)}")
    if track:
        t += Text(f"({track})", style="grey70")
    return t


def short(cmd: str, n: int = 52) -> str:
    cmd = " ".join(cmd.split())
    return cmd if len(cmd) <= n else cmd[: n - 1] + "…"


class Poller:
    """Background refresh of the job snapshot (SSH) and machine info (provider API)."""

    def __init__(self, cfg: Config, pool: str = "timing"):
        self.cfg = cfg
        self.pool = pool
        self.snap: dict | None = None
        self.pod = None
        self.err = ""
        self._stop = False
        threading.Thread(target=self._snap_loop, daemon=True).start()
        threading.Thread(target=self._pod_loop, daemon=True).start()

    def _snap_loop(self):
        while not self._stop:
            try:
                st = load_state(self.pool)
                p = ssh.run(st, self.cfg.private_key_path, f"{JOBRUN} snapshot", timeout=20)
                self.snap, self.err = json.loads(p.stdout), ""
            except Exception as e:  # noqa: BLE001 — keep the dashboard alive on any transient failure
                self.err = f"snapshot: {type(e).__name__}"
            time.sleep(2)

    def _pod_loop(self):
        from .pod import client, sync_state
        try:
            rp = client(self.cfg.pool(self.pool) or self.cfg, load_state(self.pool))
        except SystemExit:  # no API key: telemetry and jobs still work
            return
        while not self._stop:
            try:
                st = load_state(self.pool)
                self.pod = rp.get(st.pod_id) if st.pod_id else None
                if self.pod:
                    sync_state(self.pod, pool=self.pool)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(15)

    def close(self):
        self._stop = True


class Monitor:
    """Live data for one pool's pod: GPU telemetry stream + job snapshot/pod-info pollers."""

    def __init__(self, cfg: Config, pool: str):
        self.pool = pool
        st = load_state(pool)
        self.tel = Telemetry(ssh.ssh_argv(st, cfg.private_key_path, SMI))
        self.poll = Poller(cfg, pool)

    def close(self):
        self.tel.close()
        self.poll.close()


def pool_block(cfg: Config, mon: Monitor, idx: int, width: int) -> list:
    pool, tel, poll = mon.pool, mon.tel, mon.poll
    st = load_state(pool)
    now = time.time()
    rows: list = []
    pod = poll.pod
    status = pod.status if pod else ("RUNNING" if tel.latest else "UNKNOWN")
    up = fmt_dur(pod.uptime_s) if pod and pod.uptime_s else "—"
    h = Text() + Text(f" {idx} ", style="bold black on grey70") + Text(f" {pool.upper()} ", style="bold magenta")
    h += Text(f" {status} ", style=STATUS_STYLE.get(status, "bold black on grey50"))
    h += Text(f"  {st.gpu or '?'}", style="bold")
    h += Text(f"  {st.provider or cfg.provider}{'/' + st.cloud if st.cloud else ''}  ", style="grey70")
    h += Text(f"${st.cost_per_hr:.2f}/hr", style="bold cyan") + Text(f"  up {up}", style="grey70")
    rows.append(h)

    s = tel.latest
    bw = max(10, min(24, width // 6))
    if s:
        g = Text() + Text(" GPU  ", style="bold") + bar(s.util / 100, bw) + Text(f" {s.util:3.0f}%   ", style="bold")
        g += Text("MEM ", style="bold") + bar(s.mem_used_gb / max(s.mem_total_gb, 1), bw // 2, "cyan")
        g += Text(f" {s.mem_used_gb:.1f}/{s.mem_total_gb:.0f} GB   ")
        g += Text("PWR ", style="bold") + Text(f"{s.power_w:.0f}W", style="orange1")
        g += Text(f"/{s.power_limit_w:.0f}  ", style="grey50")
        tstyle = "red" if s.temp_c >= 80 else "yellow" if s.temp_c >= 70 else "green"
        g += Text("TEMP ", style="bold") + Text(f"{s.temp_c:.0f}°C  ", style=tstyle)
        g += Text("SM ", style="bold") + Text(f"{s.sm_mhz:.0f}MHz", style="grey70")
        rows.append(g)
        spark_w = max(10, width - 30)
        rows.append(Text("      ") + Text(sparkline(tel.util, spark_w), style="bright_green")
                    + Text(f"  util · last {len(list(tel.util)[-spark_w:])}s", style="grey50"))
    else:
        rows.append(Text() + Text(" GPU  ", style="bold") + Text(tel.error or "connecting…", style="grey50"))

    snap = poll.snap or {}
    jobs = snap.get("jobs") or []
    holder = snap.get("holder")
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold", width=6)
    t.add_column()
    t.add_column(style="grey70")
    t.add_column(justify="right")
    if holder:
        running = next((j for j in jobs if j["id"] == holder.get("id")), {})
        usual = running.get("usual_s")
        el = now - holder["since"]
        style = "bold red" if usual and el > 3 * max(usual, 10) else "bold green"  # suspiciously long → red
        hint = f" (usual ~{fmt_dur(usual)})" if usual else ""
        t.add_row("LOCK", agent_text(holder["agent"], holder.get("track")), short(holder["cmd"]),
                  Text(f"running {fmt_dur(el)}{hint}", style=style))
    else:
        t.add_row("LOCK", Text("free", style="green"), "", "")
    for j in [j for j in jobs if j["status"] == "queued"]:
        t.add_row("QUEUE", agent_text(j["agent"], j["track"]), short(j["cmd"]),
                  Text(f"#{j.get('queue_pos', '?')} waiting {fmt_dur(now - j['created'])}", style="yellow"))
    for j in [j for j in jobs if j["status"] == "running" and j["mode"] == "cpu"]:
        t.add_row("CPU", agent_text(j["agent"], j["track"]), short(j["cmd"]),
                  Text(f"{fmt_dur(now - (j['started'] or now))}", style="cyan"))
    rows.append(t)

    reaper = snap.get("reaper") or {}
    if reaper.get("busy") or holder or any(j["status"] in ("queued", "running") for j in jobs):
        idle = Text("      idle-stop  —  (busy)", style="grey50")
    elif reaper.get("stop_at"):
        left = reaper["stop_at"] - now
        idle = Text() + Text("      idle-stop in ", style="grey50") + Text(
            fmt_dur(max(0, left)), style="bold yellow" if left < 300 else "bold")
    else:
        idle = Text("      idle-stop  ?", style="grey50")
    rows.append(idle + Text("   " + (poll.err or ""), style="red"))
    rows.append(Text(""))
    return rows


def render(cfg: Config, monitors: list, width: int, idle_pools: list | None = None) -> Group:
    now = time.time()
    rows: list = []
    with ledger.connect() as db:
        today = ledger.spend(db, ledger.today_start())
        u = ledger.utilization(db, ledger.today_start())
        per_agent = ledger.agent_spend(db, ledger.today_start())
        recent = db.execute("select * from runs where status not in ('queued','running') "
                            "order by ended desc limit 5").fetchall()
    burn = sum(load_state(m.pool).cost_per_hr for m in monitors if (m.poll.pod is None or m.poll.pod.status == "RUNNING"))
    cap_style = "red" if today >= cfg.budget_usd else "yellow" if today >= 0.8 * cfg.budget_usd else "green"
    h = Text() + Text(" ogpu ", style="bold black on bright_magenta") + Text(f"  {len(monitors)} pod(s)  ", style="bold")
    h += Text("burn ", style="grey70") + Text(f"${burn:.2f}/hr", style="bold cyan")
    h += Text("   today ", style="grey70") + Text(f"${today:.2f}", style=f"bold {cap_style}")
    h += Text(f" / ${cfg.budget_usd:.0f} cap", style="grey50")
    rows += [h, Text("")]

    for i, mon in enumerate(monitors, 1):
        rows += pool_block(cfg, mon, i, width)
    for pool in idle_pools or []:
        rows.append(Text(f"  ○ {pool.upper()}  no pod running", style="grey50")
                    + Text("  (auto-starts on --fidelity correctness)" if pool == "correctness" else "", style="grey35"))
    if idle_pools:
        rows.append(Text(""))

    rt = Table.grid(padding=(0, 2))
    rt.add_column(style="bold", width=6)
    rt.add_column()
    for i, r in enumerate(recent):
        mark = Text("⚡", style="yellow") if r["cached_from"] else (
            Text("✔", style="bold green") if r["status"] == "done" and not r["failure"] else Text("✘", style="bold red"))
        line = Text() + mark + Text(" ") + agent_text(r["agent"], r["track"])
        if (r["pool"] or "timing") != "timing":
            line += Text(f" [{r['pool']}]", style="magenta")
        hl = ledger.headline(json.loads(r["metrics"] or "{}"))
        if r["failure"] and not r["cached_from"]:
            hl = f"{r['failure']}  {hl}".strip()
        if hl:
            line += Text(f"  {hl}", style="bold")
        line += Text(f"  {fmt_dur(r['wall_s'] or 0)} · ${r['cost_usd'] or 0:.2f} · {fmt_dur(now - (r['ended'] or now))} ago",
                     style="grey50")
        rt.add_row("RECENT" if i == 0 else "", line)
    if recent:
        rows += [rt, Text("")]

    if u["paid_s"]:
        frac = u["gpu_busy_frac"]
        fs = "bold green" if frac >= 0.6 else "bold yellow" if frac >= 0.3 else "bold red"
        line = Text() + Text("TODAY   ", style="bold") + Text("GPU busy ", style="grey70") + Text(f"{frac:.0%}", style=fs)
        line += Text(" of paid time · idle ", style="grey70") + Text(f"${u['idle_usd']:.2f}", style="bold yellow")
        for a, c in list(per_agent.items())[:4]:
            line += Text("  ") + agent_text(a) + Text(f" ${c:.2f}", style="grey70")
        rows += [line, Text("")]

    keys = Text("q quit · s stop a pod · r refresh", style="grey50")
    rows.append(Text(" " * max(1, width - len(keys.plain) - 1)) + keys)
    return Group(*rows)


def run_top(cfg: Config, pool: str | None = None) -> None:
    """All pools with a pod (or just `pool`), refreshed live; pools that come up later are picked up."""
    from .config import POOLS
    console = Console()
    wanted = [pool] if pool else [p for p in POOLS if cfg.pool(p) and cfg.pool(p).provider not in SERVERLESS]
    monitors: dict[str, Monitor] = {}

    def refresh_monitors():
        for p in wanted:
            if p not in monitors and load_state(p).reachable:
                monitors[p] = Monitor(cfg, p)

    refresh_monitors()
    if not monitors:
        raise SystemExit("no pod recorded yet — run `ogpu up` first")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd) if sys.stdin.isatty() else None
    last_scan = time.time()
    try:
        if old:
            tty.setcbreak(fd)
        view_of = lambda: render(cfg, list(monitors.values()), console.width,  # noqa: E731
                                 [p for p in wanted if p not in monitors])
        with Live(view_of(), console=console, refresh_per_second=4, screen=False) as live:
            stopping, confirm = False, None
            while True:
                key = ""
                if old and select.select([sys.stdin], [], [], 0.5)[0]:
                    key = sys.stdin.read(1).lower()
                elif not old:
                    time.sleep(0.5)
                if time.time() - last_scan > 10:
                    refresh_monitors()
                    last_scan = time.time()
                mons = list(monitors.values())
                if key == "q":
                    break
                if key == "s":
                    if confirm is not None:  # second s confirms
                        from .pod import client
                        st = load_state(confirm)
                        client(cfg.pool(confirm) or cfg, st).stop(st.pod_id)
                        stopping, confirm = False, None
                    else:
                        stopping = True
                elif stopping and key.isdigit() and 1 <= int(key) <= len(mons):
                    confirm = mons[int(key) - 1].pool
                elif key:
                    stopping, confirm = False, None
                view = view_of()
                if confirm is not None:
                    view = Group(view, Text(f" press s again to STOP the {confirm} pod (disk is kept) · any key cancels",
                                            style="bold black on yellow"))
                elif stopping:
                    pick = " · ".join(f"{i} {m.pool}" for i, m in enumerate(mons, 1))
                    view = Group(view, Text(f" stop which pod?  {pick}  · any other key cancels",
                                            style="bold black on yellow"))
                live.update(view)
    finally:
        if old:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        for m in monitors.values():
            m.close()
