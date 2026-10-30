"""`ogpu top`: a small, colorful live view of the pod — GPU telemetry, who holds the H200, who is queued, spend."""
from __future__ import annotations

import json
import os
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
    """Background refresh of the job snapshot (SSH) and pod info (RunPod API)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.snap: dict | None = None
        self.pod = None
        self.err = ""
        self._stop = False
        threading.Thread(target=self._snap_loop, daemon=True).start()
        threading.Thread(target=self._pod_loop, daemon=True).start()

    def _snap_loop(self):
        while not self._stop:
            try:
                st = load_state()
                p = ssh.run(st, self.cfg.private_key_path, f"{JOBRUN} snapshot", timeout=20)
                self.snap, self.err = json.loads(p.stdout), ""
            except Exception as e:  # noqa: BLE001 — keep the dashboard alive on any transient failure
                self.err = f"snapshot: {type(e).__name__}"
            time.sleep(2)

    def _pod_loop(self):
        if not os.environ.get("RUNPOD_API_KEY"):
            return
        from .pod import client, sync_state
        rp = client()
        while not self._stop:
            try:
                st = load_state()
                self.pod = rp.pod(st.pod_id) if st.pod_id else None
                if self.pod:
                    sync_state(self.pod)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(15)

    def close(self):
        self._stop = True


def render(cfg: Config, tel: Telemetry, poll: Poller, width: int) -> Group:
    st = load_state()
    now = time.time()
    rows: list = []

    # header: pod · status · cloud · price · uptime · spend vs cap
    pod = poll.pod
    status = pod.status if pod else ("RUNNING" if tel.latest else "UNKNOWN")
    up = fmt_dur(pod.uptime_s) if pod and pod.uptime_s else "—"
    with ledger.connect() as db:
        today = ledger.spend(db, ledger.today_start())
    cap_style = "red" if today >= cfg.budget_usd else "yellow" if today >= 0.8 * cfg.budget_usd else "green"
    h = Text() + Text(" ogpu ", style="bold black on bright_magenta") + Text(f" {cfg.pod_name} ", style="bold")
    h += Text(f" {status} ", style=STATUS_STYLE.get(status, "bold black on grey50"))
    h += Text(f"  runpod/{st.cloud or '?'}  ", style="grey70") + Text(f"${st.cost_per_hr:.2f}/hr", style="bold cyan")
    h += Text(f"  up {up}  ", style="grey70") + Text("today ", style="grey70")
    h += Text(f"${today:.2f}", style=f"bold {cap_style}") + Text(f" / ${cfg.budget_usd:.0f} cap", style="grey50")
    rows.append(h)
    rows.append(Text(""))

    # GPU telemetry
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
        rows.append(Text(f"      {s.name}", style="grey50"))
    else:
        rows.append(Text() + Text(" GPU  ", style="bold") + Text(tel.error or "connecting…", style="grey50"))
    rows.append(Text(""))

    # who's using the pod
    snap = poll.snap or {}
    jobs = snap.get("jobs") or []
    holder = snap.get("holder")
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold", width=6)
    t.add_column()
    t.add_column(style="grey70")
    t.add_column(justify="right")
    if holder:
        t.add_row("LOCK", agent_text(holder["agent"], holder.get("track")), short(holder["cmd"]),
                  Text(f"running {fmt_dur(now - holder['since'])}", style="bold green"))
    else:
        t.add_row("LOCK", Text("free", style="green"), "", "")
    for j in [j for j in jobs if j["status"] == "queued"]:
        t.add_row("QUEUE", agent_text(j["agent"], j["track"]), short(j["cmd"]),
                  Text(f"#{j.get('queue_pos', '?')} waiting {fmt_dur(now - j['created'])}", style="yellow"))
    for j in [j for j in jobs if j["status"] == "running" and j["mode"] == "cpu"]:
        t.add_row("CPU", agent_text(j["agent"], j["track"]), short(j["cmd"]),
                  Text(f"{fmt_dur(now - (j['started'] or now))}", style="cyan"))
    rows.append(t)
    rows.append(Text(""))

    # recent results (from the local ledger, so they carry parsed metrics)
    with ledger.connect() as db:
        recent = db.execute("select * from runs where status not in ('queued','running') "
                            "order by ended desc limit 5").fetchall()
    rt = Table.grid(padding=(0, 2))
    rt.add_column(style="bold", width=6)
    rt.add_column()
    for i, r in enumerate(recent):
        mark = Text("⚡", style="yellow") if r["cached_from"] else (
            Text("✔", style="bold green") if r["status"] == "done" and not r["failure"] else Text("✘", style="bold red"))
        line = Text() + mark + Text(" ") + agent_text(r["agent"], r["track"])
        hl = ledger.headline(json.loads(r["metrics"] or "{}"))
        if r["failure"] and not r["cached_from"]:
            hl = f"{r['failure']}  {hl}".strip()
        if hl:
            line += Text(f"  {hl}", style="bold")
        line += Text(f"  {fmt_dur(r['wall_s'] or 0)} · ${r['cost_usd'] or 0:.2f} · {fmt_dur(now - (r['ended'] or now))} ago",
                     style="grey50")
        rt.add_row("RECENT" if i == 0 else "", line)
    if recent:
        rows.append(rt)
        rows.append(Text(""))

    # where today's money went
    with ledger.connect() as db:
        u = ledger.utilization(db, ledger.today_start())
        per_agent = ledger.agent_spend(db, ledger.today_start())
    if u["paid_s"]:
        frac = u["gpu_busy_frac"]
        fs = "bold green" if frac >= 0.6 else "bold yellow" if frac >= 0.3 else "bold red"
        line = Text() + Text("TODAY   ", style="bold") + Text("GPU busy ", style="grey70") + Text(f"{frac:.0%}", style=fs)
        line += Text(" of paid time · idle ", style="grey70") + Text(f"${u['idle_usd']:.2f}", style="bold yellow")
        for a, c in list(per_agent.items())[:4]:
            line += Text("  ") + agent_text(a) + Text(f" ${c:.2f}", style="grey70")
        rows.append(line)
        rows.append(Text(""))

    # idle-stop countdown + keys
    reaper = snap.get("reaper") or {}
    if reaper.get("busy") or holder or any(j["status"] in ("queued", "running") for j in jobs):
        idle = Text("idle-stop  —  (busy)", style="grey50")
    elif reaper.get("stop_at"):
        left = reaper["stop_at"] - now
        idle = Text() + Text("idle-stop in ", style="grey50") + Text(fmt_dur(max(0, left)),
                                                             style="bold yellow" if left < 300 else "bold")
    else:
        idle = Text("idle-stop  ?", style="grey50")
    foot = Text() + idle + Text("   " + (poll.err or ""), style="red")
    keys = Text("q quit · s stop pod · r refresh", style="grey50")
    pad = max(1, width - len(foot.plain) - len(keys.plain) - 1)
    rows.append(foot + Text(" " * pad) + keys)
    return Group(*rows)


def run_top(cfg: Config) -> None:
    console = Console()
    st = load_state()
    if not st.reachable:
        raise SystemExit("no pod recorded yet — run `ogpu up` first")
    tel = Telemetry(ssh.ssh_argv(st, cfg.private_key_path, SMI))
    poll = Poller(cfg)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd) if sys.stdin.isatty() else None
    try:
        if old:
            tty.setcbreak(fd)
        with Live(render(cfg, tel, poll, console.width), console=console, refresh_per_second=4, screen=False) as live:
            confirm_stop = False
            while True:
                key = ""
                if old and select.select([sys.stdin], [], [], 0.5)[0]:
                    key = sys.stdin.read(1).lower()
                elif not old:
                    time.sleep(0.5)
                if key == "q":
                    break
                if key == "s":
                    if confirm_stop:
                        from .pod import client
                        client().stop(load_state().pod_id)
                        confirm_stop = False
                    else:
                        confirm_stop = True
                elif key:
                    confirm_stop = False
                view = render(cfg, tel, poll, console.width)
                if confirm_stop:
                    view = Group(view, Text(" press s again to STOP the pod (disk is kept) · any key cancels",
                                            style="bold black on yellow"))
                live.update(view)
    finally:
        if old:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        tel.close()
        poll.close()
