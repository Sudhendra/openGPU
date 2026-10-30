"""ogpu — a personal H200 broker + shared bench service for agentic kernel hill-climbing."""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import agents, ledger, runner, ssh
from .config import CONFIG_PATH, LOG_DIR, load_config, load_state

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich",
                  help="[bold magenta]ogpu[/] — get an H200, share it safely between agents, never pay for idle.")
console = Console(stderr=True)

# Default $/hr for Modal when the ledger needs a cost estimate (Modal bills per second; override via env).
MODAL_RATE = {"H200": float(os.environ.get("OGPU_MODAL_H200_HR", 4.54)),
              "H100": float(os.environ.get("OGPU_MODAL_H100_HR", 3.95)), "none": 0.35}


@app.command()
def up(retry: float = typer.Option(0, help="keep retrying for capacity this many minutes")):
    """Reuse, resume, or create the H200 pod and wait until SSH works."""
    from . import pod
    pod.up(load_config(), console, retry_minutes=retry)


@app.command()
def down(terminate: bool = typer.Option(False, "--terminate", help="delete the pod and its disk (not just stop)")):
    """Stop the pod now (disk + caches kept). --terminate deletes it."""
    from . import pod
    from .config import State, save_state
    rp = pod.client()
    p = pod.current(rp, load_config())
    if not p:
        console.print("[grey50]no pod[/]")
        return
    if terminate:
        rp.terminate(p.id)
        save_state(State())
        console.print(f"[red]✘[/] terminated {p.id}")
    else:
        rp.stop(p.id)
        console.print(f"[yellow]■[/] stopped {p.id} — `ogpu up` resumes it")


@app.command()
def status(as_json: bool = typer.Option(False, "--json")):
    """Pod state, price, uptime, and today's spend."""
    from . import pod
    cfg = load_config()
    p = pod.current(pod.client(), cfg)
    with ledger.connect() as db:
        if p and p.uptime_s:
            ledger.record_session(db, p.id, p.uptime_s, p.cost_per_hr)
        today = ledger.spend(db, ledger.today_start())
    if p:
        pod.sync_state(p)
    d = {"pod": p.__dict__ if p else None, "today_usd": round(today, 2), "budget_usd": cfg.budget_usd}
    if as_json:
        print(json.dumps(d))
        return
    if not p:
        console.print("[grey50]no pod — `ogpu up`[/]")
    else:
        color = {"RUNNING": "green", "EXITED": "yellow"}.get(p.status, "red")
        console.print(f"[{color}]●[/] [bold]{p.id}[/] {p.status} · {p.gpu} · ${p.cost_per_hr:.2f}/hr · "
                      f"up {runner.fmt_dur(p.uptime_s or 0)} · ssh root@{p.ssh_host}:{p.ssh_port}")
    console.print(f"  today ${today:.2f} / ${cfg.budget_usd:.0f} cap")


@app.command(context_settings={"allow_interspersed_args": False})
def run(cmd: list[str] = typer.Argument(..., help="command to run on the pod (put it after --)"),
        gpu: bool = typer.Option(True, "--gpu/--cpu", help="--gpu takes the exclusive H200 lock; --cpu runs now, in parallel"),
        track: Path = typer.Option(Path("."), "--track", "-t", help="dir to sync and run in (default: cwd)"),
        with_: list[Path] = typer.Option([], "--with", "-w", help="extra dirs to sync next to the track"),
        force: bool = typer.Option(False, "--force", "-f", help="ignore the content-hash cache"),
        as_json: bool = typer.Option(False, "--json", help="machine output for agents (one JSON line)"),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="don't stream the log")):
    """Sync the track, run CMD on the pod (GPU-locked or CPU-parallel), stream the log, record the result."""
    command = cmd[0] if len(cmd) == 1 else shlex.join(cmd)
    rc = runner.run(load_config(), track=track, extra=with_, cmd=command, mode="gpu" if gpu else "cpu", force=force,
                    as_json=as_json, quiet=quiet)
    raise typer.Exit(rc)


@app.command()
def top():
    """Live dashboard: GPU telemetry, lock holder, queue, agents, spend, idle-stop countdown."""
    from .tui import run_top
    run_top(load_config())


@app.command()
def logs(job_id: str, follow: bool = typer.Option(False, "--follow", "-f")):
    """Print a job's log (local copy if present, else from the pod). -f follows a running job."""
    cfg = load_config()
    local = LOG_DIR / f"{job_id}.log"
    if follow:
        st = load_state()
        tmp = LOG_DIR / f"{job_id}.follow.log"
        tmp.unlink(missing_ok=True)
        status, rc, *_ = runner.poll_loop(console, st, cfg.private_key_path, job_id, tmp, stream=True)
        console.print(f"[bold]{status}[/] exit {rc}")
        return
    if local.exists():
        sys.stdout.write(local.read_text(errors="replace"))
        return
    p = ssh.run(load_state(), cfg.private_key_path, f"cat /workspace/.ogpu/jobs/{shlex.quote(job_id)}.log", check=False)
    sys.stdout.write(p.stdout or p.stderr)


@app.command()
def cancel(job_id: str):
    """Kill a queued or running job on the pod."""
    cfg = load_config()
    ssh.run(load_state(), cfg.private_key_path, f"{runner.JOBRUN} cancel {shlex.quote(job_id)}")
    console.print(f"[yellow]■[/] cancelled {job_id}")


@app.command("ledger")
def ledger_cmd(track: Optional[str] = typer.Option(None, "--track", "-t"),
               best: bool = typer.Option(False, "--best", help="passing runs sorted by best score"),
               agent: Optional[str] = typer.Option(None, "--agent"),
               limit: int = typer.Option(20, "--limit", "-n"),
               as_json: bool = typer.Option(False, "--json")):
    """What's been tried and what won — one place for every agent's results."""
    q, args = "select * from runs where 1=1", []
    if track:
        q, args = q + " and track=?", args + [track]
    if agent:
        q, args = q + " and agent=?", args + [agent]
    with ledger.connect() as db:
        rows = [dict(r) | {"metrics": json.loads(r["metrics"] or "{}")} for r in db.execute(q + " order by ended desc", args)]
    if best:
        def score(r):
            m = r["metrics"]
            vals = list((m.get("score") or {}).values()) + [s.get("ratio_vs_s0") for s in (m.get("sides") or {}).values()
                                                            if s.get("ratio_vs_s0")]
            return max(vals) if vals else -1
        rows = [r for r in rows if r["status"] == "done" and r["metrics"].get("checks", "PASS") == "PASS"
                and not r["cached_from"] and score(r) > 0]
        rows.sort(key=score, reverse=True)
    rows = rows[:limit]
    if as_json:
        print(json.dumps(rows))
        return
    t = Table(box=None, header_style="bold magenta", pad_edge=False)
    for c in ("when", "agent", "track", "mode", "result", "wall", "$", "id", "cmd"):
        t.add_column(c, overflow="ellipsis", no_wrap=c != "cmd", max_width=48 if c == "cmd" else None)
    now = time.time()
    for r in rows:
        mark = "⚡" if r["cached_from"] else ("[green]✔[/]" if r["status"] == "done" else "[red]✘[/]")
        t.add_row(runner.fmt_dur(now - (r["ended"] or now)) + " ago", f"[{agents.color(r['agent'])}]{r['agent']}[/]",
                  r["track"], r["mode"], f"{mark} {ledger.headline(r['metrics'])}", runner.fmt_dur(r["wall_s"] or 0),
                  f"{r['cost_usd'] or 0:.2f}", r["id"], r["cmd"])
    Console().print(t)


@app.command()
def cost():
    """Spend: today, last 7 days, all time, and the burn rate right now."""
    cfg = load_config()
    st = load_state()
    with ledger.connect() as db:
        today = ledger.spend(db, ledger.today_start())
        week = ledger.spend(db, time.time() - 7 * 86400)
        total = ledger.spend(db)
        runs = db.execute("select count(*) n, sum(cached_from is not null) hits, sum(cost_usd) c from runs").fetchone()
    console.print(f"[bold]today[/] ${today:.2f} / ${cfg.budget_usd:.0f} cap   [bold]7d[/] ${week:.2f}   "
                  f"[bold]all[/] ${total:.2f}   [bold]rate[/] ${st.cost_per_hr:.2f}/hr while running")
    console.print(f"[grey50]{runs['n'] or 0} runs · {runs['hits'] or 0} cache hits · ${runs['c'] or 0:.2f} attributed to jobs "
                  "(the rest is idle/boot time)[/]")


@app.command()
def gpus(filter_: str = typer.Argument("H", metavar="FILTER")):
    """List RunPod GPU type ids + prices (to fill gpu_type_ids in the config)."""
    from . import pod
    t = Table(box=None, header_style="bold magenta")
    for c in ("id", "memory", "secure $/hr", "community $/hr"):
        t.add_column(c)
    for g in pod.client().gpu_types():
        if filter_.lower() in (g["id"] + g["displayName"]).lower():
            t.add_row(g["id"], f"{g['memoryInGb']} GB", str(g.get("securePrice") or "—"), str(g.get("communityPrice") or "—"))
    Console().print(t)


@app.command()
def fanout(cmds_file: Path,
           backend: str = typer.Option("modal", "--backend", "-b", help="modal (parallel containers) or runpod (queued on the pod)"),
           gpu: str = typer.Option("H200", "--gpu", help="modal GPU: H200, H100 (same sm_90a; correctness/compile), none"),
           track: Path = typer.Option(Path("."), "--track", "-t")):
    """Run one command per line of CMDS_FILE in parallel and record every result in the ledger."""
    cfg = load_config()
    track = track.resolve()
    cmds = [ln.strip() for ln in cmds_file.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    agent = agents.detect()
    if backend == "runpod":
        argv0 = [sys.executable, "-m", "ogpu", "run", "--quiet", "--json", "--track", str(track), "--"]
        with cf.ThreadPoolExecutor(len(cmds)) as ex:
            futs = {ex.submit(subprocess.run, argv0 + [c], capture_output=True, text=True): c for c in cmds}
            for f in cf.as_completed(futs):
                out = f.result().stdout.strip().splitlines()
                r = json.loads(out[-1]) if out else {"status": "error"}
                mark = "[green]✔[/]" if r.get("status") == "done" else "[red]✘[/]"
                console.print(f"{mark} {futs[f][:60]} · "
                              f"{ledger.headline(r.get('metrics') or {})}")
        return
    stamp = time.strftime("%m%d_%H%M%S")
    results_path = track / "output" / "ogpu-modal" / stamp / "results.json"
    env = os.environ | {"OGPU_TRACK": str(track), "OGPU_MODAL_GPU": gpu, "OGPU_RESULTS": str(results_path),
                        "OGPU_IMAGE": cfg.image}
    app_path = Path(__file__).parent / "providers" / "modal_app.py"
    console.print(f"[cyan]fanout[/] {len(cmds)} jobs → modal {gpu} · track {track.name}")
    rc = subprocess.run(["modal", "run", str(app_path), "--cmds", str(cmds_file.resolve())], env=env).returncode
    if not results_path.exists():
        raise typer.Exit(rc or 1)
    rate = MODAL_RATE.get(gpu, MODAL_RATE["H200"])
    with ledger.connect() as db:
        for r in json.loads(results_path.read_text()):
            if "error" in r:
                console.print(f"[red]✘[/] {r['error']}")
                continue
            log = Path(r["log_path"]).read_text(errors="replace")
            m = ledger.parse_metrics(log)
            key = ledger.content_key([track], r["cmd"], f"modal-{gpu}", cfg.image)
            row = {"id": f"m{stamp}-{r['i']}-{uuid.uuid4().hex[:4]}", "key": key, "agent": agent, "track": track.name,
                   "cmd": r["cmd"], "mode": f"modal-{gpu}", "backend": "modal", "status": "done" if r["rc"] == 0 else "failed",
                   "exit_code": r["rc"], "started": time.time() - r["wall_s"], "ended": time.time(), "wall_s": r["wall_s"],
                   "cost_usd": r["wall_s"] / 3600 * rate, "metrics": m, "log_path": r["log_path"], "cached_from": None}
            ledger.record(db, **row)
            runner.result_line(console, row)


@app.command()
def config():
    """Show the config file path and contents."""
    load_config()
    console.print(f"[bold]{CONFIG_PATH}[/]")
    print(CONFIG_PATH.read_text())


if __name__ == "__main__":
    app()
