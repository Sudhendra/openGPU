"""ogpu — the GPU savings engine for coding agents: cheapest capable machine, shared safely, never idle."""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shlex
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import agents, ledger, runner, ssh
from .config import CONFIG_PATH, LOG_DIR, REMOTE_OGPU, load_config, load_state
from .providers import SERVERLESS
from .providers import modal as modal_backend

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich",
                  help="[bold magenta]ogpu[/] — cheap GPUs for coding agents: shared safely, never paid for idle.")
console = Console(stderr=True)

POOL_OPT = typer.Option("timing", "--pool", "-p", help="timing (target GPU) or correctness (cheaper GPU pool)")


def _pool_cfg(pool: str, machine: bool = True):
    from .config import POOLS
    if pool not in POOLS:
        raise typer.BadParameter(f"unknown pool {pool!r} (choose from {', '.join(POOLS)})")
    pcfg = load_config().pool(pool)
    if pcfg is None:
        raise SystemExit(f"pool {pool!r} is not configured in {CONFIG_PATH}")
    if machine and pcfg.provider in SERVERLESS:
        raise SystemExit(f"the {pool} pool runs on {pcfg.provider} (serverless): there is no machine to manage")
    return pcfg


@app.command()
def up(retry: float = typer.Option(0, help="keep retrying for capacity this many minutes"), pool: str = POOL_OPT):
    """Reuse, resume, or create the pool's machine and wait until SSH works."""
    import fcntl

    from . import pod
    from .config import HOME
    pcfg = _pool_cfg(pool)
    with open(HOME / pod.up_lock(pool), "w") as lk:  # same lock as auto-up and background warm-up
        fcntl.flock(lk, fcntl.LOCK_EX)
        pod.up(pcfg, console, retry_minutes=retry, pool=pool)


@app.command()
def down(terminate: bool = typer.Option(False, "--terminate", help="delete the pod and its disk (not just stop)"),
         pool: str = POOL_OPT):
    """Stop the pool's machine now (disk + caches kept). --terminate deletes it."""
    from . import pod
    from .config import State, save_state
    pcfg = _pool_cfg(pool)
    rp = pod.client(pcfg)
    p = pod.current(rp, pcfg, pool)
    if not p:
        console.print(f"[grey50]no {pool} pod[/]")
        return
    if p.uptime_s:  # last sample of paid time before the pod goes away
        with ledger.connect() as db:
            ledger.record_session(db, p.id, p.uptime_s, p.cost_per_hr)
    if terminate:
        rp.terminate(p.id)
        save_state(State(), pool)
        console.print(f"[red]✘[/] terminated {pool} pod {p.id}")
    else:
        rp.stop(p.id)
        console.print(f"[yellow]■[/] stopped {pool} pod {p.id} — `ogpu up --pool {pool}` resumes it")


@app.command()
def status(as_json: bool = typer.Option(False, "--json")):
    """Every pool's machine: provider, state, price, uptime; plus today's spend."""
    from . import pod
    from .config import POOLS
    cfg = load_config()
    pods, serverless = {}, {}
    for name in POOLS:
        pcfg = cfg.pool(name)
        if pcfg is None:
            continue
        if pcfg.provider in SERVERLESS:
            serverless[name] = pcfg.provider
            continue
        p = pod.current(pod.client(pcfg), pcfg, name)
        if p:
            pod.sync_state(p, pool=name, provider=pcfg.provider)
        pods[name] = p
    with ledger.connect() as db:
        runner.settle_serverless(cfg, db)
        today = ledger.spend(db, ledger.today_start())
    if as_json:
        print(json.dumps({"pods": {n: (p.__dict__ if p else None) for n, p in pods.items()}, "serverless": serverless,
                          "today_usd": round(today, 2), "budget_usd": cfg.budget_usd}))
        return
    for name, prov in serverless.items():
        console.print(f"[cyan]◇[/] [magenta]{name:<11}[/] {prov} (serverless: billed per job, nothing idles)")
    for name, p in pods.items():
        if not p:
            console.print(f"[grey50]○ {name:<11} no machine[/]")
            continue
        color = {"RUNNING": "green", "EXITED": "yellow"}.get(p.status, "red")
        console.print(f"[{color}]●[/] [magenta]{name:<11}[/] [bold]{p.id}[/] {p.status} · {p.gpu} · "
                      f"${p.cost_per_hr:.2f}/hr · up {runner.fmt_dur(p.uptime_s or 0)} · "
                      f"ssh {p.ssh_user}@{p.ssh_host}:{p.ssh_port}")
    console.print(f"  today ${today:.2f} / ${cfg.budget_usd:.0f} cap")


CMD_ARG = typer.Argument(..., help="command to run on the GPU machine (put it after --)")
GPU_OPT = typer.Option(True, "--gpu/--cpu", help="--gpu takes the exclusive GPU lock; --cpu runs now, in parallel")
TRACK_OPT = typer.Option(Path("."), "--track", "-t", help="dir to sync and run in (default: cwd)")
WITH_OPT = typer.Option([], "--with", "-w", help="extra dirs to sync next to the track")
CACHE_OPT = typer.Option(None, "--cache/--no-cache", help="reuse an identical earlier run instead of paying again "
                         "(default: `cache` in the config, off unless set)")
NOTE_OPT = typer.Option("", "--note", "-m", help="hypothesis / what changed (stored in the shared ledger)")
TIMEOUT_OPT = typer.Option(None, "--timeout", help="kill the job after this many seconds (exit 124)")
FIDELITY_OPT = typer.Option("timing", "--fidelity", help="timing: run on the target GPU. correctness: may run on the "
                            "cheaper correctness pool — use for checks / numerics, never for timings you trust")


def _fidelity(f: str) -> str:
    if f not in ("timing", "correctness"):
        raise typer.BadParameter("--fidelity must be timing or correctness")
    return f


def _job_pool(job_id: str) -> str:
    with ledger.connect() as db:
        row = ledger.get(db, job_id)
    return (row["pool"] if row and row["pool"] else "timing")


def _command(cmd: list[str]) -> str:
    return cmd[0] if len(cmd) == 1 else shlex.join(cmd)


@app.command(context_settings={"allow_interspersed_args": False})
def run(cmd: list[str] = CMD_ARG, gpu: bool = GPU_OPT, track: Path = TRACK_OPT, with_: list[Path] = WITH_OPT,
        cache: Optional[bool] = CACHE_OPT, note: str = NOTE_OPT, timeout: Optional[float] = TIMEOUT_OPT,
        fidelity: str = FIDELITY_OPT, as_json: bool = typer.Option(False, "--json", help="machine output for agents (one JSON line)"),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="don't stream the log")):
    """Sync the track, run CMD on the GPU machine (GPU-locked or CPU-parallel), stream the log, record the result.

    Exit code: the job's own, 2 = daily budget cap hit, 3 = benchmark passed but correctness checks FAILED."""
    rc = runner.run(load_config(), track=track, extra=with_, cmd=_command(cmd), mode="gpu" if gpu else "cpu",
                    cache=cache, as_json=as_json, quiet=quiet, note=note, timeout_s=timeout,
                    fidelity=_fidelity(fidelity))
    raise typer.Exit(rc)


@app.command(context_settings={"allow_interspersed_args": False})
def submit(cmd: list[str] = CMD_ARG, gpu: bool = GPU_OPT, track: Path = TRACK_OPT, with_: list[Path] = WITH_OPT,
           cache: Optional[bool] = CACHE_OPT, note: str = NOTE_OPT, timeout: Optional[float] = TIMEOUT_OPT,
           fidelity: str = FIDELITY_OPT):
    """Queue CMD and return immediately with its id (JSON). Collect with `ogpu wait <id>...`.

    With --cache, a hit returns the finished result right away (status done/failed)."""
    con = Console(stderr=True)
    try:
        r = runner.submit(load_config(), con, track=track, extra=with_, cmd=_command(cmd),
                          mode="gpu" if gpu else "cpu", cache=cache, note=note, timeout_s=timeout,
                          fidelity=_fidelity(fidelity))
    except runner.BudgetExceeded as e:
        print(json.dumps({"status": "budget_exceeded", "error": e.msg}))
        raise typer.Exit(2)
    print(json.dumps({k: r.get(k) for k in runner.RESULT_KEYS if k in r} | {"deduped": r.get("deduped", False)}))


@app.command()
def wait(ids: list[str] = typer.Argument(..., help="job ids from `ogpu submit`"),
         as_json: bool = typer.Option(True, "--json/--pretty", help="one JSON line per job (default) or pretty")):
    """Wait for submitted jobs; print one result per job as each finishes (exit = worst job exit code)."""
    cfg = load_config()
    con = Console(stderr=True, quiet=as_json)
    worst = 0
    with cf.ThreadPoolExecutor(max(1, len(ids))) as ex:
        quiet = Console(stderr=True, quiet=True)  # parallel waits: no spinners (Rich allows one live display)
        futs = {ex.submit(runner.wait, cfg, quiet, i, stream=False): i for i in ids}
        for f in cf.as_completed(futs):
            try:
                r = f.result()
                rc = runner.emit(con, r, as_json)
            except SystemExit as e:
                print(json.dumps({"id": futs[f], "status": "error", "error": str(e)}))
                rc = 1
            worst = max(worst, rc)
    raise typer.Exit(worst)


@app.command(context_settings={"allow_interspersed_args": True})
def sweep(cands: list[str] = typer.Argument(..., help="candidate files / variants (file.py@NAME=VAL) to compare"),
          best: Optional[str] = typer.Option(None, "--best", "-b", help="current best (timed alongside, never dropped)"),
          bench: Optional[str] = typer.Option(None, "--bench", help="bench command; candidates are appended (or "
                                              "replace {files}). Required unless profile = okc"),
          compile_tpl: Optional[str] = typer.Option(None, "--compile", help="per-candidate CPU gate; {file} = path"),
          check_flag: str = typer.Option("--check", "--check-flag", help="flag that makes the bench run checks"),
          correct_args: str = typer.Option("--reps 3", "--correct-args", help="extra args for the correctness stage"),
          quick_args: str = typer.Option("--reps 5", "--quick-args", help="args for the quick timing stage"),
          full_args: str = typer.Option("--reps 20", "--full-args", help="args for the full timing stage"),
          top_k: int = typer.Option(2, "--top-k", "-k", help="finalists kept after quick timing"),
          noise: float = typer.Option(0.015, "--noise", help="relative win needed to beat best / stay in contention"),
          track: Path = TRACK_OPT, with_: list[Path] = WITH_OPT, note: str = NOTE_OPT,
          timeout: Optional[float] = TIMEOUT_OPT, cache: Optional[bool] = CACHE_OPT,
          as_json: bool = typer.Option(False, "--json", help="one JSON report on stdout")):
    """Tournament: compile (CPU) → correctness (cheap pool) → quick timing → full timing for finalists → ranked verdict.

    Most candidates never reach the expensive GPU, and each stage is one batched job (prepare paid once)."""
    from . import sweep as sw
    con = Console(stderr=True, quiet=as_json)
    cfg = load_config()
    bench = bench or ("python3 pod/bench.py" if cfg.profile == "okc" else None)
    if not bench:
        raise typer.BadParameter("--bench is required (e.g. --bench 'python3 bench.py {files} --fast')")
    try:
        run = sw.Sweep(cfg, con, track=track.resolve(), extra=with_, cands=cands, best=best, bench=bench,
                       compile_tpl=compile_tpl, check_flag=check_flag, correct_args=correct_args,
                       quick_args=quick_args, full_args=full_args, top_k=top_k, noise=noise, note=note,
                       timeout_s=timeout, cache=cache)
        rep = run.run()
    except runner.BudgetExceeded as e:
        print(json.dumps({"status": "budget_exceeded", "error": e.msg}))
        raise typer.Exit(2)
    if as_json:
        print(json.dumps(rep))
    else:
        sw.print_report(Console(), rep)


@app.command()
def top(pool: Optional[str] = typer.Option(None, "--pool", "-p", help="show only this pool (default: all)")):
    """Live dashboard for every machine: GPU telemetry, lock holder, queue, agents, spend, idle-stop countdowns."""
    from .tui import run_top
    if pool:
        _pool_cfg(pool)
    run_top(load_config(), pool)


@app.command()
def logs(job_id: str, follow: bool = typer.Option(False, "--follow", "-f")):
    """Print a job's log (local copy if present, else from the machine). -f follows a running job."""
    cfg = load_config()
    local = LOG_DIR / f"{job_id}.log"
    if follow:
        pool = _job_pool(job_id)
        tmp = LOG_DIR / f"{job_id}.follow.log"
        tmp.unlink(missing_ok=True)
        status, rc, *_ = runner.poll_loop(console, load_state(pool), cfg.private_key_path, job_id, tmp, stream=True,
                                          pool=pool)
        console.print(f"[bold]{status}[/] exit {rc}")
        return
    if local.exists():
        sys.stdout.write(local.read_text(errors="replace"))
        return
    p = ssh.run(load_state(_job_pool(job_id)), cfg.private_key_path,
                f"cat {REMOTE_OGPU}/jobs/{shlex.quote(job_id)}.log", check=False)
    sys.stdout.write(p.stdout or p.stderr)


@app.command()
def cancel(job_id: str):
    """Kill a queued or running job."""
    cfg = load_config()
    with ledger.connect() as db:
        row = ledger.get(db, job_id)
    if row and row["handle"]:  # serverless: interrupting `modal run` stops its containers
        try:
            os.killpg(json.loads(row["handle"])["pid"], signal.SIGINT)
        except OSError:
            pass
        if row["status"] not in ledger.TERMINAL:
            with ledger.connect() as db:
                ledger.update(db, job_id, status="cancelled", exit_code=-15, ended=time.time())
    else:
        ssh.run(load_state(_job_pool(job_id)), cfg.private_key_path, f"{runner.JOBRUN} cancel {shlex.quote(job_id)}")
    console.print(f"[yellow]■[/] cancelled {job_id}")


@app.command("ledger")
def ledger_cmd(track: Optional[str] = typer.Option(None, "--track", "-t"),
               best: bool = typer.Option(False, "--best", help="passing runs sorted by best score"),
               failed: bool = typer.Option(False, "--failed", help="only failures (with failure type + note)"),
               agent: Optional[str] = typer.Option(None, "--agent"),
               limit: int = typer.Option(20, "--limit", "-n"),
               as_json: bool = typer.Option(False, "--json")):
    """What's been tried and what won — one place for every agent's results and hypotheses."""
    q, args = "select * from runs where status not in ('queued','running')", []
    if track:
        q, args = q + " and track=?", args + [track]
    if agent:
        q, args = q + " and agent=?", args + [agent]
    if failed:
        q += " and failure is not null and cached_from is null"
    with ledger.connect() as db:
        rows = [runner.as_result(r) for r in db.execute(q + " order by ended desc", args)]
    if best:
        rows = [r for r in rows if r["status"] == "done" and not r.get("failure") and not r["cached_from"]
                and ledger.best_score(r["metrics"]) is not None]
        rows.sort(key=lambda r: ledger.best_score(r["metrics"]), reverse=True)
    rows = rows[:limit]
    if as_json:
        keep = ("id", "agent", "track", "cmd", "mode", "pool", "status", "failure", "error", "note", "metrics",
                "wall_s", "cost_usd", "ended", "cached_from")
        print(json.dumps([{k: r.get(k) for k in keep} for r in rows]))
        return
    t = Table(box=None, header_style="bold magenta", pad_edge=False)
    for c in ("when", "agent", "track", "result", "note", "$", "id"):
        t.add_column(c, overflow="ellipsis", no_wrap=c != "note", max_width=40 if c == "note" else None)
    now = time.time()
    for r in rows:
        if r["cached_from"]:
            mark = "⚡"
        elif r.get("failure"):
            mark = f"[red]✘ {r['failure']}[/]"
        else:
            mark = "[green]✔[/]"
        t.add_row(runner.fmt_dur(now - (r["ended"] or now)) + " ago", f"[{agents.color(r['agent'])}]{r['agent']}[/]",
                  r["track"], f"{mark} {ledger.headline(r['metrics'])}", r.get("note") or "",
                  f"{r['cost_usd'] or 0:.2f}", r["id"])
    Console().print(t)


@app.command()
def show(job_id: str, as_json: bool = typer.Option(False, "--json")):
    """Everything about one run: command, note, metrics, failure + error excerpt, log tail."""
    with ledger.connect() as db:
        row = ledger.get(db, job_id)
    if not row:
        raise SystemExit(f"unknown job {job_id}")
    r = runner.as_result(row)
    if as_json:
        print(json.dumps(r))
        return
    c = Console()
    c.print(f"[bold]{r['id']}[/] [{agents.color(r['agent'])}]{r['agent']}[/] {r['track']} · {r['mode']} · {r['status']}"
            + (f" · [red]{r['failure']}[/]" if r.get("failure") else ""))
    c.print(Text("$ ", style="grey50") + Text(r["cmd"]))
    if r.get("note"):
        c.print(f"[bold]note[/] {r['note']}")
    if r["metrics"]:
        c.print(f"[bold]metrics[/] {ledger.headline(r['metrics'])}")
        c.print_json(json.dumps(r["metrics"]))
    if r.get("error"):
        c.print("[bold red]error[/]")
        c.print(r["error"], markup=False, highlight=False, style="red")
    lp = Path(r["log_path"] or "")
    if lp.is_file():
        c.print(f"[bold]log tail[/] [grey50]{lp}[/]")
        c.print("\n".join(lp.read_text(errors="replace").splitlines()[-15:]), markup=False, highlight=False)


@app.command()
def cost(as_json: bool = typer.Option(False, "--json")):
    """Spend today / 7d / all, how much of the paid time the GPU was actually busy, and spend per agent."""
    cfg = load_config()
    st = load_state()
    t0 = ledger.today_start()
    with ledger.connect() as db:
        runner.settle_serverless(cfg, db)
        today, week, total = ledger.spend(db, t0), ledger.spend(db, time.time() - 7 * 86400), ledger.spend(db)
        u = ledger.utilization(db, t0)
        per_agent = ledger.agent_spend(db, t0)
        runs = db.execute("select count(*) n, sum(cached_from is not null) hits from runs where ended > ?",
                          (t0,)).fetchone()
    if as_json:
        print(json.dumps({"today_usd": today, "week_usd": week, "total_usd": total, "budget_usd": cfg.budget_usd,
                          "utilization_today": u, "agent_usd_today": per_agent}))
        return
    console.print(f"[bold]today[/] ${today:.2f} / ${cfg.budget_usd:.0f} cap   [bold]7d[/] ${week:.2f}   "
                  f"[bold]all[/] ${total:.2f}   [bold]rate[/] ${st.cost_per_hr:.2f}/hr while running")
    frac = u["gpu_busy_frac"]
    style = "green" if frac >= 0.6 else "yellow" if frac >= 0.3 else "red"
    console.print(f"[bold]GPU busy[/] [{style}]{frac:.0%}[/] of paid time today · [bold]idle[/] "
                  f"${u['idle_usd']:.2f} (boot, agents thinking, reaper wait) · {runs['n'] or 0} runs, "
                  f"{runs['hits'] or 0} cache hits")
    if per_agent:
        console.print("[bold]by agent[/] " + "  ".join(f"[{agents.color(a)}]{a}[/] ${c:.2f}" for a, c in per_agent.items()))


@app.command()
def gpus(filter_: str = typer.Argument("", metavar="FILTER"),
         provider: Optional[str] = typer.Option(None, "--provider", help="default: the config's provider")):
    """List a provider's GPU ids + prices (to fill gpu_type_ids in the config)."""
    from . import providers
    cfg = load_config()
    name = provider or cfg.provider
    rows = modal_backend.gpu_types(cfg) if name == "modal" else providers.load(name, cfg).gpu_types()
    t = Table(box=None, header_style="bold magenta")
    for c in ("id", "memory", "$/hr"):
        t.add_column(c)
    for g in rows:
        if filter_.lower() in str(g["id"]).lower():
            t.add_row(str(g["id"]), f"{g['memory_gb']} GB" if g["memory_gb"] else "", str(g["price"] or "—"))
    Console().print(t)


@app.command()
def fanout(cmds_file: Path,
           provider: Optional[str] = typer.Option(None, "--provider", help="modal: one container per command; a machine "
                                                  "provider: queued on the pool's machine (default: the config's)"),
           gpu: Optional[str] = typer.Option(None, "--gpu", help="modal GPU (H200, H100, L4, ..., none)"),
           track: Path = typer.Option(Path("."), "--track", "-t")):
    """Run one command per line of CMDS_FILE in parallel and record every result in the ledger."""
    cfg = load_config()
    track = track.resolve()
    cmds = [ln.strip() for ln in cmds_file.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    provider = provider or cfg.provider
    if provider not in SERVERLESS:
        quiet = Console(stderr=True, quiet=True)
        subs = [runner.submit(cfg, console, track=track, extra=[], cmd=c, mode="gpu") for c in cmds]
        with cf.ThreadPoolExecutor(max(1, len(subs))) as ex:
            futs = [ex.submit(lambda s: s if s["status"] != "queued" else runner.wait(cfg, quiet, s["id"], stream=False),
                              s) for s in subs]
            for f in cf.as_completed(futs):
                runner.emit(console, f.result(), False)
        return
    gpu = gpu or (cfg.gpu_type_ids[0] if cfg.provider == "modal" and cfg.gpu_type_ids else modal_backend.DEFAULT_GPU)
    try:
        with ledger.connect() as db:
            runner.check_budget(cfg, db)
    except runner.BudgetExceeded as e:
        console.print(f"[bold red]✘ budget[/] {e.msg}")
        raise typer.Exit(2)
    stamp = time.strftime("%m%d_%H%M%S")
    results_path = track / "output" / "ogpu-modal" / stamp / "results.json"
    console.print(f"[cyan]fanout[/] {len(cmds)} jobs → modal {gpu} · track {track.name}")
    rc = modal_backend.launch(cfg, track=track, extra=[], cmds=cmds, gpu=gpu, results=results_path, log=Path(),
                              timeout_s=None, detach=False)
    if not results_path.exists():
        raise typer.Exit(rc or 1)
    agent = agents.detect()
    with ledger.connect() as db:
        for r in json.loads(results_path.read_text()):
            if "error" in r:
                console.print(f"[red]✘[/] {r['error']}")
                continue
            log = Path(r["log_path"]).read_text(errors="replace")
            ended = r.get("ended") or time.time()
            row = {"id": f"m{stamp}-{r['i']}-{uuid.uuid4().hex[:4]}", "key": None, "agent": agent, "track": track.name,
                   "cmd": r["cmd"], "mode": "gpu" if gpu != "none" else "cpu", "backend": "modal",
                   "status": "done" if r["rc"] == 0 else "failed", "exit_code": r["rc"], "started": ended - r["wall_s"],
                   "ended": ended, "wall_s": r["wall_s"],
                   "cost_usd": r.get("billed_s", r["wall_s"]) / 3600 * modal_backend.rate(gpu, *modal_backend.container(cfg)),
                   "metrics": ledger.parse_metrics(log, cfg.profile), "log_path": r["log_path"], "cached_from": None}
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
