"""`ogpu run` / `submit` / `wait`: sync → submit to the on-machine runner → poll (survives SSH drops) → pull out/ →
classify → ledger. Serverless pools (Modal) launch one container per job instead. `run` = submit + wait."""
from __future__ import annotations

import base64
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from rich.console import Console
from rich.text import Text

from . import agents, failures, ledger, notify, pod, ssh
from .config import HOME, LOG_DIR, REMOTE_OGPU, Config, State, load_state
from .providers import SERVERLESS
from .providers import modal as modal_backend

JOBRUN = f"python3 {REMOTE_OGPU}/bin/jobrun.py"
RESULT_KEYS = ("id", "status", "failure", "error", "exit_code", "wall_s", "cost_usd", "metrics", "note", "pool",
               "confirm_on_timing", "advice", "log_path", "cached_from")
NEAR_BOUND = 0.8  # correctness-pool passes above this fraction of the bound get re-confirmed on the timing pool


def remote_dir(cfg: Config, local: Path) -> str:
    return f"{cfg.remote_root}/{local.resolve().name}"


def submit_argv(spec: dict) -> str:
    return f"{JOBRUN} start {base64.b64encode(json.dumps(spec).encode()).decode()}"


def fmt_dur(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def as_result(row) -> dict:
    r = dict(row)
    r["metrics"] = json.loads(r["metrics"] or "{}") if isinstance(r["metrics"], str) else (r["metrics"] or {})
    return r


def result_line(console: Console, r: dict) -> None:
    mark = "[bold green]✔[/]" if r["status"] == "done" and not r.get("failure") else "[bold red]✘[/]"
    if r.get("cached_from"):
        mark = "[bold yellow]⚡[/] cache hit"
    tail = ledger.headline(r["metrics"])
    fail = f" · [red]{r['failure']}[/]" if r.get("failure") else ""
    console.print(f"{mark} [{agents.color(r['agent'])}]{r['agent']}[/] {r['track']} · exit {r['exit_code']}{fail} · "
                  f"{fmt_dur(r['wall_s'] or 0)} · ${r['cost_usd'] or 0:.3f}" + (f" · [bold]{tail}[/]" if tail else ""))
    if r.get("failure") and r.get("error") and not r.get("cached_from"):
        console.print(Text("\n".join("  " + l for l in r["error"].splitlines()[-8:]), style="red"))
    console.print(f"  [dim]id {r['id']} · log {r['log_path']}[/]")


def emit(console: Console, r: dict, as_json: bool) -> int:
    if as_json:
        print(json.dumps({k: r.get(k) for k in RESULT_KEYS}))
    else:
        result_line(console, r)
    rc = r["exit_code"]
    if r.get("failure") == "correctness_fail" and rc == 0:
        return 3  # benchmark exited cleanly but the candidate is wrong: don't let agents read that as success
    return rc if isinstance(rc, int) and 0 <= rc < 256 else 1


# ---------------------------------------------------------------- submit


def submit(cfg: Config, console: Console, *, track: Path, extra: list[Path], cmd: str, mode: str,
           cache: bool | None = None, note: str = "", timeout_s: float | None = None, fidelity: str = "timing") -> dict:
    """Returns either a finished result (cache hit) or {"id", "status": "queued"} for `wait`. `cache` (default: the
    config's) reuses an identical earlier run and joins an identical in-flight one instead of paying twice."""
    agent = agents.detect()
    track = track.resolve()
    roots = [track, *[e.resolve() for e in extra]]
    cache = cfg.cache if cache is None else cache
    # fidelity and the target hardware are part of the key: an L4 result must never answer an H100 request
    target = (cfg.pool("correctness") if fidelity == "correctness" else None) or cfg
    key = ledger.content_key(roots, cmd, mode if fidelity == "timing" else f"{mode}:{fidelity}", cfg.image,
                             f"{target.provider}:{','.join(target.gpu_type_ids)}") if cache else None
    db = ledger.connect()
    now = time.time()

    if cache and (hit := ledger.cached(db, key)):
        r = as_result(hit) | {"id": uuid.uuid4().hex[:10], "agent": agent, "cached_from": hit["id"], "cost_usd": 0.0,
                              "wall_s": 0.0, "backend": "cache", "started": now, "ended": now, "submitted": now,
                              "note": note or hit["note"]}
        ledger.record(db, **{k: r.get(k) for k in ("id", "key", "agent", "track", "cmd", "mode", "backend", "status",
                                                   "exit_code", "started", "ended", "wall_s", "cost_usd", "metrics",
                                                   "log_path", "cached_from", "track_path", "note", "failure",
                                                   "error", "submitted", "pool")})
        return r
    if cache and (fly := ledger.in_flight(db, key)):
        console.print(f"[yellow]≡[/] identical job {fly['id']} already in flight ({fly['agent']}) — waiting on it")
        return {"id": fly["id"], "status": "queued", "deduped": True}

    check_budget(cfg, db)
    pool = choose_pool(cfg, console, mode, fidelity)
    jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    timeout_s = timeout_s or cfg.default_timeout_s or None
    handle = None
    while True:
        try:
            if cfg.pool(pool).provider in SERVERLESS:
                handle = start_serverless(cfg.pool(pool), jid, track, roots[1:], cmd, mode, timeout_s)
            else:
                push_and_start(cfg, console, pool, jid, agent, track, roots, cmd, mode, key, timeout_s, db)
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            err = (getattr(e, "stderr", None) or "").strip().splitlines()[-3:]
            detail = " | ".join(err) or type(e).__name__
            if pool != "timing":  # a broken cheap pod must never block work: run it on the timing pod instead
                console.print(f"[yellow]↪[/] could not start the job on the {pool} pod ({detail}) — "
                              f"running it on the timing pod")
                pool = "timing"
                continue
            raise SystemExit(f"could not start the job on the {pool} pod: {detail}") from None
    ledger.record(db, id=jid, key=key, agent=agent, track=track.name, cmd=cmd, mode=mode,
                  backend=cfg.pool(pool).provider, status="queued", handle=json.dumps(handle) if handle else None,
                  exit_code=None, started=None, ended=None, wall_s=None, cost_usd=None, metrics={},
                  log_path=str(LOG_DIR / f"{jid}.log"), cached_from=None, track_path=str(track), note=note,
                  extra_paths=json.dumps([str(e.resolve()) for e in extra]),
                  failure=None, error=None, timeout_s=timeout_s, submitted=now, pool=pool)
    return {"id": jid, "status": "queued"}


def push_and_start(cfg: Config, console: Console, pool: str, jid: str, agent: str, track: Path, roots: list,
                   cmd: str, mode: str, key: str, timeout_s, db) -> None:
    pcfg = cfg.pool(pool)
    st = pod.ensure(pcfg, console, pool)
    k = cfg.private_key_path
    spec = {"id": jid, "agent": agent, "track": track.name, "cmd": cmd, "mode": mode, "cwd": remote_dir(pcfg, track),
            "roots": [remote_dir(pcfg, r) for r in roots], "key": key, "timeout_s": timeout_s, "quiet": cfg.quiet_timing,
            **hang_policy(cfg, db, track.name, cmd, pool)}
    # push + submit (which snapshots the synced dirs on the pod) must not interleave with another agent's push
    with open(HOME / f"push-{pool}-{track.name}.lock", "w") as lk, console.status(f"[cyan]syncing {track.name}…"):
        fcntl.flock(lk, fcntl.LOCK_EX)
        for r in roots:
            ssh.push(st, k, r, remote_dir(pcfg, r))
        ssh.run(st, k, submit_argv(spec), timeout=120)


def check_budget(cfg: Config, db) -> None:
    """Every new job (GPU or CPU, any provider) is refused once today's spend reaches the cap. Spend includes
    serverless jobs still running and ones nobody waited on (settled here first)."""
    settle_serverless(cfg, db)
    spent = ledger.spend(db, ledger.today_start())
    if spent >= cfg.budget_usd:
        raise BudgetExceeded(spent, cfg.budget_usd)
    if spent >= 0.8 * cfg.budget_usd:
        notify.once("budget80", "ogpu budget", f"${spent:.2f} of ${cfg.budget_usd:.0f} daily cap used")


def settle_serverless(cfg: Config, db) -> None:
    """Record serverless jobs that finished while nobody was waiting on them (submit without wait), so their cost
    reaches the ledger and the budget. Jobs another process is already waiting on are left to it."""
    rows = db.execute("select id, handle from runs where handle is not null and status in ('queued','running')")
    for row in rows.fetchall():
        try:
            h = json.loads(row["handle"])
        except ValueError:
            continue
        if not Path(h["results"]).exists() and pid_alive(h["pid"]):
            continue
        (HOME / "locks").mkdir(parents=True, exist_ok=True)
        with open(HOME / "locks" / f"{row['id']}.lock", "w") as lk:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue  # a `wait` is on it
        finalize(cfg, Console(stderr=True, quiet=True), row["id"], stream=False)


def start_serverless(pcfg: Config, jid: str, track: Path, extra: list[Path], cmd: str, mode: str, timeout_s) -> dict:
    gpu = modal_backend.gpu_for(pcfg, mode)
    results = LOG_DIR / f"{jid}.modal" / "results.json"
    launcher_log = LOG_DIR / f"{jid}.modal.log"
    p = modal_backend.launch(pcfg, track=track, extra=extra, cmds=[cmd], gpu=gpu, results=results, log=launcher_log,
                             timeout_s=timeout_s, detach=True)
    return {"pid": p.pid, "results": str(results), "log": str(launcher_log), "gpu": gpu,
            "rate": modal_backend.rate(gpu, *modal_backend.container(pcfg))}


def hang_policy(cfg: Config, db, track: str, cmd: str, pool: str) -> dict:
    """What 'hung' means for this job, from how long the same command took before (watchdog on the pod)."""
    usual, n = ledger.usual_runtime(db, track, cmd, pool)
    policy = {"spin_silent_s": cfg.spin_silent_s, "hang_silent_s": 60}
    if n >= 2 and usual > 0:
        policy |= {"usual_s": round(usual, 1), "hang_after_s": max(cfg.hang_floor_s, cfg.hang_factor * usual)}
    return policy


def choose_pool(cfg: Config, console: Console, mode: str, fidelity: str) -> str:
    """Correctness GPU jobs go to the cheap same-ISA pool when its pod is reachable. If it isn't, start warming it
    in the background and run this job on the timing pod: a correctness check never waits on a cold boot."""
    if mode != "gpu" or fidelity != "correctness" or (pcfg := cfg.pool("correctness")) is None:
        return "timing"
    if pcfg.provider in SERVERLESS:  # nothing to warm up
        return "correctness"
    if ssh.reachable(load_state("correctness"), pcfg.private_key_path):
        return "correctness"
    warm_pool_in_background("correctness")
    console.print("[yellow]↪[/] correctness pool is warming up — this job runs on the timing pod")
    return "timing"


def warm_pool_in_background(pool: str) -> bool:
    """Spawn a detached `ogpu up --pool <pool>` unless one is already running (it holds the pool's up-lock)."""
    lock_path = HOME / f"up-{pool}.lock"
    with open(lock_path, "w") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False  # someone is already bringing it up
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / f"up-{pool}.log", "ab") as log:
        subprocess.Popen([sys.executable, "-m", "ogpu", "up", "--pool", pool], stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    return True


class BudgetExceeded(SystemExit):
    def __init__(self, spent: float, cap: float):
        super().__init__(2)
        self.msg = f"today's spend ${spent:.2f} ≥ cap ${cap:.2f} (raise budget_usd in ~/.ogpu/config.toml)"


# ---------------------------------------------------------------- wait / finalize


def finalize(cfg: Config, console: Console, jid: str, *, stream: bool) -> dict:
    """Wait for a job, then pull out/, classify and record it. Safe to call from several processes at once: the
    first one polls; the others block on the per-job lock and read the recorded result."""
    (HOME / "locks").mkdir(parents=True, exist_ok=True)
    with open(HOME / "locks" / f"{jid}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        db = ledger.connect()
        row = ledger.get(db, jid)
        if row is None:
            raise SystemExit(f"unknown job {jid} (not in this machine's ledger)")
        if row["status"] in ledger.TERMINAL:
            return as_result(row)
        pool = row["pool"] or "timing"
        log_path = Path(row["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(b"")  # poll_loop re-reads from offset 0
        out_local = Path(row["track_path"] or ".") / "output" / "ogpu" / jid
        if row["handle"]:  # serverless
            h = json.loads(row["handle"])
            status, rc, started, ended, last = wait_serverless(console, h, log_path, out_local, stream=stream)
            rate = h.get("rate") or modal_backend.rate(h["gpu"])
            billed = last.get("billed_s")
        else:
            st = load_state(pool)
            k = cfg.private_key_path
            status, rc, started, ended, last = poll_loop(console, st, k, jid, log_path, stream=stream, pool=pool)
            if ssh.pull(st, k, f"{REMOTE_OGPU}/runs/{jid}/{row['track']}/out", out_local) and not any(out_local.iterdir()):
                out_local.rmdir()
            rate, billed = st.cost_per_hr, None
        log = log_path.read_text(errors="replace")
        metrics = ledger.parse_metrics(log, cfg.profile)
        failure, error = failures.classify(status, rc, log, metrics, hang=last.get("hang"))
        if last.get("launch_failed"):  # the job never ran: image build, auth or quota, not the user's code
            failure, error = "launch_error", failures.excerpt(log, 8)
        if status == "lost" and (last.get("runner_signal") or last.get("orphan_killed")):
            error = (error or "") + (f"\nrunner got {last['runner_signal']}" if last.get("runner_signal") else "") + \
                    ("\norphaned job process was killed" if last.get("orphan_killed") else "")
        wall = (ended or time.time()) - (started or time.time()) if started else 0.0
        prev_best = ledger.track_best(db, row["track"], exclude_id=jid)
        ledger.update(db, jid, status=status, exit_code=rc, started=started, ended=ended or time.time(), wall_s=wall,
                      cost_usd=(billed or wall) / 3600 * rate, metrics=metrics, failure=failure, error=error)
        if not row["handle"]:
            record_uptime(cfg.pool(pool) or cfg, st, db)
        r = as_result(ledger.get(db, jid))
        if pool != "timing" and not failure and metrics.get("check_margin", 0) > NEAR_BOUND:
            # measured: an H100 shifts error values by up to ~17% of the bound vs the H200
            r["confirm_on_timing"] = True
            r["advice"] = (f"checks passed on the {pool} pool but the worst error is at "
                           f"{metrics['check_margin']:.0%} of the bound — confirm with --fidelity timing")
        b = ledger.best_score(metrics)
        if failure == "hang":
            notify.send("ogpu · hung job killed", f"{row['track']} ({row['agent']}): {(last.get('hang') or {}).get('reason', '')}")
        if not failure and b is not None and (prev_best is None or b > prev_best):
            r["new_best"] = True
            notify.send("ogpu · new best", f"{row['track']}: {b:.4g} ({row['agent']})"
                        + (f" — was {prev_best:.4g}" if prev_best else ""))
        return r


def wait_serverless(console: Console, h: dict, log_path: Path, out_local: Path, *, stream: bool):
    """Follow a detached `modal run` until its results file appears (or the launcher dies without one)."""
    launcher_log, results = Path(h["log"]), Path(h["results"])
    offset = 0
    spinner = console.status("[cyan]starting container…")
    spinner.start()
    try:
        while True:
            done = results.exists() or not pid_alive(h["pid"])
            if stream and launcher_log.is_file():
                with open(launcher_log, "rb") as f:
                    f.seek(offset)
                    chunk = f.read()
                if chunk:
                    spinner.stop()
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.flush()
                    offset += len(chunk)
            if done:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        console.print("\n[yellow]detached[/] — the job keeps running. `ogpu wait <id>` / `ogpu cancel <id>`")
        raise SystemExit(130)
    finally:
        spinner.stop()
    tail = launcher_log.read_text(errors="replace")[-4000:] if launcher_log.is_file() else ""
    if not results.exists():  # `modal run` itself failed (image build, auth, quota): retrying won't help
        log_path.write_text(tail or "modal run exited without a result\n")
        return "failed", 1, None, time.time(), {"launch_failed": True}
    r = (json.loads(results.read_text()) or [{"error": "no result"}])[0]
    if "error" in r:
        log_path.write_text(tail + "\n" + r["error"] + "\n")
        return "failed", 1, None, time.time(), {}
    log_path.write_text(Path(r["log_path"]).read_text(errors="replace"))
    src = results.parent / str(r["i"])
    if src.is_dir() and any(src.iterdir()):
        shutil.copytree(src, out_local, dirs_exist_ok=True)
    ended = r.get("ended") or results.stat().st_mtime
    return ("done" if r["rc"] == 0 else "failed"), r["rc"], ended - r["wall_s"], ended, \
        {"billed_s": r.get("billed_s")}


def pid_alive(pid) -> bool:
    try:  # our own child (`ogpu run` launched it): reap it, or its zombie looks alive forever
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except (ChildProcessError, TypeError):
        pass
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def record_uptime(pcfg: Config, st: State, db) -> None:
    """Paid time comes from the provider's uptime; sample it after every job so spend (and the budget cap) never
    depends on someone running `status` or `top`. Best effort: never fails a job."""
    if not st.pod_id:
        return
    try:
        p = pod.client(pcfg, st).get(st.pod_id)
        if p and p.uptime_s:
            ledger.record_session(db, p.id, p.uptime_s, p.cost_per_hr)
    except Exception:  # noqa: BLE001
        pass


def wait(cfg: Config, console: Console, jid: str, *, stream: bool, retry_lost: int = 1) -> dict:
    r = finalize(cfg, console, jid, stream=stream)
    while r["status"] == "lost" and retry_lost > 0:
        retry_lost -= 1
        console.print(f"[yellow]↻[/] job {jid} was lost (pod restarted) — resubmitting once")
        extra = [Path(p) for p in json.loads(r.get("extra_paths") or "[]")]
        fidelity = "correctness" if r.get("pool") == "correctness" else "timing"
        sub = submit(cfg, console, track=Path(r["track_path"]), extra=extra, cmd=r["cmd"], mode=r["mode"], cache=False,
                     note=r.get("note") or "", timeout_s=r.get("timeout_s"), fidelity=fidelity)
        r = finalize(cfg, console, sub["id"], stream=stream) if sub["status"] == "queued" else sub
    return r


def run(cfg: Config, *, track: Path, extra: list[Path], cmd: str, mode: str, cache: bool | None, as_json: bool,
        quiet: bool, note: str = "", timeout_s: float | None = None, fidelity: str = "timing") -> int:
    console = Console(stderr=True, quiet=as_json)
    try:
        sub = submit(cfg, console, track=track, extra=extra, cmd=cmd, mode=mode, cache=cache, note=note,
                     timeout_s=timeout_s, fidelity=fidelity)
    except BudgetExceeded as e:
        if as_json:
            print(json.dumps({"status": "budget_exceeded", "error": e.msg}))
        console.print(f"[bold red]✘ budget[/] {e.msg}")
        return 2
    if sub["status"] != "queued":  # cache hit
        if not (as_json or quiet) and Path(sub["log_path"]).exists():
            sys.stdout.write(Path(sub["log_path"]).read_text(errors="replace"))
        return emit(console, sub, as_json)
    r = wait(cfg, console, sub["id"], stream=not (as_json or quiet or sub.get("deduped")))
    return emit(console, r, as_json)


def poll_loop(console: Console, st: State, key: str, jid: str, log_path: Path, *, stream: bool,
              pool: str = "timing"):
    offset, failures_ = 0, 0
    status = None
    spinner = console.status("[cyan]submitted…")
    spinner.start()
    try:
        with open(log_path, "ab") as log:
            while True:
                try:
                    st = load_state(pool) if failures_ else st  # pod may have been resumed on a new port
                    p = ssh.run(st, key, f"{JOBRUN} poll {shlex.quote(jid)} {offset}", timeout=60)
                    d = json.loads(p.stdout)
                    failures_ = 0
                except (subprocess.SubprocessError, ValueError) as e:
                    failures_ += 1
                    spinner.update(f"[yellow]connection lost ({failures_}) · job keeps running on the machine · retrying")
                    if failures_ > 40:
                        raise SystemExit(f"lost contact with the machine for too long; job {jid} may still be running "
                                         f"(`ogpu wait {jid}`)") from e
                    time.sleep(min(15, 2 * failures_))
                    continue
                status = d["status"]
                chunk = base64.b64decode(d.get("chunk") or "")
                if chunk:
                    log.write(chunk)
                    log.flush()
                    if stream:
                        spinner.stop()
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.flush()
                offset = d.get("offset", offset)
                if status == "queued":
                    h = d.get("holder") or {}
                    who = f" · holder {h.get('agent')}({h.get('track')}) {fmt_dur(time.time() - h['since'])}" if h else ""
                    spinner.update(f"[cyan]waiting for the GPU lock · position {d.get('queue_pos')}{who}")
                    spinner.start()
                if status in ledger.TERMINAL:
                    return status, d.get("exit_code"), d.get("started"), d.get("ended"), d
                time.sleep(0.25 if chunk else 1.0)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]detached[/] — job {jid} keeps running. `ogpu wait {jid}` / `ogpu cancel {jid}`")
        raise SystemExit(130)
    finally:
        spinner.stop()
