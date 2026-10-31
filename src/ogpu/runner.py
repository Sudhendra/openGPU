"""`ogpu run` / `submit` / `wait`: sync → submit to the on-pod runner → poll (survives SSH drops) → pull out/ →
classify → ledger. `run` = submit + wait in one call."""
from __future__ import annotations

import base64
import fcntl
import json
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

from rich.console import Console
from rich.text import Text

from . import agents, failures, ledger, notify, pod, ssh
from .config import HOME, LOG_DIR, REMOTE_OGPU, REMOTE_ROOT, Config, State, load_state

JOBRUN = f"python3 {REMOTE_OGPU}/bin/jobrun.py"
RESULT_KEYS = ("id", "status", "failure", "error", "exit_code", "wall_s", "cost_usd", "metrics", "note", "pool",
               "log_path", "cached_from")


def remote_dir(local: Path) -> str:
    return f"{REMOTE_ROOT}/{local.resolve().name}"


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


def submit(cfg: Config, console: Console, *, track: Path, extra: list[Path], cmd: str, mode: str, force: bool,
           note: str = "", timeout_s: float | None = None, fidelity: str = "timing") -> dict:
    """Returns either a finished result (cache hit) or {"id", "status": "queued"} for `wait`."""
    agent = agents.detect()
    track = track.resolve()
    roots = [track, *[e.resolve() for e in extra]]
    # fidelity is part of the key: a correctness run on an H100 must not satisfy a timing request on the H200
    key = ledger.content_key(roots, cmd, mode if fidelity == "timing" else f"{mode}:{fidelity}", cfg.image)
    db = ledger.connect()
    now = time.time()

    if not force and (hit := ledger.cached(db, key)):
        r = as_result(hit) | {"id": uuid.uuid4().hex[:10], "agent": agent, "cached_from": hit["id"], "cost_usd": 0.0,
                              "wall_s": 0.0, "backend": "cache", "started": now, "ended": now, "submitted": now,
                              "note": note or hit["note"]}
        ledger.record(db, **{k: r.get(k) for k in ("id", "key", "agent", "track", "cmd", "mode", "backend", "status",
                                                   "exit_code", "started", "ended", "wall_s", "cost_usd", "metrics",
                                                   "log_path", "cached_from", "track_path", "note", "failure",
                                                   "error", "submitted")})
        return r
    if not force and (fly := ledger.in_flight(db, key)):
        console.print(f"[yellow]≡[/] identical job {fly['id']} already in flight ({fly['agent']}) — waiting on it")
        return {"id": fly["id"], "status": "queued", "deduped": True}

    if mode == "gpu":
        spent = ledger.spend(db, ledger.today_start())
        if spent >= cfg.budget_usd:
            raise BudgetExceeded(spent, cfg.budget_usd)
        if spent >= 0.8 * cfg.budget_usd:
            notify.once("budget80", "ogpu budget", f"${spent:.2f} of ${cfg.budget_usd:.0f} daily cap used")

    pool = choose_pool(cfg, console, mode, fidelity)
    pcfg = cfg.pool(pool)
    st = pod.ensure(pcfg, console, pool)
    k = cfg.private_key_path
    jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    spec = {"id": jid, "agent": agent, "track": track.name, "cmd": cmd, "mode": mode, "cwd": remote_dir(track),
            "roots": [remote_dir(r) for r in roots], "key": key, "timeout_s": timeout_s, "quiet": cfg.quiet_timing}
    # push + submit (which snapshots the synced dirs on the pod) must not interleave with another agent's push
    with open(HOME / f"push-{pool}-{track.name}.lock", "w") as lk, console.status(f"[cyan]syncing {track.name}…"):
        fcntl.flock(lk, fcntl.LOCK_EX)
        for r in roots:
            ssh.push(st, k, r, remote_dir(r))
        ssh.run(st, k, submit_argv(spec), timeout=120)
    ledger.record(db, id=jid, key=key, agent=agent, track=track.name, cmd=cmd, mode=mode, backend="runpod",
                  status="queued", exit_code=None, started=None, ended=None, wall_s=None, cost_usd=None, metrics={},
                  log_path=str(LOG_DIR / f"{jid}.log"), cached_from=None, track_path=str(track), note=note,
                  extra_paths=json.dumps([str(e.resolve()) for e in extra]),
                  failure=None, error=None, timeout_s=timeout_s, submitted=now, pool=pool)
    return {"id": jid, "status": "queued"}


def choose_pool(cfg: Config, console: Console, mode: str, fidelity: str) -> str:
    """Correctness GPU jobs go to the cheap same-ISA pool when its pod is reachable. If it isn't, start warming it
    in the background and run this job on the timing pod: a correctness check never waits on a cold boot."""
    if mode != "gpu" or fidelity != "correctness" or (pcfg := cfg.pool("correctness")) is None:
        return "timing"
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
        st = load_state(pool)
        k = cfg.private_key_path
        log_path = Path(row["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(b"")  # poll_loop re-reads from offset 0
        status, rc, started, ended = poll_loop(console, st, k, jid, log_path, stream=stream, pool=pool)
        track = Path(row["track_path"] or ".")
        out_local = track / "output" / "ogpu" / jid
        if ssh.pull(st, k, f"{REMOTE_ROOT}/.runs/{jid}/{row['track']}/out", out_local) and not any(out_local.iterdir()):
            out_local.rmdir()
        log = log_path.read_text(errors="replace")
        metrics = ledger.parse_metrics(log)
        failure, error = failures.classify(status, rc, log, metrics)
        wall = (ended or time.time()) - (started or time.time()) if started else 0.0
        prev_best = ledger.track_best(db, row["track"], exclude_id=jid)
        ledger.update(db, jid, status=status, exit_code=rc, started=started, ended=ended or time.time(), wall_s=wall,
                      cost_usd=wall / 3600 * st.cost_per_hr, metrics=metrics, failure=failure, error=error)
        r = as_result(ledger.get(db, jid))
        b = ledger.best_score(metrics)
        if not failure and b is not None and (prev_best is None or b > prev_best):
            r["new_best"] = True
            notify.send("ogpu · new best", f"{row['track']}: {b:.4f}× ({row['agent']})"
                        + (f" — was {prev_best:.4f}×" if prev_best else ""))
        return r


def wait(cfg: Config, console: Console, jid: str, *, stream: bool, retry_lost: int = 1) -> dict:
    r = finalize(cfg, console, jid, stream=stream)
    while r["status"] == "lost" and retry_lost > 0:
        retry_lost -= 1
        console.print(f"[yellow]↻[/] job {jid} was lost (pod restarted) — resubmitting once")
        extra = [Path(p) for p in json.loads(r.get("extra_paths") or "[]")]
        fidelity = "correctness" if r.get("pool") == "correctness" else "timing"
        sub = submit(cfg, console, track=Path(r["track_path"]), extra=extra, cmd=r["cmd"], mode=r["mode"], force=True,
                     note=r.get("note") or "", timeout_s=r.get("timeout_s"), fidelity=fidelity)
        r = finalize(cfg, console, sub["id"], stream=stream) if sub["status"] == "queued" else sub
    return r


def run(cfg: Config, *, track: Path, extra: list[Path], cmd: str, mode: str, force: bool, as_json: bool, quiet: bool,
        note: str = "", timeout_s: float | None = None, fidelity: str = "timing") -> int:
    console = Console(stderr=True, quiet=as_json)
    try:
        sub = submit(cfg, console, track=track, extra=extra, cmd=cmd, mode=mode, force=force, note=note,
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
                    spinner.update(f"[yellow]connection lost ({failures_}) · job keeps running on the pod · retrying")
                    if failures_ > 40:
                        raise SystemExit(f"lost contact with pod for too long; job {jid} may still be running "
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
                    spinner.update(f"[cyan]waiting for H200 lock · position {d.get('queue_pos')}{who}")
                    spinner.start()
                if status in ledger.TERMINAL:
                    return status, d.get("exit_code"), d.get("started"), d.get("ended")
                time.sleep(0.25 if chunk else 1.0)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]detached[/] — job {jid} keeps running. `ogpu wait {jid}` / `ogpu cancel {jid}`")
        raise SystemExit(130)
    finally:
        spinner.stop()
