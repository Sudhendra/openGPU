"""`ogpu run`: sync → submit to the on-pod runner → poll (survives SSH drops) → pull out/ → ledger."""
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

from . import agents, ledger, pod, ssh
from .config import HOME, LOG_DIR, REMOTE_OGPU, REMOTE_ROOT, Config, State

JOBRUN = f"python3 {REMOTE_OGPU}/bin/jobrun.py"


def remote_dir(local: Path) -> str:
    return f"{REMOTE_ROOT}/{local.resolve().name}"


def submit_argv(spec: dict) -> str:
    return f"{JOBRUN} start {base64.b64encode(json.dumps(spec).encode()).decode()}"


def fmt_dur(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def result_line(console: Console, r: dict) -> None:
    ok = r["status"] == "done"
    mark = "[bold green]✔[/]" if ok else "[bold red]✘[/]"
    if r.get("cached_from"):
        mark = "[bold yellow]⚡[/] cache hit"
    tail = ledger.headline(r["metrics"])
    console.print(f"{mark} [{agents.color(r['agent'])}]{r['agent']}[/] {r['track']} · exit {r['exit_code']} · "
                  f"{fmt_dur(r['wall_s'] or 0)} · ${r['cost_usd']:.3f}" + (f" · [bold]{tail}[/]" if tail else ""))
    console.print(f"  [dim]log {r['log_path']}[/]")


def run(cfg: Config, *, track: Path, extra: list[Path], cmd: str, mode: str, force: bool, as_json: bool,
        quiet: bool) -> int:
    console = Console(stderr=True, quiet=as_json)
    agent = agents.detect()
    track = track.resolve()
    roots = [track, *[e.resolve() for e in extra]]
    key = ledger.content_key(roots, cmd, mode, cfg.image)
    db = ledger.connect()

    if not force and (hit := ledger.cached(db, key)):
        r = dict(hit) | {"metrics": json.loads(hit["metrics"]), "agent": agent, "cached_from": hit["id"],
                         "cost_usd": 0.0, "wall_s": 0.0}
        rid = uuid.uuid4().hex[:10]
        r["id"] = rid
        ledger.record(db, **{k: r[k] for k in ("key", "track", "cmd", "mode", "status", "exit_code", "metrics",
                                              "log_path", "cached_from")} | {"id": rid, "agent": agent,
                      "backend": "cache", "started": time.time(), "ended": time.time(), "wall_s": 0.0, "cost_usd": 0.0})
        return finish(console, r, as_json, quiet)

    if mode == "gpu":
        spent = ledger.spend(db, ledger.today_start())
        if spent >= cfg.budget_usd:
            console.print(f"[bold red]✘ budget[/] today's spend ${spent:.2f} ≥ cap ${cfg.budget_usd:.2f} "
                          "(raise budget_usd in ~/.ogpu/config.toml)")
            return 2

    st = pod.ensure(cfg, console)
    k = cfg.private_key_path
    jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    spec = {"id": jid, "agent": agent, "track": track.name, "cmd": cmd, "mode": mode, "cwd": remote_dir(track),
            "roots": [remote_dir(r) for r in roots], "key": key}
    # push + submit (which snapshots the synced dirs on the pod) must not interleave with another agent's push
    with open(HOME / f"push-{track.name}.lock", "w") as lk, console.status(f"[cyan]syncing {track.name}…"):
        fcntl.flock(lk, fcntl.LOCK_EX)
        for r in roots:
            ssh.push(st, k, r, remote_dir(r))
        ssh.run(st, k, submit_argv(spec), timeout=120)
    log_path = LOG_DIR / f"{jid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status, rc, started, ended = poll_loop(console, st, k, jid, log_path, stream=not (as_json or quiet))

    out_local = track / "output" / "ogpu" / jid
    if ssh.pull(st, k, f"{REMOTE_ROOT}/.runs/{jid}/{track.name}/out", out_local) and not any(out_local.iterdir()):
        out_local.rmdir()
    wall = (ended or time.time()) - (started or time.time())
    metrics = ledger.parse_metrics(log_path.read_text(errors="replace"))
    r = {"id": jid, "key": key, "agent": agent, "track": track.name, "cmd": cmd, "mode": mode, "backend": "runpod",
         "status": status, "exit_code": rc, "started": started, "ended": ended, "wall_s": wall,
         "cost_usd": wall / 3600 * st.cost_per_hr, "metrics": metrics, "log_path": str(log_path),
         "cached_from": None}
    ledger.record(db, **r)
    return finish(console, r, as_json, quiet)


def finish(console: Console, r: dict, as_json: bool, quiet: bool) -> int:
    if as_json:
        print(json.dumps({k: r.get(k) for k in ("id", "status", "exit_code", "wall_s", "cost_usd", "metrics",
                                                 "log_path", "cached_from")}))
    else:
        if r.get("cached_from") and not quiet:
            sys.stdout.write(Path(r["log_path"]).read_text(errors="replace") if Path(r["log_path"]).exists() else "")
        result_line(console, r)
    rc = r["exit_code"]
    return rc if isinstance(rc, int) and 0 <= rc < 256 else 1


def poll_loop(console: Console, st: State, key: str, jid: str, log_path: Path, *, stream: bool):
    offset, failures = 0, 0
    status = None
    spinner = console.status("[cyan]submitted…")
    spinner.start()
    try:
        with open(log_path, "ab") as log:
            while True:
                try:
                    p = ssh.run(st, key, f"{JOBRUN} poll {shlex.quote(jid)} {offset}", timeout=60)
                    d = json.loads(p.stdout)
                    failures = 0
                except (subprocess.SubprocessError, ValueError) as e:
                    failures += 1
                    spinner.update(f"[yellow]connection lost ({failures}) · job keeps running on the pod · retrying")
                    if failures > 40:
                        raise SystemExit(f"lost contact with pod for too long; job {jid} may still be running "
                                         f"(`ogpu logs {jid}`)") from e
                    time.sleep(min(15, 2 * failures))
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
                if status in ("done", "failed", "lost", "cancelled", "missing"):
                    return status, d.get("exit_code"), d.get("started"), d.get("ended")
                time.sleep(0.25 if chunk else 1.0)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]detached[/] — job {jid} keeps running. `ogpu logs {jid}` / `ogpu cancel {jid}`")
        raise SystemExit(130)
    finally:
        spinner.stop()
