import base64
import fcntl
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path

from test_core import fake_pod, wait_status  # noqa: F401 — reuse the local on-pod runner fixture

REMOTE = Path(__file__).parent.parent / "src/ogpu/remote"


def load(name, path, inject=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if inject:
        sys.modules.update(inject)
    spec.loader.exec_module(mod)
    return mod


reaper = load("reaper_under_test", REMOTE / "reaper.py")


def test_hang_reason_rules():
    job = {"mode": "gpu", "usual_s": 12.0, "hang_after_s": 180, "hang_silent_s": 60, "spin_silent_s": 900}
    assert reaper.hang_reason(job, 100, 500, None) is None              # under the limit
    assert reaper.hang_reason(job, 200, 30, None) is None               # over the limit but still printing
    assert "usually takes ~12s" in reaper.hang_reason(job, 200, 90, None)
    fresh = {"mode": "gpu", "spin_silent_s": 900}                        # no history yet
    spin = (100.0, 118.0, 700.0)                                         # the real hang we saw: 100% util at 118W
    assert "busy-waiting" in reaper.hang_reason(fresh, 2000, 1200, spin)
    assert reaper.hang_reason(fresh, 2000, 1200, (100.0, 550.0, 700.0)) is None  # real work draws power
    assert reaper.hang_reason(fresh, 2000, 600, spin) is None            # not silent long enough
    assert reaper.hang_reason({**fresh, "mode": "cpu"}, 2000, 1200, spin) is None


def start_spec(call, track, jid, cmd, **extra):
    spec = {"id": jid, "agent": "claude", "track": "track", "cmd": cmd, "mode": "gpu", "cwd": str(track),
            "roots": [str(track)], **extra}
    return call("start", base64.b64encode(json.dumps(spec).encode()).decode())


def patched_jobrun(track):
    return load("jobrun", track.parent.parent / "jobrun.py")  # the fixture's path-patched copy


def test_watchdog_kills_hung_job(fake_pod):  # noqa: F811
    call, start, track = fake_pod
    start_spec(call, track, "hung", "echo built; sleep 60", usual_s=0.2, hang_after_s=1, hang_silent_s=0)
    wait_status(call, "hung", {"running"})
    time.sleep(1.5)
    wd = load("reaper_wd", REMOTE / "reaper.py", inject={"jobrun": patched_jobrun(track)})
    wd.watchdog()
    d = wait_status(call, "hung", {"failed"}, timeout=15)
    assert d["hang"]["reason"].startswith("running") and "usually takes" in d["hang"]["reason"]
    assert d["hang"]["term_at"]


def test_orphan_keeps_lock_then_is_killed(fake_pod, tmp_path):  # noqa: F811
    call, start, track = fake_pod
    start_spec(call, track, "orph", "sleep 30")
    wait_status(call, "orph", {"running"})
    job = json.loads((track.parent.parent / "og" / "jobs" / "orph.json").read_text())
    os.kill(job["runner_pid"], signal.SIGKILL)  # runner dies without a chance to clean up
    time.sleep(0.3)
    lock = open(track.parent.parent / "h200.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held = False
    except BlockingIOError:
        held = True
    assert held, "the orphaned job must keep the GPU lock while it runs"
    d = call("poll", "orph", "0")  # reconcile: runner dead → lost, orphan killed
    assert d["status"] == "lost" and d["orphan_killed"] is True
    for _ in range(50):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(0.1)
    else:
        raise AssertionError("lock not released after the orphan was killed")


def test_classify_hang_and_usual_runtime(tmp_path):
    from ogpu import failures, ledger
    f, e = failures.classify("failed", -15, "built\n", {}, hang={"reason": "running 1200s; usually ~12s"})
    assert f == "hang" and "usually ~12s" in e
    db = ledger.connect(tmp_path / "l.db")
    for i, w in enumerate([10, 12, 11, 13, 40]):
        ledger.record(db, id=str(i), key="k", agent="a", track="T", cmd="c", mode="gpu", backend="runpod",
                      status="done", exit_code=0, started=0, ended=i, wall_s=w, cost_usd=0, metrics={}, log_path="",
                      cached_from=None)
    usual, n = ledger.usual_runtime(db, "T", "c", "timing")
    assert n == 5 and usual == 40  # p90 of 5 samples = the max: conservative
