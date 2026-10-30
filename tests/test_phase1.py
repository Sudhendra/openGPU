import sqlite3
import time

import pytest

from ogpu import failures, ledger
from test_core import fake_pod, wait_status  # noqa: F401 — reuse the local on-pod runner fixture

TRACEBACK_LOG = """prepare cand.py: 12s
Traceback (most recent call last):
  File "pod/bench.py", line 88, in <module>
    out = fn(x)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
"""
COMPILE_LOG = """building extension sk2...
/root/.cache/torch_extensions/sk2/sk2.cu(141): error: identifier "tile_k" is undefined
ninja: build stopped: subcommand failed.
"""


@pytest.mark.parametrize("status,rc,log,metrics,want", [
    ("done", 0, "SCORE a: 1.02\nCHECKS: ALL PASS", {"checks": "PASS"}, None),
    ("failed", 1, TRACEBACK_LOG, {}, "oom"),
    ("failed", 1, COMPILE_LOG, {}, "compile_error"),
    ("failed", 1, "RuntimeError: CUDA error: an illegal memory access was encountered", {}, "cuda_error"),
    ("failed", 1, "ModuleNotFoundError: No module named 'sk2'", {}, "import_error"),
    ("failed", 124, "still compiling...", {}, "timeout"),
    ("done", 0, "CHECK d1s cand.py judge: FAIL rel L2 2e-1\nCHECKS: 1 FAIL", {"checks": "FAIL"}, "correctness_fail"),
    ("lost", -1, "", {}, "lost"),
    ("failed", 2, "something odd happened", {}, "runtime_error"),
])
def test_classify(status, rc, log, metrics, want):
    failure, error = failures.classify(status, rc, log, metrics)
    assert failure == want
    if want == "oom":
        assert error.startswith("Traceback") and "OutOfMemoryError" in error
    if want == "correctness_fail":
        assert "CHECK d1s cand.py judge: FAIL" in error


def test_excerpt_trims_long_tracebacks():
    log = "Traceback (most recent call last):\n" + "\n".join(f"  frame {i}" for i in range(100)) + "\nValueError: x"
    ex = failures.excerpt(log, max_lines=10)
    assert ex.splitlines()[0].startswith("Traceback") and "ValueError: x" in ex and len(ex.splitlines()) == 10


def row(db, rid, **kw):
    base = dict(id=rid, key="k", agent="claude", track="t", cmd="c", mode="gpu", backend="runpod", status="done",
                exit_code=0, started=None, ended=None, wall_s=0, cost_usd=0, metrics={}, log_path="x",
                cached_from=None)
    ledger.record(db, **(base | kw))


def test_utilization_and_agent_spend(tmp_path):
    db = ledger.connect(tmp_path / "l.db")
    now = time.time()
    start = round((now - 3600) / 60) * 60
    ledger.record_session(db, "pod", now - start, 4.0)  # one paid hour
    row(db, "a", agent="claude", started=start + 0, ended=start + 900, cost_usd=1.0)     # 15 min GPU
    row(db, "b", agent="codex", started=start + 600, ended=start + 1200, cost_usd=0.5)    # overlaps a by 5 min
    row(db, "c", agent="codex", mode="cpu", started=start + 1800, ended=start + 2100)     # 5 min CPU-only
    u = ledger.utilization(db, start)
    assert u["gpu_busy_s"] == pytest.approx(1200, abs=1)        # union, not sum
    assert u["any_busy_s"] == pytest.approx(1500, abs=1)
    assert u["gpu_busy_frac"] == pytest.approx(1200 / u["paid_s"], rel=1e-3)
    assert u["idle_usd"] == pytest.approx((u["paid_s"] - 1500) / 3600 * 4.0, rel=1e-2)
    assert ledger.agent_spend(db, start) == {"claude": 1.0, "codex": 0.5}


def test_in_flight_dedupe_and_track_best(tmp_path):
    db = ledger.connect(tmp_path / "l.db")
    row(db, "q", key="K", status="queued", submitted=time.time())
    assert ledger.in_flight(db, "K")["id"] == "q"
    assert ledger.cached(db, "K") is None  # queued runs are not cache hits
    row(db, "old", key="K2", status="queued", submitted=time.time() - 4 * 3600)
    assert ledger.in_flight(db, "K2") is None  # stale submissions don't block forever
    row(db, "p", track="T", metrics={"score": {"a.py": 1.05}, "checks": "PASS"})
    row(db, "f", track="T", metrics={"score": {"b.py": 1.30}, "checks": "FAIL"})  # fast but wrong: ignored
    assert ledger.track_best(db, "T") == 1.05
    assert ledger.track_best(db, "T", exclude_id="p") is None


def test_migrates_v0_ledger(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("create table runs (id text primary key, key text, agent text, track text, cmd text, mode text,"
                      " backend text, status text, exit_code int, started real, ended real, wall_s real, cost_usd real,"
                      " metrics text, log_path text, cached_from text);"
                      "insert into runs (id, status, metrics) values ('x', 'done', '{}');")
    old.commit()
    old.close()
    db = ledger.connect(path)
    cols = {r["name"] for r in db.execute("pragma table_info(runs)")}
    assert {"note", "failure", "error", "track_path", "submitted", "extra_paths"} <= cols
    assert ledger.get(db, "x")["status"] == "done"


def test_jobrun_timeout(fake_pod):  # noqa: F811
    call, start, track = fake_pod
    import base64
    import json
    spec = {"id": "slow", "agent": "claude", "track": "track", "cmd": "sleep 30", "mode": "cpu", "cwd": str(track),
            "roots": [str(track)], "timeout_s": 1}
    call("start", base64.b64encode(json.dumps(spec).encode()).decode())
    d = wait_status(call, "slow", {"failed"}, timeout=15)
    assert d["exit_code"] == 124
