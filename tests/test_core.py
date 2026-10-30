import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ogpu import agents, ledger, telemetry
from ogpu.config import State
from ogpu.providers.runpod import Enum, Pod, gql_literal
from ogpu.runner import submit_argv
from ogpu.ssh import rsync_argv, ssh_argv

BENCH_LOG = """\
prepare submission_best.py: 41s
CHECK d1s submission_best.py judge: PASS rel L2 8.0e-03 [bound 0.095]
CHECK d1s submission_sk.py judge: FAIL rel L2 2.1e-01 [bound 0.095]
CASE d1s (local 0, sglang 51.2): submission_best.py 47.1us | submission_sk.py 46.0us
SCORE submission_best.py: 1.0831  d1=1.090 d8=1.070
SCORE submission_sk.py: 1.1012  d1=1.120 d8=1.090
CHECKS: 1 FAIL [('d1s', 'submission_sk.py')]
"""

PAIR_LOG = 'SIDE s1: {"score_vs_sglang": 1.21, "ratio_vs_s0": 1.034, "ci95": [1.021, 1.047]}\nOGPU_METRIC tflops=512.5\n'


def test_parse_bench_metrics():
    m = ledger.parse_metrics(BENCH_LOG)
    assert m["score"] == {"submission_best.py": 1.0831, "submission_sk.py": 1.1012}
    assert m["checks"] == "FAIL" and m["check_pass"] == 1 and m["check_fail"] == 1
    assert "submission_sk.py 1.1012×" in ledger.headline(m)


def test_parse_pair_bench_and_custom_metrics():
    m = ledger.parse_metrics(PAIR_LOG)
    assert m["sides"]["s1"]["ratio_vs_s0"] == 1.034
    assert m["tflops"] == 512.5
    assert "s1 1.0340× vs s0 [1.021,1.047]" in ledger.headline(m)


def test_content_key_tracks_sources_not_outputs(tmp_path):
    (tmp_path / "submission.py").write_text("x = 1\n")
    (tmp_path / "output").mkdir()
    k1 = ledger.content_key([tmp_path], "python3 bench.py", "gpu", "img")
    (tmp_path / "output" / "noise.log").write_text("ignored")
    (tmp_path / "notes.md").write_text("ignored, not a source ext")
    assert ledger.content_key([tmp_path], "python3 bench.py", "gpu", "img") == k1
    (tmp_path / "submission.py").write_text("x = 2\n")
    k2 = ledger.content_key([tmp_path], "python3 bench.py", "gpu", "img")
    assert k2 != k1
    assert ledger.content_key([tmp_path], "python3 bench.py", "cpu", "img") != k2


def test_ledger_cache_and_spend(tmp_path):
    db = ledger.connect(tmp_path / "l.db")
    ledger.record(db, id="a", key="k", agent="claude", track="t", cmd="c", mode="gpu", backend="runpod", status="done",
                  exit_code=0, started=1, ended=2, wall_s=1, cost_usd=0.1, metrics={"checks": "PASS"}, log_path="x",
                  cached_from=None)
    assert ledger.cached(db, "k")["id"] == "a"
    assert ledger.cached(db, "other") is None
    ledger.record_session(db, "pod1", 3600, 4.0)
    ledger.record_session(db, "pod1", 3610, 4.0)  # same session, later poll: upsert, not double count
    assert ledger.spend(db) == pytest.approx(4.0 * 3610 / 3600, rel=1e-3)


def test_nvidia_smi_parse_and_sparkline():
    s = telemetry.parse_line("78, 62668, 143771, 512.33, 700.00, 63, 1980, NVIDIA H200")
    assert s.util == 78 and round(s.mem_total_gb) == 140 and s.name == "NVIDIA H200"
    assert telemetry.parse_line("[N/A], 1, 2, [N/A], 700, 40, 1000, NVIDIA H200").util == 0
    assert telemetry.parse_line("garbage") is None
    assert telemetry.sparkline([0, 50, 100], 5) == "  ▁▅█"


def test_agent_classify_and_env(monkeypatch):
    assert agents.classify(["zsh", "codex", "claude"]) == "codex"  # nearest parent wins
    assert agents.classify(["node", "claude"]) == "claude"
    assert agents.classify(["opencode-darwin-arm64"]) == "opencode"
    assert agents.classify(["bash", "zsh"]) is None
    monkeypatch.setenv("OGPU_AGENT", "codex-bf16")
    assert agents.detect() == "codex-bf16"
    assert agents.color("claude") == "orange1"
    assert agents.color("x") == agents.color("x")


def test_gql_literal_and_pod_parse():
    lit = gql_literal({"name": 'a"b', "cloudType": Enum("SECURE"), "gpuCount": 1, "x": None, "public": True,
                       "env": [{"key": "K", "value": "v"}]})
    assert lit == '{name: "a\\"b", cloudType: SECURE, gpuCount: 1, public: true, env: [{key: "K", value: "v"}]}'
    pod = Pod.from_api({"id": "p", "desiredStatus": "RUNNING", "costPerHr": "4.31",
                        "runtime": {"uptimeInSeconds": 5, "ports": [
                            {"ip": "10.0.0.1", "isIpPublic": False, "privatePort": 22, "publicPort": 22, "type": "tcp"},
                            {"ip": "203.0.113.9", "isIpPublic": True, "privatePort": 22, "publicPort": 40123, "type": "tcp"}]}})
    assert (pod.ssh_host, pod.ssh_port, pod.cost_per_hr) == ("203.0.113.9", 40123, 4.31)


def test_ssh_and_rsync_commands():
    st = State(pod_id="p", host="203.0.113.9", port=40123)
    a = ssh_argv(st, "/k", "true")
    assert a[0] == "ssh" and "40123" in a and a[-2:] == ["root@203.0.113.9", "true"]
    assert any(o.startswith("ControlPath=") for o in a)
    r = rsync_argv(st, "/k", "src/", "root@h:/dst/", delete=True)
    assert "--delete" in r and "--exclude" in r and r[-2:] == ["src/", "root@h:/dst/"]
    spec = {"id": "j", "cmd": "echo 'hi' && ls"}
    b64 = submit_argv(spec).split()[-1]
    assert json.loads(base64.b64decode(b64)) == spec  # quoting-proof transport


@pytest.fixture
def fake_pod(tmp_path, monkeypatch):
    """Run the real on-pod jobrun.py locally against a temp /workspace."""
    src = (Path(__file__).parent.parent / "src/ogpu/remote/jobrun.py").read_text()
    og = tmp_path / "og"
    src = (src.replace('"/workspace/.ogpu"', repr(str(og))).replace('"/tmp/h200.lock"', repr(str(tmp_path / "h200.lock")))
              .replace('"/workspace/okc/.runs"', repr(str(tmp_path / "runs"))))
    script = tmp_path / "jobrun.py"
    script.write_text(src)
    (og / "jobs").mkdir(parents=True)
    track = tmp_path / "okc" / "track"
    track.mkdir(parents=True)
    (track / "bench.py").write_text("print('SCORE cand: 1.05')\n")

    def call(*args):
        return json.loads(subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                                         check=True).stdout or "null")

    def start(jid, cmd, mode="gpu"):
        spec = {"id": jid, "agent": "claude", "track": "track", "cmd": cmd, "mode": mode, "cwd": str(track),
                "roots": [str(track)]}
        return call("start", base64.b64encode(json.dumps(spec).encode()).decode())

    return call, start, track


def wait_status(call, jid, want, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = call("poll", jid, "0")
        if d["status"] in want:
            return d
        time.sleep(0.1)
    raise AssertionError(f"{jid} never reached {want}: {d}")


def test_jobrun_gpu_lock_serializes_and_snapshots(fake_pod):
    call, start, track = fake_pod
    start("j1", "sleep 1.5; python3 bench.py")
    wait_status(call, "j1", {"running"})
    start("j2", "python3 bench.py", mode="gpu")
    start("c1", "echo cpu-ok", mode="cpu")
    (track / "bench.py").write_text("print('SCORE cand: 9.99')\n")  # pushed after j2 was submitted
    d2 = call("poll", "j2", "0")
    assert d2["status"] == "queued" and d2["queue_pos"] == 1 and d2["holder"]["id"] == "j1"
    assert wait_status(call, "c1", {"done"})["exit_code"] == 0  # CPU job does not wait for the GPU lock
    snap = call("snapshot")
    assert {j["id"] for j in snap["jobs"]} == {"j1", "j2", "c1"}
    d1 = wait_status(call, "j1", {"done", "failed"})
    d2 = wait_status(call, "j2", {"done", "failed"})
    assert d2["started"] >= d1["ended"] - 0.05  # never overlapped
    assert b"SCORE cand: 1.05" in base64.b64decode(d2["chunk"])  # ran the snapshot, not the later push


def test_jobrun_failure_and_cancel(fake_pod):
    call, start, _ = fake_pod
    start("bad", "exit 3")
    assert wait_status(call, "bad", {"failed"})["exit_code"] == 3
    start("long", "sleep 30")
    wait_status(call, "long", {"running"})
    call("cancel", "long")
    assert call("poll", "long", "0")["status"] == "cancelled"
