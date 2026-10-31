import json

import pytest
from rich.console import Console

from ogpu import config, runner


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(config, "HOME", tmp_path)
    monkeypatch.setattr(runner, "HOME", tmp_path)
    monkeypatch.setattr(runner, "LOG_DIR", tmp_path / "logs")
    return tmp_path


def test_pool_config_defaults_and_overrides():
    cfg = config.Config(pod_name="ogpu-h200", idle_minutes=20)
    assert cfg.pool("timing") is cfg
    c = cfg.pool("correctness")
    assert c.pod_name == "ogpu-h200-correctness" and c.idle_minutes == 10
    assert c.gpu_type_ids[0] == "NVIDIA H100 NVL" and c.image == cfg.image  # same image, cheaper GPU
    c2 = config.Config(pools={"correctness": {"gpu_type_ids": ["NVIDIA H100 NVL"], "idle_minutes": 5}}).pool("correctness")
    assert c2.gpu_type_ids == ["NVIDIA H100 NVL"] and c2.idle_minutes == 5
    assert config.Config(pools={"correctness": {"enabled": False}}).pool("correctness") is None
    assert cfg.pool("nope") is None


def test_state_per_pool_and_legacy_file(home):
    (home / "state.json").write_text(json.dumps({"pod_id": "old", "host": "1.2.3.4", "port": 22}))  # v0 format
    assert config.load_state().pod_id == "old"  # legacy file = timing pool
    assert config.load_state("correctness").pod_id == ""
    config.save_state(config.State(pod_id="h100", host="5.6.7.8", port=2222), "correctness")
    assert config.load_state().pod_id == "old"  # timing preserved through the migration
    assert config.load_state("correctness").port == 2222
    assert "pools" in json.loads((home / "state.json").read_text())


def test_choose_pool_routes_and_falls_back(home, monkeypatch):
    cfg = config.Config()
    con = Console(quiet=True)
    warmed = []
    monkeypatch.setattr(runner, "warm_pool_in_background", lambda p: warmed.append(p) or True)

    monkeypatch.setattr(runner.ssh, "reachable", lambda st, k: False)
    assert runner.choose_pool(cfg, con, "gpu", "correctness") == "timing"  # cold pool: never wait on a boot
    assert warmed == ["correctness"]

    monkeypatch.setattr(runner.ssh, "reachable", lambda st, k: True)
    assert runner.choose_pool(cfg, con, "gpu", "correctness") == "correctness"
    assert runner.choose_pool(cfg, con, "gpu", "timing") == "timing"
    assert runner.choose_pool(cfg, con, "cpu", "correctness") == "timing"  # CPU gates stay with the code
    disabled = config.Config(pools={"correctness": {"enabled": False}})
    assert runner.choose_pool(disabled, con, "gpu", "correctness") == "timing"


def test_warm_up_is_single_flight(home, monkeypatch):
    import fcntl
    spawned = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: spawned.append(a[0]))
    with open(home / "up-correctness.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)  # an `ogpu up --pool correctness` is already running
        assert runner.warm_pool_in_background("correctness") is False
    assert spawned == []
    assert runner.warm_pool_in_background("correctness") is True
    assert spawned[0][-3:] == ["up", "--pool", "correctness"]


def test_fidelity_is_part_of_cache_key(tmp_path):
    from ogpu import ledger
    (tmp_path / "a.py").write_text("x=1")
    timing = ledger.content_key([tmp_path], "bench", "gpu", "img")
    correctness = ledger.content_key([tmp_path], "bench", "gpu:correctness", "img")
    assert timing != correctness  # an H100 correctness run must never answer an H200 timing request


def test_resume_failure_recreates_then_deletes_stuck_pod(home, monkeypatch):
    from ogpu import pod as podmod
    from ogpu.providers.runpod import Pod, RunPodError

    def mk(pid, status):
        return Pod(id=pid, name="ogpu-h200", status=status, cost_per_hr=5.29, gpu="H200 SXM", location="",
                   uptime_s=None, ssh_host="", ssh_port=0)

    events = []

    class FakeRP:
        def resume(self, pid):
            events.append(("resume", pid))
            raise RunPodError("There are not enough free GPUs on the host machine to start this pod.")

        def terminate(self, pid):
            events.append(("terminate", pid))

    monkeypatch.setattr(podmod, "client", lambda: FakeRP())
    monkeypatch.setattr(podmod, "current", lambda rp, cfg, pool: mk("stuck", "EXITED"))
    monkeypatch.setattr(podmod, "create_with_fallback",
                        lambda rp, cfg, con, retry, pool: events.append(("create", pool)) or mk("fresh", "RUNNING"))
    monkeypatch.setattr(podmod, "wait_reachable",
                        lambda rp, pid, cfg, con, pool: events.append(("reachable", pid)) or config.State(pod_id=pid))
    monkeypatch.setattr(podmod, "install_helpers", lambda st, cfg: None)
    monkeypatch.setattr(podmod, "import_pod_sessions", lambda st, cfg: None)
    monkeypatch.setattr(podmod, "gpu_health", lambda st, cfg: (True, "OK"))
    st = podmod.up(config.Config(), Console(quiet=True))
    assert st.pod_id == "fresh"
    # the stuck pod is only deleted after the replacement is reachable
    assert events == [("resume", "stuck"), ("create", "timing"), ("reachable", "fresh"), ("terminate", "stuck")]

    events.clear()
    with pytest.raises(SystemExit):
        podmod.up(config.Config(recreate_on_resume_failure=False), Console(quiet=True))
    assert events == [("resume", "stuck")]


def test_up_replaces_pod_that_fails_gpu_check(monkeypatch):
    """The Oct 30 H100 NVL had a CUDA 12.8 driver: every job died at torch CUDA init. `up` must never hand such a
    pod to agents — it deletes it and lands on another host."""
    from ogpu import pod as podmod
    from ogpu.providers.runpod import Pod
    mk = lambda pid, status: Pod(id=pid, name="n", status=status, cost_per_hr=3.19, gpu="H100 NVL", location="",
                                 uptime_s=None, ssh_host="h", ssh_port=22)  # noqa: E731
    events = []

    class FakeRP:
        def resume(self, pid):
            events.append(("resume", pid))

        def terminate(self, pid):
            events.append(("terminate", pid))

    health = iter([(False, "RuntimeError: The NVIDIA driver on your system is too old (found version 12080)"),
                   (True, "OK NVIDIA H100 NVL torch cuda 13.0")])
    monkeypatch.setattr(podmod, "client", lambda: FakeRP())
    monkeypatch.setattr(podmod, "current", lambda rp, cfg, pool: mk("old-driver", "EXITED"))
    monkeypatch.setattr(podmod, "create_with_fallback",
                        lambda rp, cfg, con, retry, pool: events.append(("create", pool)) or mk("good", "RUNNING"))
    monkeypatch.setattr(podmod, "wait_reachable", lambda rp, pid, cfg, con, pool: config.State(pod_id=pid))
    monkeypatch.setattr(podmod, "install_helpers", lambda st, cfg: events.append(("install", st.pod_id)))
    monkeypatch.setattr(podmod, "import_pod_sessions", lambda st, cfg: None)
    monkeypatch.setattr(podmod, "gpu_health", lambda st, cfg: next(health))
    monkeypatch.setattr(podmod, "save_state", lambda st, pool: None)
    st = podmod.up(config.Config().pool("correctness"), Console(quiet=True), pool="correctness")
    assert st.pod_id == "good"
    assert events == [("resume", "old-driver"), ("terminate", "old-driver"), ("create", "correctness"),
                      ("install", "good")]


def test_check_margin_parsed_from_judge_lines():
    from ogpu import ledger
    log = ("CHECK d1f s.py judge: PASS update relative L2 error 4.780e-03 (worst row 4.780e-03) [bound 0.070]; "
           "attention input relative L2 error 3.850e-03 (worst row 3.850e-03) [bound 0.070]\n"
           "CHECK v1f best.py judge: PASS relative L2 error 2.976e-02 (worst row 3.1e-02) [bound 0.064: 2.5 x the ...]\n")
    m = ledger.parse_metrics(log)
    assert m["check_margin"] == round(2.976e-02 / 0.064, 4)  # worst ratio across all checks/terms
    assert m["checks"] == "PASS"


def test_session_with_explicit_start_merges_with_polled_samples(tmp_path):
    import time as _t
    from ogpu import ledger
    db = ledger.connect(tmp_path / "l.db")
    start = _t.time() - 3600
    ledger.record_session(db, "pod", 1800, 4.0)                  # polled mid-session (start ≈ now-1800)
    ledger.record_session(db, "pod", 3600, 4.0, start=start)     # reaper's end-of-session record for an earlier start
    ledger.record_session(db, "pod", 3590, 4.0, start=start + 5)  # same session seen again: merged, not doubled
    rows = db.execute("select count(*) n, sum(uptime_s) u from sessions").fetchone()
    assert rows["n"] == 2 and rows["u"] == 1800 + 3600
