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
    assert c.gpu_type_ids[0] == "NVIDIA H100 PCIe" and c.image == cfg.image  # same image, cheaper GPU
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
    st = podmod.up(config.Config(), Console(quiet=True))
    assert st.pod_id == "fresh"
    # the stuck pod is only deleted after the replacement is reachable
    assert events == [("resume", "stuck"), ("create", "timing"), ("reachable", "fresh"), ("terminate", "stuck")]

    events.clear()
    with pytest.raises(SystemExit):
        podmod.up(config.Config(recreate_on_resume_failure=False), Console(quiet=True))
    assert events == [("resume", "stuck")]
