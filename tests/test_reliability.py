"""Oct 31 incidents: a correctness pod on a too-old driver, a dead reaper that let the H200 idle for hours, and job
records that outlive their container (recycled pids)."""
import importlib.util
import json
import sys
from pathlib import Path

from ogpu import config, pod

REMOTE = Path(__file__).parent.parent / "src/ogpu/remote"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_health():
    assert pod.parse_health(0, "OK NVIDIA H100 NVL torch cuda 13.0\n") == (True, "OK NVIDIA H100 NVL torch cuda 13.0")
    assert pod.parse_health(0, "SKIP no torch: x\n")[0] is True
    ok, why = pod.parse_health(1, "Traceback...\nRuntimeError: The NVIDIA driver on your system is too old "
                                  "(found version 12080).\n  more\n")
    assert not ok and "driver on your system is too old" in why
    assert pod.parse_health(255, "")[0] is False


def test_allowed_cuda_versions():
    assert config.Config().allowed_cuda_versions[0] == "13.0" and "12.8" not in config.Config().allowed_cuda_versions
    assert config.Config(min_cuda="").allowed_cuda_versions is None
    assert config.Config().pool("correctness").allowed_cuda_versions == config.Config().allowed_cuda_versions


def test_reaper_alive():
    assert pod.reaper_alive("1000", "1100")
    assert not pod.reaper_alive("1000", "1200")
    assert not pod.reaper_alive("", "1200")  # no reaper.json at all


def test_remote_files_include_guard():
    assert "reaperguard.sh" in pod.REMOTE_FILES
    guard = (REMOTE / "reaperguard.sh").read_text()
    assert "flock -n 9" in guard and "setsid" in guard and "9>&-" in guard  # the reaper must not inherit the lock


def fake_proc(tmp_path, start_ticks, btime=1_000_000):
    proc = tmp_path / f"proc{start_ticks}"
    (proc / "1").mkdir(parents=True)
    fields = ["S"] + ["0"] * 18 + [str(start_ticks)] + ["0"] * 10
    (proc / "1" / "stat").write_text("1 (sleep) " + " ".join(fields))
    (proc / "stat").write_text(f"cpu 0 0\nbtime {btime}\n")
    return str(proc)


def test_stale_boot_job_is_lost_without_signalling(tmp_path, monkeypatch):
    jr = load("jobrun_rel", REMOTE / "jobrun.py")
    import os
    hz = os.sysconf("SC_CLK_TCK")
    assert jr.boot_id(fake_proc(tmp_path, 50 * hz)) == 1_000_050
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(jr, "JOBS", str(jobs))
    monkeypatch.setattr(jr, "BOOT", 2_000_000)
    killed = []
    monkeypatch.setattr(jr, "kill_group", lambda pid, sig: killed.append(pid))
    me = os.getpid()  # a "recycled" pid that is alive in this container
    old = {"id": "old", "status": "running", "created": 1_500_000, "boot": 1_500_000, "runner_pid": me, "pid": me,
           "mode": "gpu", "exit_code": None, "ended": None}
    jr.write_job(old)
    j = jr.reconcile(old)
    assert j["status"] == "lost" and j["lost_reason"] == "pod restarted" and killed == []
    legacy = dict(old, id="legacy", status="queued", ended=None)
    legacy.pop("boot")
    jr.write_job(legacy)
    assert jr.reconcile(legacy)["status"] == "lost"  # no boot tag, created before this container started
    fresh = dict(old, id="fresh", status="running", created=2_000_100, boot=2_000_000, ended=None)
    jr.write_job(fresh)
    assert jr.reconcile(fresh)["status"] == "running"  # same container, live runner: untouched
    assert jr.same_boot({"boot": 2_000_001}) and not jr.same_boot({"since": 1_000})


def test_reaper_busy_ignores_records_from_earlier_container(tmp_path, monkeypatch):
    jr = load("jobrun", REMOTE / "jobrun.py")
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(jr, "JOBS", str(jobs))
    monkeypatch.setattr(jr, "BOOT", 2_000_000)
    import os
    jr.write_job({"id": "ghost", "status": "running", "created": 1_500_000, "boot": 1_500_000,
                  "runner_pid": os.getpid(), "pid": None, "mode": "gpu", "exit_code": None, "ended": None})
    sys.modules["jobrun"] = jr
    try:
        rp = load("reaper_rel", REMOTE / "reaper.py")
        assert rp.busy() is False  # before the fix, the live recycled pid kept the pod "busy" forever
        jr.write_job({"id": "real", "status": "running", "created": 2_000_100, "boot": 2_000_000,
                      "runner_pid": os.getpid(), "pid": None, "mode": "gpu", "exit_code": None, "ended": None})
        assert rp.busy() is True
    finally:
        sys.modules.pop("jobrun", None)


def test_reaper_tick_survives_errors_and_writes_heartbeat(tmp_path, monkeypatch):
    rp = load("reaper_tick", REMOTE / "reaper.py")
    monkeypatch.setattr(rp, "STATE", str(tmp_path / "reaper.json"))
    monkeypatch.setattr(rp, "busy", lambda: True)
    monkeypatch.setattr(rp, "gpu_sample", lambda: None)
    monkeypatch.setattr(rp, "watchdog", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(rp, "idle_minutes", lambda: 20.0)
    assert rp.tick(123.0) == 15
    st = json.loads((tmp_path / "reaper.json").read_text())
    assert st["busy"] and st["checked"] == 123.0 and "pid" in st
