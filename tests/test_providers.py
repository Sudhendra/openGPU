"""Providers (built in + user plugins), profiles, opt-in cache, and the security fixes."""
import base64
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest
from rich.console import Console

from ogpu import config, ledger, providers, runner, ssh
from test_core import fake_pod, wait_status  # noqa: F401 — reuse the local on-machine runner fixture

REMOTE = Path(__file__).parent.parent / "src/ogpu/remote"


class MyCloud(providers.Provider):  # a user's own provider, loaded through [providers.mycloud] module = ...
    name = "mycloud"


def test_load_builtin_custom_and_unknown_providers():
    cfg = config.Config()
    assert providers.load("runpod", cfg).name == "runpod"
    custom = providers.load("mycloud", config.Config(providers={"mycloud": {"module": "test_providers:MyCloud",
                                                                            "region": "x"}}))
    assert isinstance(custom, MyCloud) and custom.opts == {"region": "x"}
    with pytest.raises(SystemExit, match="unknown provider"):
        providers.load("nope", cfg)
    with pytest.raises(SystemExit, match="serverless"):
        providers.load("modal", cfg)


def test_static_ssh_host_provider():
    p = providers.load("ssh", config.Config(providers={"ssh": {"host": "10.0.0.5", "port": 2200, "user": "me",
                                                               "cost_per_hr": 0.4, "stop_cmd": "true"}}))
    m = p.create(config.Config(), "", "")
    assert (m.ssh_host, m.ssh_port, m.ssh_user, m.status, m.cost_per_hr) == ("10.0.0.5", 2200, "me", "RUNNING", 0.4)
    assert p.reaper_stop() == {"provider": "ssh", "stop_cmd": "true"}


def test_config_is_generic_by_default_and_okc_is_a_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOME", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    cfg = config.load_config()  # writes the default file
    assert cfg.profile == "" and cfg.cache is False and cfg.provider == "runpod"
    assert "sglang" not in cfg.image and cfg.pool("correctness") is None and cfg.remote_root == "/workspace/src"
    (tmp_path / "config.toml").write_text('profile = "okc"\nidle_minutes = 7\n[pools.correctness]\nidle_minutes = 3\n')
    okc = config.load_config()
    assert okc.image.startswith("lmsysorg/sglang") and okc.gpu_type_ids == ["NVIDIA H200"]
    assert okc.idle_minutes == 7  # the file beats the profile
    c = okc.pool("correctness")
    assert c.idle_minutes == 3 and c.gpu_type_ids[0] == "NVIDIA H100 NVL"  # pool keys merge with the profile's
    (tmp_path / "config.toml").write_text('profile = "nope"\n')
    with pytest.raises(SystemExit, match="unknown profile"):
        config.load_config()


@pytest.fixture
def local_submit(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "HOME", tmp_path)
    monkeypatch.setattr(runner, "LOG_DIR", tmp_path / "logs")
    real = ledger.connect
    monkeypatch.setattr(ledger, "connect", lambda *a, **k: real(tmp_path / "l.db"))
    started = []
    monkeypatch.setattr(runner, "push_and_start", lambda *a: started.append(a[3]))
    track = tmp_path / "t"
    track.mkdir()
    (track / "a.py").write_text("x = 1\n")
    db = real(tmp_path / "l.db")
    return track, started, db


def test_cache_is_opt_in(local_submit):
    track, started, db = local_submit
    cfg = config.Config(budget_usd=1e9)
    con = Console(quiet=True)
    first = runner.submit(cfg, con, track=track, extra=[], cmd="x", mode="gpu", cache=True)
    db.execute("update runs set status='done', exit_code=0, ended=1 where id=?", (first["id"],))
    db.commit()
    assert runner.submit(cfg, con, track=track, extra=[], cmd="x", mode="gpu")["status"] == "queued"  # default: off
    hit = runner.submit(cfg, con, track=track, extra=[], cmd="x", mode="gpu", cache=True)
    assert hit["cached_from"] == first["id"]
    assert runner.submit(config.Config(budget_usd=1e9, cache=True), con, track=track, extra=[], cmd="x",
                         mode="gpu")["cached_from"] == first["id"]  # or on for everything via the config
    assert len(started) == 2


def test_spend_counts_serverless_jobs_and_numeric_scores(tmp_path):
    db = ledger.connect(tmp_path / "l.db")
    ledger.record(db, id="m", key=None, agent="a", track="t", cmd="c", mode="gpu", backend="modal", status="done",
                  exit_code=0, started=time.time() - 10, ended=time.time(), wall_s=10, cost_usd=0.5,
                  metrics={"score": 2.5}, log_path="x", cached_from=None)
    assert ledger.spend(db) == pytest.approx(0.5)  # no machine session, but it still counts toward the budget
    assert ledger.best_score({"score": 2.5}) == 2.5 and "score 2.5" in ledger.headline({"score": 2.5})


def test_jobrun_quotes_cwd_and_rejects_bad_ids(fake_pod, tmp_path):  # noqa: F811
    call, start, track = fake_pod
    evil = track.parent / "$(touch PWNED)"
    evil.mkdir()
    spec = {"id": "j1", "agent": "a", "track": evil.name, "cmd": "pwd", "mode": "cpu", "cwd": str(evil),
            "roots": [str(evil)]}
    call("start", base64.b64encode(json.dumps(spec).encode()).decode())
    d = wait_status(call, "j1", ("done", "failed"))
    assert d["status"] == "done"
    assert not list(tmp_path.rglob("PWNED")) and not Path("PWNED").exists()  # the dir name was never executed
    for bad in ("../x", "a/b", ""):
        with pytest.raises(Exception):
            call("poll", bad, "0")


def test_private_dirs_are_locked_down(tmp_path):
    d = tmp_path / "socks"
    d.mkdir(mode=0o755)
    ssh.private_dir(d)
    assert os.stat(d).st_mode & 0o777 == 0o700


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_reaper_stops_through_the_configured_provider(monkeypatch):
    rp = load("reaper_stop_test", REMOTE / "reaper.py")
    calls = []
    monkeypatch.setattr(rp, "http", lambda method, url, key, body: calls.append((method, url, key, body)) or b"{}")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod1")
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(rp, "load_cfg", lambda: {"provider": "runpod"})
    rp.stop_pod()
    assert calls[0][1] == "https://api.runpod.io/graphql" and "api_key" not in calls[0][1]  # key in a header
    ran = []
    monkeypatch.setattr(rp.subprocess, "run", lambda argv, **k: ran.append(argv) or type("P", (), {"returncode": 0}))
    monkeypatch.setattr(rp, "load_cfg", lambda: {"provider": "ssh", "stop_cmd": "sudo shutdown -h now"})
    rp.stop_pod()
    assert ran == [["bash", "-c", "sudo shutdown -h now"]]


def test_pid_alive_reaps_own_finished_child():
    import subprocess
    p = subprocess.Popen(["true"])
    time.sleep(0.3)  # exited but not waited on: a zombie, which os.kill(pid, 0) alone reports as alive
    assert runner.pid_alive(p.pid) is False
    q = subprocess.Popen(["sleep", "5"])
    assert runner.pid_alive(q.pid) is True
    q.kill()


def test_cache_key_includes_target_hardware(local_submit):
    track, started, db = local_submit
    con = Console(quiet=True)
    h100 = config.Config(budget_usd=1e9, gpu_type_ids=["NVIDIA H100 80GB HBM3"])
    first = runner.submit(h100, con, track=track, extra=[], cmd="x", mode="gpu", cache=True)
    db.execute("update runs set status='done', exit_code=0, ended=1 where id=?", (first["id"],))
    db.commit()
    l4 = config.Config(budget_usd=1e9, provider="modal", gpu_type_ids=["L4"])
    assert runner.submit(h100, con, track=track, extra=[], cmd="x", mode="gpu", cache=True).get("cached_from")
    monkey_serverless = runner.start_serverless
    runner.start_serverless = lambda *a: {"pid": 0, "results": "/nonexistent", "log": "", "gpu": "L4", "rate": 1.0}
    try:
        assert runner.submit(l4, con, track=track, extra=[], cmd="x", mode="gpu", cache=True)["status"] == "queued"
    finally:
        runner.start_serverless = monkey_serverless


def test_cache_never_replays_infrastructure_failures(tmp_path):
    db = ledger.connect(tmp_path / "l.db")
    base = dict(agent="a", track="t", cmd="c", mode="gpu", backend="runpod", status="failed", exit_code=1, started=1,
                ended=2, wall_s=1, cost_usd=0, metrics={}, log_path="", cached_from=None)
    for i, failure in enumerate(("launch_error", "timeout", "hang", "oom", "cuda_error")):
        ledger.record(db, id=f"x{i}", key="K", failure=failure, **base)
    assert ledger.cached(db, "K") is None
    ledger.record(db, id="y", key="K", failure="compile_error", **base)  # the code's own fault: replaying is fine
    assert ledger.cached(db, "K")["id"] == "y"


def test_config_typos_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOME", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text("budget = 1.0\n")
    with pytest.raises(SystemExit, match="did you mean 'budget_usd'"):
        config.load_config()
    (tmp_path / "config.toml").write_text("[pools.correctness]\nprovder = 'modal'\n")
    with pytest.raises(SystemExit, match="did you mean 'provider'"):
        config.load_config()
    (tmp_path / "config.toml").write_text("[pools.cheap]\nprovider = 'modal'\n")
    with pytest.raises(SystemExit, match="unknown pool"):
        config.load_config()
    (tmp_path / "config.toml").write_text("[providers.modal]\ncpu = 1\n[pools.correctness]\nenabled = false\n")
    assert config.load_config().providers == {"modal": {"cpu": 1}}  # provider options are free-form


def test_budget_applies_to_cpu_jobs_and_in_flight_serverless(local_submit):
    track, started, db = local_submit
    now = time.time()
    ledger.record(db, id="m", key=None, agent="a", track="t", cmd="c", mode="cpu", backend="modal", status="running",
                  exit_code=None, started=None, ended=None, wall_s=None, cost_usd=None, metrics={}, log_path="",
                  cached_from=None, submitted=now - 3600,
                  handle=json.dumps({"pid": os.getpid(), "results": "/nonexistent", "log": "", "gpu": "none",
                                     "rate": 2.0}))
    assert ledger.spend(db, now - 86400) == pytest.approx(2.0, rel=0.01)  # still running: billed as it goes
    with pytest.raises(runner.BudgetExceeded):
        runner.submit(config.Config(budget_usd=1.5), Console(quiet=True), track=track, extra=[], cmd="x", mode="cpu")


def test_unwaited_serverless_job_is_settled(local_submit, tmp_path):
    track, started, db = local_submit
    results = tmp_path / "res" / "results.json"
    results.parent.mkdir()
    (results.parent / "0.log").write_text("OGPU_METRIC score=3\n")
    results.write_text(json.dumps([{"i": 0, "cmd": "x", "rc": 0, "wall_s": 10, "billed_s": 36, "ended": time.time(),
                                    "log_path": str(results.parent / "0.log")}]))
    ledger.record(db, id="s1", key=None, agent="a", track="t", cmd="x", mode="gpu", backend="modal", status="queued",
                  exit_code=None, started=None, ended=None, wall_s=None, cost_usd=None, metrics={},
                  log_path=str(tmp_path / "s1.log"), cached_from=None, track_path=str(track), submitted=time.time(),
                  handle=json.dumps({"pid": 999999, "results": str(results), "log": "", "gpu": "T4", "rate": 1.0}))
    runner.settle_serverless(config.Config(), db)
    row = ledger.get(db, "s1")
    assert row["status"] == "done" and row["cost_usd"] == pytest.approx(0.01)  # 36 billed seconds at $1/hr
    assert json.loads(row["metrics"])["score"] == 3


def test_modal_container_is_small_and_priced_in_full():
    from ogpu.providers import modal
    cfg = config.Config()
    assert modal.container(cfg) == (2.0, 8)
    assert modal.rate("T4", 2, 8) == pytest.approx(0.59 + 2 * 0.04716 + 8 * 0.007992, rel=1e-3)
    big = config.Config(providers={"modal": {"cpu": 8, "memory_gb": 32}})
    assert modal.rate("T4", *modal.container(big)) > 1.2  # CPU + memory can cost more than a small GPU
