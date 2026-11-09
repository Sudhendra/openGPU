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
from ogpu.providers import vast
from test_core import fake_pod, wait_status  # noqa: F401 — reuse the local on-machine runner fixture

REMOTE = Path(__file__).parent.parent / "src/ogpu/remote"


class MyCloud(providers.Provider):  # a user's own provider, loaded through [providers.mycloud] module = ...
    name = "mycloud"


def test_load_builtin_custom_and_unknown_providers():
    cfg = config.Config()
    assert providers.load("runpod", cfg).name == "runpod"
    assert providers.load("vast", cfg).name == "vast"
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


def test_vast_instance_parse():
    d = {"id": 42, "label": "ogpu", "intended_status": "running", "actual_status": "running", "dph_total": 1.87,
         "gpu_name": "H100 SXM", "public_ipaddr": "198.51.100.7\n", "ports": {"22/tcp": [{"HostPort": "41022"}]},
         "start_date": time.time() - 60}
    m = vast.machine(d)
    assert (m.id, m.status, m.ssh_host, m.ssh_port, m.cost_per_hr) == ("42", "RUNNING", "198.51.100.7", 41022, 1.87)
    assert 55 < m.uptime_s < 70
    proxied = vast.machine({"id": 1, "intended_status": "stopped", "actual_status": "exited",
                            "ssh_host": "ssh4.vast.ai", "ssh_port": 12345})
    assert (proxied.status, proxied.ssh_host, proxied.ssh_port, proxied.uptime_s) == ("EXITED", "ssh4.vast.ai", 12345,
                                                                                      None)


def test_vast_offer_query():
    q = vast.offer_query(config.Config(min_cuda="12.4"), "H100 SXM", "verified")
    assert q["gpu_name"] == {"eq": "H100 SXM"} and q["verified"] == {"eq": True}
    assert q["cuda_max_good"] == {"gte": 12.4} and q["order"] == [["dph_total", "asc"]]
    assert "verified" not in vast.offer_query(config.Config(), "H100 SXM", "any")


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
    monkeypatch.setenv("CONTAINER_ID", "42")
    monkeypatch.setenv("CONTAINER_API_KEY", "scoped")
    monkeypatch.setattr(rp, "load_cfg", lambda: {"provider": "vast"})
    rp.stop_pod()
    assert calls == [("PUT", "https://console.vast.ai/api/v0/instances/42/", "scoped", {"state": "stopped"})]
    calls.clear()
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
