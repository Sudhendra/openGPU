import base64
import importlib.util
from pathlib import Path

from test_core import fake_pod, wait_status  # noqa: F401 — reuse the local on-pod runner fixture

SRC = Path(__file__).parent.parent / "src/ogpu/remote/jobrun.py"
spec = importlib.util.spec_from_file_location("jobrun", SRC)
jobrun = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobrun)


def test_cpu_budget_cgroup_v1_quota(tmp_path):
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("2040000\n")  # the real H200 pod: 192 cores visible, 20.4 CPUs
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    allowed, eff = jobrun.cpu_budget(str(tmp_path))
    assert eff == min(20.4, len(allowed))


def test_cpu_budget_cgroup_v2_and_unlimited(tmp_path):
    (tmp_path / "cpu.max").write_text("800000 100000\n")
    assert jobrun.cpu_budget(str(tmp_path))[1] == min(8.0, len(jobrun.cpu_budget(str(tmp_path))[0]))
    (tmp_path / "cpu.max").write_text("max 100000\n")
    allowed, eff = jobrun.cpu_budget(str(tmp_path))
    assert eff == len(allowed)  # no quota → visible cores


def test_isolation_rules():
    allowed = list(range(192))
    env, cores, nice = jobrun.isolation("gpu", allowed, 20.4)
    assert env["MAX_JOBS"] == "20" and cores is None and nice == 0  # GPU jobs: quota-sized builds, unrestricted
    env, cores, nice = jobrun.isolation("cpu", allowed, 20.4)
    assert env["MAX_JOBS"] == "16" and cores == list(range(176, 192)) and nice == 19  # 4 CPUs of headroom kept
    assert 0 not in cores  # reserved headroom is the low cores
    env, cores, nice = jobrun.isolation("cpu", allowed, 20.4, quiet=False)
    assert env["MAX_JOBS"] == "20" and cores is None and nice == 0
    env, cores, _ = jobrun.isolation("cpu", [0, 1], 2.0)
    assert env["MAX_JOBS"] == "1" and cores == [1]  # tiny boxes still keep one core free


def test_cpu_job_gets_build_limits(fake_pod):  # noqa: F811
    call, start, track = fake_pod
    start("lim", 'echo "MAX_JOBS=$MAX_JOBS MAKEFLAGS=$MAKEFLAGS"', mode="cpu")
    d = wait_status(call, "lim", {"done"})
    out = base64.b64decode(d["chunk"]).decode()
    assert "MAX_JOBS=" in out and "MAKEFLAGS=-j" in out and "MAX_JOBS= " not in out
    snap = call("snapshot")
    iso = next(j for j in snap["jobs"] if j["id"] == "lim")["isolation"]
    assert iso["nice"] == 19 and iso["max_jobs"] >= 1


def test_user_max_jobs_wins(fake_pod):  # noqa: F811
    call, start, track = fake_pod
    start("own", 'MAX_JOBS=3 bash -c \'echo got=$MAX_JOBS\'', mode="gpu")
    d = wait_status(call, "own", {"done"})
    assert "got=3" in base64.b64decode(d["chunk"]).decode()
