#!/usr/bin/env python3
"""On-pod job runner (stdlib only; copied to /workspace/.ogpu/bin by `ogpu up`).

  jobrun.py start <job.json b64>   register a job and run it detached (survives SSH drops)
  jobrun.py poll <id> <offset>     JSON: status, queue position, lock holder, log bytes from offset
  jobrun.py snapshot               JSON: all recent jobs + lock holder + reaper state (for `ogpu top`)
  jobrun.py cancel <id>            kill a job's process group

GPU jobs serialize on an flock over /tmp/h200.lock so timings from different agents never overlap; CPU jobs
(ptxas/nvcc gates) run immediately and in parallel.
"""
import base64
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

OG = "/workspace/.ogpu"
JOBS = f"{OG}/jobs"
LOCK = "/tmp/h200.lock"
HOLDER = f"{OG}/lock_holder.json"
ACTIVITY = f"{OG}/last_activity"
RECENT_S = 6 * 3600
RUNS = "/workspace/okc/.runs"   # per-job snapshots: a queued job never sees code pushed after it was submitted
KEEP_RUNS_S = 24 * 3600


def jpath(jid, ext="json"):
    return f"{JOBS}/{jid}.{ext}"


def read_job(jid):
    try:
        with open(jpath(jid)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_job(job):
    tmp = jpath(job["id"]) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(job, f)
    os.replace(tmp, jpath(job["id"]))


def touch():
    with open(ACTIVITY, "w") as f:
        f.write(str(int(time.time())))


def all_jobs():
    out = []
    for n in os.listdir(JOBS):
        if n.endswith(".json"):
            j = read_job(n[:-5])
            if j:
                out.append(j)
    return sorted(out, key=lambda j: j["created"])


def boot_id(proc="/proc"):
    """This container's start time (PID 1's start). Job records live on the persistent volume and outlive the
    container; after a stop/resume, PIDs are recycled, so a pid is only meaningful together with the boot it
    was recorded in."""
    try:
        with open(f"{proc}/1/stat") as f:
            ticks = int(f.read().rsplit(")", 1)[1].split()[19])
        with open(f"{proc}/stat") as f:
            btime = next(int(l.split()[1]) for l in f if l.startswith("btime"))
        return round(btime + ticks / os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, StopIteration, IndexError):
        return None


BOOT = boot_id()


def same_boot(rec):
    """False when the record was written by an earlier container (its pids now belong to unrelated processes)."""
    if BOOT is None:
        return True
    b = rec.get("boot")
    if b is not None:
        return abs(b - BOOT) <= 2
    return (rec.get("created") or rec.get("since") or 0) >= BOOT  # legacy record: judge by its timestamp


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def reconcile(j):
    """A job whose runner died is marked lost instead of hanging forever — and its own processes are killed, so an
    orphan can never keep using the GPU (or hold the inherited GPU lock) behind everyone's back."""
    if j["status"] in ("queued", "running") and not same_boot(j):
        # the container restarted under it: its pids are recycled, so never signal them — just close the record
        j.update(status="lost", ended=time.time(), exit_code=-1, orphan_killed=False, lost_reason="pod restarted")
        write_job(j)
        return j
    starting = j.get("runner_pid") is None and time.time() - j["created"] < 30
    if j["status"] in ("queued", "running") and not starting and not alive(j.get("runner_pid")):
        orphan = alive(j.get("pid"))
        if orphan:
            kill_group(j["pid"], signal.SIGKILL)
        j = read_job(j["id"]) or j  # merge with anything written since (e.g. runner_signal)
        j.update(status="lost", ended=time.time(), exit_code=-1, orphan_killed=orphan)
        write_job(j)
    return j


def kill_group(pid, sig):
    try:
        os.killpg(pid, sig)
        return True
    except (OSError, TypeError):
        return False


def queue_position(job, jobs):
    if job["status"] != "queued" or job["mode"] != "gpu":
        return 0
    q = [j for j in jobs if j["status"] == "queued" and j["mode"] == "gpu"]
    return 1 + [j["id"] for j in q].index(job["id"]) if job["id"] in [j["id"] for j in q] else 0


def holder():
    try:
        with open(HOLDER) as f:
            h = json.load(f)
        return h if same_boot(h) and alive(h.get("runner_pid")) else None
    except (OSError, ValueError):
        return None


def cmd_start(spec_b64):
    job = json.loads(base64.b64decode(spec_b64))
    os.makedirs(JOBS, exist_ok=True)
    prune_runs()
    run_dir = f"{RUNS}/{job['id']}"
    os.makedirs(run_dir)
    for root in job.get("roots") or [job["cwd"]]:
        subprocess.run(["cp", "-a", root, run_dir + "/"], check=True)
    job["cwd"] = f"{run_dir}/{os.path.basename(job['cwd'])}"
    job.update(status="queued", created=time.time(), started=None, ended=None, exit_code=None, runner_pid=None,
               boot=BOOT)
    write_job(job)
    open(jpath(job["id"], "log"), "wb").close()
    touch()
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "_exec", job["id"]],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    print(json.dumps({"id": job["id"], "runner_pid": p.pid}))


def prune_runs():
    if not os.path.isdir(RUNS):
        return
    now = time.time()
    for n in os.listdir(RUNS):
        p = f"{RUNS}/{n}"
        if now - os.path.getmtime(p) > KEEP_RUNS_S:
            subprocess.run(["rm", "-rf", p])


def wait_with_timeout(p, timeout_s, log):
    """Exit 124 on timeout like coreutils `timeout`: SIGTERM the job's process group, SIGKILL 30s later."""
    try:
        return p.wait(timeout=float(timeout_s) if timeout_s else None)
    except subprocess.TimeoutExpired:
        log.write(f"\nogpu: timed out after {timeout_s}s\n".encode())
        for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 5)):
            try:
                os.killpg(p.pid, sig)
            except OSError:
                pass
            try:
                p.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        return 124


def cpu_budget(cgroup_root="/sys/fs/cgroup"):
    """→ (allowed cpu ids, effective cpus). Containers often see every host core (nproc=192) but are throttled to a
    CFS quota (e.g. 20.4 CPUs); build tools that size themselves from nproc then thrash the quota and throttle
    everything in the container — including a timing run."""
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    eff = float(len(allowed))
    try:
        with open(f"{cgroup_root}/cpu.max") as f:  # cgroup v2: "<quota> <period>" or "max <period>"
            q, per = f.read().split()[:2]
        if q != "max":
            eff = min(eff, int(q) / int(per))
    except (OSError, ValueError):
        try:  # cgroup v1
            with open(f"{cgroup_root}/cpu/cpu.cfs_quota_us") as f:
                q = int(f.read())
            with open(f"{cgroup_root}/cpu/cpu.cfs_period_us") as f:
                per = int(f.read())
            if q > 0:
                eff = min(eff, q / per)
        except (OSError, ValueError):
            pass
    return allowed, max(1.0, eff)


def isolation(mode, allowed, eff, quiet=True):
    """Build parallelism sized to the real CPU budget for every job; CPU jobs additionally leave headroom for
    timing (reserved cores, lowest priority). → (env overrides, cpu ids to pin or None, nice)"""
    total = max(1, int(eff))
    if mode != "cpu" or not quiet:
        n, cores, nice = total, None, 0
    else:
        reserve = max(1, min(4, total // 4))
        n = max(1, total - reserve)
        cores, nice = allowed[-n:], 19  # reserved headroom = the lowest-numbered cores
    env = {"MAX_JOBS": str(n), "CMAKE_BUILD_PARALLEL_LEVEL": str(n), "MAKEFLAGS": f"-j{n}"}
    return env, cores, nice


def cmd_exec(jid):
    job = read_job(jid)
    job["runner_pid"] = os.getpid()
    write_job(job)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # a dropped SSH session must never take a runner down

    def on_term(signum, _frame):  # record why the runner is going away, then take the job with it
        j = read_job(jid) or job
        j["runner_signal"] = signal.Signals(signum).name
        write_job(j)
        if job.get("pid"):
            kill_group(job["pid"], signal.SIGKILL)
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    lockf = None
    if job["mode"] == "gpu":
        lockf = open(LOCK, "a")
        fcntl.flock(lockf, fcntl.LOCK_EX)
        with open(HOLDER, "w") as f:
            json.dump({k: job.get(k) for k in ("id", "agent", "track", "cmd")} | {"runner_pid": os.getpid(),
                      "since": time.time(), "boot": BOOT}, f)
    job = read_job(jid) or job
    job.update(status="running", started=time.time())
    write_job(job)
    touch()
    allowed, eff = cpu_budget()
    env_over, cores, nice = isolation(job["mode"], allowed, eff, job.get("quiet", True))
    job["isolation"] = {"cpu_budget": round(eff, 1), "max_jobs": int(env_over["MAX_JOBS"]),
                        "pinned_cores": len(cores) if cores else None, "nice": nice}
    write_job(job)
    exports = " ".join(f"{k}={v}" for k, v in env_over.items())
    # env.sh first (container env), then our overrides so they win; a user's own MAX_JOBS in the cmd still wins
    shell = (f"[ -f {OG}/env.sh ] && source {OG}/env.sh; export {exports}; "
             f"cd {json.dumps(job['cwd'])} && {job['cmd']}")

    def limit():
        if cores and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cores)
        if nice:
            os.nice(nice)

    with open(jpath(jid, "log"), "ab", buffering=0) as log:
        try:
            # the job inherits the GPU lock's file description: if this runner dies, the lock stays held until
            # the job itself exits, so a running orphan can never overlap another agent's timing
            p = subprocess.Popen(["bash", "-c", shell], stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True, preexec_fn=limit,
                                 pass_fds=(lockf.fileno(),) if lockf else ())
            job["pid"] = p.pid
            write_job(job)
            rc = wait_with_timeout(p, job.get("timeout_s"), log)
            kill_group(p.pid, signal.SIGKILL)  # reap leftovers (background children) so nothing outlives the job
        except OSError as e:
            log.write(f"ogpu: could not start job: {e}\n".encode())
            rc = 127
    job = read_job(jid) or job  # keep fields the watchdog may have added (hang evidence)
    if job.get("status") != "cancelled":
        job.update(status="done" if rc == 0 else "failed", ended=time.time(), exit_code=rc)
        write_job(job)
    if lockf:
        try:
            os.remove(HOLDER)
        except OSError:
            pass
        fcntl.flock(lockf, fcntl.LOCK_UN)
    touch()


def cmd_poll(jid, offset):
    job = read_job(jid)
    if not job:
        print(json.dumps({"status": "missing"}))
        return
    job = reconcile(job)
    with open(jpath(jid, "log"), "rb") as f:
        f.seek(int(offset))
        chunk = f.read(1 << 20)
    jobs = [reconcile(j) for j in all_jobs()]
    print(json.dumps({"status": job["status"], "exit_code": job["exit_code"], "queue_pos": queue_position(job, jobs),
                      "holder": holder(), "chunk": base64.b64encode(chunk).decode(), "offset": int(offset) + len(chunk),
                      "started": job["started"], "ended": job["ended"], "hang": job.get("hang"),
                      "runner_signal": job.get("runner_signal"), "orphan_killed": job.get("orphan_killed")}))


def cmd_snapshot():
    now = time.time()
    jobs = [reconcile(j) for j in all_jobs() if j["status"] in ("queued", "running") or now - (j["ended"] or now) < RECENT_S]
    for j in jobs:
        j["queue_pos"] = queue_position(j, jobs)
    try:
        with open(f"{OG}/reaper.json") as f:
            reaper = json.load(f)
    except (OSError, ValueError):
        reaper = None
    print(json.dumps({"now": now, "jobs": jobs, "holder": holder(), "reaper": reaper}))


def cmd_cancel(jid):
    job = read_job(jid)
    job.update(status="cancelled", ended=time.time(), exit_code=-15)  # status first: the runner won't overwrite it
    write_job(job)
    for pid in (job.get("pid"), job.get("runner_pid")):
        kill_group(pid, signal.SIGTERM)


if __name__ == "__main__":
    a = sys.argv[1:]
    {"start": lambda: cmd_start(a[1]), "_exec": lambda: cmd_exec(a[1]), "poll": lambda: cmd_poll(a[1], a[2]),
     "snapshot": cmd_snapshot, "cancel": lambda: cmd_cancel(a[1])}[a[0]]()
