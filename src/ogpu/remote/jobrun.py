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


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def reconcile(j):
    """A job whose runner died (pod restart, OOM-kill) is marked lost instead of hanging forever."""
    starting = j.get("runner_pid") is None and time.time() - j["created"] < 30
    if j["status"] in ("queued", "running") and not starting and not alive(j.get("runner_pid")):
        j.update(status="lost", ended=time.time(), exit_code=-1)
        write_job(j)
    return j


def queue_position(job, jobs):
    if job["status"] != "queued" or job["mode"] != "gpu":
        return 0
    q = [j for j in jobs if j["status"] == "queued" and j["mode"] == "gpu"]
    return 1 + [j["id"] for j in q].index(job["id"]) if job["id"] in [j["id"] for j in q] else 0


def holder():
    try:
        with open(HOLDER) as f:
            h = json.load(f)
        return h if alive(h.get("runner_pid")) else None
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
    job.update(status="queued", created=time.time(), started=None, ended=None, exit_code=None, runner_pid=None)
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


def cmd_exec(jid):
    job = read_job(jid)
    job["runner_pid"] = os.getpid()
    write_job(job)
    lockf = None
    if job["mode"] == "gpu":
        lockf = open(LOCK, "a")
        fcntl.flock(lockf, fcntl.LOCK_EX)
        with open(HOLDER, "w") as f:
            json.dump({k: job.get(k) for k in ("id", "agent", "track", "cmd")} | {"runner_pid": os.getpid(),
                      "since": time.time()}, f)
    job.update(status="running", started=time.time())
    write_job(job)
    touch()
    shell = f"[ -f {OG}/env.sh ] && source {OG}/env.sh; cd {json.dumps(job['cwd'])} && {job['cmd']}"
    with open(jpath(jid, "log"), "ab", buffering=0) as log:
        p = subprocess.Popen(["bash", "-c", shell], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        job["pid"] = p.pid
        write_job(job)
        rc = p.wait()
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
                      "started": job["started"], "ended": job["ended"]}))


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
    for pid in (job.get("pid"), job.get("runner_pid")):
        try:
            os.killpg(pid, signal.SIGTERM)
        except (OSError, TypeError):
            pass
    job.update(status="cancelled", ended=time.time(), exit_code=-15)
    write_job(job)


if __name__ == "__main__":
    a = sys.argv[1:]
    {"start": lambda: cmd_start(a[1]), "_exec": lambda: cmd_exec(a[1]), "poll": lambda: cmd_poll(a[1], a[2]),
     "snapshot": cmd_snapshot, "cancel": lambda: cmd_cancel(a[1])}[a[0]]()
