#!/usr/bin/env python3
"""On-pod idle reaper (stdlib only). Stops this pod via the RunPod API once no job has been queued or running for
`idle_minutes`. Stopping keeps /workspace (code, compiled-extension cache, job history); `ogpu up` resumes it."""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request

OG = "/workspace/.ogpu"
PIDFILE = f"{OG}/reaper.pid"
STATE = f"{OG}/reaper.json"


try:  # same directory on the pod; tests load it explicitly
    import jobrun
except ImportError:  # pragma: no cover
    jobrun = None

TERM_GRACE_S = 20


def gpu_sample():
    """(util %, power W, power limit W) of GPU 0, or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,power.limit",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        u, p, lim = (float(x) for x in out.splitlines()[0].split(",")[:3])
        return u, p, lim
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def hang_reason(job, elapsed, silent, gpu):
    """Why this running job looks hung, or None. Two independent signals:
    - history: it has run far longer than this exact command usually takes, and its log has gone quiet;
    - spin: a GPU job whose log has been silent for long while the GPU sits at ~100% util on low power
      (busy-waiting kernels — a deadlocked barrier/cluster — not real work)."""
    after = job.get("hang_after_s")
    if after and elapsed > after and silent > job.get("hang_silent_s", 60):
        usual = job.get("usual_s")
        return f"running {elapsed:.0f}s; this command usually takes ~{usual:.0f}s" if usual else \
            f"running {elapsed:.0f}s (limit {after:.0f}s) with a silent log"
    spin_after = job.get("spin_silent_s", 900)
    if job.get("mode") == "gpu" and gpu and silent > spin_after:
        util, power, limit = gpu
        if util >= 95 and limit and power < 0.3 * limit:
            return (f"GPU busy-waiting: {util:.0f}% util at {power:.0f}W/{limit:.0f}W with no log output "
                    f"for {silent / 60:.0f} min (deadlocked kernel?)")
    return None


def watchdog(now=None, gpu=None):
    """Kill hung jobs (TERM, then KILL after a grace period) and clean up orphans, even when nobody is polling."""
    if jobrun is None:
        return
    now = now or time.time()
    for j in jobrun.all_jobs():
        j = jobrun.reconcile(j)
        if j["status"] != "running" or not j.get("started") or not j.get("pid"):
            continue
        hang = j.get("hang") or {}
        if hang.get("term_at"):
            if now - hang["term_at"] > TERM_GRACE_S and jobrun.alive(j["pid"]):
                jobrun.kill_group(j["pid"], signal.SIGKILL)
            continue
        try:
            silent = now - os.path.getmtime(jobrun.jpath(j["id"], "log"))
        except OSError:
            silent = 0.0
        reason = hang_reason(j, now - j["started"], silent, gpu)
        if reason:
            j["hang"] = {"reason": reason, "term_at": now, "elapsed_s": round(now - j["started"]),
                         "silent_s": round(silent), "gpu": gpu}
            jobrun.write_job(j)
            print(time.ctime(), "hang:", j["id"], reason, flush=True)
            jobrun.kill_group(j["pid"], signal.SIGTERM)


def idle_minutes():
    try:
        with open(f"{OG}/config.json") as f:
            return float(json.load(f).get("idle_minutes", 20))
    except (OSError, ValueError):
        return 20.0


def busy():
    for n in os.listdir(f"{OG}/jobs"):
        if n.endswith(".json"):
            try:
                with open(f"{OG}/jobs/{n}") as f:
                    j = json.load(f)
            except (OSError, ValueError):
                continue
            if j.get("status") in ("queued", "running"):
                try:
                    os.kill(j.get("runner_pid"), 0)
                    return True
                except (OSError, TypeError):
                    pass
    return False


def last_activity():
    try:
        with open(f"{OG}/last_activity") as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return time.time()


def container_started_at():
    """Start time of PID 1 (the container's init) — /proc/uptime would be the host's."""
    try:
        with open("/proc/1/stat") as f:
            ticks = int(f.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/stat") as f:
            btime = next(int(l.split()[1]) for l in f if l.startswith("btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, StopIteration, IndexError):
        return None


def log_session_end():
    """Paid time between the last poll and this stop is invisible to the provider API once the pod is down, so
    record it on the persistent disk; `ogpu` imports it into the ledger on its next connection."""
    start = container_started_at()
    if start is None:
        return
    with open(f"{OG}/sessions.jsonl", "a") as f:
        f.write(json.dumps({"pod_id": os.environ.get("RUNPOD_POD_ID"), "start": start, "end": time.time()}) + "\n")


def stop_pod():
    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if pod and key:
        q = json.dumps({"query": f'mutation {{ podStop(input: {{podId: "{pod}"}}) {{ id }} }}'}).encode()
        req = urllib.request.Request("https://api.runpod.io/graphql?" + urllib.parse.urlencode({"api_key": key}),
                                     data=q, headers={"Content-Type": "application/json"})
        print(time.ctime(), "stopping pod", pod, urllib.request.urlopen(req, timeout=30).read()[:200], flush=True)
    elif pod:
        subprocess.run(["runpodctl", "stop", "pod", pod])
    else:
        print(time.ctime(), "no RUNPOD_POD_ID; cannot stop", flush=True)


def single_instance():
    try:
        with open(PIDFILE) as f:
            os.kill(int(f.read()), 0)
        sys.exit(0)  # another reaper is alive
    except (OSError, ValueError):
        pass
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))


def main():
    single_instance()
    while True:
        limit = idle_minutes() * 60
        now = time.time()
        try:
            watchdog(now, gpu_sample())
        except Exception as e:  # noqa: BLE001 — the reaper must keep running
            print(time.ctime(), "watchdog error:", e, flush=True)
        if busy():
            state = {"busy": True, "idle_since": None, "stop_at": None}
        else:
            since = last_activity()
            state = {"busy": False, "idle_since": since, "stop_at": since + limit}
            if limit > 0 and now - since >= limit:
                state["stopping"] = True
                with open(STATE, "w") as f:
                    json.dump(state, f)
                log_session_end()
                stop_pod()
                time.sleep(120)
                continue
        with open(STATE, "w") as f:
            json.dump(state | {"checked": now, "idle_minutes": limit / 60}, f)
        time.sleep(15)


if __name__ == "__main__":
    main()
