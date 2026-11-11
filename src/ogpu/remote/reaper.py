#!/usr/bin/env python3
"""On-machine idle reaper (stdlib only). Stops this machine through its provider once no job has been queued or
running for `idle_minutes`. Stopping keeps /workspace (code, compiled-extension cache, job history); `ogpu up`
resumes it. config.json (written by `ogpu up`): idle_minutes, machine_id, provider, optional stop_cmd."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
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
HEARTBEAT_LOG_S = 1800  # a periodic "alive" line, so a silent death shows up as a gap in reaper.log


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


def load_cfg():
    try:
        with open(f"{OG}/config.json") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def idle_minutes():
    try:
        return float(load_cfg().get("idle_minutes", 20))
    except (TypeError, ValueError):
        return 20.0


def busy():
    # reconcile closes records left by an earlier container (recycled pids) and dead runners
    return any(jobrun.reconcile(j)["status"] in ("queued", "running")
               for j in jobrun.all_jobs() if j["status"] in ("queued", "running"))


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
        machine = load_cfg().get("machine_id") or os.environ.get("RUNPOD_POD_ID")
        f.write(json.dumps({"pod_id": machine, "start": start, "end": time.time()}) + "\n")


def http(method, url, key, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method=method,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    return urllib.request.urlopen(req, timeout=30).read()[:200]


def stop_runpod():
    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")  # pod-scoped key, set by RunPod
    if not (pod and key):
        return "no RUNPOD_POD_ID/RUNPOD_API_KEY"
    q = {"query": 'mutation { podStop(input: {podId: %s}) { id } }' % json.dumps(pod)}
    try:
        out = http("POST", "https://api.runpod.io/graphql", key, q)
        if b'"errors"' not in out:
            return out
    except urllib.error.HTTPError:
        pass
    # a key the header form rejects may still work as a query parameter (TLS, straight to RunPod): the stop matters
    return http("POST", "https://api.runpod.io/graphql?" + urllib.parse.urlencode({"api_key": key}), key, q)


def stop_pod():
    cfg = load_cfg()
    if cfg.get("stop_cmd"):
        out = subprocess.run(["bash", "-c", cfg["stop_cmd"]], capture_output=True, text=True).returncode
    else:
        stop = {"runpod": stop_runpod}.get(cfg.get("provider", "runpod"))
        out = stop() if stop else f"provider {cfg.get('provider')!r} has no self-stop; set stop_cmd"
    print(time.ctime(), "stopping machine:", out, flush=True)


def single_instance():
    """Exit if another reaper of THIS container is alive. The pidfile lives on the persistent volume, so a pid from
    an earlier container may now belong to an unrelated process (sshd, sleep) — that must not keep us from running."""
    boot = container_started_at()
    try:
        with open(PIDFILE) as f:
            parts = f.read().split()
        pid, pboot = int(parts[0]), float(parts[1]) if len(parts) > 1 else None
        if pboot is not None and boot is not None and abs(pboot - boot) > 2:
            raise ValueError("pidfile from an earlier container")
        os.kill(pid, 0)
        sys.exit(0)  # another reaper is alive
    except (OSError, ValueError, IndexError):
        pass
    with open(PIDFILE, "w") as f:
        f.write(f"{os.getpid()} {boot or ''}")


def on_signal(signum, _frame):
    print(time.ctime(), f"reaper exiting on {signal.Signals(signum).name}", flush=True)
    sys.exit(128 + signum)


def tick(now):
    """One pass: kill hung jobs, then stop the pod if it has been idle for too long. → seconds to sleep."""
    limit = idle_minutes() * 60
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
            with open(STATE, "w") as f:
                json.dump(state | {"stopping": True, "checked": now, "pid": os.getpid()}, f)
            log_session_end()
            stop_pod()
            return 120
    with open(STATE, "w") as f:  # `checked` doubles as the heartbeat `ogpu` checks before every job
        json.dump(state | {"checked": now, "idle_minutes": limit / 60, "pid": os.getpid()}, f)
    return 15


def main():
    single_instance()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    print(time.ctime(), "reaper started pid", os.getpid(), "idle_minutes", idle_minutes(), flush=True)
    last_beat = time.time()
    while True:
        now = time.time()
        try:
            delay = tick(now)
        except Exception:  # noqa: BLE001 — log and keep going: a dead reaper means a pod that bills forever
            print(time.ctime(), "reaper error:\n" + traceback.format_exc(), flush=True)
            delay = 30
        if now - last_beat >= HEARTBEAT_LOG_S:
            print(time.ctime(), "alive", flush=True)
            last_beat = now
        time.sleep(delay)


if __name__ == "__main__":
    main()
