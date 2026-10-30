#!/usr/bin/env python3
"""On-pod idle reaper (stdlib only). Stops this pod via the RunPod API once no job has been queued or running for
`idle_minutes`. Stopping keeps /workspace (code, compiled-extension cache, job history); `ogpu up` resumes it."""
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

OG = "/workspace/.ogpu"
PIDFILE = f"{OG}/reaper.pid"
STATE = f"{OG}/reaper.json"


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
        if busy():
            state = {"busy": True, "idle_since": None, "stop_at": None}
        else:
            since = last_activity()
            state = {"busy": False, "idle_since": since, "stop_at": since + limit}
            if limit > 0 and now - since >= limit:
                state["stopping"] = True
                with open(STATE, "w") as f:
                    json.dump(state, f)
                stop_pod()
                time.sleep(120)
                continue
        with open(STATE, "w") as f:
            json.dump(state | {"checked": now, "idle_minutes": limit / 60}, f)
        time.sleep(15)


if __name__ == "__main__":
    main()
