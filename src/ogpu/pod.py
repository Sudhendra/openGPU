"""Pod lifecycle per pool: get a reachable GPU pod (reuse → resume → create with fallbacks), install the helpers."""
from __future__ import annotations

import fcntl
import hashlib
import json
import shlex
import time
from importlib import resources

import requests
from rich.console import Console

from . import ledger, ssh
from .config import HOME, REMOTE_OGPU, Config, State, api_key, load_state, save_state
from .providers.runpod import Pod, RunPod, RunPodError

REMOTE_FILES = ("jobrun.py", "reaper.py", "reaperguard.sh")
REAPER_STALE_S = 120   # reaper.json not refreshed for this long = the reaper is dead or wedged
HEALTH_PROBE = r"""[ -f /workspace/.ogpu/env.sh ] && source /workspace/.ogpu/env.sh
python3 - <<'PY'
try:
    import torch
except Exception as e:
    print("SKIP no torch:", e)
    raise SystemExit(0)
torch.zeros(1, device="cuda").add_(1)
torch.cuda.synchronize()
print("OK", torch.cuda.get_device_name(), "torch cuda", torch.version.cuda)
PY"""


def client() -> RunPod:
    return RunPod(api_key())


def current(rp: RunPod, cfg: Config, pool: str = "timing") -> Pod | None:
    st = load_state(pool)
    pod = None
    if st.pod_id:
        try:
            pod = rp.pod(st.pod_id)
        except RunPodError:
            pod = None
    return pod or rp.find(cfg.pod_name)


def sync_state(pod: Pod, cloud: str = "", pool: str = "timing") -> State:
    st = load_state(pool)
    st = State(pod_id=pod.id, host=pod.ssh_host, port=pod.ssh_port, cost_per_hr=pod.cost_per_hr,
               gpu=pod.gpu or st.gpu, cloud=cloud or st.cloud)
    save_state(st, pool)
    if pod.uptime_s:
        with ledger.connect() as db:
            ledger.record_session(db, pod.id, pod.uptime_s, pod.cost_per_hr)
    return st


def helpers_version() -> str:
    src = resources.files("ogpu.remote")
    return hashlib.sha256(b"".join(src.joinpath(n).read_bytes() for n in REMOTE_FILES)).hexdigest()[:16]


def install_helpers(st: State, cfg: Config, *, reset_idle: bool = True) -> None:
    """Copy jobrun/reaper to the pod, write the reaper config, and make sure exactly one reaper (plus the guard that
    restarts it) runs. `reset_idle=False` when only reviving a dead reaper: that must not extend the idle clock."""
    key = cfg.private_key_path
    src = resources.files("ogpu.remote")
    files = {name: src.joinpath(name).read_text() for name in REMOTE_FILES}
    script = [f"mkdir -p {REMOTE_OGPU}/bin {REMOTE_OGPU}/jobs"]
    for name, body in files.items():
        script.append(f"cat > {REMOTE_OGPU}/bin/{name} <<'OGPU_EOF'\n{body}\nOGPU_EOF")
    script.append(f"echo {helpers_version()} > {REMOTE_OGPU}/bin/VERSION")
    cfg_json = json.dumps({"idle_minutes": cfg.idle_minutes})
    script.append(f"echo {shlex.quote(cfg_json)} > {REMOTE_OGPU}/config.json")
    if reset_idle:  # a stale boot-time stamp would make the reaper stop the pod right after `up`
        script.append(f"date +%s > {REMOTE_OGPU}/last_activity")
    # restart guard + reaper so they run the current code (patterns anchored so they can't match this shell).
    # setsid: nothing from this SSH session (hangup, session teardown) can reach them afterwards.
    # kill the guard's whole process group (it runs under setsid): a surviving `sleep` child would keep holding
    # the guard's lock and make the new guard exit at once
    script.append(f"for p in $(pgrep -f '^bash {REMOTE_OGPU}/bin/reaperguard.sh'); do kill -9 -- -$p; done 2>/dev/null; true")
    script.append(f"pkill -f '^python3 {REMOTE_OGPU}/bin/reaper.py' || true; rm -f {REMOTE_OGPU}/reaper.pid")
    script.append(f"bash -c 'set -a; [ -f {REMOTE_OGPU}/env.sh ] && source {REMOTE_OGPU}/env.sh; set +a; "
                  f"setsid nohup python3 {REMOTE_OGPU}/bin/reaper.py >> {REMOTE_OGPU}/reaper.log 2>&1 < /dev/null & "
                  f"sleep 1; setsid nohup bash {REMOTE_OGPU}/bin/reaperguard.sh >/dev/null 2>&1 < /dev/null &' ")
    ssh.run(st, key, "\n".join(script), timeout=60)


def wait_reachable(rp: RunPod, pod_id: str, cfg: Config, console: Console, timeout_s: int = 900,
                   pool: str = "timing") -> State:
    t0 = time.time()
    with console.status("[cyan]booting pod · waiting for public SSH…") as s:
        while time.time() - t0 < timeout_s:
            try:
                pod = rp.pod(pod_id)
            except (RunPodError, requests.RequestException):  # transient API hiccup: keep waiting
                time.sleep(5)
                continue
            if pod and pod.ssh_port:
                st = sync_state(pod, pool=pool)
                err = f" · [red]{ssh.last_error}[/]" if ssh.last_error and time.time() - t0 > 60 else ""
                s.update(f"[cyan]pod up at {st.host}:{st.port} · waiting for sshd (first boot installs it)…{err}")
                if ssh.reachable(st, cfg.private_key_path):
                    return st
            time.sleep(5)
    raise SystemExit(f"pod {pod_id} did not become reachable in {timeout_s}s — check the RunPod console logs")


def up(cfg: Config, console: Console, *, retry_minutes: float = 0, pool: str = "timing") -> State:
    """`cfg` is the pool's config (Config.pool(name))."""
    rp = client()
    pod = current(rp, cfg, pool)
    if pod and pod.status == "RUNNING":
        console.print(f"[green]●[/] reusing running pod [bold]{pod.id}[/] ({pod.gpu or 'GPU'}, ${pod.cost_per_hr:.2f}/hr)")
    stuck = None
    if pod and pod.status != "RUNNING":
        console.print(f"[yellow]●[/] resuming stopped pod [bold]{pod.id}[/] (disk + caches preserved)…")
        try:
            rp.resume(pod.id)
        except RunPodError as e:
            if not cfg.recreate_on_resume_failure:
                raise SystemExit(f"resume failed: {e}\n  The host may have no free GPU right now. Retry, or "
                                 f"`ogpu down --terminate` and `ogpu up` to land on a new machine (loses the pod disk).")
            # the host's GPU was rented while we were stopped: move to a new machine rather than fail the job
            console.print(f"[yellow]↻[/] resume failed ({e}) — creating a new {pool} pod; "
                          f"the stuck one is deleted once the new one is up (only caches are lost)")
            stuck, pod = pod, None
    if pod is None:
        pod = create_with_fallback(rp, cfg, console, retry_minutes, pool)
    st = wait_reachable(rp, pod.id, cfg, console, pool=pool)
    if stuck is not None:
        try:
            rp.terminate(stuck.id)
            console.print(f"[grey50]deleted stuck pod {stuck.id}[/]")
        except RunPodError as e:
            console.print(f"[yellow]could not delete stuck pod {stuck.id}: {e} — remove it in the RunPod console[/]")
    for attempt in range(3):
        ok, detail = gpu_health(st, cfg)
        if ok:
            break
        # a host whose driver can't run the image's CUDA fails every job (the H100 NVL on driver 12.8 did, all
        # evening): never hand it to agents — replace it with a pod on another host
        console.print(f"[red]✘[/] {pool} pod {st.pod_id} failed the GPU check: {detail}")
        if attempt == 2:
            raise SystemExit(f"{pool}: 3 pods in a row failed the GPU check; last: {detail}")
        console.print("[yellow]↻[/] deleting it and creating a pod on another host…")
        try:
            rp.terminate(st.pod_id)
        except RunPodError as e:
            console.print(f"[yellow]could not delete {st.pod_id}: {e} — remove it in the RunPod console[/]")
        save_state(State(), pool)
        pod = create_with_fallback(rp, cfg, console, retry_minutes, pool)
        st = wait_reachable(rp, pod.id, cfg, console, pool=pool)
    import_pod_sessions(st, cfg)
    install_helpers(st, cfg)
    console.print(f"[green]✔[/] ready  [magenta]{pool}[/]  [bold]{st.gpu}[/]  root@{st.host}:{st.port}  ${st.cost_per_hr:.2f}/hr  "
                  f"idle-stop {cfg.idle_minutes}m")
    return st


def create_with_fallback(rp: RunPod, cfg: Config, console: Console, retry_minutes: float, pool: str = "timing") -> Pod:
    deadline = time.time() + retry_minutes * 60
    attempt = 0
    while True:
        attempt += 1
        errors = []
        for gpu in cfg.gpu_type_ids:
            for cloud in cfg.cloud_types:
                try:
                    pod = create_one(rp, cfg, gpu, cloud)
                    console.print(f"[green]✔[/] created pod [bold]{pod.id}[/] · {gpu} · {cloud.lower()} cloud")
                    sync_state(pod, cloud=cloud.lower(), pool=pool)
                    return pod
                except RunPodError as e:
                    errors.append(f"{gpu}/{cloud.lower()}: {e}")
        if time.time() >= deadline:
            raise SystemExit("no capacity:\n  " + "\n  ".join(errors) + "\n  try `ogpu up --retry 30` to keep polling")
        console.print(f"[yellow]…[/] no capacity (attempt {attempt}); retrying in 30s")
        time.sleep(30)


def create_one(rp: RunPod, cfg: Config, gpu: str, cloud: str) -> Pod:
    kw = dict(name=cfg.pod_name, image=cfg.image, gpu_type_id=gpu, cloud=cloud,
              container_disk_gb=cfg.container_disk_gb, volume_gb=cfg.volume_gb,
              network_volume_id=cfg.network_volume_id, public_key=cfg.public_key,
              min_vcpu=cfg.min_vcpu, min_memory_gb=cfg.min_memory_gb)
    try:
        return rp.create(**kw, allowed_cuda=cfg.allowed_cuda_versions)
    except RunPodError as e:
        if "cuda" not in str(e).lower():
            raise
        return rp.create(**kw)  # the API rejected the filter itself: the post-boot GPU check still guards us


def parse_health(rc: int, out: str) -> tuple[bool, str]:
    """→ (usable, detail) from HEALTH_PROBE's exit code and output."""
    lines = [l for l in out.strip().splitlines() if l.strip()]
    last = lines[-1] if lines else f"exit {rc}"
    if rc == 0 and lines and (last.startswith("OK") or last.startswith("SKIP")):
        return True, last
    hint = next((l for l in reversed(lines) if "driver" in l.lower() or "cuda" in l.lower()), last)
    return False, hint[:300]


def gpu_health(st: State, cfg: Config) -> tuple[bool, str]:
    """Allocate on the GPU through the image's own torch, retrying once (a GPU can be slow right after boot)."""
    detail = ""
    for i in range(2):
        try:
            p = ssh.run(st, cfg.private_key_path, HEALTH_PROBE, check=False, timeout=180)
            if p.returncode == 255:  # ssh's own failure code: no verdict on the GPU
                return True, "check skipped (ssh error)"
            ok, detail = parse_health(p.returncode, p.stdout + p.stderr)
        except Exception as e:  # noqa: BLE001 — SSH trouble is not a verdict on the GPU
            return True, f"check skipped ({type(e).__name__})"
        if ok:
            return True, detail
        if i == 0:
            time.sleep(10)
    return False, detail


def import_pod_sessions(st: State, cfg: Config) -> None:
    """Pull session-end records the reaper wrote before stopping (the idle tail the API can't report)."""
    try:
        out = ssh.run(st, cfg.private_key_path, f"cat {REMOTE_OGPU}/sessions.jsonl 2>/dev/null && "
                      f"mv {REMOTE_OGPU}/sessions.jsonl {REMOTE_OGPU}/sessions.imported.$(date +%s)", check=False).stdout
    except Exception:  # noqa: BLE001
        return
    with ledger.connect() as db:
        for line in out.splitlines():
            try:
                rec = json.loads(line)
                ledger.record_session(db, rec.get("pod_id") or st.pod_id, rec["end"] - rec["start"],
                                      st.cost_per_hr, start=rec["start"])
            except (ValueError, KeyError, TypeError):
                continue


def reaper_alive(heartbeat: str, now: str) -> bool:
    try:
        return int(now) - int(float(heartbeat)) <= REAPER_STALE_S
    except ValueError:
        return False


def upgrade_helpers(st: State, cfg: Config, console: Console | None = None) -> None:
    """Push new jobrun/reaper versions after an `ogpu` update (running jobs keep the code they started with), and
    revive the reaper if its heartbeat went stale — a dead reaper means the pod never stops (Oct 31: ~4.7 idle hours)."""
    try:
        out = ssh.run(st, cfg.private_key_path,
                      f"echo v=$(cat {REMOTE_OGPU}/bin/VERSION 2>/dev/null); "
                      f"echo hb=$(stat -c %Y {REMOTE_OGPU}/reaper.json 2>/dev/null); echo now=$(date +%s)",
                      check=False).stdout
    except Exception:  # noqa: BLE001
        return
    kv = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
    version, heartbeat, now = kv.get("v", "").strip(), kv.get("hb", "").strip(), kv.get("now", "").strip()
    if version != helpers_version():
        install_helpers(st, cfg)
    elif not reaper_alive(heartbeat, now):
        if console:
            console.print("[yellow]↻[/] idle reaper was not running on the pod — restarted it")
        install_helpers(st, cfg, reset_idle=False)


def ensure(cfg: Config, console: Console, pool: str = "timing") -> State:
    """Fast path for `ogpu run`: trust the pool's state if SSH answers; otherwise bring its pod up (if auto_up).
    `cfg` is the pool's config."""
    st = load_state(pool)
    if ssh.reachable(st, cfg.private_key_path):
        upgrade_helpers(st, cfg, console)
        return st
    if not cfg.auto_up:
        raise SystemExit(f"{pool} pod not reachable; run `ogpu up --pool {pool}` (or set auto_up = true)")
    # several agents may hit a stopped pod at once: only one brings it up, the rest reuse the result
    lock_name = "up.lock" if pool == "timing" else f"up-{pool}.lock"
    with open(HOME / lock_name, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = load_state(pool)
        if ssh.reachable(st, cfg.private_key_path):
            return st
        return up(cfg, console, pool=pool)
