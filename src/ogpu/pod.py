"""Pod lifecycle: get a reachable H200 (reuse → resume → create with fallbacks), install the remote helpers."""
from __future__ import annotations

import fcntl
import json
import shlex
import time
from importlib import resources

from rich.console import Console

from . import ledger, ssh
from .config import HOME, REMOTE_OGPU, Config, State, api_key, load_state, save_state
from .providers.runpod import Pod, RunPod, RunPodError

REMOTE_FILES = ("jobrun.py", "reaper.py")


def client() -> RunPod:
    return RunPod(api_key())


def current(rp: RunPod, cfg: Config) -> Pod | None:
    st = load_state()
    pod = None
    if st.pod_id:
        try:
            pod = rp.pod(st.pod_id)
        except RunPodError:
            pod = None
    return pod or rp.find(cfg.pod_name)


def sync_state(pod: Pod, cloud: str = "") -> State:
    st = load_state()
    st = State(pod_id=pod.id, host=pod.ssh_host, port=pod.ssh_port, cost_per_hr=pod.cost_per_hr,
               gpu=pod.gpu or st.gpu, cloud=cloud or st.cloud)
    save_state(st)
    if pod.uptime_s:
        with ledger.connect() as db:
            ledger.record_session(db, pod.id, pod.uptime_s, pod.cost_per_hr)
    return st


def install_helpers(st: State, cfg: Config) -> None:
    """Copy jobrun/reaper to the pod, write the reaper config, and make sure exactly one reaper runs."""
    key = cfg.private_key_path
    src = resources.files("ogpu.remote")
    files = {name: src.joinpath(name).read_text() for name in REMOTE_FILES}
    script = [f"mkdir -p {REMOTE_OGPU}/bin {REMOTE_OGPU}/jobs"]
    for name, body in files.items():
        script.append(f"cat > {REMOTE_OGPU}/bin/{name} <<'OGPU_EOF'\n{body}\nOGPU_EOF")
    cfg_json = json.dumps({"idle_minutes": cfg.idle_minutes})
    script.append(f"echo {shlex.quote(cfg_json)} > {REMOTE_OGPU}/config.json")
    # reset the idle clock: a stale boot-time stamp would make the reaper stop the pod right after `up`
    script.append(f"date +%s > {REMOTE_OGPU}/last_activity")
    script.append(f"bash -c 'set -a; [ -f {REMOTE_OGPU}/env.sh ] && source {REMOTE_OGPU}/env.sh; "
                  f"nohup python3 {REMOTE_OGPU}/bin/reaper.py >> {REMOTE_OGPU}/reaper.log 2>&1 < /dev/null &' ")
    ssh.run(st, key, "\n".join(script), timeout=60)


def wait_reachable(rp: RunPod, pod_id: str, cfg: Config, console: Console, timeout_s: int = 900) -> State:
    t0 = time.time()
    with console.status("[cyan]booting pod · waiting for public SSH…") as s:
        while time.time() - t0 < timeout_s:
            pod = rp.pod(pod_id)
            if pod and pod.ssh_port:
                st = sync_state(pod)
                err = f" · [red]{ssh.last_error}[/]" if ssh.last_error and time.time() - t0 > 60 else ""
                s.update(f"[cyan]pod up at {st.host}:{st.port} · waiting for sshd (first boot installs it)…{err}")
                if ssh.reachable(st, cfg.private_key_path):
                    return st
            time.sleep(5)
    raise SystemExit(f"pod {pod_id} did not become reachable in {timeout_s}s — check the RunPod console logs")


def up(cfg: Config, console: Console, *, retry_minutes: float = 0) -> State:
    rp = client()
    pod = current(rp, cfg)
    if pod and pod.status == "RUNNING":
        console.print(f"[green]●[/] reusing running pod [bold]{pod.id}[/] ({pod.gpu or 'GPU'}, ${pod.cost_per_hr:.2f}/hr)")
    elif pod:
        console.print(f"[yellow]●[/] resuming stopped pod [bold]{pod.id}[/] (disk + caches preserved)…")
        try:
            rp.resume(pod.id)
        except RunPodError as e:
            raise SystemExit(f"resume failed: {e}\n  The host may have no free H200 right now. Retry, or "
                             f"`ogpu down --terminate` and `ogpu up` to land on a new machine (loses the pod disk).")
    else:
        pod = create_with_fallback(rp, cfg, console, retry_minutes)
    st = wait_reachable(rp, pod.id, cfg, console)
    install_helpers(st, cfg)
    console.print(f"[green]✔[/] ready  [bold]{st.gpu}[/]  root@{st.host}:{st.port}  ${st.cost_per_hr:.2f}/hr  "
                  f"idle-stop {cfg.idle_minutes}m")
    return st


def create_with_fallback(rp: RunPod, cfg: Config, console: Console, retry_minutes: float) -> Pod:
    deadline = time.time() + retry_minutes * 60
    attempt = 0
    while True:
        attempt += 1
        errors = []
        for gpu in cfg.gpu_type_ids:
            for cloud in cfg.cloud_types:
                try:
                    pod = rp.create(name=cfg.pod_name, image=cfg.image, gpu_type_id=gpu, cloud=cloud,
                                    container_disk_gb=cfg.container_disk_gb, volume_gb=cfg.volume_gb,
                                    network_volume_id=cfg.network_volume_id, public_key=cfg.public_key,
                                    min_vcpu=cfg.min_vcpu, min_memory_gb=cfg.min_memory_gb)
                    console.print(f"[green]✔[/] created pod [bold]{pod.id}[/] · {gpu} · {cloud.lower()} cloud")
                    sync_state(pod, cloud=cloud.lower())
                    return pod
                except RunPodError as e:
                    errors.append(f"{gpu}/{cloud.lower()}: {e}")
        if time.time() >= deadline:
            raise SystemExit("no capacity:\n  " + "\n  ".join(errors) + "\n  try `ogpu up --retry 30` to keep polling")
        console.print(f"[yellow]…[/] no capacity (attempt {attempt}); retrying in 30s")
        time.sleep(30)


def ensure(cfg: Config, console: Console) -> State:
    """Fast path for `ogpu run`: trust state.json if SSH answers; otherwise bring the pod up (if auto_up)."""
    st = load_state()
    if ssh.reachable(st, cfg.private_key_path):
        return st
    if not cfg.auto_up:
        raise SystemExit("pod not reachable; run `ogpu up` (or set auto_up = true)")
    # several agents may hit a stopped pod at once: only one brings it up, the rest reuse the result
    with open(HOME / "up.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = load_state()
        if ssh.reachable(st, cfg.private_key_path):
            return st
        return up(cfg, console)
