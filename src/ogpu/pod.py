"""Machine lifecycle per pool, for any provider: get a reachable GPU machine (reuse → resume → create with
fallbacks), check its GPU, install the on-machine helpers."""
from __future__ import annotations

import fcntl
import hashlib
import json
import shlex
import time
from importlib import resources

import requests
from rich.console import Console

from . import ledger, providers, ssh
from .config import HOME, REMOTE_OGPU, Config, State, load_state, save_state
from .providers import Machine, Provider, ProviderError

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


def client(cfg: Config, st: State | None = None) -> Provider:
    """The provider of a pool (`cfg` is the pool's config), or of the machine recorded in `st`."""
    return providers.load((st.provider if st else "") or cfg.provider, cfg)


def current(prov: Provider, cfg: Config, pool: str = "timing") -> Machine | None:
    st = load_state(pool)
    m = None
    if st.pod_id and (st.provider or cfg.provider) == prov.name:
        try:
            m = prov.get(st.pod_id)
        except ProviderError:
            m = None
    return m or prov.find(cfg.pod_name)


def sync_state(m: Machine, cloud: str = "", pool: str = "timing", provider: str = "") -> State:
    st = load_state(pool)
    st = State(pod_id=m.id, host=m.ssh_host, port=m.ssh_port, cost_per_hr=m.cost_per_hr, gpu=m.gpu or st.gpu,
               cloud=cloud or st.cloud, provider=provider or st.provider, user=m.ssh_user)
    save_state(st, pool)
    if m.uptime_s:
        with ledger.connect() as db:
            ledger.record_session(db, m.id, m.uptime_s, m.cost_per_hr)
    return st


def helpers_version() -> str:
    src = resources.files("ogpu.remote")
    return hashlib.sha256(b"".join(src.joinpath(n).read_bytes() for n in REMOTE_FILES)).hexdigest()[:16]


def install_helpers(st: State, cfg: Config, *, reset_idle: bool = True) -> None:
    """Copy jobrun/reaper to the machine, write the reaper config, and make sure exactly one reaper (plus the guard
    that restarts it) runs. `reset_idle=False` when only reviving a dead reaper: that must not extend the idle clock."""
    src = resources.files("ogpu.remote")
    script = [f"mkdir -p {REMOTE_OGPU}/bin {REMOTE_OGPU}/jobs && chmod 700 {REMOTE_OGPU}"]
    for name in REMOTE_FILES:
        script.append(f"cat > {REMOTE_OGPU}/bin/{name} <<'OGPU_EOF'\n{src.joinpath(name).read_text()}\nOGPU_EOF")
    script.append(f"echo {helpers_version()} > {REMOTE_OGPU}/bin/VERSION")
    reaper_cfg = {"idle_minutes": cfg.idle_minutes, "machine_id": st.pod_id} | client(cfg, st).reaper_stop()
    script.append(f"echo {shlex.quote(json.dumps(reaper_cfg))} > {REMOTE_OGPU}/config.json")
    if reset_idle:  # a stale boot-time stamp would make the reaper stop the machine right after `up`
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
    ssh.run(st, cfg.private_key_path, "\n".join(script), timeout=60)


def wait_reachable(prov: Provider, machine_id: str, cfg: Config, console: Console, timeout_s: int = 900,
                   pool: str = "timing") -> State:
    t0 = time.time()
    with console.status("[cyan]booting machine · waiting for public SSH…") as s:
        while time.time() - t0 < timeout_s:
            try:
                m = prov.get(machine_id)
            except (ProviderError, requests.RequestException):  # transient API hiccup: keep waiting
                time.sleep(5)
                continue
            if m and m.ssh_port:
                st = sync_state(m, pool=pool, provider=prov.name)
                err = f" · [red]{ssh.last_error}[/]" if ssh.last_error and time.time() - t0 > 60 else ""
                s.update(f"[cyan]machine up at {st.host}:{st.port} · waiting for sshd (first boot installs it)…{err}")
                if ssh.reachable(st, cfg.private_key_path):
                    return st
            time.sleep(5)
    raise SystemExit(f"machine {machine_id} did not become reachable in {timeout_s}s — check the {prov.name} console")


def up(cfg: Config, console: Console, *, retry_minutes: float = 0, pool: str = "timing") -> State:
    """`cfg` is the pool's config (Config.pool(name))."""
    prov = client(cfg)
    m = current(prov, cfg, pool)
    if m and m.status == "RUNNING":
        console.print(f"[green]●[/] reusing running machine [bold]{m.id}[/] ({m.gpu or 'GPU'}, ${m.cost_per_hr:.2f}/hr)")
    stuck = None
    if m and m.status != "RUNNING":
        console.print(f"[yellow]●[/] resuming stopped machine [bold]{m.id}[/] (disk + caches preserved)…")
        try:
            prov.resume(m.id)
            ssh.forget_host(State(pod_id=m.id))
        except ProviderError as e:
            if not cfg.recreate_on_resume_failure:
                raise SystemExit(f"resume failed: {e}\n  The host may have no free GPU right now. Retry, or "
                                 f"`ogpu down --terminate` and `ogpu up` to land on a new machine (loses its disk).")
            # the host's GPU was rented while we were stopped: move to a new machine rather than fail the job
            console.print(f"[yellow]↻[/] resume failed ({e}) — creating a new {pool} machine; "
                          f"the stuck one is deleted once the new one is up (only caches are lost)")
            stuck, m = m, None
    if m is None:
        m = create_with_fallback(prov, cfg, console, retry_minutes, pool)
    st = wait_reachable(prov, m.id, cfg, console, pool=pool)
    if stuck is not None:
        try:
            prov.terminate(stuck.id)
            console.print(f"[grey50]deleted stuck machine {stuck.id}[/]")
        except ProviderError as e:
            console.print(f"[yellow]could not delete stuck machine {stuck.id}: {e} — remove it in the console[/]")
    for attempt in range(3):
        ok, detail = gpu_health(st, cfg)
        if ok:
            break
        # a host whose driver can't run the image's CUDA fails every job: never hand it to agents — replace it
        console.print(f"[red]✘[/] {pool} machine {st.pod_id} failed the GPU check: {detail}")
        if attempt == 2:
            raise SystemExit(f"{pool}: 3 machines in a row failed the GPU check; last: {detail}")
        console.print("[yellow]↻[/] deleting it and creating a machine on another host…")
        try:
            prov.terminate(st.pod_id)
        except ProviderError as e:
            console.print(f"[yellow]could not delete {st.pod_id}: {e} — remove it in the console[/]")
        save_state(State(), pool)
        m = create_with_fallback(prov, cfg, console, retry_minutes, pool)
        st = wait_reachable(prov, m.id, cfg, console, pool=pool)
    import_pod_sessions(st, cfg)
    install_helpers(st, cfg)
    console.print(f"[green]✔[/] ready  [magenta]{pool}[/]  [bold]{st.gpu}[/]  {prov.name}  {ssh.target(st)}:{st.port}  "
                  f"${st.cost_per_hr:.2f}/hr  idle-stop {cfg.idle_minutes}m")
    return st


def create_with_fallback(prov: Provider, cfg: Config, console: Console, retry_minutes: float,
                         pool: str = "timing") -> Machine:
    deadline = time.time() + retry_minutes * 60
    attempt = 0
    while True:
        attempt += 1
        errors = []
        for gpu in cfg.gpu_type_ids or prov.default_gpus:
            for tier in cfg.cloud_types or prov.default_tiers:
                try:
                    m = prov.create(cfg, gpu, tier)
                    console.print(f"[green]✔[/] created machine [bold]{m.id}[/] · {prov.name} · {gpu}"
                                  + (f" · {tier.lower()}" if tier else ""))
                    ssh.forget_host(State(pod_id=m.id))
                    sync_state(m, cloud=tier.lower(), pool=pool, provider=prov.name)
                    return m
                except ProviderError as e:
                    errors.append(f"{gpu}/{tier.lower() or prov.name}: {e}")
        if time.time() >= deadline:
            raise SystemExit("no capacity:\n  " + "\n  ".join(errors) + "\n  try `ogpu up --retry 30` to keep polling")
        console.print(f"[yellow]…[/] no capacity (attempt {attempt}); retrying in 30s")
        time.sleep(30)


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
    revive the reaper if its heartbeat went stale — a dead reaper means the machine never stops."""
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
            console.print("[yellow]↻[/] idle reaper was not running on the machine — restarted it")
        install_helpers(st, cfg, reset_idle=False)


def ensure(cfg: Config, console: Console, pool: str = "timing") -> State:
    """Fast path for `ogpu run`: trust the pool's state if SSH answers; otherwise bring its machine up (if auto_up).
    `cfg` is the pool's config."""
    st = load_state(pool)
    if ssh.reachable(st, cfg.private_key_path):
        upgrade_helpers(st, cfg, console)
        return st
    if not cfg.auto_up:
        raise SystemExit(f"{pool} machine not reachable; run `ogpu up --pool {pool}` (or set auto_up = true)")
    # several agents may hit a stopped machine at once: only one brings it up, the rest reuse the result
    with open(HOME / up_lock(pool), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = load_state(pool)
        if ssh.reachable(st, cfg.private_key_path):
            return st
        return up(cfg, console, pool=pool)


def up_lock(pool: str) -> str:
    return "up.lock" if pool == "timing" else f"up-{pool}.lock"
