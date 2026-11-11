"""~/.ogpu/config.toml + ~/.ogpu/state.json. Secrets come from the environment, never the config file."""
from __future__ import annotations

import difflib
import json
import os
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

HOME = Path(os.environ.get("OGPU_HOME", Path.home() / ".ogpu"))
CONFIG_PATH = HOME / "config.toml"
STATE_PATH = HOME / "state.json"
LEDGER_PATH = HOME / "ledger.db"
LOG_DIR = HOME / "logs"

REMOTE_OGPU = "/workspace/.ogpu"    # jobs/, runs/, bin/, reaper state

DEFAULT_CONFIG = """\
# ogpu config. API keys come from the environment (RUNPOD_API_KEY; Modal uses `modal token new`).
provider = "runpod"          # runpod | modal | ssh | your own under [providers.<name>]
gpu_type_ids = []            # provider-native GPU ids, tried in order ([] = provider default; `ogpu gpus` lists ids)
image = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
idle_minutes = 10            # the machine stops itself after this much idle time
budget_usd = 25.0            # daily cap: new jobs (GPU or CPU, any provider) are refused beyond it
cache = false                # true: an identical run (same files + command + image + GPU) returns the earlier result
profile = ""                 # "okc": Open Kernel Challenge defaults (judge image, H200 + H100 correctness pool)

# A cheaper pool for `--fidelity correctness` jobs (any provider), e.g.:
# [pools.correctness]
# provider = "modal"
# gpu_type_ids = ["L4"]
"""

POOLS = ("timing", "correctness")
CUDA_VERSIONS = ["12.4", "12.5", "12.6", "12.7", "12.8", "12.9", "13.0", "13.1", "13.2", "13.3"]


@dataclass
class Config:
    provider: str = "runpod"
    profile: str = ""
    pod_name: str = "ogpu"
    image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
    gpu_type_ids: list[str] = field(default_factory=list)   # [] = the provider's default
    cloud_types: list[str] = field(default_factory=list)    # provider tiers; [] = the provider's default
    container_disk_gb: int = 50
    volume_gb: int = 50
    network_volume_id: str = ""
    min_vcpu: int = 8
    min_memory_gb: int = 32
    ssh_public_key: str = "~/.ssh/id_ed25519.pub"
    ssh_private_key: str = "~/.ssh/id_ed25519"
    idle_minutes: int = 10
    budget_usd: float = 25.0
    auto_up: bool = True
    cache: bool = False
    quiet_timing: bool = True
    recreate_on_resume_failure: bool = True
    default_timeout_s: float = 3600
    hang_factor: float = 5
    hang_floor_s: float = 180
    spin_silent_s: float = 900
    min_cuda: str = "12.4"
    remote_root: str = "/workspace/src"   # synced dirs live here on the machine
    pools: dict = field(default_factory=dict)
    providers: dict = field(default_factory=dict)  # [providers.<name>]: options, or module = "pkg.mod:Class"

    def pool(self, name: str) -> "Config | None":
        """The config for one pool: `timing` is the top level; others override it. None if not configured."""
        if name == "timing":
            return self
        over = dict(self.pools.get(name, {}))
        if not over or over.pop("enabled", True) is False:
            return None
        known = Config.__dataclass_fields__
        over.setdefault("pod_name", f"{self.pod_name}-{name}")
        return replace(self, **{k: v for k, v in over.items() if k in known and k != "pools"})

    @property
    def allowed_cuda_versions(self) -> list[str] | None:
        """Host filters want an explicit list: every known CUDA version ≥ min_cuda."""
        if not self.min_cuda:
            return None
        key = lambda v: tuple(int(x) for x in v.split("."))  # noqa: E731
        return [v for v in CUDA_VERSIONS if key(v) >= key(self.min_cuda)] or [self.min_cuda]

    @property
    def public_key(self) -> str:
        return Path(self.ssh_public_key).expanduser().read_text().strip()

    @property
    def private_key_path(self) -> str:
        return str(Path(self.ssh_private_key).expanduser())


def profile_defaults(name: str) -> dict:
    if not name:
        return {}
    if name == "okc":
        from . import okc
        return okc.DEFAULTS
    raise SystemExit(f"unknown profile {name!r} in {CONFIG_PATH} (known: okc)")


def ensure_home() -> None:
    HOME.mkdir(mode=0o700, parents=True, exist_ok=True)


def check_keys(data: dict, where: str) -> None:
    """A typo (`budget = 5` for `budget_usd`) must not silently fall back to a default — least of all the budget."""
    known = set(Config.__dataclass_fields__) | ({"enabled"} if where != "top level" else set())
    for k in data:
        if k not in known:
            near = difflib.get_close_matches(k, known, n=1)
            raise SystemExit(f"unknown config key {k!r} ({where} of {CONFIG_PATH})"
                             + (f" — did you mean {near[0]!r}?" if near else ""))


def load_config() -> Config:
    ensure_home()
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG)
    data = tomllib.loads(CONFIG_PATH.read_text())
    check_keys(data, "top level")
    for name, pool in data.get("pools", {}).items():
        if name not in POOLS:
            raise SystemExit(f"unknown pool {name!r} in {CONFIG_PATH} (pools: {', '.join(POOLS)})")
        check_keys(pool, f"[pools.{name}]")
    base = profile_defaults(data.get("profile", ""))
    pools = {n: dict(base.get("pools", {}).get(n, {})) | p for n, p in data.get("pools", {}).items()}
    merged = base | data | {"pools": dict(base.get("pools", {})) | pools}
    known = Config.__dataclass_fields__
    return Config(**{k: v for k, v in merged.items() if k in known})


@dataclass
class State:
    pod_id: str = ""
    host: str = ""
    port: int = 0
    cost_per_hr: float = 0.0
    gpu: str = ""
    cloud: str = ""
    provider: str = ""
    user: str = "root"

    @property
    def reachable(self) -> bool:
        return bool(self.host and self.port)


def _read_states() -> dict:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text()).get("pools", {})


def load_state(pool: str = "timing") -> State:
    known = State.__dataclass_fields__
    return State(**{k: v for k, v in _read_states().get(pool, {}).items() if k in known})


def save_state(s: State, pool: str = "timing") -> None:
    ensure_home()
    states = _read_states()
    states[pool] = asdict(s)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"pools": states}, indent=1))
    tmp.replace(STATE_PATH)
