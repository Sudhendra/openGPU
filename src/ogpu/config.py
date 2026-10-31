"""~/.ogpu/config.toml + ~/.ogpu/state.json. Secrets come from the environment, never the config file."""
from __future__ import annotations

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

REMOTE_ROOT = "/workspace/okc"      # synced tracks live here on the pod
REMOTE_OGPU = "/workspace/.ogpu"    # jobs/, lock_holder.json, bin/, reaper state

DEFAULT_CONFIG = """\
# ogpu config. Secrets (RUNPOD_API_KEY) come from the environment.
pod_name = "ogpu-h200"
image = "lmsysorg/sglang:dev-muse-spark-open"
gpu_type_ids = ["NVIDIA H200"]        # = H200 SXM (the judge GPU). Not "NVIDIA H200 NVL". `ogpu gpus` lists ids
cloud_types = ["COMMUNITY", "SECURE"]  # tried in order; community H200 SXM ~$1.70/hr cheaper
container_disk_gb = 60
volume_gb = 100                        # pod volume at /workspace, survives stop/resume
network_volume_id = ""                 # optional: RunPod network volume (pins the datacenter)
min_vcpu = 8
min_memory_gb = 32
ssh_public_key = "~/.ssh/id_ed25519.pub"
ssh_private_key = "~/.ssh/id_ed25519"
idle_minutes = 20                      # on-pod reaper stops the pod after this much idle time
budget_usd = 60.0                      # daily cap: `ogpu run --gpu` refuses new work beyond it
auto_up = true                         # `ogpu run` resumes/creates the pod if it is not running
quiet_timing = true                    # CPU jobs run niced on non-reserved cores so they can't disturb GPU timings
recreate_on_resume_failure = true      # stopped pod's host has no free GPU → create a new pod instead of failing
default_timeout_s = 3600               # jobs without --timeout are killed after this long
hang_factor = 5                        # hang = running > factor × this command's usual (p90) runtime, log quiet
hang_floor_s = 180                     # ...but never flag a job as hung before this many seconds
spin_silent_s = 900                    # GPU at ~100% util on low power with a silent log this long = deadlock

# Pools: the settings above are the `timing` pool. Other pools override any of them.
[pools.correctness]                    # `--fidelity correctness` jobs: cheap GPUs with the same ISA (sm_90a, like H200)
# 132-SM parts first (same SM count as H200 → same launch configs); 114-SM PCIe last
gpu_type_ids = ["NVIDIA H100 NVL", "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"]
idle_minutes = 10
"""

POOLS = ("timing", "correctness")
DEFAULT_POOLS = {"correctness": {"gpu_type_ids": ["NVIDIA H100 NVL", "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"],
                                 "idle_minutes": 10}}


@dataclass
class Config:
    pod_name: str = "ogpu-h200"
    image: str = "lmsysorg/sglang:dev-muse-spark-open"
    gpu_type_ids: list[str] = field(default_factory=lambda: ["NVIDIA H200"])
    cloud_types: list[str] = field(default_factory=lambda: ["COMMUNITY", "SECURE"])
    container_disk_gb: int = 60
    volume_gb: int = 100
    network_volume_id: str = ""
    min_vcpu: int = 8
    min_memory_gb: int = 32
    ssh_public_key: str = "~/.ssh/id_ed25519.pub"
    ssh_private_key: str = "~/.ssh/id_ed25519"
    idle_minutes: int = 20
    budget_usd: float = 60.0
    auto_up: bool = True
    quiet_timing: bool = True
    recreate_on_resume_failure: bool = True
    default_timeout_s: float = 3600
    hang_factor: float = 5
    hang_floor_s: float = 180
    spin_silent_s: float = 900
    pools: dict = field(default_factory=dict)

    def pool(self, name: str) -> "Config | None":
        """The config for one pool: `timing` is the top level; others override it. None if disabled."""
        if name == "timing":
            return self
        over = dict(DEFAULT_POOLS.get(name, {})) | dict(self.pools.get(name, {}))
        if not over or over.pop("enabled", True) is False:
            return None
        known = Config.__dataclass_fields__
        over.setdefault("pod_name", f"{self.pod_name}-{name}")
        return replace(self, **{k: v for k, v in over.items() if k in known and k != "pools"})

    @property
    def public_key(self) -> str:
        return Path(self.ssh_public_key).expanduser().read_text().strip()

    @property
    def private_key_path(self) -> str:
        return str(Path(self.ssh_private_key).expanduser())


def load_config() -> Config:
    HOME.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG)
    data = tomllib.loads(CONFIG_PATH.read_text())
    known = Config.__dataclass_fields__
    return Config(**{k: v for k, v in data.items() if k in known})


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set (export it in your shell profile).")
    return key


@dataclass
class State:
    pod_id: str = ""
    host: str = ""
    port: int = 0
    cost_per_hr: float = 0.0
    gpu: str = ""
    cloud: str = ""

    @property
    def reachable(self) -> bool:
        return bool(self.host and self.port)


def _read_states() -> dict:
    if not STATE_PATH.exists():
        return {}
    raw = json.loads(STATE_PATH.read_text())
    return raw.get("pools", {}) if "pools" in raw else {"timing": raw}  # pre-pool state files = timing pool


def load_state(pool: str = "timing") -> State:
    known = State.__dataclass_fields__
    return State(**{k: v for k, v in _read_states().get(pool, {}).items() if k in known})


def save_state(s: State, pool: str = "timing") -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    states = _read_states()
    states[pool] = asdict(s)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"pools": states}, indent=1))
    tmp.replace(STATE_PATH)
