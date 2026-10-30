"""~/.ogpu/config.toml + ~/.ogpu/state.json. Secrets come from the environment, never the config file."""
from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass, field
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
"""


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


def load_state() -> State:
    if not STATE_PATH.exists():
        return State()
    known = State.__dataclass_fields__
    return State(**{k: v for k, v in json.loads(STATE_PATH.read_text()).items() if k in known})


def save_state(s: State) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(asdict(s), indent=1))
