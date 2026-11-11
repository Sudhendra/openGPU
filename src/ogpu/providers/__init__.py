"""Where machines come from.

Built in: `runpod` (machines with SSH and stop/resume), `ssh` (a host you already have) and `modal`
(serverless: each job is its own container, so there is no machine to keep up).

Add your own provider without touching ogpu: subclass `Provider` and point the config at it

    [providers.mycloud]
    module = "my_pkg.mycloud:MyCloud"   # or register an `ogpu.providers` entry point named "mycloud"
    region = "us-east"                  # every other key is passed to the provider as `opts`

then set `provider = "mycloud"` (top level or per pool).
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from importlib.metadata import entry_points

SERVERLESS = {"modal"}
BUILTIN = {"runpod": "ogpu.providers.runpod:RunPod", "ssh": "ogpu.providers.ssh:StaticHost"}


class ProviderError(RuntimeError):
    pass


@dataclass
class Machine:
    id: str
    name: str
    status: str            # RUNNING / EXITED (stopped, disk kept) / anything else = not usable yet
    cost_per_hr: float
    gpu: str
    location: str = ""
    uptime_s: int | None = None
    ssh_host: str = ""
    ssh_port: int = 0
    ssh_user: str = "root"


class Provider:
    """A machine provider. A machine is a container or VM with SSH (key auth), `python3`, a writable /workspace that
    survives stop/resume, and a provider-specific way to stop itself (see `reaper_stop`)."""
    name = ""
    env_key = ""                   # environment variable holding the API key, if any
    default_gpus: list[str] = []   # used when the config's gpu_type_ids is empty
    default_tiers: list[str] = [""]

    def __init__(self, opts: dict | None = None):
        self.opts = opts or {}

    @property
    def key(self) -> str:
        key = os.environ.get(self.env_key, "")
        if not key:
            raise SystemExit(f"{self.env_key} is not set (export it in your shell profile).")
        return key

    def get(self, machine_id: str) -> Machine | None:
        raise NotImplementedError

    def find(self, name: str) -> Machine | None:
        raise NotImplementedError

    def create(self, cfg, gpu: str, tier: str) -> Machine:
        """Start a machine with cfg.image on one `gpu`, name cfg.pod_name, cfg.public_key authorized for SSH, and
        remote/bootstrap.sh run at boot. Raise ProviderError when there is no capacity (the next gpu/tier is tried)."""
        raise NotImplementedError

    def resume(self, machine_id: str) -> None:
        raise NotImplementedError

    def stop(self, machine_id: str) -> None:
        raise NotImplementedError

    def terminate(self, machine_id: str) -> None:
        raise NotImplementedError

    def gpu_types(self) -> list[dict]:
        """[{"id", "memory_gb", "price"}] for `ogpu gpus`."""
        return []

    def reaper_stop(self) -> dict:
        """How the on-machine reaper stops an idle machine: {"provider": name} uses its built-in API call
        (runpod); {"stop_cmd": "<shell>"} runs a command instead."""
        return {"provider": self.name}


def load(name: str, cfg) -> Provider:
    if name in SERVERLESS:
        raise SystemExit(f"{name} is serverless: there is no machine to manage (jobs run in their own containers)")
    opts = dict(cfg.providers.get(name, {}))
    target = opts.pop("module", None) or BUILTIN.get(name)
    if target is None:
        eps = [ep for ep in entry_points(group="ogpu.providers") if ep.name == name]
        if not eps:
            known = ", ".join(sorted({*BUILTIN, *SERVERLESS}))
            raise SystemExit(f"unknown provider {name!r} (built in: {known}; or set [providers.{name}] module = ...)")
        cls = eps[0].load()
    else:
        mod, _, attr = target.partition(":")
        cls = getattr(importlib.import_module(mod), attr)
    return cls(opts)
