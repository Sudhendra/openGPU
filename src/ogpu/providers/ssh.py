"""A machine you already have (your own GPU box, a reserved instance): ogpu never creates, resumes or deletes it.

    [providers.ssh]
    host = "203.0.113.9"
    port = 22
    user = "root"            # needs write access to /workspace
    gpu = "RTX 4090"
    cost_per_hr = 0.0        # what an hour of it costs you, for the ledger and budget
    stop_cmd = ""            # optional: what the reaper runs when idle (e.g. "sudo shutdown -h now")

It is also the smallest example of a provider to copy when writing your own.
"""
from __future__ import annotations

from . import Machine, Provider


class StaticHost(Provider):
    name = "ssh"

    def _machine(self) -> Machine:
        o = self.opts
        if not o.get("host"):
            raise SystemExit("provider ssh needs [providers.ssh] host = ... in the config")
        return Machine(id=f"{o['host']}:{o.get('port', 22)}", name=o.get("name", "ssh"), status="RUNNING",
                       cost_per_hr=float(o.get("cost_per_hr", 0)), gpu=o.get("gpu", ""), ssh_host=o["host"],
                       ssh_port=int(o.get("port", 22)), ssh_user=o.get("user", "root"))

    def get(self, machine_id: str) -> Machine | None:
        return self._machine()

    def find(self, name: str) -> Machine | None:
        return self._machine()

    def create(self, cfg, gpu: str, tier: str) -> Machine:
        return self._machine()

    def resume(self, machine_id: str) -> None:
        pass

    def stop(self, machine_id: str) -> None:
        pass

    def terminate(self, machine_id: str) -> None:
        pass

    def reaper_stop(self) -> dict:
        return {"provider": self.name, "stop_cmd": self.opts.get("stop_cmd", "")}
