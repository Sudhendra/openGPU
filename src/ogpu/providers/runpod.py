"""RunPod GraphQL client: find/create/resume/stop pods with a real public SSH port (no PTY proxy)."""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from importlib import resources

import requests

API = "https://api.runpod.io/graphql"
POD_FIELDS = """
  id name desiredStatus costPerHr imageName gpuCount
  machine { gpuDisplayName location }
  runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
"""


class Enum(str):
    """A GraphQL enum literal (serialized unquoted)."""


class RunPodError(RuntimeError):
    pass


def gql_literal(v) -> str:
    """Serialize a Python value as a GraphQL input literal (JSON strings are valid GraphQL strings)."""
    if isinstance(v, Enum):
        return str(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return json.dumps(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(gql_literal(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {gql_literal(x)}" for k, x in v.items() if x is not None) + "}"
    raise TypeError(f"cannot serialize {type(v)}")


@dataclass
class Pod:
    id: str
    name: str
    status: str            # desiredStatus: RUNNING / EXITED / ...
    cost_per_hr: float
    gpu: str
    location: str
    uptime_s: int | None
    ssh_host: str
    ssh_port: int

    @classmethod
    def from_api(cls, d: dict) -> "Pod":
        rt = d.get("runtime") or {}
        host, port = "", 0
        for p in rt.get("ports") or []:
            if p.get("privatePort") == 22 and p.get("isIpPublic") and p.get("type") == "tcp":
                host, port = p["ip"], int(p["publicPort"])
        m = d.get("machine") or {}
        return cls(id=d["id"], name=d.get("name", ""), status=d.get("desiredStatus", "?"),
                   cost_per_hr=float(d.get("costPerHr") or 0), gpu=m.get("gpuDisplayName") or "",
                   location=m.get("location") or "", uptime_s=rt.get("uptimeInSeconds"),
                   ssh_host=host, ssh_port=port)


class RunPod:
    def __init__(self, key: str):
        self.key = key

    def _q(self, query: str) -> dict:
        r = requests.post(API, params={"api_key": self.key}, json={"query": query}, timeout=30)
        try:
            body = r.json()
        except ValueError:
            raise RunPodError(f"HTTP {r.status_code}: {r.text[:300]}")
        if body.get("errors"):
            raise RunPodError("; ".join(e.get("message", str(e)) for e in body["errors"]))
        return body["data"]

    def pods(self) -> list[Pod]:
        me = self._q(f"query {{ myself {{ pods {{ {POD_FIELDS} }} }} }}")["myself"]
        if me is None:
            raise RunPodError("unauthorized — check RUNPOD_API_KEY")
        return [Pod.from_api(p) for p in me["pods"]]

    def pod(self, pod_id: str) -> Pod | None:
        d = self._q(f'query {{ pod(input: {{podId: {json.dumps(pod_id)}}}) {{ {POD_FIELDS} }} }}')["pod"]
        return Pod.from_api(d) if d else None

    def find(self, name: str) -> Pod | None:
        return next((p for p in self.pods() if p.name == name), None)

    def gpu_types(self) -> list[dict]:
        q = "query { gpuTypes { id displayName memoryInGb secureCloud communityCloud securePrice communityPrice } }"
        return self._q(q)["gpuTypes"]

    def create(self, *, name: str, image: str, gpu_type_id: str, cloud: str, container_disk_gb: int,
               volume_gb: int, network_volume_id: str, public_key: str, min_vcpu: int = 8,
               min_memory_gb: int = 32) -> Pod:
        boot = resources.files("ogpu.remote").joinpath("bootstrap.sh").read_bytes()
        inp = {
            "name": name, "imageName": image, "gpuTypeId": gpu_type_id, "gpuCount": 1,
            "cloudType": Enum(cloud), "supportPublicIp": True, "startSsh": True,
            "containerDiskInGb": container_disk_gb,
            "volumeInGb": 0 if network_volume_id else volume_gb,
            "networkVolumeId": network_volume_id or None,
            "volumeMountPath": "/workspace", "ports": "22/tcp",
            "minVcpuCount": min_vcpu, "minMemoryInGb": min_memory_gb,
            # custom images ignore startSsh, so the start command installs sshd itself (see remote/bootstrap.sh)
            "dockerArgs": "bash -c 'echo $OGPU_BOOTSTRAP | base64 -d > /ogpu_boot.sh && exec bash /ogpu_boot.sh'",
            "env": [{"key": "OGPU_BOOTSTRAP", "value": base64.b64encode(boot).decode()},
                    {"key": "OGPU_PUBKEY", "value": public_key}],
        }
        d = self._q(f"mutation {{ podFindAndDeployOnDemand(input: {gql_literal(inp)}) {{ id }} }}")
        return self.pod(d["podFindAndDeployOnDemand"]["id"])

    def resume(self, pod_id: str) -> None:
        self._q(f'mutation {{ podResume(input: {{podId: {json.dumps(pod_id)}, gpuCount: 1}}) {{ id }} }}')

    def stop(self, pod_id: str) -> None:
        self._q(f'mutation {{ podStop(input: {{podId: {json.dumps(pod_id)}}}) {{ id }} }}')

    def terminate(self, pod_id: str) -> None:
        self._q(f'mutation {{ podTerminate(input: {{podId: {json.dumps(pod_id)}}}) }}')
