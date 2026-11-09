"""RunPod GraphQL client: find/create/resume/stop pods with a real public SSH port (no PTY proxy)."""
from __future__ import annotations

import base64
import json
from importlib import resources

import requests

from . import Machine, Provider, ProviderError

API = "https://api.runpod.io/graphql"
POD_FIELDS = """
  id name desiredStatus costPerHr imageName gpuCount
  machine { gpuDisplayName location }
  runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
"""


class Enum(str):
    """A GraphQL enum literal (serialized unquoted)."""


def gql_literal(v) -> str:
    """Serialize a Python value as a GraphQL input literal (JSON strings are valid GraphQL strings)."""
    if isinstance(v, Enum):
        return str(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float, str)):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(gql_literal(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {gql_literal(x)}" for k, x in v.items() if x is not None) + "}"
    raise TypeError(f"cannot serialize {type(v)}")


def machine(d: dict) -> Machine:
    rt = d.get("runtime") or {}
    host, port = "", 0
    for p in rt.get("ports") or []:
        if p.get("privatePort") == 22 and p.get("isIpPublic") and p.get("type") == "tcp":
            host, port = p["ip"], int(p["publicPort"])
    m = d.get("machine") or {}
    return Machine(id=d["id"], name=d.get("name", ""), status=d.get("desiredStatus", "?"),
                   cost_per_hr=float(d.get("costPerHr") or 0), gpu=m.get("gpuDisplayName") or "",
                   location=m.get("location") or "", uptime_s=rt.get("uptimeInSeconds"), ssh_host=host, ssh_port=port)


def bootstrap_env(public_key: str) -> list[dict]:
    boot = resources.files("ogpu.remote").joinpath("bootstrap.sh").read_bytes()
    return [{"key": "OGPU_BOOTSTRAP", "value": base64.b64encode(boot).decode()},
            {"key": "OGPU_PUBKEY", "value": public_key}]


class RunPod(Provider):
    name = "runpod"
    env_key = "RUNPOD_API_KEY"
    default_gpus = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe"]
    default_tiers = ["COMMUNITY", "SECURE"]   # community is usually cheaper

    def _q(self, query: str) -> dict:
        # the key goes in a header, never the URL (URLs end up in proxy and server logs)
        r = requests.post(API, headers={"Authorization": f"Bearer {self.key}"}, json={"query": query}, timeout=30)
        try:
            body = r.json()
        except ValueError:
            raise ProviderError(f"HTTP {r.status_code}: {r.text[:300]}")
        if body.get("errors"):
            raise ProviderError("; ".join(e.get("message", str(e)) for e in body["errors"]))
        if "data" not in body:  # rate limits / gateway errors come back as JSON without data
            raise ProviderError(f"HTTP {r.status_code}: {str(body)[:300]}")
        return body["data"]

    def pods(self) -> list[Machine]:
        me = self._q(f"query {{ myself {{ pods {{ {POD_FIELDS} }} }} }}")["myself"]
        if me is None:
            raise ProviderError("unauthorized — check RUNPOD_API_KEY")
        return [machine(p) for p in me["pods"]]

    def get(self, machine_id: str) -> Machine | None:
        d = self._q(f'query {{ pod(input: {{podId: {json.dumps(machine_id)}}}) {{ {POD_FIELDS} }} }}')["pod"]
        return machine(d) if d else None

    def find(self, name: str) -> Machine | None:
        return next((p for p in self.pods() if p.name == name), None)

    def gpu_types(self) -> list[dict]:
        q = "query { gpuTypes { id displayName memoryInGb securePrice communityPrice } }"
        return [{"id": g["id"], "memory_gb": g["memoryInGb"],
                 "price": " / ".join(f"{t} {g[k]}" for t, k in (("community", "communityPrice"),
                                                                 ("secure", "securePrice")) if g.get(k))}
                for g in self._q(q)["gpuTypes"]]

    def create(self, cfg, gpu: str, tier: str) -> Machine:
        inp = {
            "name": cfg.pod_name, "imageName": cfg.image, "gpuTypeId": gpu, "gpuCount": 1,
            "cloudType": Enum(tier), "supportPublicIp": True, "startSsh": True,
            "containerDiskInGb": cfg.container_disk_gb,
            "volumeInGb": 0 if cfg.network_volume_id else cfg.volume_gb,
            "networkVolumeId": cfg.network_volume_id or None,
            "volumeMountPath": "/workspace", "ports": "22/tcp",
            "minVcpuCount": cfg.min_vcpu, "minMemoryInGb": cfg.min_memory_gb,
            "allowedCudaVersions": cfg.allowed_cuda_versions,  # hosts with an older driver can't run the image
            # custom images ignore startSsh, so the start command installs sshd itself (see remote/bootstrap.sh)
            "dockerArgs": "bash -c 'echo $OGPU_BOOTSTRAP | base64 -d > /ogpu_boot.sh && exec bash /ogpu_boot.sh'",
            "env": bootstrap_env(cfg.public_key),
        }
        try:
            d = self._q(f"mutation {{ podFindAndDeployOnDemand(input: {gql_literal(inp)}) {{ id }} }}")
        except ProviderError as e:
            if "cuda" not in str(e).lower():
                raise
            # the API rejected the filter itself: the post-boot GPU check still guards us
            inp.pop("allowedCudaVersions")
            d = self._q(f"mutation {{ podFindAndDeployOnDemand(input: {gql_literal(inp)}) {{ id }} }}")
        return self.get(d["podFindAndDeployOnDemand"]["id"])

    def resume(self, machine_id: str) -> None:
        self._q(f'mutation {{ podResume(input: {{podId: {json.dumps(machine_id)}, gpuCount: 1}}) {{ id }} }}')

    def stop(self, machine_id: str) -> None:
        self._q(f'mutation {{ podStop(input: {{podId: {json.dumps(machine_id)}}}) {{ id }} }}')

    def terminate(self, machine_id: str) -> None:
        self._q(f'mutation {{ podTerminate(input: {{podId: {json.dumps(machine_id)}}}) }}')
