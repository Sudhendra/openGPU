"""Vast.ai REST client: rent the cheapest matching offer as an SSH instance; stop/start keeps its disk.

Stopped instances keep billing for storage (not GPU). Uptime comes from the instance's start date, so after a
stop/start the paid-time estimate errs high (the budget cap stays conservative)."""
from __future__ import annotations

import base64
import time
from importlib import resources

import requests

from . import Machine, Provider, ProviderError

API = "https://console.vast.ai/api/v0"
# Vast runs its own sshd for ssh runtypes, so the bootstrap only adds our key, the env snapshot and the reaper
ONSTART = ("echo $OGPU_BOOTSTRAP | base64 -d > /root/ogpu_boot.sh && "
           "OGPU_SSHD=0 OGPU_KEEPALIVE=0 bash /root/ogpu_boot.sh")


def machine(d: dict) -> Machine:
    want = d.get("intended_status") or d.get("cur_state") or ""
    status = {"running": "RUNNING", "stopped": "EXITED"}.get(want, (want or "?").upper())
    host, port = "", 0
    direct = (d.get("ports") or {}).get("22/tcp") or []
    if direct and d.get("public_ipaddr"):
        host, port = d["public_ipaddr"].strip(), int(direct[0]["HostPort"])
    elif d.get("ssh_host"):
        host, port = d["ssh_host"], int(d.get("ssh_port") or 0)
    up = time.time() - d["start_date"] if d.get("actual_status") == "running" and d.get("start_date") else None
    return Machine(id=str(d["id"]), name=d.get("label") or "", status=status, cost_per_hr=float(d.get("dph_total") or 0),
                   gpu=d.get("gpu_name") or "", location=d.get("geolocation") or "", uptime_s=up,
                   ssh_host=host, ssh_port=port)


def offer_query(cfg, gpu: str, tier: str) -> dict:
    q = {"gpu_name": {"eq": gpu}, "num_gpus": {"eq": 1}, "rentable": {"eq": True}, "rented": {"eq": False},
         "cpu_cores_effective": {"gte": cfg.min_vcpu}, "cpu_ram": {"gte": cfg.min_memory_gb * 1024},
         "disk_space": {"gte": cfg.container_disk_gb + cfg.volume_gb}, "direct_port_count": {"gte": 1},
         "order": [["dph_total", "asc"]], "type": "on-demand", "limit": 5}
    if tier == "verified":
        q["verified"] = {"eq": True}
    if cfg.min_cuda:
        q["cuda_max_good"] = {"gte": float(cfg.min_cuda)}
    return q


class Vast(Provider):
    name = "vast"
    env_key = "VAST_API_KEY"
    default_gpus = ["H100 SXM", "H100 NVL", "H100 PCIE"]
    default_tiers = ["verified", "any"]

    def _req(self, method: str, path: str, **kw):
        r = requests.request(method, API + path, headers={"Authorization": f"Bearer {self.key}"}, timeout=30, **kw)
        try:
            body = r.json()
        except ValueError:
            raise ProviderError(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400 or (isinstance(body, dict) and body.get("success") is False):
            msg = body.get("msg") or body.get("error") if isinstance(body, dict) else None
            raise ProviderError(f"HTTP {r.status_code}: {msg or str(body)[:300]}")
        return body

    def get(self, machine_id: str) -> Machine | None:
        d = self._req("GET", f"/instances/{int(machine_id)}/").get("instances")
        return machine(d) if d else None

    def find(self, name: str) -> Machine | None:
        return next((machine(d) for d in self._req("GET", "/instances/").get("instances") or []
                     if d.get("label") == name), None)

    def create(self, cfg, gpu: str, tier: str) -> Machine:
        offers = self._req("POST", "/bundles/", json=offer_query(cfg, gpu, tier)).get("offers") or []
        if not offers:
            raise ProviderError("no matching offers")
        boot = base64.b64encode(resources.files("ogpu.remote").joinpath("bootstrap.sh").read_bytes()).decode()
        body = {"client_id": "me", "image": cfg.image, "disk": cfg.container_disk_gb + cfg.volume_gb,
                "label": cfg.pod_name, "runtype": "ssh_direc ssh_proxy", "onstart": ONSTART,
                "env": {"OGPU_BOOTSTRAP": boot, "OGPU_PUBKEY": cfg.public_key}}
        err = None
        for o in offers:  # the cheapest offer may be rented by someone else between search and accept
            try:
                new = self._req("PUT", f"/asks/{o['id']}/", json=body)["new_contract"]
            except (ProviderError, KeyError) as e:
                err = e
                continue
            try:
                self._req("POST", f"/instances/{new}/ssh/", json={"ssh_key": cfg.public_key})
            except ProviderError:
                pass  # the onstart script installs the key too
            return self.get(str(new)) or Machine(id=str(new), name=cfg.pod_name, status="RUNNING",
                                                 cost_per_hr=float(o.get("dph_total") or 0), gpu=gpu)
        raise ProviderError(f"every offer was taken: {err}")

    def resume(self, machine_id: str) -> None:
        self._req("PUT", f"/instances/{int(machine_id)}/", json={"state": "running"})

    def stop(self, machine_id: str) -> None:
        self._req("PUT", f"/instances/{int(machine_id)}/", json={"state": "stopped"})

    def terminate(self, machine_id: str) -> None:
        self._req("DELETE", f"/instances/{int(machine_id)}/")

    def gpu_types(self) -> list[dict]:
        q = {"num_gpus": {"eq": 1}, "rentable": {"eq": True}, "order": [["dph_total", "asc"]], "type": "on-demand",
             "limit": 500}
        best: dict[str, dict] = {}
        for o in self._req("POST", "/bundles/", json=q).get("offers") or []:
            g = o.get("gpu_name") or "?"
            if g not in best:
                best[g] = {"id": g, "memory_gb": round((o.get("gpu_ram") or 0) / 1024),
                           "price": f"from {o.get('dph_total', 0):.2f}"}
        return list(best.values())
