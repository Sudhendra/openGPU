"""Modal (serverless): every job runs in its own container and bills per second, so nothing ever idles.
Needs `pip install 'ogpu[modal]'` and `modal token new`. GPU ids are Modal's: H200, H100, A100-80GB, L40S, L4, ...

Containers are billed for GPU + CPU cores + memory, so keep CPU and memory small unless the job needs them:

    [providers.modal]
    cpu = 2            # physical cores (default 2)
    memory_gb = 8      # default 8
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# list $/hr (modal.com/pricing). Override a GPU's rate with OGPU_MODAL_<GPU>_HR, e.g. OGPU_MODAL_A100_80GB_HR.
GPU_HR = {"B200": 6.25, "H200": 4.54, "H100": 3.95, "A100-80GB": 2.50, "A100": 2.10, "L40S": 1.95, "A10G": 1.10,
          "L4": 0.80, "T4": 0.59, "none": 0.0}
CORE_HR = 0.0000131 * 3600     # per physical core
GIB_HR = 0.00000222 * 3600     # per GiB of memory
DEFAULT_GPU = "H100"
DEFAULT_CPU, DEFAULT_MEMORY_GB = 2.0, 8
APP = Path(__file__).with_name("modal_app.py")


def container(cfg) -> tuple[float, int]:
    """(cpu cores, memory GiB) each job's container gets."""
    o = cfg.providers.get("modal", {})
    return float(o.get("cpu", DEFAULT_CPU)), int(o.get("memory_gb", DEFAULT_MEMORY_GB))


def rate(gpu: str, cpu: float = DEFAULT_CPU, memory_gb: float = DEFAULT_MEMORY_GB) -> float:
    override = os.environ.get(f"OGPU_MODAL_{gpu.replace('-', '_').upper()}_HR")
    gpu_hr = float(override) if override else GPU_HR.get(gpu.split(":")[0], GPU_HR[DEFAULT_GPU])
    return gpu_hr + cpu * CORE_HR + memory_gb * GIB_HR


def gpu_for(cfg, mode: str) -> str:
    return "none" if mode == "cpu" else (cfg.gpu_type_ids or [DEFAULT_GPU])[0]


def gpu_types(cfg) -> list[dict]:
    cpu, mem = container(cfg)
    extra = cpu * CORE_HR + mem * GIB_HR
    return [{"id": g, "memory_gb": "", "price": f"{r:.2f} (+{extra:.2f} for {cpu:g} cores / {mem} GiB)"}
            for g, r in GPU_HR.items() if g != "none"]


def launch(cfg, *, track: Path, extra: list[Path], cmds: list[str], gpu: str, results: Path, log: Path,
           timeout_s: float | None, detach: bool):
    """Start `modal run` for these commands. detach=True returns the Popen (it outlives this process); otherwise
    waits and returns the exit code."""
    if shutil.which("modal") is None:
        raise SystemExit("modal CLI not found: pip install 'ogpu[modal]' && modal token new")
    results.parent.mkdir(parents=True, exist_ok=True)
    cmds_file = results.with_name("cmds.txt")
    cmds_file.write_text("\n".join(cmds) + "\n")
    cpu, mem = container(cfg)
    env = os.environ | {"OGPU_TRACK": str(track), "OGPU_WITH": os.pathsep.join(str(e) for e in extra),
                        "OGPU_MODAL_GPU": gpu, "OGPU_RESULTS": str(results), "OGPU_IMAGE": cfg.image,
                        "OGPU_TIMEOUT": str(int(timeout_s or cfg.default_timeout_s or 3600)),
                        "OGPU_MODAL_CPU": str(cpu), "OGPU_MODAL_MEMORY_MB": str(mem * 1024)}
    argv = ["modal", "run", str(APP), "--cmds", str(cmds_file)]
    if not detach:
        return subprocess.run(argv, env=env).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        return subprocess.Popen(argv, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
