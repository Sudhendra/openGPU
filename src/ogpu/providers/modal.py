"""Modal (serverless): every job runs in its own container and bills per second, so nothing ever idles.
Needs `pip install 'ogpu[modal]'` and `modal token new`. GPU ids are Modal's: H200, H100, A100-80GB, L40S, L4, ..."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# list $/hr per GPU (Modal pricing page), plus ~CPU_HR for the 8 vCPU / 32 GiB each container gets
RATES = {"B200": 6.25, "H200": 4.54, "H100": 3.95, "A100-80GB": 2.50, "A100": 2.10, "L40S": 1.95, "A10G": 1.10,
         "L4": 0.80, "T4": 0.59, "none": 0.0}
CPU_HR = 0.35
DEFAULT_GPU = "H100"
APP = Path(__file__).with_name("modal_app.py")


def rate(gpu: str) -> float:
    override = os.environ.get(f"OGPU_MODAL_{gpu.replace('-', '_').upper()}_HR")
    return float(override) if override else RATES.get(gpu.split(":")[0], RATES[DEFAULT_GPU]) + CPU_HR


def gpu_for(cfg, mode: str) -> str:
    return "none" if mode == "cpu" else (cfg.gpu_type_ids or [DEFAULT_GPU])[0]


def gpu_types() -> list[dict]:
    return [{"id": g, "memory_gb": "", "price": f"{r:.2f} (+{CPU_HR:.2f} cpu)"} for g, r in RATES.items() if g != "none"]


def launch(cfg, *, track: Path, extra: list[Path], cmds: list[str], gpu: str, results: Path, log: Path,
           timeout_s: float | None, detach: bool):
    """Start `modal run` for these commands. detach=True returns the Popen (it outlives this process); otherwise
    waits and returns the exit code."""
    if shutil.which("modal") is None:
        raise SystemExit("modal CLI not found: pip install 'ogpu[modal]' && modal token new")
    results.parent.mkdir(parents=True, exist_ok=True)
    cmds_file = results.with_name("cmds.txt")
    cmds_file.write_text("\n".join(cmds) + "\n")
    env = os.environ | {"OGPU_TRACK": str(track), "OGPU_WITH": os.pathsep.join(str(e) for e in extra),
                        "OGPU_MODAL_GPU": gpu, "OGPU_RESULTS": str(results), "OGPU_IMAGE": cfg.image,
                        "OGPU_TIMEOUT": str(int(timeout_s or cfg.default_timeout_s or 3600))}
    argv = ["modal", "run", str(APP), "--cmds", str(cmds_file)]
    if not detach:
        return subprocess.run(argv, env=env).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        return subprocess.Popen(argv, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
