"""The Modal app behind `provider = "modal"` and `ogpu fanout`: one container per command, billed per second.

Launched by ogpu as `modal run modal_app.py --cmds cmds.txt`; the env picks the dirs, GPU and image at import time:
  OGPU_TRACK=/abs/track  OGPU_WITH=/abs/a:/abs/b  OGPU_MODAL_GPU=H100|none  OGPU_IMAGE=...  OGPU_TIMEOUT=3600
  OGPU_RESULTS=/abs/results.json

Each command runs in a fresh copy of the dirs under /root/src/; compiled torch extensions are shared through one
Volume; files written under out/ come back next to the log.
"""
import os
import shlex
import time
from pathlib import Path

import modal

TRACK = Path(os.environ.get("OGPU_TRACK", ".")).resolve()
EXTRA = [Path(p).resolve() for p in os.environ.get("OGPU_WITH", "").split(os.pathsep) if p]
GPU = os.environ.get("OGPU_MODAL_GPU", "H100")
IGNORE = ["output", "out", "profiles", "runs", ".git", "**/__pycache__", "**/.DS_Store", "*.log", ".venv",
          "node_modules"]

app = modal.App("ogpu")
image = (modal.Image.from_registry(os.environ.get("OGPU_IMAGE", "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"))
         .env({"PYTHONUNBUFFERED": "1", "TORCH_EXTENSIONS_DIR": "/tmp/torch_ext", "OGPU_ROOT": f"/root/src/{TRACK.name}"}))
for d in [TRACK, *EXTRA]:
    image = image.add_local_dir(d, f"/src/{d.name}", ignore=IGNORE)
cache = modal.Volume.from_name("ogpu-ext-cache", create_if_missing=True)


@app.function(image=image, gpu=None if GPU == "none" else GPU, cpu=8, memory=32768,
              timeout=int(float(os.environ.get("OGPU_TIMEOUT") or 3600)), volumes={"/cache": cache},
              scaledown_window=60, retries=0)
def run(job):
    import base64
    import io
    import os
    import subprocess
    import tarfile
    import time

    i, cmd = job
    root = os.environ["OGPU_ROOT"]
    subprocess.run(f"rm -rf /root/src && mkdir -p /root/src && cp -r /src/. /root/src/ && mkdir -p {shlex.quote(root)}/out "
                   "&& mkdir -p /cache/torch_ext /tmp/torch_ext && cp -r /cache/torch_ext/. /tmp/torch_ext/",
                   shell=True, check=True)
    have = set(os.listdir("/tmp/torch_ext"))
    head = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True,
                          text=True).stdout.strip() or "cpu"
    lines = [f"[{i}] gpu: {head}", f"[{i}] $ {cmd}"]
    t0 = time.time()
    p = subprocess.Popen(["bash", "-lc", cmd], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for ln in p.stdout:
        lines.append(ln.rstrip("\n"))
        print(f"[{i}] {ln}", end="", flush=True)
    rc = p.wait()
    wall = time.time() - t0
    # publish newly built extensions atomically (copy to a temp name, then rename)
    new = [d for d in os.listdir("/tmp/torch_ext") if d not in have]
    for d in new:
        if not os.path.exists(f"/cache/torch_ext/{d}"):
            tmp = shlex.quote(f"/cache/torch_ext/.{d}.{os.getpid()}")
            subprocess.run(f"cp -r {shlex.quote(f'/tmp/torch_ext/{d}')} {tmp} && "
                           f"mv -T {tmp} {shlex.quote(f'/cache/torch_ext/{d}')}", shell=True)
    if new:
        cache.commit()
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as t:
        t.add(f"{root}/out", arcname="out")
    return {"i": i, "cmd": cmd, "rc": rc, "wall_s": wall, "gpu": head, "log": "\n".join(lines),
            "out": base64.b64encode(b.getvalue()).decode()}


def extract_out(tar_bytes: bytes, dest: Path) -> None:
    """Unpack a job's out/ into dest; members that would land outside it (../, absolute paths) are skipped."""
    import io
    import tarfile
    dest = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            p = (dest / Path(m.name).relative_to("out")).resolve()
            if not p.is_relative_to(dest):
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(t.extractfile(m).read())


@app.local_entrypoint()
def main(cmds: str = ""):
    import base64
    import json

    jobs = [ln.strip() for ln in open(cmds) if ln.strip() and not ln.lstrip().startswith("#")]
    results_path = Path(os.environ.get("OGPU_RESULTS") or TRACK / "output" / "ogpu-modal" /
                        time.strftime("%m%d_%H%M%S") / "results.json")
    dest = results_path.parent
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    for r in run.map(list(enumerate(jobs)), return_exceptions=True):
        if isinstance(r, Exception):
            results.append({"error": repr(r)})
            continue
        log_path = dest / f"{r['i']}.log"
        log_path.write_text(r.pop("log") + "\n")
        extract_out(base64.b64decode(r.pop("out")), dest / str(r["i"]))
        results.append(r | {"log_path": str(log_path), "ended": time.time()})
    tmp = results_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=1))
    tmp.replace(results_path)
