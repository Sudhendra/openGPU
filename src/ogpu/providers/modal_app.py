"""One generic Modal app for parallel exploration (replaces the per-track okc_modal.py copies).

Launched by `ogpu fanout --backend modal`; the env picks the track and GPU at import time:
  OGPU_TRACK=/abs/path/to/track  OGPU_MODAL_GPU=H200|H100|none  OGPU_RESULTS=/abs/path/results.json
  modal run src/ogpu/providers/modal_app.py --cmds cmds.txt

Each command gets its own container in the judge image with the track at /root/okc/<track>; compiled torch
extensions are shared through one Volume; files written under out/ come back with the log.
"""
import os
import time
from pathlib import Path

import modal

TRACK = Path(os.environ.get("OGPU_TRACK", ".")).resolve()
ROOT = f"/root/okc/{TRACK.name}"
GPU = os.environ.get("OGPU_MODAL_GPU", "H200")
IGNORE = ["output", "out", "profiles", "runs", ".git", "**/__pycache__", "**/.DS_Store", "*.log", ".venv"]

app = modal.App("ogpu-fanout")
image = (modal.Image.from_registry(os.environ.get("OGPU_IMAGE", "lmsysorg/sglang:dev-muse-spark-open"))
         .env({"PYTHONUNBUFFERED": "1", "TORCH_EXTENSIONS_DIR": "/tmp/torch_ext"})
         .add_local_dir(TRACK, "/src", ignore=IGNORE))
cache = modal.Volume.from_name("ogpu-ext-cache", create_if_missing=True)


@app.function(image=image, gpu=None if GPU == "none" else GPU, cpu=8, memory=32768, timeout=3600,
              volumes={"/cache": cache}, scaledown_window=120, retries=0)
def run(job):
    import base64
    import io
    import os
    import subprocess
    import tarfile
    import time

    i, cmd = job
    subprocess.run(f"rm -rf {ROOT} && mkdir -p {ROOT} && cp -r /src/. {ROOT}/ && mkdir -p {ROOT}/out "
                   "&& mkdir -p /cache/torch_ext /tmp/torch_ext && cp -r /cache/torch_ext/. /tmp/torch_ext/",
                   shell=True, check=True)
    have = set(os.listdir("/tmp/torch_ext"))
    head = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True,
                          text=True).stdout.strip() or "cpu"
    lines = [f"[{i}] gpu: {head}", f"[{i}] $ {cmd}"]
    t0 = time.time()
    p = subprocess.Popen(["bash", "-lc", cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for ln in p.stdout:
        lines.append(ln.rstrip("\n"))
        print(f"[{i}] {ln}", end="", flush=True)
    rc = p.wait()
    wall = time.time() - t0
    # publish newly built extensions atomically (copy to a temp name, then rename)
    new = [d for d in os.listdir("/tmp/torch_ext") if d not in have]
    for d in new:
        if not os.path.exists(f"/cache/torch_ext/{d}"):
            subprocess.run(f"cp -r /tmp/torch_ext/{d} /cache/torch_ext/.{d}.{os.getpid()} && "
                           f"mv -T /cache/torch_ext/.{d}.{os.getpid()} /cache/torch_ext/{d}", shell=True)
    if new:
        cache.commit()
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as t:
        t.add(f"{ROOT}/out", arcname="out")
    return {"i": i, "cmd": cmd, "rc": rc, "wall_s": wall, "gpu": head, "log": "\n".join(lines),
            "out": base64.b64encode(b.getvalue()).decode()}


@app.local_entrypoint()
def main(cmds: str = "", cmd: str = ""):
    import base64
    import io
    import json
    import tarfile

    jobs = [cmd] if cmd else [ln.strip() for ln in open(cmds) if ln.strip() and not ln.lstrip().startswith("#")]
    results_path = Path(os.environ.get("OGPU_RESULTS", TRACK / "output" / "ogpu-modal" / time.strftime("%m%d_%H%M%S")
                                       / "results.json"))
    dest = results_path.parent
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    for r in run.map(list(enumerate(jobs)), return_exceptions=True):
        if isinstance(r, Exception):
            results.append({"error": repr(r)})
            continue
        log_path = dest / f"{r['i']}.log"
        log_path.write_text(r["log"] + "\n")
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(r.pop("out"))), mode="r:gz") as t:
            for m in t.getmembers():
                if m.isfile():
                    p = dest / str(r["i"]) / Path(m.name).relative_to("out")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(t.extractfile(m).read())
        r.pop("log")
        results.append(r | {"log_path": str(log_path)})
    results_path.write_text(json.dumps(results, indent=1))
