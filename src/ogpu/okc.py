"""`profile = "okc"`: Open Kernel Challenge defaults and its bench output format. Nothing here applies otherwise.

Bench lines parsed into metrics:
  SCORE <file>: <speedup>                         → metrics.score[file]
  CHECK <case> <file> <tag>: PASS|FAIL ...        → checks, check_pass/fail, check_margin (worst error ÷ bound)
  CHECKS: ALL PASS | <n> FAIL                     → checks
  SIDE <name>: {"ratio_vs_s0": ..., "ci95": ...}  → metrics.sides[name]
"""
from __future__ import annotations

import json
import re

DEFAULTS = {
    "pod_name": "ogpu-h200",
    "image": "lmsysorg/sglang:dev-muse-spark-open",   # the judge image
    "gpu_type_ids": ["NVIDIA H200"],                   # H200 SXM, the judge GPU (not "NVIDIA H200 NVL")
    "cloud_types": ["COMMUNITY", "SECURE"],
    "container_disk_gb": 60,
    "volume_gb": 100,
    "idle_minutes": 20,
    "budget_usd": 60.0,
    "min_cuda": "13.0",                                # the image's torch is cu130
    "remote_root": "/workspace/okc",
    # 132-SM parts first (same SM count as H200 → same launch configs); 114-SM PCIe last
    "pools": {"correctness": {"gpu_type_ids": ["NVIDIA H100 NVL", "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"],
                              "idle_minutes": 10}},
}

SCORE = re.compile(r"^SCORE (\S+): ([\d.]+)")
CHECK = re.compile(r"^CHECK (\S+) (\S+) (\S+): (PASS|FAIL)")
ERR = re.compile(r"relative L2 error ([\d.e+-]+)[^\[]*\[bound ([\d.e+-]+)")
CHECKS = re.compile(r"^CHECKS: (ALL PASS|(\d+) FAIL)")
SIDE = re.compile(r"^SIDE (\S+): (\{.*\})\s*$")


def parse(log: str) -> dict:
    m: dict = {}
    fails = passes = 0
    for line in log.splitlines():
        line = line.strip()
        if r := SCORE.match(line):
            m.setdefault("score", {})[r[1]] = float(r[2])
        elif r := CHECK.match(line):
            passes += r[4] == "PASS"
            fails += r[4] == "FAIL"
            for err, bound in ERR.findall(line):  # how close to the judge's bound (1.0 = at the bound)
                try:
                    ratio = float(err) / float(bound)
                except (ValueError, ZeroDivisionError):
                    continue
                m["check_margin"] = max(m.get("check_margin", 0.0), round(ratio, 4))
        elif r := CHECKS.match(line):
            m["checks"] = "PASS" if r[1] == "ALL PASS" else "FAIL"
        elif r := SIDE.match(line):
            try:
                m.setdefault("sides", {})[r[1]] = json.loads(r[2])
            except ValueError:
                pass
    if passes or fails:
        m["check_pass"], m["check_fail"] = passes, fails
        m.setdefault("checks", "FAIL" if fails else "PASS")
    return m
