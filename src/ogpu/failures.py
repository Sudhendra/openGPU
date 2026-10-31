"""Classify a finished job so agents get `failure` + a short `error` excerpt instead of reading whole logs."""
from __future__ import annotations

import re

# (failure type, pattern) — first match wins, most specific first
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("oom", re.compile(r"CUDA out of memory|OutOfMemoryError|out of memory|CUBLAS_STATUS_ALLOC_FAILED", re.I)),
    ("cuda_error", re.compile(r"illegal memory access|misaligned address|device-side assert|CUDA error|"
                              r"unspecified launch failure|cudaErrorLaunch|an illegal instruction", re.I)),
    ("compile_error", re.compile(r"Error building extension|ptxas (fatal|error)|nvcc fatal|ninja: build stopped|"
                                 r"error: identifier|error: no instance|error: expected|\.cu\(\d+\): error|"
                                 r"CompilationError|triton\.compiler\.errors|\berror: ", re.I)),
    ("import_error", re.compile(r"ModuleNotFoundError|ImportError|SyntaxError|IndentationError|NameError")),
]
ERRORISH = re.compile(r"error|Error|FAIL|Traceback|fatal|assert|Killed|exception", re.I)


def excerpt(log: str, max_lines: int = 14) -> str:
    """The last traceback, or else the last error-looking lines (deduplicated), trimmed for an agent's context."""
    lines = log.splitlines()
    tb = max((i for i, l in enumerate(lines) if l.startswith("Traceback")), default=None)
    if tb is not None:
        block = lines[tb:]
        if len(block) > max_lines:
            block = block[:3] + ["  …"] + block[-(max_lines - 4):]
        return "\n".join(l[:240] for l in block)
    hits, seen = [], set()
    for l in reversed(lines):
        key = l.strip()
        if ERRORISH.search(l) and key not in seen:
            seen.add(key)
            hits.append(l[:240])
            if len(hits) >= max_lines:
                break
    return "\n".join(reversed(hits)) or "\n".join(l[:240] for l in lines[-max_lines:])


def classify(status: str, exit_code: int | None, log: str, metrics: dict,
             hang: dict | None = None) -> tuple[str | None, str | None]:
    """→ (failure, error). failure is None for a clean pass."""
    if hang:  # the pod watchdog killed it; its evidence beats anything we could infer from the log
        return "hang", f"watchdog: {hang.get('reason')}\n" + excerpt(log, 6)
    if status == "done" and metrics.get("checks", "PASS") == "PASS":
        return None, None
    if status == "lost":
        return "lost", "the job runner died (pod restart or killed process); re-run it"
    if status == "cancelled":
        return "cancelled", None
    if exit_code == 124:
        return "timeout", excerpt(log, 6)
    if exit_code in (137, -9) and "Killed" in log[-2000:]:
        return "oom", "process was SIGKILLed (host RAM OOM?)\n" + excerpt(log, 6)
    tail = log[-20000:]
    for kind, pat in PATTERNS:
        if pat.search(tail):
            return kind, excerpt(log)
    if metrics.get("checks") == "FAIL":
        fails = [l for l in log.splitlines() if re.match(r"^CHECK .*: FAIL", l.strip()) or l.startswith("CHECKS:")]
        return "correctness_fail", "\n".join(fails[-10:])
    return "runtime_error", excerpt(log)
