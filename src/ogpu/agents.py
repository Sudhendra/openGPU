"""Which coding agent is invoking ogpu? OGPU_AGENT wins; otherwise walk the parent process chain."""
from __future__ import annotations

import hashlib
import os

import psutil

KNOWN = {"claude": "claude", "codex": "codex", "opencode": "opencode", "cursor-agent": "cursor", "aider": "aider",
         "gemini": "gemini"}
PALETTE = ["bright_magenta", "bright_cyan", "bright_green", "bright_yellow", "bright_blue", "orange1", "hot_pink",
           "spring_green1"]
FIXED = {"claude": "orange1", "codex": "bright_cyan", "opencode": "bright_green", "human": "bright_white"}


def classify(names: list[str]) -> str | None:
    """First known agent in a list of process names / cmdline basenames (nearest parent first)."""
    for n in names:
        base = os.path.basename(n).lower()
        for key, agent in KNOWN.items():
            if base == key or base.startswith(key + "-") or base.startswith(key + "."):
                return agent
    return None


def _chain() -> list[str]:
    names = []
    try:
        p = psutil.Process().parent()
        while p is not None and p.pid > 1:
            try:
                names.append(p.name())
                names += [os.path.basename(a) for a in p.cmdline()[:2]]
            except (psutil.AccessDenied, psutil.ZombieProcess):
                pass
            p = p.parent()
    except psutil.Error:
        pass
    return names


def detect() -> str:
    if os.environ.get("OGPU_AGENT"):
        return os.environ["OGPU_AGENT"]
    found = classify(_chain())
    if found:
        return found
    if os.environ.get("CLAUDECODE"):
        return "claude"
    return "human"


def color(agent: str) -> str:
    base = agent.split("(")[0].split("-")[0]
    if base in FIXED:
        return FIXED[base]
    return PALETTE[int(hashlib.md5(agent.encode()).hexdigest(), 16) % len(PALETTE)]
