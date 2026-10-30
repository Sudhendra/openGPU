"""Real SSH + rsync to the pod's public TCP port. One multiplexed master connection makes the many small calls
(polls, snapshots) cost ~tens of ms instead of a full handshake each."""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from .config import State

# Unix socket paths are capped at 104 bytes on macOS, so keep the control sockets in a short private dir.
SOCK_DIR = Path(f"/tmp/ogpu-{os.getuid()}")
last_error = ""

EXCLUDES = [".git", "output", "out", "profiles", "runs", "__pycache__", "*.pyc", ".venv", ".DS_Store", "*.log",
            "node_modules", ".pytest_cache"]


def ssh_opts(state: State, key: str) -> list[str]:
    SOCK_DIR.mkdir(mode=0o700, exist_ok=True)
    return ["-p", str(state.port), "-i", key,
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ControlMaster=auto", "-o", f"ControlPath={SOCK_DIR}/%C", "-o", "ControlPersist=10m",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "ConnectTimeout=10"]


def ssh_argv(state: State, key: str, remote_cmd: str) -> list[str]:
    return ["ssh", *ssh_opts(state, key), f"root@{state.host}", remote_cmd]


def run(state: State, key: str, remote_cmd: str, *, check: bool = True, timeout: float | None = 60,
        capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(ssh_argv(state, key, remote_cmd), capture_output=capture, text=True, check=check,
                          timeout=timeout)


def rsync_argv(state: State, key: str, src: str, dst: str, *, delete: bool = False, excludes=EXCLUDES) -> list[str]:
    e = "ssh " + " ".join(shlex.quote(o) for o in ssh_opts(state, key))
    argv = ["rsync", "-az", "--no-owner", "--no-group", "--no-perms", "-e", e]
    if delete:
        argv.append("--delete")
    for x in excludes:
        argv += ["--exclude", x]
    return argv + [src, dst]


def push(state: State, key: str, local_dir: Path, remote_dir: str) -> None:
    run(state, key, f"mkdir -p {shlex.quote(remote_dir)}")
    argv = rsync_argv(state, key, f"{local_dir}/", f"root@{state.host}:{remote_dir}/", delete=True)
    subprocess.run(argv, check=True, capture_output=True, text=True)


def pull(state: State, key: str, remote_dir: str, local_dir: Path) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    argv = rsync_argv(state, key, f"root@{state.host}:{remote_dir}/", f"{local_dir}/", excludes=[])
    return subprocess.run(argv, capture_output=True, text=True).returncode == 0


def reachable(state: State, key: str) -> bool:
    if not state.reachable:
        return False
    global last_error
    try:
        p = run(state, key, "true", check=False, timeout=20)
        last_error = p.stderr.strip().splitlines()[-1][:120] if p.returncode and p.stderr.strip() else ""
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        last_error = "ssh timed out"
        return False
