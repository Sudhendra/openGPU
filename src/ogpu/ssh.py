"""Real SSH + rsync to the machine's public TCP port. One multiplexed master connection makes the many small calls
(polls, snapshots) cost ~tens of ms instead of a full handshake each.

Host keys are pinned per machine (trust on first use): the first connection to a machine records its key in
~/.ogpu/known_hosts/<machine id>, and every later connection must present the same key. The file is reset only
when ogpu itself (re)starts that machine (see `forget_host`)."""
from __future__ import annotations

import os
import re
import shlex
import stat
import subprocess
from pathlib import Path

from . import config
from .config import State

# Unix socket paths are capped at 104 bytes on macOS, so keep the control sockets in a short private dir.
SOCK_DIR = Path(f"/tmp/ogpu-{os.getuid()}")
last_error = ""

EXCLUDES = [".git", "output", "out", "profiles", "runs", "__pycache__", "*.pyc", ".venv", ".DS_Store", "*.log",
            "node_modules", ".pytest_cache"]


def private_dir(path: Path) -> Path:
    """mkdir 0700 and refuse a dir someone else created first (e.g. a pre-planted /tmp/ogpu-<uid>)."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise SystemExit(f"{path} is not a directory owned by you; remove it and retry")
    if st.st_mode & 0o077:
        os.chmod(path, 0o700)
    return path


def known_hosts(state: State) -> Path:
    name = re.sub(r"[^A-Za-z0-9_.:-]", "_", state.pod_id or f"{state.host}_{state.port}")
    return private_dir(config.HOME / "known_hosts") / name


def forget_host(state: State) -> None:
    """Called when ogpu has just created or resumed this machine: a fresh boot may present a new host key."""
    known_hosts(state).unlink(missing_ok=True)


def ssh_opts(state: State, key: str) -> list[str]:
    private_dir(SOCK_DIR)
    return ["-p", str(state.port), "-i", key, "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known_hosts(state)}",
            "-o", "LogLevel=ERROR",
            "-o", "ControlMaster=auto", "-o", f"ControlPath={SOCK_DIR}/%C", "-o", "ControlPersist=10m",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "ConnectTimeout=10"]


def target(state: State) -> str:
    return f"{state.user or 'root'}@{state.host}"


def ssh_argv(state: State, key: str, remote_cmd: str) -> list[str]:
    return ["ssh", *ssh_opts(state, key), target(state), remote_cmd]


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
    argv = rsync_argv(state, key, f"{local_dir}/", f"{target(state)}:{remote_dir}/", delete=True)
    subprocess.run(argv, check=True, capture_output=True, text=True)


def pull(state: State, key: str, remote_dir: str, local_dir: Path) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    # --safe-links: a symlink the job wrote can't point outside local_dir
    argv = rsync_argv(state, key, f"{target(state)}:{remote_dir}/", f"{local_dir}/", excludes=[])
    argv.insert(-2, "--safe-links")
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
