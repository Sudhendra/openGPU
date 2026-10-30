"""Desktop notifications (macOS) for things worth looking up from the terminal for. Never raises."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from .config import HOME


def send(title: str, message: str) -> None:
    if os.environ.get("OGPU_NO_NOTIFY") or sys.platform != "darwin":
        return
    script = f"display notification {json.dumps(message)} with title {json.dumps(title)} sound name \"Glass\""
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass


def once(key: str, title: str, message: str, per_s: float = 6 * 3600) -> None:
    """Send at most once per `per_s` for a given key (e.g. the 80%-of-budget warning)."""
    path = HOME / "notified.json"
    try:
        seen = json.loads(path.read_text())
    except (OSError, ValueError):
        seen = {}
    if time.time() - seen.get(key, 0) < per_s:
        return
    seen[key] = time.time()
    path.write_text(json.dumps(seen))
    send(title, message)
