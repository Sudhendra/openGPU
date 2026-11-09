#!/usr/bin/env python3
"""Claude Code PreToolUse hook (Bash): block direct GPU-machine access so agents go through `ogpu`.

Blocks raw ssh/scp/rsync to machines and the providers' own CLIs (runpodctl, vastai, `modal run`).
Exit 2 + stderr = the tool call is rejected and the message is shown to the agent.
"""
import json
import re
import sys

BLOCK = [
    (re.compile(r"\brunpodctl\b"), "runpodctl"),
    (re.compile(r"\bvastai\b"), "vastai"),
    (re.compile(r"\bmodal\s+run\b"), "modal run"),
    (re.compile(r"\b(ssh|scp|sftp)\b[^|;&]*(runpod|vast\.ai|root@\d{1,3}(\.\d{1,3}){3}|-p\s*\d{4,5})"),
     "direct ssh to a GPU machine"),
    (re.compile(r"\brsync\b[^|;&]*root@"), "direct rsync to a GPU machine"),
]
HINT = ("Use the ogpu CLI instead (load the `ogpu` skill): `ogpu run --gpu --json -- <cmd>` for GPU work, "
        "`ogpu run --cpu --json -- <cmd>` for compile gates, `ogpu submit`/`ogpu wait` to batch, "
        "`ogpu ledger --best --json` for results.")


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    if event.get("tool_name") != "Bash":
        return 0
    cmd = (event.get("tool_input") or {}).get("command", "")
    if re.match(r"\s*ogpu\b", cmd):
        return 0
    for pat, what in BLOCK:
        if pat.search(cmd):
            print(f"Blocked: {what}. All GPU access goes through ogpu so agents share machines safely. {HINT}",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
