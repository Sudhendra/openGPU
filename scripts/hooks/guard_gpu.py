#!/usr/bin/env python3
"""Claude Code PreToolUse hook (Bash): block direct pod access so agents go through `ogpu`.

Blocks raw ssh/scp/rsync to pods, runpodctl, `modal run`, and the legacy OKC RunPod helper scripts.
Exit 2 + stderr = the tool call is rejected and the message is shown to the agent.
"""
import json
import re
import sys

BLOCK = [
    (re.compile(r"\brunpodctl\b"), "runpodctl"),
    (re.compile(r"\bmodal\s+run\b"), "modal run"),
    (re.compile(r"\b(ssh|scp|sftp)\b[^|;&]*(runpod|root@\d{1,3}(\.\d{1,3}){3}|-p\s*\d{4,5})"), "direct ssh to a pod"),
    (re.compile(r"\brsync\b[^|;&]*root@"), "direct rsync to a pod"),
    (re.compile(r"(tools/(pod|rpod|ssh_run|xfer|podrun)|runpod_bridge|runpod_control|pod/podc|pod/rp|okc_modal)\.py\b"),
     "legacy RunPod/Modal helper script"),
]
HINT = ("Use the ogpu CLI instead (load the `ogpu` skill): `ogpu run --gpu --json -- <cmd>` for timing/correctness, "
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
            print(f"Blocked: {what}. All GPU/pod access goes through ogpu so agents share the H200 safely. {HINT}",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
