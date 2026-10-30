#!/bin/bash
# Install the ogpu skill for Claude Code, Codex and OpenCode.
# ~/.agents/skills/ogpu -> this repo's skills/ogpu, and each agent's skills dir -> ~/.agents/skills/ogpu,
# so editing skills/ogpu/SKILL.md here updates every agent at once.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)/skills/ogpu"

link() {  # link <target> <link path>
  mkdir -p "$(dirname "$2")"
  if [ -e "$2" ] && [ ! -L "$2" ]; then
    echo "skip $2 (exists and is not a symlink)"; return
  fi
  ln -sfn "$1" "$2"
  echo "ok   $2 -> $1"
}

link "$SRC" "$HOME/.agents/skills/ogpu"
link "../../.agents/skills/ogpu" "$HOME/.claude/skills/ogpu"
link "../../.agents/skills/ogpu" "$HOME/.codex/skills/ogpu"
link "../../../.agents/skills/ogpu" "$HOME/.config/opencode/skills/ogpu"
