# openGPU · `ogpu`

A personal H200 broker and shared bench service for GPU kernel hill-climbing with coding agents (Claude Code, Codex, OpenCode). Built first for the Open Kernel Challenge.

- **Get an H200 with one command:** reuse, resume, or create a RunPod H200 SXM in the judge image, with a real public SSH port (no PTY/base64 hacks).
- **Never pay for idle:** an on-pod reaper stops the pod after `idle_minutes` with no jobs. The disk and compiled-extension cache survive, and `ogpu up` resumes.
- **Many agents, one GPU, clean timings:** `ogpu run --gpu` serializes on an exclusive lock. `--cpu` compile gates run in parallel. Each job runs from its own snapshot, so later pushes never leak into a queued job.
- **Never pay twice:** a content-hash cache (sources + command + image) returns identical runs instantly.
- **One ledger:** every run's parsed metrics (`SCORE`, `CHECK`, `SIDE`, `OGPU_METRIC`), cost, and agent go into `~/.ogpu/ledger.db`.
- **`ogpu top`:** a live view of GPU util/mem/power/temp, who holds the lock, who's queued, spend vs cap, and the idle-stop countdown.
- **Daily budget cap**, plus `ogpu fanout` to parallel Modal containers (H200, or cheaper H100 for same-arch sm_90a correctness runs).

## Setup
```bash
uv tool install --editable .          # puts `ogpu` on PATH for you and every agent
export RUNPOD_API_KEY=...             # add to ~/.zshrc
ogpu config                           # creates/prints ~/.ogpu/config.toml (budget, idle minutes, GPU, cloud order)
ogpu up                               # first boot installs sshd in the judge image (~2–5 min)
```

## Daily use
```bash
cd ~/Desktop/projects/open_kernel_challenge/muse_decode_layer
ogpu run --cpu -- python3 ../prefill_loop/harness/ptxas_check.py cand.py     # compile gate, parallel
ogpu run --gpu -- python3 pod/bench.py submission_best.py submission_sk.py --check --reps 20
ogpu top                       # live dashboard (q quit · s stop pod)
ogpu ledger --best             # what won
ogpu cost                      # today / 7d / all
ogpu fanout cmds_014.txt --backend modal --gpu H100                         # parallel correctness on Modal
ogpu down                      # stop now (or let the reaper do it)
```
## Agents
- `./scripts/install-skill.sh` installs the [`ogpu` skill](skills/ogpu/SKILL.md) for Claude Code, Codex and OpenCode (symlinks, so edits here update every agent).
- Paste the 4-line rule in [`agents/OGPU.md`](agents/OGPU.md) into each project's `AGENTS.md` / `CLAUDE.md`, so agents always reach for the skill.

## How it works
```
laptop (ogpu CLI, ledger.db) ──rsync/ssh (multiplexed)──▶ RunPod H200 pod (judge image)
                                                          /workspace/okc/<track>        latest push
                                                          /workspace/okc/.runs/<job>/   per-job snapshot (cwd)
                                                          /workspace/.ogpu/jobs/        job registry + logs
                                                          /tmp/h200.lock                GPU-exclusive flock
                                                          reaper.py                     idle → podStop
```
`ogpu run`: hash the sources (cache check) → budget check → ensure the pod → push + submit under a per-track lock → `jobrun.py` snapshots the dirs, takes the GPU lock, runs detached → the CLI polls (survives SSH drops, Ctrl-C detaches) → pull `out/` → parse metrics → ledger.

## Business plan
Parked in [`documents/feasibility-analysis.md`](documents/feasibility-analysis.md).
