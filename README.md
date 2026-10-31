# openGPU

**Cloud GPUs for builders and their coding agents. Pay for work, not for idle.**

`ogpu` is a command-line GPU broker and shared job runtime. It brings up a GPU machine on demand, lets many coding agents (Claude Code, Codex, OpenCode, …) share it without stepping on each other's measurements, caches results so identical work is never paid for twice, records every run in a queryable ledger, and shuts the machine down the moment it goes idle.

It was built to hill-climb GPU kernels with a fleet of agents on a single H200. It works for any workload of the form *"sync my code, run a command on a GPU, give me the result."*

> **Status: alpha.** RunPod is the supported provider today, plus Modal for parallel fan-out. Multi-supplier support is the next milestone (see [Roadmap](#roadmap)).

---

## Why
GPU time is expensive. Most of the bill isn't GPU work:

- **Idle machines.** Boot time, agents thinking between runs, and machines left on overnight.
- **Retail rates.** Individuals pay on-demand prices on whatever tier they clicked first.
- **Duplicate work.** Agents re-measure candidates that were already measured.
- **Fragile access.** Hand-made pods, hard-coded SSH hosts, and timings corrupted when two agents use one GPU at once.

`ogpu` removes the waste and shows you exactly where every dollar went.

## Features
| | |
|---|---|
| **One-command machines** | `ogpu up` reuses, resumes or creates a GPU machine in your image, with fallback across GPU types and cloud tiers. Real SSH, no proxies. |
| **Never pay for idle** | An on-machine reaper stops the machine after *N* idle minutes. A guard process restarts it if it dies, and every `ogpu` call checks its heartbeat. Disks and compiled caches persist, so `ogpu up` resumes warm. |
| **Safe sharing** | GPU jobs queue behind an exclusive lock, so timings never overlap. CPU jobs (compile gates) run in parallel without disturbing timings: builds are capped to the container's real CPU quota, run at lowest priority, and stay off reserved cores. Every job runs from its own code snapshot. |
| **Never pay twice** | A content-hash cache (sources + command + image) returns identical runs instantly. Identical in-flight jobs are deduplicated across agents. |
| **Shared ledger** | Every run's metrics, failure type, hypothesis note, cost and agent in one SQLite ledger: `ogpu ledger`, `ogpu show`. |
| **Agent-native** | One-line JSON results, failure classification with error excerpts, async `submit`/`wait`, meaningful exit codes, and a drop-in agent skill and guard hook. |
| **Cost visibility** | `ogpu cost` and `ogpu top` show spend, GPU-busy % of paid time, idle dollars and spend per agent. Daily budget cap. |
| **Live dashboard** | `ogpu top`: GPU utilization, memory, power and temperature, lock holder, queue, recent results and the idle-stop countdown. |
| **Parallel fan-out** | `ogpu fanout` runs a list of commands across parallel serverless containers (Modal). |

```
 ogpu  ogpu-h200  RUNNING   runpod/secure  $5.29/hr  up 2h14m  today $9.63 / $60 cap

 GPU  ███████████████░░░░░  78%   MEM ████░░░░░░ 61.2/140 GB   PWR 512W/700  TEMP 63°C  SM 1980MHz
      ▁▂▂▃▄▄▅▅▆▆▇▇█▁▁▂▂▃▃▄▅▅▆▆▇▇██▁▂▂▃▃▄▄▅▅▆▇▇██▁▁▂▃▃▄▄▅▅▆▆▇██  util · last 90s

LOCK    claude(decode)     python3 pod/bench.py best.py cand_sk.py --check      running 3m12s
QUEUE   codex(bf16_proj)   python3 verifiable_loop/run.py cand_071.py           #1 waiting 0m41s
CPU     codex(prefill)     python3 harness/ptxas_check.py candidates/c12.py               0m22s

RECENT  ✔ claude(decode)   cand_sk.py 1.0831× · checks PASS   3m32s · $0.31 · 4m ago
        ✘ codex(bf16_proj) correctness_fail                   3m32s · $0.31 · 8m ago
TODAY   GPU busy 64% of paid time · idle $3.40   claude $4.10  codex $2.13

idle-stop  —  (busy)                                            q quit · s stop pod · r refresh
```

## Quickstart
**Requirements:** Python ≥ 3.11, `rsync`, an SSH key, and a [RunPod](https://runpod.io) API key.

```bash
git clone <repo-url> openGPU && cd openGPU
uv tool install --editable .            # puts `ogpu` on your PATH

export RUNPOD_API_KEY=...               # add to your shell profile (with `export`)
ogpu config                             # creates ~/.ogpu/config.toml: image, GPU, budget, idle minutes
ogpu up                                 # bring up a machine (first boot installs sshd: ~2–5 min)

cd path/to/your/project
ogpu run --gpu -- python3 bench.py      # sync, run under the GPU lock, stream the log, record the result
ogpu top                                # live dashboard (in another terminal)
```

## Usage
```bash
# run
ogpu run --gpu  -m "tile 128x64" -- python3 bench.py cand.py --check   # exclusive GPU lock, queued
ogpu run --cpu  -- python3 ptxas_check.py cand.py                      # no lock, runs immediately
ogpu run --gpu  --with ../shared --timeout 900 -- ...                  # sync extra dirs, cap runtime

# async
ogpu submit --gpu -m "hypothesis" -- python3 bench.py a.py b.py         # → {"id": "...", "status": "queued"}
ogpu wait <id> [<id> ...]                                              # one JSON result per job

# results
ogpu ledger [--track T] [--best | --failed] [--json]
ogpu show <id>                                                         # command, note, metrics, error, log tail
ogpu logs <id> [-f]
ogpu cancel <id>

# machines & money
ogpu up [--retry MINUTES]   ogpu down [--terminate]   ogpu status
ogpu cost                   ogpu top                  ogpu gpus [FILTER]
ogpu fanout cmds.txt --backend modal --gpu H100
```

### Results
With `--json`, every run ends with one line:
```json
{"id": "1030-164825-d47177", "status": "done", "failure": null, "error": null, "exit_code": 0,
 "wall_s": 212.4, "cost_usd": 0.31, "metrics": {"score": {"cand.py": 1.0831}, "checks": "PASS"},
 "note": "tile 128x64", "log_path": "~/.ogpu/logs/1030-164825-d47177.log", "cached_from": null}
```
- **`failure`:** `null`, or one of `compile_error`, `import_error`, `correctness_fail`, `oom`, `cuda_error`, `timeout`, `runtime_error`, `lost`, `cancelled`. `error` carries the relevant excerpt.
- **`metrics`:** parsed from your output. Print `OGPU_METRIC name=value` from any script to add your own.
- **Exit codes:** the job's own code, `3` = ran but correctness checks failed, `2` = daily budget cap reached.
- **Files** your command writes to `out/` are copied back to `<project>/output/ogpu/<id>/`.

## Using it with coding agents
```bash
./scripts/install-skill.sh      # installs the `ogpu` skill for Claude Code, Codex and OpenCode
```
- **Skill:** [`skills/ogpu/SKILL.md`](skills/ogpu/SKILL.md) teaches agents the commands, the result format and a hill-climbing loop.
- **Project rule:** paste the four lines in [`agents/OGPU.md`](agents/OGPU.md) into your project's `AGENTS.md` / `CLAUDE.md`, so agents always use `ogpu`.
- **Guard hook (Claude Code):** [`scripts/hooks/guard_gpu.py`](scripts/hooks/guard_gpu.py), registered as a `PreToolUse` hook, blocks raw `ssh`/`scp` to machines, `runpodctl` and `modal run`, and points the agent to `ogpu`.
- **Agent labels:** detected automatically from the process tree. Set `OGPU_AGENT=<name>` for an explicit label in `ogpu top` and the ledger.

## Configuration
`~/.ogpu/config.toml`. Secrets come from the environment, never this file.

| Key | Default | |
|---|---|---|
| `image` | `lmsysorg/sglang:dev-muse-spark-open` | Container image for the machine |
| `gpu_type_ids` | `["NVIDIA H200"]` | Tried in order (`ogpu gpus` lists ids) |
| `cloud_types` | `["COMMUNITY", "SECURE"]` | Tried in order. Community is usually cheaper. |
| `idle_minutes` | `20` | Reaper stops the machine after this much idle time |
| `budget_usd` | `60` | Daily cap. New GPU jobs are refused beyond it. |
| `volume_gb` / `container_disk_gb` | `100` / `60` | Persistent volume at `/workspace` / container disk |
| `min_vcpu` / `min_memory_gb` | `8` / `32` | Host requirements |
| `auto_up` | `true` | `ogpu run` brings the machine up if needed |
| `quiet_timing` | `true` | Size build parallelism to the container's real CPU quota; CPU jobs run niced on non-reserved cores so they can't disturb GPU timings |

Environment: `RUNPOD_API_KEY` (required), `OGPU_AGENT`, `OGPU_HOME` (default `~/.ogpu`), `OGPU_NO_NOTIFY`.

## How it works
```
 laptop                                          GPU machine (your image)
 ┌───────────────────────┐   rsync + ssh   ┌────────────────────────────────────────────┐
 │ ogpu CLI              │ ──────────────▶ │ /workspace/okc/<project>     latest push    │
 │  · content-hash cache │  (multiplexed)  │ /workspace/okc/.runs/<job>/  job snapshot   │
 │  · ledger.db (SQLite) │ ◀── poll/logs ──│ jobrun.py   queue + GPU flock + timeouts    │
 │  · ogpu top           │                 │ reaper.py   idle → stop via provider API    │
 └───────────────────────┘                 └────────────────────────────────────────────┘
```
1. `ogpu run` hashes the sources (a cache hit returns immediately), checks the budget, and ensures the machine is up.
2. It pushes your project and submits the job under a per-project lock. The runner snapshots the code, takes the GPU lock and runs the job detached, so SSH drops and Ctrl-C don't kill it.
3. The CLI polls the log, pulls back `out/`, parses metrics, classifies failures and writes the ledger.
4. The reaper watches for idle time and stops the machine. Persistent disks keep code and caches for the next `ogpu up`.
5. Every new or resumed machine must pass a GPU check (a real allocation through the image's own framework) before agents get it. Hosts whose driver is older than `min_cuda` are filtered out at creation, and a machine that still fails is replaced automatically.

Only Python's standard library runs on the machine, so any image with `python3` works.

## Cost accounting
- **Paid time** comes from the provider's reported uptime and price.
- **Busy time** is the union of job intervals. *Idle $* = paid time with no job × rate.
- **Per-job cost** = wall time × rate. Cache hits cost $0.
- `ogpu cost` reports today, the last 7 days and all time, plus GPU-busy % and spend per agent. Desktop notifications (macOS) fire on a new best result and at 80% of the daily budget.

## Security notes
- Your API key stays on your machine. The on-machine reaper uses the provider's pod-scoped credentials to stop itself.
- Machines accept SSH with your public key only (root, key auth). Host keys aren't pinned because machines are ephemeral.
- The guard hook and skill are guardrails for cooperative agents, not a sandbox.

## Roadmap
- **Release prep:** a provider interface, per-project `.ogpu.toml` profiles, pluggable metric parsers, `ogpu init`, CI, packaging.
- **Multi-supplier:** Vast.ai, Lambda, Shadeform and bring-your-own SSH hosts; `ogpu offers` price comparison; cheapest-capable placement.
- **Cost engine:** spot tiers with auto-retry, migration to cheaper capacity, a cheap same-architecture tier for correctness runs, `ogpu savings`.
- **Teams:** hosted shared queue and ledger, team budgets, dashboards.

Details: [`documents/platform-plan.md`](documents/platform-plan.md) · [`documents/roadmap.md`](documents/roadmap.md)

## Development
```bash
uv sync
uv run pytest -q          # unit tests, including the on-machine runner exercised locally
```
Layout: `src/ogpu/` (CLI, runner, ledger, TUI), `src/ogpu/providers/` (RunPod, Modal), `src/ogpu/remote/` (code that runs on the machine), `skills/`, `scripts/`, `tests/`.

## License
To be decided before the first public release.
