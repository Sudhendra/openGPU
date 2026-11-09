# openGPU

**A GPU savings engine for coding agents. Pay for work, not for idle.**

`ogpu` is a command-line GPU broker and shared job runtime. Agents (Claude Code, Codex, OpenCode, …) iterating on experiments call `ogpu run -- <cmd>`; ogpu puts the job on the cheapest capable GPU across **RunPod, Vast.ai and Modal** (or your own provider), lets many agents share one machine without corrupting each other's measurements, records every run in a queryable ledger, and stops the machine the moment it goes idle.

> **Status: alpha.** RunPod is the most battle-tested provider; Vast.ai and Modal are new.

---

## Where the savings come from
| | |
|---|---|
| **Never pay for idle** | An on-machine reaper stops the machine after *N* idle minutes (a guard restarts the reaper if it dies; every `ogpu` call checks its heartbeat). Disks and compiled caches persist, so the next job resumes warm. Serverless Modal jobs bill per second and never idle at all. |
| **Cheapest capable GPU** | Pick the provider per pool, list GPU types and tiers in your price order (`ogpu gpus` shows prices), and ogpu falls back down the list on capacity errors. Vast offers are taken cheapest first. A broken or too-old host is detected by a real GPU allocation and replaced before any agent gets it. |
| **Multi-fidelity** | `--fidelity correctness` jobs go to a cheaper pool (e.g. Modal L4 or a Vast H100) so the expensive target GPU only times candidates that are already correct. `ogpu sweep` runs a whole tournament this way. |
| **Batch & share** | GPU jobs queue behind an exclusive lock so timings never overlap; CPU jobs (compile gates) run in parallel at low priority, sized to the container's real CPU quota. Every job runs from its own code snapshot. |
| **Opt-in result cache** | With `cache = true` (or `--cache`), an identical run (same sources + command + image) returns the earlier result for $0, and identical in-flight jobs from different agents are deduplicated. |
| **Cost visibility** | `ogpu cost` / `ogpu top`: spend, GPU-busy % of paid time, idle dollars, spend per agent. A daily budget cap refuses new GPU work beyond it. |
| **Agent-native** | One-line JSON results, failure classification with error excerpts, async `submit`/`wait`, meaningful exit codes, a drop-in agent skill and guard hook. |

## Quickstart
**Requirements:** Python ≥ 3.11, `rsync`, an SSH key, and an account with at least one provider.

```bash
git clone <repo-url> openGPU && cd openGPU
uv tool install --editable .            # puts `ogpu` on your PATH  (add `--with modal` for Modal)

export RUNPOD_API_KEY=...               # and/or VAST_API_KEY=...; Modal: `modal token new`
ogpu config                             # creates ~/.ogpu/config.toml: provider, GPU, budget, idle minutes
ogpu gpus H100                          # GPU ids + prices for the configured provider (--provider vast|modal)

cd path/to/your/project
ogpu run --gpu -- python3 train.py      # sync, run under the GPU lock, stream the log, record the result
ogpu top                                # live dashboard (in another terminal)
```

## Usage
```bash
# run
ogpu run --gpu -m "lr 3e-4" -- python3 train.py --steps 200      # exclusive GPU lock, queued
ogpu run --cpu -- python3 compile_check.py                       # no lock, runs immediately
ogpu run --gpu --fidelity correctness -- python3 test_kernel.py  # cheaper pool, if configured
ogpu run --gpu --with ../shared --timeout 900 --cache -- ...     # sync extra dirs, cap runtime, reuse results

# async
ogpu submit --gpu -m "hypothesis" -- python3 bench.py a.py b.py   # → {"id": "...", "status": "queued"}
ogpu wait <id> [<id> ...]                                        # one JSON result per job

# tournaments: correctness on the cheap pool → quick timing → full timing for finalists
ogpu sweep --bench "python3 bench.py {files}" --best best.py cand_*.py

# results
ogpu ledger [--track T] [--best | --failed] [--json]
ogpu show <id>   ogpu logs <id> [-f]   ogpu cancel <id>

# machines & money
ogpu up [--pool P] [--retry MIN]   ogpu down [--terminate]   ogpu status
ogpu cost   ogpu top   ogpu gpus [FILTER] [--provider P]
ogpu fanout cmds.txt --provider modal --gpu L4                   # one container per line, in parallel
```

### Results
With `--json`, every run ends with one line:
```json
{"id": "1030-164825-d47177", "status": "done", "failure": null, "error": null, "exit_code": 0,
 "wall_s": 212.4, "cost_usd": 0.31, "metrics": {"score": 1.083, "loss": 0.412}, "note": "lr 3e-4",
 "pool": "timing", "log_path": "~/.ogpu/logs/1030-164825-d47177.log", "cached_from": null}
```
- **`metrics`:** print `OGPU_METRIC name=value` from your script. `score` (higher = better) ranks runs in `ogpu ledger --best`.
- **`failure`:** `null`, or one of `compile_error`, `import_error`, `correctness_fail`, `oom`, `cuda_error`, `timeout`, `hang`, `runtime_error`, `lost`, `cancelled`. `error` carries the relevant excerpt.
- **Exit codes:** the job's own code, `3` = ran but correctness checks failed, `2` = daily budget cap reached.
- **Files** your command writes to `out/` are copied back to `<project>/output/ogpu/<id>/`.

## Providers
| Provider | Kind | Key | Notes |
|---|---|---|---|
| `runpod` | machine | `RUNPOD_API_KEY` | Tiers `COMMUNITY`, `SECURE`. Host CUDA filter from `min_cuda`. |
| `vast` | machine | `VAST_API_KEY` | Cheapest matching offer; tiers `verified`, `any`. Stopped instances still bill storage. |
| `modal` | serverless | `modal token new` | One container per job, per-second billing, nothing to stop. GPU ids: `H200`, `H100`, `A100-80GB`, `L40S`, `L4`, … |
| `ssh` | machine | — | A host you already have: `[providers.ssh] host/port/user/cost_per_hr/stop_cmd`. |

Set `provider` at the top level and override it per pool, e.g. timing on RunPod H100s, correctness on Modal L4s:
```toml
provider = "runpod"
gpu_type_ids = ["NVIDIA H100 80GB HBM3"]

[pools.correctness]
provider = "modal"
gpu_type_ids = ["L4"]
```

**Your own provider:** subclass `ogpu.providers.Provider` (get/find/create/resume/stop/terminate; see [`providers/ssh.py`](src/ogpu/providers/ssh.py) for the smallest example) and point the config at it:
```toml
provider = "mycloud"
[providers.mycloud]
module = "my_pkg.mycloud:MyCloud"   # or an `ogpu.providers` entry point named "mycloud"
region = "us-east"                  # other keys are passed to the provider as `opts`
```

## Configuration
`~/.ogpu/config.toml`. Secrets come from the environment, never this file.

| Key | Default | |
|---|---|---|
| `provider` | `runpod` | `runpod`, `vast`, `modal`, `ssh`, or your own |
| `gpu_type_ids` | provider default (H100s) | Provider-native ids, tried in order (`ogpu gpus`) |
| `cloud_types` | provider default | Provider tiers, tried in order |
| `image` | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` | Any image with `python3` |
| `idle_minutes` | `10` | The machine stops itself after this much idle time |
| `budget_usd` | `25` | Daily cap. New GPU jobs are refused beyond it. |
| `cache` | `false` | Reuse identical runs (`--cache/--no-cache` per run) |
| `min_cuda` | `12.4` | Only hosts whose driver supports this CUDA |
| `profile` | `""` | `"okc"`: Open Kernel Challenge preset (below) |
| `volume_gb` / `container_disk_gb` | `50` / `50` | Persistent volume at `/workspace` / container disk |
| `auto_up` / `quiet_timing` | `true` / `true` | Bring machines up on demand / keep CPU jobs off the timing cores |
| `[pools.correctness]` | none | Any of the keys above, for `--fidelity correctness` jobs |

Environment: provider keys, `OGPU_AGENT` (label in the ledger), `OGPU_HOME` (default `~/.ogpu`), `OGPU_NO_NOTIFY`.

### Profiles
`profile = "okc"` presets ogpu for the Open Kernel Challenge: the judge image on an H200 timing pool, an H100 correctness pool, `min_cuda = 13.0`, and the OKC bench output (`SCORE`, `CHECK`, `CHECKS`, `SIDE` lines) parsed into metrics. Anything you set in the file still wins.

## Using it with coding agents
```bash
./scripts/install-skill.sh      # installs the `ogpu` skill for Claude Code, Codex and OpenCode
```
- **Skill:** [`skills/ogpu/SKILL.md`](skills/ogpu/SKILL.md) teaches agents the commands, the result format and an experiment loop.
- **Project rule:** paste [`agents/OGPU.md`](agents/OGPU.md) into your project's `AGENTS.md` / `CLAUDE.md`.
- **Guard hook (Claude Code):** [`scripts/hooks/guard_gpu.py`](scripts/hooks/guard_gpu.py) as a `PreToolUse` hook blocks raw `ssh`/`scp` to machines, `runpodctl`, `vastai` and `modal run`, and points the agent to `ogpu`.

## How it works
```
 laptop                                          GPU machine (your image)
 ┌───────────────────────┐   rsync + ssh   ┌──────────────────────────────────────────────┐
 │ ogpu CLI              │ ──────────────▶ │ /workspace/src/<project>       latest push    │
 │  · ledger.db (SQLite) │  (multiplexed,  │ /workspace/.ogpu/runs/<job>/   job snapshot   │
 │  · opt-in cache       │   pinned keys)  │ jobrun.py   queue + GPU flock + timeouts      │
 │  · ogpu top           │ ◀── poll/logs ──│ reaper.py   idle → stop via the provider     │
 └───────────────────────┘                 └──────────────────────────────────────────────┘
```
1. `ogpu run` checks the budget (and the cache, if enabled) and ensures the pool's machine is up.
2. It pushes your project and submits the job. The runner snapshots the code, takes the GPU lock and runs the job detached, so SSH drops and Ctrl-C don't kill it. A watchdog kills hung jobs.
3. The CLI polls the log, pulls back `out/`, parses metrics, classifies failures and writes the ledger.
4. The reaper stops the machine when idle. Serverless pools (Modal) skip all of this: one container per job.

Only Python's standard library runs on the machine, so any image with `python3` works.

## Security
- API keys stay on your machine and are sent in `Authorization` headers, never URLs. The on-machine reaper stops its machine with the provider's machine-scoped credentials.
- Machines accept SSH with your public key only (no passwords). Host keys are pinned per machine on first use (`~/.ogpu/known_hosts/`), persisted on the machine's volume so they survive stop/resume, and only re-learned when ogpu itself (re)starts the machine.
- `~/.ogpu` and the SSH control-socket directory are private (0700); a pre-existing socket dir owned by someone else is refused.
- The guard hook and skill are guardrails for cooperative agents, not a sandbox. Custom providers are Python code you choose to load.

## Development
```bash
uv sync
uv run pytest -q          # unit tests, including the on-machine runner exercised locally
```
Layout: `src/ogpu/` (CLI, runner, ledger, TUI), `src/ogpu/providers/` (RunPod, Vast, Modal, SSH + the plugin interface), `src/ogpu/remote/` (code that runs on the machine), `src/ogpu/okc.py` (the OKC profile), `skills/`, `scripts/`, `tests/`. Plans: [`documents/`](documents/).

## License
To be decided before the first public release.
