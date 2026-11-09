# openGPU

**A GPU savings engine for coding agents.**

`ogpu` is a CLI that coding agents (Claude Code, Codex, OpenCode) use to run experiments on cloud GPUs. It runs jobs on RunPod, Modal, a host you already have, or your own provider. Agents share a machine without disturbing each other's measurements, every run is recorded in a queryable ledger, idle machines stop themselves, and a daily budget cap limits spend.

> **Status: alpha.** RunPod is the most exercised provider. Modal is newer; see [Known limitations](#known-limitations).

## Features

- **No paying for idle.** An on-machine reaper stops the machine after a configurable idle period, and disks and compiled caches persist for the next job. Modal jobs bill per second and have nothing to stop.
- **GPU fallback.** List GPU types and tiers in the order you prefer. ogpu moves down the list on capacity errors and replaces hosts that fail a real GPU allocation.
- **Multi-fidelity runs.** Send `--fidelity correctness` jobs to a cheaper pool so the expensive GPU only times candidates that already work. `ogpu sweep` runs a whole tournament this way.
- **Safe sharing.** GPU jobs queue behind an exclusive lock so timings never overlap. CPU jobs run in parallel at low priority, and each job runs from its own code snapshot.
- **Budget cap.** Beyond the daily limit, new jobs are refused. Modal jobs count toward spend while still running.
- **Opt-in result cache.** Identical runs return the earlier result for free. Infrastructure failures are never replayed.
- **Built for agents.** One-line JSON results, classified failures with error excerpts, async `submit`/`wait`, meaningful exit codes, and a shared ledger of what has been tried.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). RunPod and SSH hosts also need `rsync` and an SSH key.

```bash
git clone https://github.com/Sudhendra/openGPU.git
cd openGPU

# RunPod or SSH hosts only
uv tool install .

# With Modal support (installs the `modal` CLI too)
uv tool install '.[modal]' --with-executables-from modal
```

Then authenticate with your provider:

```bash
export RUNPOD_API_KEY=...    # RunPod
modal token new              # Modal
```

## Quickstart

```bash
ogpu config                               # creates ~/.ogpu/config.toml
ogpu gpus H100                            # GPU ids and prices for your provider

cd path/to/your/project
ogpu run --gpu -- python3 train.py        # sync, run under the GPU lock, stream the log
ogpu cost                                 # spend, GPU-busy time, spend per agent
```

`ogpu top` opens a live dashboard of machines, queues and spend.

## Usage

```bash
# Run
ogpu run --gpu -m "lr 3e-4" -- python3 train.py --steps 200      # exclusive GPU lock, queued
ogpu run --cpu -- python3 compile_check.py                       # no lock, runs immediately
ogpu run --gpu --fidelity correctness -- python3 test_kernel.py  # cheaper pool, if configured
ogpu run --gpu --with ../shared --timeout 900 --cache -- ...     # extra dirs, runtime cap, reuse results

# Queue now, collect later
ogpu submit --gpu -m "hypothesis" -- python3 bench.py a.py b.py
ogpu wait <id> [<id> ...]

# Tournament: correctness on the cheap pool, quick timing, then full timing for finalists
ogpu sweep --bench "python3 bench.py {files}" --best best.py cand_*.py

# One command per line, in parallel
ogpu fanout cmds.txt --provider modal --gpu L4

# Results
ogpu ledger [--track T] [--best | --failed] [--json]
ogpu show <id>
ogpu logs <id> [-f]
ogpu cancel <id>

# Machines and money
ogpu up [--pool P] [--retry MIN]
ogpu down [--terminate]
ogpu status
ogpu cost
ogpu gpus [FILTER] [--provider P]
```

### Result format

With `--json`, every run ends with one line:

```json
{"id": "1030-164825-d47177", "status": "done", "failure": null, "error": null, "exit_code": 0,
 "wall_s": 212.4, "cost_usd": 0.31, "metrics": {"score": 1.083, "loss": 0.412}, "note": "lr 3e-4",
 "pool": "timing", "log_path": "~/.ogpu/logs/1030-164825-d47177.log", "cached_from": null}
```

| Field | Meaning |
|---|---|
| `metrics` | Print `OGPU_METRIC name=value` from your script. `score` (higher is better) ranks `ogpu ledger --best`. |
| `failure` | `null`, or one of `compile_error`, `import_error`, `correctness_fail`, `oom`, `cuda_error`, `timeout`, `hang`, `runtime_error`, `launch_error`, `lost`, `cancelled`. |
| `error` | The relevant excerpt from the log. |
| `cached_from` | The earlier run this result came from, if the cache was used. |

**Exit codes:** the job's own code, `2` if the daily budget cap was reached, `3` if the job ran but correctness checks failed.

**Output files:** anything your command writes to `out/` is copied back to `<project>/output/ogpu/<id>/`.

## Providers

| Provider | Type | Auth | Notes |
|---|---|---|---|
| `runpod` | Machine | `RUNPOD_API_KEY` | Tiers `COMMUNITY` and `SECURE`. Filters hosts by `min_cuda`. |
| `modal` | Serverless | `modal token new` | One container per job. GPU ids include `H200`, `H100`, `A100-80GB`, `L40S`, `L4`, `T4`. |
| `ssh` | Machine | SSH key | A host you already have. Set `host`, `port`, `user`, `cost_per_hr` and `stop_cmd` under `[providers.ssh]`. |

Set `provider` at the top level and override it per pool. For example, timing on RunPod and correctness on Modal:

```toml
provider = "runpod"
gpu_type_ids = ["NVIDIA H100 80GB HBM3"]

[pools.correctness]
provider = "modal"
gpu_type_ids = ["L4"]
```

Modal bills for CPU and memory as well as the GPU. Containers default to 2 cores and 8 GiB; raise them only if a job needs more:

```toml
[providers.modal]
cpu = 2
memory_gb = 8
```

### Adding a provider

Subclass `ogpu.providers.Provider` and implement `get`, `find`, `create`, `resume`, `stop` and `terminate`. [`providers/ssh.py`](src/ogpu/providers/ssh.py) is the smallest example. Then point the config at it:

```toml
provider = "mycloud"

[providers.mycloud]
module = "my_pkg.mycloud:MyCloud"   # or an `ogpu.providers` entry point named "mycloud"
region = "us-east"                  # other keys are passed to the provider as `opts`
```

## Configuration

Settings live in `~/.ogpu/config.toml`. Secrets come from the environment, never this file. Unknown keys are rejected, so a typo can't silently fall back to a default.

| Key | Default | Description |
|---|---|---|
| `provider` | `runpod` | `runpod`, `modal`, `ssh`, or your own |
| `gpu_type_ids` | provider default | Provider-native GPU ids, tried in order |
| `cloud_types` | provider default | Provider tiers, tried in order |
| `image` | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` | Any image with `python3` |
| `idle_minutes` | `10` | Idle time before a machine stops itself |
| `budget_usd` | `25` | Daily spend cap |
| `cache` | `false` | Reuse identical runs (`--cache/--no-cache` per run) |
| `min_cuda` | `12.4` | Only use hosts whose driver supports this CUDA version |
| `profile` | `""` | Named preset, e.g. `okc` |
| `volume_gb` | `50` | Persistent volume at `/workspace` |
| `container_disk_gb` | `50` | Container disk |
| `auto_up` | `true` | Bring machines up on demand |
| `quiet_timing` | `true` | Keep CPU jobs off the timing cores |
| `[pools.correctness]` | none | Any of the keys above, for `--fidelity correctness` jobs |

**Environment variables:** `OGPU_AGENT` (label in the ledger), `OGPU_HOME` (default `~/.ogpu`), `OGPU_NO_NOTIFY`, `OGPU_MODAL_<GPU>_HR` (override a Modal GPU's hourly rate).

### Result cache

With `cache = true` or `--cache`, a run with the same synced files, command, image, provider and GPU types returns the earlier result, and identical in-flight jobs are joined rather than run twice. Passes and failures caused by the code itself are reused. Launch errors, timeouts, hangs, out-of-memory and CUDA errors always re-run.

### Profiles

`profile = "okc"` presets ogpu for the Open Kernel Challenge: the judge image on an H200 timing pool, an H100 correctness pool, `min_cuda = 13.0`, and parsing of the OKC bench output into metrics. Anything you set in the config still takes precedence.

## Using with coding agents

```bash
./scripts/install-skill.sh    # installs the `ogpu` skill for Claude Code, Codex and OpenCode
```

- **Skill:** [`skills/ogpu/SKILL.md`](skills/ogpu/SKILL.md) teaches agents the commands, the result format and an experiment loop.
- **Project rule:** paste [`agents/OGPU.md`](agents/OGPU.md) into your project's `AGENTS.md` or `CLAUDE.md`.
- **Guard hook (Claude Code):** use [`scripts/hooks/guard_gpu.py`](scripts/hooks/guard_gpu.py) as a `PreToolUse` hook to block raw `ssh`/`scp` to machines, `runpodctl` and `modal run`.

## How it works

```
 your machine                                  GPU machine
 ┌──────────────────────┐   rsync + ssh   ┌─────────────────────────────────────────────┐
 │ ogpu CLI             │ ──────────────▶ │ /workspace/src/<project>      latest sync   │
 │ ledger.db (SQLite)   │                 │ /workspace/.ogpu/runs/<job>/  job snapshot  │
 │ ogpu top             │ ◀── poll/logs ──│ jobrun.py   GPU lock, timeouts, watchdog    │
 │                      │                 │ reaper.py   idle → stop via provider        │
 └──────────────────────┘                 └─────────────────────────────────────────────┘
```

1. `ogpu run` checks the budget and, if enabled, the cache, then makes sure the pool's machine is up.
2. It syncs your project and submits the job. The runner snapshots the code, takes the GPU lock and runs detached, so SSH drops and Ctrl-C don't kill it.
3. The CLI polls the log, pulls back `out/`, parses metrics, classifies failures and writes the ledger.
4. The reaper stops the machine once it has been idle long enough.

On Modal there is no machine to manage: each job gets its own container, and results come back when it finishes.

## Known limitations

- A Modal job is tied to the local `modal run` process, so it stops if your laptop sleeps or that process is killed.
- `ogpu logs -f` can't follow a running Modal job, and `ogpu wait` has no deadline for Modal jobs.
- Modal timeouts are reported as `runtime_error` rather than `timeout`.
- Modal cost estimates don't include container start-up time, so they run slightly low.
- Projects with the same directory name collide on a shared machine.

## Security

- API keys are sent in `Authorization` headers, never URLs.
- Machines accept SSH with your public key only. Host keys are pinned per machine in `~/.ogpu/known_hosts/`. If a provider restarts a machine outside ogpu and its key changes, delete that machine's file there to reconnect.
- `~/.ogpu` and the SSH control-socket directory are private (`0700`).
- Jobs on a shared machine run as root. The lock, guard hook and skill are for cooperating agents; they are not a sandbox, so don't share a machine with code you don't trust.
- Custom providers are Python code that runs on your machine. Only load ones you trust.

## Development

```bash
uv sync
uv run pytest -q
```

| Path | Contents |
|---|---|
| `src/ogpu/` | CLI, job runner, ledger, dashboard |
| `src/ogpu/providers/` | RunPod, Modal, SSH and the plugin interface |
| `src/ogpu/remote/` | Code that runs on the GPU machine (standard library only) |
| `src/ogpu/okc.py` | The Open Kernel Challenge profile |
| `skills/`, `agents/`, `scripts/` | Agent integrations |
| `tests/` | Test suite, including the on-machine runner exercised locally |

## License

No license has been chosen yet.
