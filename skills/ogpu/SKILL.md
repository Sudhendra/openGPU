---
name: ogpu
description: Run GPU work (training/eval runs, benchmarks, kernel timing, correctness checks, nvcc/ptxas compile gates, profiling) on shared cloud GPUs through the `ogpu` CLI. Use whenever a task needs a GPU, or would otherwise SSH to a GPU machine or use runpodctl, vastai or `modal run` by hand.
---

# ogpu: shared, cost-capped cloud GPUs

Every GPU command goes through `ogpu`. It syncs your working dir to a GPU machine (RunPod, Vast.ai, Modal or the user's own provider), queues GPU jobs behind an exclusive lock so different agents never overlap, records every result in a shared ledger, and stops machines when idle. Other agents may be using the same GPU at the same time as you.

## Hard rules
- **Never** `ssh`/`scp` to a GPU machine, run `runpodctl`/`vastai`, create machines, or call `modal run` directly. Use `ogpu`.
- **Never** run `ogpu down`, `ogpu down --terminate`, or `ogpu cancel` on a job you didn't start. The idle reaper stops machines for you.
- **Never** run `ogpu top`. It's an interactive dashboard for the human.
- Always pass `--json` and read the one-line result. Don't scrape the streamed log for numbers.

## Commands
Run from the project directory (or pass `--track DIR`). Everything after `--` is the command, executed in a snapshot of that dir on the machine.

```bash
# compile / static gates: no GPU lock, runs immediately and in parallel
ogpu run --cpu --json -- python3 compile_check.py cand.py

# GPU work (exclusive lock; you may queue, just wait). Always say what you're testing with -m.
ogpu run --gpu --json -m "lr 3e-4, warmup 200" -- python3 train.py --steps 500

# numerics / correctness only: the cheaper pool if one is configured. Never trust its timings.
ogpu run --gpu --fidelity correctness --json -m "fused softmax" -- python3 test_kernel.py --check

# sync a sibling dir too (repeatable); cap runaway jobs (seconds)
ogpu run --gpu --json --with ../common --timeout 900 -- python3 eval.py

# async: queue now, keep working, collect later (one JSON line per job as each finishes)
ogpu submit --gpu -m "hypothesis A" -- python3 bench.py a.py b.py   # -> {"id": ...}
ogpu wait <id> [<id> ...]

# shared memory: check BEFORE proposing a new experiment
ogpu ledger --track <dir_name> --best --json      # what's winning
ogpu ledger --track <dir_name> --failed --json    # what failed and why: don't repeat it
ogpu show <id>                                    # one run: cmd, note, metrics, error excerpt, log tail
ogpu logs <id>  ·  ogpu cancel <id> (yours only)  ·  ogpu status --json (machines + today's spend vs the cap)
```

**Save money.** Batch variants into one command (`bench.py best.py a.py b.py`) instead of one job each: every job pays its setup cost. Check correctness on `--fidelity correctness` first and spend the target GPU only on survivors. Use `--cache` to re-ask for a result you already have instead of re-running it.

**Always pass `-m/--note`** with the hypothesis. It's stored in the shared ledger so other agents (and you, after a context reset) can see what was tried.

## Sweeps (tournaments)
```bash
ogpu sweep --json --bench "python3 bench.py {files}" --best best.py cand_a.py cand_b.py 'cand_c.py@BLOCK=128' -m "<hypothesis>"
```
Stages: optional `--compile "<cmd> {file}"` on CPU → correctness of all candidates in ONE job on the cheap pool → quick timing of best + survivors → full timing of best + finalists. Wrong candidates are never timed. `recommend` is set **only** if a candidate beats `--best` by more than `--noise` (default 0.015). Bench contract: candidates are arguments; per candidate print `CHECK <case> <name> <tag>: PASS|FAIL` and `SCORE <name>: <x>` (higher = better).

## The result line (`--json`)
```json
{"id": "1030-164825-d47177", "status": "done", "failure": null, "error": null, "exit_code": 0, "wall_s": 212.4,
 "cost_usd": 0.31, "metrics": {"score": 1.083, "loss": 0.412}, "note": "lr 3e-4", "pool": "timing",
 "log_path": "~/.ogpu/logs/1030-164825-d47177.log", "cached_from": null}
```
- `failure`: `null` on a clean pass, else `compile_error`, `import_error`, `correctness_fail`, `oom`, `cuda_error`, `timeout`, `hang`, `runtime_error`, `lost`, `cancelled`. `error` holds the relevant excerpt, so read it instead of the whole log.
- `status`: `done` (exit 0), `failed`, `cancelled`, `lost` (the machine restarted mid-job; ogpu already retried once).
- `metrics`: print `OGPU_METRIC name=value` from your script to add numbers. `score` (higher = better) ranks `ledger --best`.
- `pool`: where it ran. Never compare timings across pools.
- `cached_from` set means this exact code + command already ran (only with `--cache`), so the result is free.
- Files written under `out/` come back to `<project>/output/ogpu/<id>/`.
- **Exit codes:** the job's own code, `3` = ran but correctness FAILED, `2` = daily budget cap hit (stop and tell the human, don't retry).

## Experiment loop
1. `ogpu ledger --track T --best --json` and `--failed --json`: know the current best and the dead ends.
2. Write the hypothesis down where the project keeps experiments.
3. `ogpu run --cpu --json -- <compile/static gate>`: fix build errors for free.
4. `ogpu run --gpu --fidelity correctness --json -m "<hypothesis>" -- <check, few reps>`: drop everything that fails.
5. `ogpu run --gpu --json -m "<hypothesis>" -- <current best + survivors, full measurement>` on the target GPU.
6. Keep it only if checks passed and the win clears noise. Otherwise discard it and log why.

## When things go wrong
- **Queued for a long time:** another agent holds the lock. Normal, so wait. Don't cancel other jobs.
- **"machine not reachable" / slow first call:** ogpu resumes or creates the machine automatically (2–5 min). Let it finish.
- **Interrupted (Ctrl-C, tool timeout):** the job keeps running. Use `ogpu wait <id>`; don't resubmit duplicates.
- **Your tool call has a time limit shorter than the job:** `ogpu submit`, then `ogpu wait <id>` (re-run `wait` if it times out too).
- **`failure: lost` after the automatic retry, budget cap (exit 2), or repeated capacity errors:** stop and tell the human.

## Open Kernel Challenge (`profile = "okc"`)
When the user's config has `profile = "okc"`: timing runs on the H200 (judge GPU), `--fidelity correctness` on an H100 pool, and `metrics` include `score` per file, `checks`, `check_margin` and `sides` from the OKC bench. A correctness-pool pass with `check_margin > 0.8` returns `"confirm_on_timing": true`: include `--check` in the H200 run too. Don't spend `okc check` / `okc submit` quota until an H200 result shows `checks: PASS` and beats the current best (`ogpu sweep` defaults to `--bench "python3 pod/bench.py"` under this profile; check only its `recommend`ed candidate).
