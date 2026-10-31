---
name: ogpu
description: Run GPU work (H200 benchmarks, kernel timing, correctness checks, nvcc/ptxas compile gates, profiling) on the shared remote pod through the `ogpu` CLI. Use whenever a task needs a GPU, needs to time or validate a CUDA/Triton kernel, or would otherwise SSH to RunPod, use runpodctl, or launch Modal by hand, including Open Kernel Challenge (okc) work before spending `okc check`/`okc submit` quota.
---

# ogpu: the shared H200 (+ a cheap correctness pool)

Every GPU command goes through `ogpu`. It syncs your working dir to a shared H200 pod (timing) or a cheaper H100 pod (correctness), queues GPU jobs behind an exclusive lock so timings from different agents never overlap, caches identical runs, records every result in a shared ledger, and stops the pod when idle. Other agents are using the same GPU at the same time as you.

## Hard rules
- **Never** `ssh`/`scp` to a pod, run `runpodctl`, create pods, or call `modal run` directly. Use `ogpu`.
- **Never** run `ogpu down`, `ogpu down --terminate`, or `ogpu cancel` on a job you didn't start. Other agents' work is on that pod. The idle reaper stops it for you.
- **Never** run `ogpu top`. It's an interactive dashboard for the human.
- Always pass `--json` and read the one-line result. Don't scrape the streamed log for numbers.
- Don't spend `okc check` / `okc submit` quota until an `ogpu run --gpu` result says the candidate passes and beats the current best.

## Commands
Run from the track directory (or pass `--track DIR`). Everything after `--` is the command, executed in a snapshot of that dir on the pod.

```bash
# compile / ptxas / static gates: no GPU lock, runs immediately and in parallel
ogpu run --cpu --json -- python3 harness/ptxas_check.py candidates/c12.py

# STEP A, correctness first: cheaper H100 pool (same sm_90a ISA), few reps. Its timings mean nothing; only read checks.
ogpu run --gpu --fidelity correctness --json -m "split-K=4 for d8 shapes" -- python3 pod/bench.py submission_new.py --check --reps 3

# STEP B, timing, only for candidates that passed STEP A: the H200 (exclusive lock; you may queue, just wait)
ogpu run --gpu --json -m "split-K=4 for d8 shapes" -- python3 pod/bench.py submission_best.py submission_new.py --reps 20

# command imports a sibling dir? sync it too (repeatable). Cap runaway jobs with --timeout (seconds).
ogpu run --gpu --json --with ../tools --timeout 900 -- python3 pod/bench.py submission_new.py --check

# async: queue now, keep working, collect later (one JSON line per job as each finishes)
ogpu submit --gpu -m "hypothesis A" -- python3 pod/bench.py submission_best.py cand_a.py cand_b.py --check   # -> {"id": ...}
ogpu wait <id> [<id> ...]

# shared memory: check BEFORE proposing a new candidate
ogpu ledger --track <track_dir_name> --best --json      # what's winning
ogpu ledger --track <track_dir_name> --failed --json    # what failed and why (failure type + note): don't repeat it
ogpu show <id>                                          # one run: cmd, note, metrics, error excerpt, log tail

ogpu logs <id>        # full log of a finished job
ogpu cancel <id>      # only for your own job
ogpu status --json    # pod state + today's spend vs the daily cap
```

**Split correctness from timing.** The H200 is the bottleneck shared by every agent. Don't spend it on candidates that turn out to be wrong: verify on `--fidelity correctness` first (STEP A), then time only the survivors on the H200 (STEP B). If the correctness pool is cold, ogpu runs that job on the H200 and warms the pool in the background (first boot takes ~15 min), so you never wait. Just keep using `--fidelity correctness`.

**Batch candidates.** Every benchmark invocation pays the `prepare` cost (model shapes, weights, inputs, and sometimes minutes of compiling). Compare several candidates in ONE command (`bench.py best.py a.py b.py c.py`) rather than one job each. `metrics.score` reports every file separately.

**Always pass `-m/--note`** with the hypothesis. It's stored in the shared ledger so other agents (and you, after a context reset) can see what was tried.

## The result line (`--json`)
```json
{"id": "1030-164825-d47177", "status": "done", "failure": null, "error": null, "exit_code": 0, "wall_s": 212.4,
 "cost_usd": 0.31, "metrics": {"score": {"submission_new.py": 1.0831}, "checks": "PASS", "check_pass": 12,
 "check_fail": 0, "sides": {"s1": {"ratio_vs_s0": 1.034, "ci95": [1.021, 1.047]}}},
 "note": "split-K=4 for d8 shapes", "pool": "timing", "log_path": "~/.ogpu/logs/1030-164825-d47177.log",
 "cached_from": null}
```
- `failure`: `null` on a clean pass. Otherwise one of `compile_error`, `import_error`, `correctness_fail`, `oom`, `cuda_error`, `timeout`, `runtime_error`, `lost`, `cancelled`. `error` holds the relevant excerpt (last traceback or failing CHECK lines), so read it instead of the whole log.
- `status`: `done` (exit 0), `failed`, `cancelled`, `lost` (the pod restarted mid-job; ogpu already retried once).
- `metrics.score`: geomean speedup vs SGLang per file, from `SCORE <file>: <x>` lines. Higher is better.
- `metrics.checks`: `PASS`/`FAIL`. **A FAIL means the candidate is invalid, whatever its speed.**
- `pool`: where it ran (`timing` = H200, `correctness` = H100). Never compare or keep timings from `correctness`.
- `metrics.check_margin`: worst error ÷ bound across all checks (1.0 = at the judge's bound). On the correctness pool, a pass with margin > 0.8 returns `"confirm_on_timing": true` and `advice`: re-run the check on the H200 before trusting it (H100 error values shift by up to ~17% of the bound).
- `metrics.sides`: paired-bench ratio vs the reference side (`s0`) with a 95% CI. Treat it as a real win only if the CI's lower bound is > 1.0.
- Need a number the parser doesn't know? Print `OGPU_METRIC name=value` from your script and it appears in `metrics`.
- `cached_from` set means this exact code + command already ran, so the result is free. If another agent is already running the identical job, ogpu waits on that one instead of running it twice. Pass `--force` only to deliberately re-measure noise.
- Files your command writes under `out/` come back to `<track>/output/ogpu/<id>/`.
- **Exit codes:** the job's own code, `3` = the benchmark ran but correctness FAILED, `2` = daily budget cap hit (stop and tell the human, don't retry).

## Hill-climbing loop
1. `ogpu ledger --track T --best --json` and `--failed --json`: know the current best and the dead ends.
2. Write the hypothesis down where your track keeps experiments (e.g. `EXPERIMENTS.md`).
3. `ogpu run --cpu --json -- <compile/ptxas gate>`. Fix compile errors and register/spill problems here, for free.
4. `ogpu run --gpu --fidelity correctness --json -m "<hypothesis>" -- <bench with --check, few reps>`. Batch variants into one invocation. Drop every candidate that fails. If `confirm_on_timing` is true, include `--check` in step 5 too.
5. `ogpu run --gpu --json -m "<hypothesis>" -- <bench, current best + survivors, full reps>` on the H200.
6. Keep it only if checks passed and the win clears noise (CI lower bound > 1.0, or ≥ ~1.5% on `score`). Otherwise discard it and log why.
7. Only then `okc check`, then `okc submit`.

## When things go wrong
- **Queued for a long time:** another agent holds the lock. Normal, so wait. Don't cancel other jobs.
- **First run in a track is slow:** extensions compile once, then they're cached on the pod disk.
- **"pod not reachable" / slow first call:** `ogpu` resumes or creates the pod automatically (2–5 min). Let it finish.
- **Interrupted (Ctrl-C, tool timeout):** the job keeps running on the pod. Use `ogpu wait <id>`. Don't resubmit duplicates.
- **Your tool call has a time limit shorter than the benchmark:** use `ogpu submit`, then `ogpu wait <id>` (re-run `wait` if it times out too; the job keeps going).
- **`failure: lost` after the automatic retry:** the pod is unstable. Tell the human.
- **Budget cap (exit 2) or repeated capacity errors:** stop and report to the human.
