---
name: ogpu
description: Run GPU work (H200 benchmarks, kernel timing, correctness checks, nvcc/ptxas compile gates, profiling) on the shared remote pod through the `ogpu` CLI. Use whenever a task needs a GPU, needs to time or validate a CUDA/Triton kernel, or would otherwise SSH to RunPod, use runpodctl, or launch Modal by hand, including Open Kernel Challenge (okc) work before spending `okc check`/`okc submit` quota.
---

# ogpu: the shared H200

Every GPU command goes through `ogpu`. It syncs your working dir to one shared H200 pod, queues GPU jobs behind an exclusive lock so timings from different agents never overlap, caches identical runs, records every result in a shared ledger, and stops the pod when idle. Other agents are using the same GPU at the same time as you.

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

# timing + correctness: exclusive H200 lock (you may queue behind other agents; just wait)
ogpu run --gpu --json -- python3 pod/bench.py submission_best.py submission_new.py --check --reps 20

# command imports a sibling dir? sync it too (repeatable)
ogpu run --gpu --json --with ../tools -- python3 pod/bench.py submission_new.py --check

# what has already been tried / what's winning: check BEFORE proposing a new candidate
ogpu ledger --track <track_dir_name> --best --json
ogpu ledger --track <track_dir_name> -n 30 --json

ogpu logs <id>        # full log of a finished job
ogpu logs <id> -f     # re-attach to a still-running job (e.g. after an interruption)
ogpu cancel <id>      # only for your own job
ogpu status --json    # pod state + today's spend vs the daily cap
```

## The result line (`--json`)
```json
{"id": "1030-164825-d47177", "status": "done", "exit_code": 0, "wall_s": 212.4, "cost_usd": 0.31,
 "metrics": {"score": {"submission_new.py": 1.0831}, "checks": "PASS", "check_pass": 12, "check_fail": 0,
             "sides": {"s1": {"ratio_vs_s0": 1.034, "ci95": [1.021, 1.047]}}},
 "log_path": "~/.ogpu/logs/1030-164825-d47177.log", "cached_from": null}
```
- `status`: `done` (exit 0), `failed`, `cancelled`, or `lost` (the pod restarted mid-job; just re-run).
- `metrics.score`: geomean speedup vs SGLang per file, from `SCORE <file>: <x>` lines. Higher is better.
- `metrics.checks`: `PASS`/`FAIL`, from `CHECK ...` / `CHECKS:` lines. **A FAIL means the candidate is invalid, whatever its speed.**
- `metrics.sides`: paired-bench ratio vs the reference side (`s0`) with a 95% CI. Treat it as a real win only if the CI's lower bound is > 1.0.
- Need a number the parser doesn't know? Print `OGPU_METRIC name=value` from your script and it appears in `metrics`.
- `cached_from` set means this exact code + command already ran, so the result is free. Pass `--force` only to deliberately re-measure noise.
- Files your command writes under `out/` come back to `<track>/output/ogpu/<id>/`.
- Exit code: `ogpu` exits with the job's exit code. `2` means the daily budget cap was hit: stop and tell the human, don't retry.

## Hill-climbing loop
1. `ogpu ledger --track T --best --json`: know the current best and what's been tried.
2. Write the hypothesis down where your track keeps experiments (e.g. `EXPERIMENTS.md`).
3. `ogpu run --cpu --json -- <compile/ptxas gate>`. Fix compile errors and register/spill problems here, for free.
4. `ogpu run --gpu --json -- <bench with --check>`, comparing the candidate against the current best in the same run.
5. Keep it only if `checks == PASS` and the win clears noise (CI lower bound > 1.0, or ≥ ~1.5% on `score`). Otherwise discard it and log why.
6. Only then `okc check`, then `okc submit`.

## When things go wrong
- **Queued for a long time:** another agent holds the lock. Normal, so wait. Don't cancel other jobs.
- **First run in a track is slow:** extensions compile once, then they're cached on the pod disk.
- **"pod not reachable" / slow first call:** `ogpu` resumes or creates the pod automatically (2–5 min). Let it finish.
- **Interrupted (Ctrl-C, tool timeout):** the job keeps running on the pod. Use `ogpu logs <id> -f`. Don't resubmit duplicates.
- **`status: lost`:** the pod was restarted. Re-run the same command.
- **Budget cap (exit 2) or repeated capacity errors:** stop and report to the human.
