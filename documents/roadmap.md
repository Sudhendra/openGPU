# ogpu roadmap

## Phase 1 (done 2026-10-30)
- `ogpu submit` / `ogpu wait`, so agents queue batches and keep working
- Identical in-flight jobs are deduped (a second agent waits on the first run)
- Failure classification in every result: `compile_error | import_error | correctness_fail | oom | cuda_error | timeout | runtime_error | lost`, plus an error excerpt
- Exit code 3 when a benchmark passes but correctness checks fail
- `--note` hypotheses in the shared ledger, `ogpu ledger --failed`, `ogpu show <id>`
- `--timeout`, one automatic retry on `lost`, pod helpers auto-upgrade when ogpu changes
- GPU busy %, idle $ and per-agent spend in `ogpu cost` and `ogpu top`
- macOS notifications: new best per track, 80% of daily budget
- Claude Code PreToolUse guard hook (`scripts/hooks/guard_gpu.py`) that blocks raw ssh/runpodctl/modal run

## Phase 2 (next)
- Cheap correctness tier: route `--check`-only jobs to an H100 (same sm_90a; $1.99–2.69/hr vs $3.59–5.29). Timing stays on the H200.
- `ogpu migrate`: move to community capacity when it frees up (−32%). Needs a portable ext cache (network volume or sync).
- `okc` calibration: ingest `okc check`/`submit` results, learn local→judge score mapping, warn before likely-wasted quota.
- Custom image = judge image + sshd (faster boot; only addition is sshd).
- Per-agent budgets and priority (final timing ahead of exploration).

## Phase 3
- Warm bench server: keep CUDA context, weights and inputs resident and hot-swap candidates (seconds per eval instead of minutes). Needs harness integration and a fidelity check against `bench.py`.
- Multi-provider H200 capacity/price probe and failover (Shadeform, Vast, Nodus, Jarvislabs).
- A host with ncu counters (RunPod blocks `ERR_NVGPUCTRPERM`).
- Serving-track variant (SGLang server benchmarks).
