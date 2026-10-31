## GPU access
All GPU work (benchmarks, timing, correctness checks, compile/ptxas gates) goes through the `ogpu` CLI. Load the **ogpu** skill for details.
Correctness checks go to `--fidelity correctness` (cheap H100 pool); time only passing candidates on the H200.
Never SSH to pods, run `runpodctl`, or `modal run` directly, and never stop or terminate the pod.
Before `okc check`/`okc submit`, a `ogpu run --gpu --json` result must show `checks: PASS` and a win over the current best (`ogpu ledger --best --json`).
