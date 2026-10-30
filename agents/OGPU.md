## GPU access
All GPU work (benchmarks, timing, correctness checks, compile/ptxas gates) goes through the `ogpu` CLI. Load the **ogpu** skill for details.
Never SSH to pods, run `runpodctl`, or `modal run` directly, and never stop or terminate the pod.
Before `okc check`/`okc submit`, a `ogpu run --gpu --json` result must show `checks: PASS` and a win over the current best (`ogpu ledger --best --json`).
