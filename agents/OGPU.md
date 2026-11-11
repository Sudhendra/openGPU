## GPU access
All GPU work (benchmarks, training/eval runs, timing, correctness checks, compile gates) goes through the `ogpu` CLI. Load the **ogpu** skill for details.
Check numerics with `--fidelity correctness` (cheaper pool, if configured) before spending the target GPU on timing.
Never SSH to GPU machines or call provider CLIs (`runpodctl`, `modal run`) directly, and never stop or delete machines.
Before acting on a result, the `ogpu run --json` line must show `failure: null` (and, for comparisons, a win over `ogpu ledger --best --json`).
