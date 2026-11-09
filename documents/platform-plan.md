# From challenge helper to compute broker: platform plan

*2026-10-30. Builds on [`feasibility-analysis.md`](feasibility-analysis.md).*

## 1. Where we are
`ogpu` started as a way to win one kernel competition. Along the way it became the **Layer 1 "efficiency layer"** from the feasibility analysis:

| Feasibility-analysis promise | What exists today |
|---|---|
| Zero-friction GPU sessions | `ogpu up`: reuse → resume → create with GPU/cloud fallback, real SSH, auto-installed runtime |
| Auto-shutdown when idle | On-pod reaper stops the pod after N idle minutes. Disk and caches persist. |
| Agent cost controls / per-task budgets | Daily budget cap, content-hash cache, in-flight dedupe, `--timeout`, per-agent spend |
| Measure the actual $ saved | Ledger records paid time, GPU-busy time, idle $, cache hits and per-job cost |
| Many users sharing capacity | Exclusive GPU lock + queue, per-job snapshots, multi-agent identity |

The core insight from the analysis still holds: **the raw GPU-hour isn't the problem; the waste around it is.** Agentic sessions pay for boot time, for the minutes agents spend thinking between runs, and for the idle-stop wait. They pay retail on-demand rates and re-run identical work. (Our own measurements so far are a small sample, so the first job of the ledger is to quantify this per user.) `ogpu` attacks all three, and now **proves it with numbers**. That's the trust that supply-first DePIN networks never earned.

What's missing for a general product: **it only speaks RunPod**, and a few defaults (judge image, OKC metric parsers) are challenge-specific.

## 2. Product direction: a supplier-neutral compute broker for builders and their agents
One CLI and runtime, any GPU supplier. The user declares *what* they need (GPU type or min VRAM, image, budget). `ogpu` picks *where* (the cheapest supplier with stock that satisfies the requirements), runs the job safely next to other agents, and shuts it down when idle.

### Supplier landscape (targets)
| Family | Suppliers | Notes |
|---|---|---|
| Container pods (choose image) | RunPod ✅, Vast.ai, TensorDock, Hyperbolic, Prime Intellect | Closest to today's model. Stop/resume semantics vary. |
| VMs (Ubuntu + Docker) | Lambda, Nebius, CUDO, Crusoe, hyperscalers | Run the image with `docker run --gpus all`. Some can't stop, only terminate. |
| Aggregators | Shadeform (one API over many clouds) | Fastest way to broad H100/H200 coverage |
| Serverless | Modal ✅ (fanout), Beam | Job-level, no SSH. Good for parallel correctness sweeps. |
| Bring-your-own | Any SSH host: university cluster node, home box, reserved server | Zero supplier cost. Biggest win for students and researchers. |

### Architecture change: a small provider surface
Everything above the provider (job runner, lock, reaper logic, ledger, cache, TUI, skill) only needs **an SSH-reachable machine with a GPU and the right environment**. So the provider interface stays small:

```python
class Provider(Protocol):
    name: str
    caps: Capabilities            # public_ssh, stop_resume, persistent_disk, spot, custom_image, profiler_counters
    def offers(self, req: GpuRequest) -> list[Offer]   # price/hr, stock, region, tier (secure/community/spot)
    def create(self, offer: Offer, spec: MachineSpec) -> Machine
    def get(self, id) -> Machine | None
    def list(self) -> list[Machine]
    def resume(self, id) -> None   # NotSupported → placement treats stop as terminate
    def stop(self, id) -> None
    def terminate(self, id) -> None
    def stop_command(self, machine) -> str   # what the on-machine reaper runs (scoped token, never the account key)
```

Plus two orthogonal pieces:
- **Runtime:** `native` (the image *is* the machine, like RunPod or Vast) or `docker` (a VM runs the image via `docker exec`). `jobrun.py` already runs everything through one shell entry point, so this is a thin wrapper.
- **Cache portability:** for suppliers without stop/resume, sync the compiled-extension cache to object storage or a local mirror so a fresh machine starts warm.

### Placement policy (the money engine)
`ogpu offers h200` shows every supplier's live price and stock side by side. `ogpu up` picks by policy:
1. Resume an existing stopped machine (warm caches) if its host still has capacity.
2. Otherwise take the cheapest offer that meets the requirements, preferring tiers the user allows (`spot` when jobs are retry-safe, which ours are thanks to `lost` → auto-retry).
3. Migrate when a much cheaper offer appears and the GPU is idle (e.g. secure → community H200 is −32% on RunPod alone).
4. Route by requirement within a pool: correctness-only jobs → cheapest same-ISA GPU (H100 for sm_90a). Timing jobs → the exact target GPU.

### Generalizing the challenge-specific bits
- **Metric parsers become plugins:** `okc` (today's `SCORE`/`CHECK`/`SIDE`), `pytest`, `json-lines`, and the universal `OGPU_METRIC k=v`. Selected per project in `.ogpu.toml`.
- **Project profiles:** a `.ogpu.toml` in the repo sets the image, GPU requirement, parser, excludes and budget, so a project, not the user's global config, describes its environment.

## 3. The financial picture, updated
### What a user saves (the pitch, now measurable)
For one developer running agents on an H200:

| Lever | Mechanism in ogpu | Typical effect |
|---|---|---|
| Idle time | Reaper + busy % visibility | Removes paid-but-idle hours (the largest waste we observe) |
| Rate | Supplier placement + community/spot tiers | −30–45% vs. a default secure on-demand H200 ($5.29 → $3.59 or lower) |
| Duplicate work | Content-hash cache + in-flight dedupe | Repeat measurements cost $0 |
| Wrong hardware | Same-ISA cheap tier for correctness | Correctness runs at ~40–60% of the H200 rate |
| Wasted human/agent time | Failure classes, shared ledger, notes | Fewer repeated dead ends (not priced, but real) |

The ledger lets us replace "typical effect" with each user's own numbers: `ogpu cost` already reports idle $ and busy %, and a `savings` report compares actual spend with the counterfactual (no reaper, no cache, default tier).

### Revenue model (open-core, in order)
1. **Open-source core, free forever** for individuals: CLI, runtime, provider adapters, local ledger. This is the adoption engine and keeps the mission ("compute for talented individuals") credible.
2. **Hosted control plane for teams and labs (paid, per seat):** a shared queue and ledger across machines and people, team budgets and approvals, dashboards, SSO, audit. Same CLI; the ledger moves from SQLite to the service.
3. **Brokerage:** routing spend across suppliers. Revenue from supplier partner/referral programs and volume deals. **The rule is neutrality:** ranking is by user cost and is fully transparent, and any referral relationship is disclosed in `ogpu offers`.
4. **Pooled capacity (feasibility Layer 2):** once aggregate demand is visible in the control plane, buy reserved blocks (25–50% below on-demand) and resell at cost-plus to members, backfilling idle hours. The ledger's utilization data sizes these commitments, which is the key risk control from the analysis (≥80% utilization or don't sign).
5. **Verified results and sponsored compute (Layer 3):** every run already has a content hash, command, environment and metrics. Signing these records makes **reproducible proof-of-work** that a developer can show anyone. Sponsors (labs, startups hiring, GPU vendors) fund compute pools for challenge participants and discover talent through verified results rather than résumés. This closes the loop on the original frustration: talented people who can't prove themselves.

### Metrics that decide each step
- **Adoption:** weekly active users, machines launched, supplier mix.
- **Value:** measured savings % per user (target ≥30%), GPU busy % trend, cache-hit rate.
- **Brokerage readiness:** monthly routed spend (target ≥$20k/mo before supplier deals).
- **Pooling readiness:** aggregate steady-state GPU-hours per type (size reserved blocks only at ≥80% projected utilization).

## 4. Phased plan
| Phase | Scope | Exit criterion |
|---|---|---|
| **R0: release prep** | Rename (see below), license, `Provider` interface with RunPod refactored onto it, parser plugins, `.ogpu.toml` project profiles, `ogpu init` setup wizard, CI, PyPI release | A fresh user goes from `pip install` to a first GPU job in <10 min on RunPod |
| **R1: multi-supplier** | Vast.ai, Lambda (docker runtime), BYO SSH adapters; `ogpu offers`; placement policy; cache portability | Same job runs on 3 suppliers with no config changes; `offers` shows live prices |
| **R2: cost engine** | Shadeform aggregator, spot tiers with auto-retry, migration, multi-machine pools with requirement routing, `ogpu savings` | Measured savings ≥30% vs. counterfactual for pilot users |
| **R3: teams** | Hosted control plane (shared ledger/queue, budgets, dashboards, SSO) | 3 paying teams |
| **R4: pooling & sponsorship** | Reserved-capacity pool, signed run records, sponsor-funded compute for challenges | First reserved block at ≥80% utilization; first sponsored challenge |

## 5. Release blockers found
- **Name collision.** "OpenGPU" is an existing decentralized-GPU crypto network (OpenGPU Network, reportedly a registered trademark of OpenGPU Management S.à r.l.), with a token traded as **OGPU**. The PyPI name **`ogpu` is taken by their SDK** ("OpenGPU SDK for distributed AI task deployment", v0.2.3). Same domain, same name: we must rename both the project and the CLI before any public release.
- **No license yet.** Pick one before publishing (Apache-2.0 recommended: permissive, with a patent grant, common for infra tools).
