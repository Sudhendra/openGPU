# Strategy: open-source core, savings engine, then revenue

*2026-10-30. Decisions from the founder:*
1. *Open source comes first: visibility, credibility, community, and no paywall that scares users away.*
2. *The fundamental core is a **savings engine**: optimization loops and queues that move work around with one goal, maximal savings, and prove how much was saved compared with not using the tool.*
3. *What gets sold comes after (a deep financial analysis later).*

Related: [`platform-plan.md`](platform-plan.md) (suppliers, revenue overview), [`roadmap.md`](roadmap.md), [`feasibility-analysis.md`](feasibility-analysis.md).

---

## 1. What stays open source

**The rule: anything that saves an individual money runs locally and is free, forever.** Paid offerings only add things that need *our servers, other people, or money handling*. The open-source tool must never feel crippled. A user who never pays should still get the full savings engine.

| Open source (Apache-2.0) | Paid later (needs a server, other people, or money) |
|---|---|
| CLI, TUI, job runtime, ledger | Hosted control plane: shared queue and ledger across machines and people |
| **The full savings engine**: scheduler, autoscaler, adaptive stop, placement, migration, multi-fidelity tournaments | Team budgets, approvals, SSO and roles, audit logs |
| All provider adapters + plugin API (community adds suppliers) | Managed fleet: one wallet across suppliers, consolidated billing, volume pricing |
| `audit`, `savings`, `offers`, arrival probe, spending firewall, capacity sniper | Pooled reserved capacity at cost-plus |
| Trace recorder + simulator + policy benchmarks | Global host-reputation and performance service (aggregated across users) |
| Agent integrations: skill, MCP server, hooks, CI runner | Verified run records (signing service), sponsored-compute marketplace |
| No telemetry by default; the opt-in contribution client is open source | Priority support and SLAs |

**Community levers:**
- Provider and parser plugins: like SkyPilot's clouds, but easier to write.
- A public, anonymized **trace dataset** of agent GPU workloads.
- A **policy zoo**: anyone can submit a scheduling policy and see its savings on the standard traces.
- An opt-in savings leaderboard.
- Shareable savings receipts.

---

## 2. The core: the savings engine

### 2.1 Objective
Minimize **total dollars per unit of useful work**, subject to the user's constraints:
- **Fidelity:** timing runs need the exact target GPU; correctness runs can use the same architecture; compile runs need no GPU.
- **Latency:** an agent waiting interactively vs. an overnight batch.
- **Budget:** hard caps.
- **Reliability:** whether the job is retry-safe.

**Agent waiting time is also a cost.** An idle agent burns tokens and wall-clock time. The engine weighs both, with a user-set dollars-per-agent-minute knob.

### 2.2 Job model: agents declare intent, not machines
```
ogpu submit --fidelity {timing|correctness|compile|any} --priority {interactive|batch|background}
            [--deadline 10m] [--preemptible] [--sweep cand_*.py] -- <cmd>
```
- **`fidelity`** decides which hardware is acceptable. This is the biggest lever after idle time.
- **`priority` / `deadline`** decide how long the engine may hold a job to batch it or wait for cheaper capacity.
- **`preemptible`** allows spot capacity (our runner already retries `lost` jobs).
- **Expected runtime is learned, not declared.** A predictor is trained from the ledger, per command template, host type and track.

### 2.3 The loops
```
            ┌──────────────── learn (every run) ────────────────┐
            ▼                                                   │
 agents → QUEUES ──schedule──▶ FLEET (machines/slots) ──run──▶ LEDGER/TRACES
            ▲        ▲               ▲      │                    │
            │        └── place/route │      └─ adaptive stop     │
            └──── tournament promote ┘ autoscale/migrate ◀───────┘
```

**Loop 1: Scheduler (seconds).** Several queues: interactive > batch > background. Within a class, **shortest-expected-job-first**, weighted by priority and age (with aging so nothing starves). This minimizes total agent waiting. Jobs for one track are **coalesced** inside a short batch window, so one benchmark process measures several candidates and pays `prepare` once. Machines with several GPUs get one lock per GPU (`CUDA_VISIBLE_DEVICES`), so N jobs run in parallel on one node, often cheaper per GPU.

**Loop 2: Fidelity routing (per job).** Each job runs on the cheapest slot that satisfies its fidelity:
- compile → CPU
- correctness → cheapest same-architecture GPU (H100 at $1.99–2.69 for an sm_90a target)
- timing → exact target GPU

**Loop 3: Adaptive stop (minutes). Replaces the fixed `idle_minutes`.** Keeping a machine warm vs. stopping it is the classic **ski-rental problem**:
- Restarting costs *boot time × price + cache warm-up*.
- Staying up costs *price × idle time*.
- The deterministic break-even rule (stop once idle cost equals restart cost) is guaranteed within 2× of the hindsight optimum. The randomized version is within e/(e−1) ≈ 1.58×.
- **We do better with learning:** model each agent fleet's arrival process from the ledger (time between submissions, time of day, a "session ending" signal) and stop when the expected cost of staying exceeds the expected restart cost.
- It degrades gracefully to the break-even rule when there's little data.

**Loop 4: Autoscaler and placement (minutes).** Given the queue backlog, predicted runtimes and arrival rate:
- **Scale up** when the expected cost of agents waiting exceeds a new machine's cost.
- **Choose where** by *effective* cost per job: price × predicted runtime ÷ host performance factor + amortized boot + failure-rate penalty + data-transfer cost.
- **Spot vs. on-demand:** expected spot cost = spot price × (1 + preemption rate × fraction of work lost).
- **Migrate** an idle or stoppable machine when a cheaper equivalent offer appears and the savings over the expected remaining session exceed the migration cost.
- **Capacity sniper:** when the required fidelity isn't available, watch every supplier and grab capacity under a price ceiling.

**Loop 5: Multi-fidelity tournaments (per sweep). The state-of-the-art loop for agents.** Kernel hill-climbing, hyperparameter search and evals all share one shape: many candidates, few winners. So run a **successive-halving / Hyperband**-style tournament:
1. Compile all candidates on CPU and drop failures (free).
2. Correctness-check survivors on the cheap same-architecture GPU and drop failures.
3. Run a quick timing pass (few reps) on the target GPU and keep the top *k*, plus anything within noise of the leader.
4. Run full timing reps with confidence intervals on the finalists only.
5. Return a ranked result. Only a winner whose interval clears the current best is recommended for the expensive external check (e.g. `okc check`).

The agent submits one sweep and gets back one ranked answer. Most candidates never touch the expensive GPU at full reps.

**Loop 6: Learning (every run).** Every finished job updates the estimates the other loops use:
- runtime per command template and host type
- host performance factor (from the arrival probe and real runtimes)
- boot time per supplier and image
- preemption and loss rate per supplier and tier
- agent arrival rate

Moving averages and simple Bayesian estimates, all local in v1. **Opt-in sharing** of these estimates across users later becomes the network effect behind the reputation service.

### 2.4 Proving it: savings accounting
The headline number is **actual spend vs. a counterfactual "without ogpu" baseline**. The method is public and deliberately conservative:
- **Baseline:** one on-demand machine of the target GPU on the user's default supplier tier. It's on from first job to last job of each session, with **no forgotten-machine tail by default** (reported separately), every job re-run (no cache), every job on the target GPU (no fidelity routing), and no batching.
- **Breakdown by lever:** idle avoided, placement and tier, cache and dedupe, fidelity routing, batching and tournaments, spot.
- **Outputs:** `ogpu savings` (terminal), `ogpu savings --card` (a shareable image), and per-run `saved_usd` in the JSON result.

### 2.5 Developing policies with a trace-driven simulator
- Every real session is recorded as a **trace**: job arrivals, fidelity, runtimes, machine prices, boot times.
- A simulator replays traces against policies (naive, fixed auto-stop, break-even, learned, ogpu full) and reports cost and agent wait.
- **We design policies offline, prove them in tests, and publish the benchmark.** That gives a reproducible "how much ogpu saves" number the world can check. It's both our development method and our credibility.
- Seed traces: our own Open Kernel Challenge sessions, which are real multi-agent hill-climbing traces.

### 2.6 What agents get (protocol)
- Declarative `submit` with fidelity, priority and sweep. `wait --any` returns the first finished. Results carry `cost_usd`, `saved_usd`, `queue_wait_s` and **advice** ("batch these 3", "this could run at fidelity=correctness").
- An MCP server exposing the same operations as typed tools.
- Spending firewall: per-agent and per-project budgets, hard caps with a lock period, approval above a threshold, and anomaly kill for jobs running far longer than predicted.

---

## 3. What gets sold (outline; deep analysis later)
In rough order of when each becomes viable:
1. **Team control plane** (per seat). Comes once individuals bring the tool to work.
2. **Managed fleet / one wallet** (margin on routed spend). Needs supplier agreements and a track record of routed volume.
3. **Host reputation and performance service** (API and enterprise). Needs opt-in data volume.
4. **Pooled reserved capacity** (cost-plus). Needs proof of ≥80% projected utilization from aggregate traces.
5. **Verified records and sponsored compute** (sponsor-funded). Needs a community and challenges.

Every one of these is fed by the open-source engine's adoption and data. None requires taking anything away from the open-source tool.

---

## 4. Build order
| Milestone | Scope | Proof |
|---|---|---|
| **M1: savings engine core** | Job classes and fidelity; runtime predictor; multi-queue SJF scheduler with aging and coalescing; per-GPU slots; **adaptive stop** (break-even + learned); trace recorder; simulator with baseline policies; `ogpu savings` with lever breakdown | Simulator on our OKC traces shows the savings vs. the naive baseline. Live savings report. |
| **M2: tournaments + fidelity routing** | `--sweep` successive-halving tournaments; heterogeneous fleet (H200 + cheap same-arch tier); provider interface; Vast.ai + BYO-SSH adapters | One sweep run end to end; % of candidates that never reach full H200 reps |
| **M3: placement engine** | `offers` with effective cost per job; arrival probe and host scores; spot with auto-retry; migration; `audit` | Savings breakdown shows placement and spot levers on real runs |
| **M4: guardrails + reach** | Spending firewall, capacity sniper, MCP server, CI runner, `ogpu dev` | External users onboard without help |
| **R0 (in parallel)** | Rename, license, packaging, docs | Public release |
