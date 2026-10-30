# openGPU — Feasibility Analysis & Preliminary Business Model

*Drafted 2026-10-30. Parked while we build openGPU v0 for personal use. Revisit later.*

## Context
Talented independent developers can't compete in the agentic era: getting near the top (e.g. the Open Kernel Challenge) takes a lot of money for compute and frontier tokens. The goal of openGPU is to make compute per unit of intelligence affordable for individuals, with much better infra access and setup. Research is from web sources dated Jun–Oct 2026. Many of them are vendor blogs, so figures are directional.

---

## 1. What the market actually looks like (Oct 2026)

| Signal | Data | Implication |
|---|---|---|
| H100 rent | Marketplace floor ~$1.79–2.40/hr; median ~$3.84–4.20; Lambda/Together $3.99; CoreWeave ~$6.16; Nebius $4.50 | Prices are **rising**, not falling (Lambda $2.99→$3.99, CoreWeave +25% in Jul). The supply shortage is real. |
| H200 rent | ~$2.89–6.30/hr on-demand; median ~$4.65 across 37 providers; Modal ~$4.54; RunPod ~$3.59–5.29 | The judge GPU for OKC; availability is sparse on marketplaces |
| B200 rent | $3.69–8.19/hr, avg ~$6.39 | Newer silicon gives better $/token but costs more per hour |
| Reserved discount | 25–50% off on-demand for multi-year deals | **This is the gap**: individuals can't sign multi-year contracts, and companies can |
| Neocloud margins | CoreWeave: ~59% adj. EBITDA but a $626M net loss (interest + depreciation) | Owning GPUs is a debt-financed bet on utilization. Don't start there. |
| DePIN supply vs demand | io.net claims 139k GPUs and ~$20M ARR (self-reported); Akash 33.7% GPU util (84/334 GPUs); sector ~$180–220M on-chain revenue | **Supply is easy to get and demand is the hard part.** Token-subsidized supply-first networks stall. |
| Consumer GPUs | Salad ~60k daily active consumer GPUs; inference only (low VRAM, home bandwidth, nodes drop off) | Fine for batch/small-model inference. Useless for frontier-scale training. |
| Open-weight tokens | DeepSeek V4 Flash ~$0.044/M blended; GLM-5.2 ~$0.46/M; MiniMax M3 $0.10/$1.21 | **Open-weight intelligence is already cheap.** Tokens are not the bottleneck for open models. |
| Dev agentic spend | $150–600/dev/month. Copilot went metered (Jun 2026), Cursor moved off flat rate, Claude Max is still flat | The real pain is the **frontier-model bill plus the GPU bill for experiments** |
| Free programs | Fragmented: Azure $100 student, Modal academic up to $10k, HF ZeroGPU 5 min/day, startup credits for companies only | No one serves the **"talented individual with no company"** segment well |

**Core insight:** The problem isn't the price of a raw GPU-hour. It's that (a) individuals pay on-demand/retail rates, roughly 2× what contract buyers pay, (b) setup friction burns money (idle instances, wrong GPU, no batching, no caching), and (c) there's no path from "I'm good" to "someone funds my compute."

---

## 2. Feasibility verdict

| Approach | Feasible? | Why |
|---|---|---|
| Build or own a GPU cloud | ❌ Not at start | Capital-intensive, debt-financed, margins are utilization bets. CoreWeave loses money at scale. |
| Another DePIN/token network | ❌ | Crowded (io.net, Akash, Aethir, Render, Salad). Demand is the bottleneck and crypto hurts credibility with devs. |
| Cheaper open-model API | ⚠️ Commodity | OpenRouter, DeepSeek, Together already sell at near-zero margin. No moat. |
| **Demand aggregator / compute co-op + efficiency layer + talent-funded compute** | ✅ **Recommended** | Captures the reserved-vs-retail spread, removes setup waste, and adds a subsidy source that isn't the developer's wallet. Capital-light at first. |

---

## 3. Recommended model: "Compute co-op for builders"

Three layers, built in order:

### Layer 1 — Efficiency layer (weeks 0–8, near-zero capex)
A CLI/SDK plus dashboard that makes every dollar go further on *existing* clouds:
- **Smart routing:** each job or agent call goes to the cheapest capable target (marketplace spot GPU, reserved pool, open-weight API, or frontier API only when needed).
- **Agent cost controls:** prompt caching, context compression, using a cheap model for the planner and frontier only for hard steps, and per-task budgets.
- **Zero-friction GPU sessions:** one-command dev env, auto-shutdown when idle, checkpoint and resume across spot preemption (spot is often 40–60% cheaper).
- Revenue: a small margin on routed spend (5–10%) or a $10–20/mo pro tier.
- Purpose: get the demand and trust that DePINs never had.

### Layer 2 — Pooled capacity (once demand exists, ~month 3–6)
- Pool members' demand and sign **1-yr reserved contracts** (~$2.0–2.5/hr H100-equivalent vs $3.99 retail).
- Resell to members at about cost +15%, in tiers: guaranteed (higher price), preemptible (cheaper), batch/overnight (cheapest).
- **Backfill idle hours** with batch inference (serving open-weight coding models to members' agents). This keeps utilization high and is the key risk control.

Unit economics sketch for one 8×H100 node on a 1-yr reserved contract at $2.20/hr:
- Cost: 8 × 8,760 × $2.20 ≈ **$154k/yr**
- Members at a blended $2.90/hr:
  - 75% util ≈ $152k (break-even)
  - 85% util ≈ $173k (+12%)
  - 90% util ≈ $183k (+19%)
- Members still save ~27% vs retail. **Utilization ≥80% is the whole game.** That's why Layer 1 demand and inference backfill come first.

### Layer 3 — "Prove yourself" compute (the mission wedge, ~month 4+)
- Public challenges, evals and builds on the platform. Strong results earn **sponsored compute credits**.
- **Sponsors** (labs, AI startups hiring, model providers wanting adoption, GPU vendors) pay for (a) credits and (b) access to verified talent signals: what someone built and their eval scores, not their resume.
- Revenue: sponsorship pools plus recruiting/placement fees. This subsidizes compute for people who can't pay and directly answers "can't prove themselves."
- Optional later: a nonprofit/fiscal-sponsor arm for grants (NSF/NAIRR-style public compute, foundations).

### Revenue mix (target by year 2)
1. Pooled-capacity margin: 10–20% on compute resale (main volume)
2. Efficiency-layer pro subscriptions and routing fees
3. Sponsor and recruiting revenue (high margin, mission-aligned)

---

## 4. Key risks & mitigations
- **Utilization risk on reserved capacity:** start with month-to-month or 3-month blocks from neoclouds and marketplaces, fill with backfill inference, and only scale contracts once 80%+ utilization is proven.
- **Commodity pressure:** the moat is community, sponsor network, talent signal and workflow tooling, not price alone.
- **Rising GPU prices:** the reserved pool *hedges* members against this. That's a selling point.
- **Abuse/fraud** (crypto mining, credit farming): KYC-lite, workload monitoring, credits tied to verified output.
- **Frontier-model dependence:** costs can be cut (routing, caching) but not commoditized. Be honest that open-weight models carry most of the volume.
- **Provider ToS on resale:** check reseller terms with each upstream (many neoclouds have partner/reseller programs).

---

## 5. 90-day validation plan
1. **Demand discovery (wk 1–3):** interview or survey 50–100 indie and agentic devs about monthly spend, what they'd run with 2× the compute, and willingness to prepay or commit.
2. **Prototype Layer 1 (wk 2–8):** CLI that launches auto-shutdown GPU sessions on 2–3 marketplaces, plus an OpenAI-compatible gateway with routing and caching for agent calls. Measure the actual $ saved per user. *(openGPU v0, `ogpu`, is the first slice of this.)*
3. **Supply quotes (wk 4–6):** get reserved/partner pricing from 3–5 neoclouds (Lambda, Nebius, CUDO, Shadeform, Hyperbolic) for 1–2 nodes.
4. **Sponsor LOIs (wk 6–10):** pitch 5–10 AI startups and model providers on sponsored compute plus talent access.
5. **Go/no-go metrics:** ≥200 active devs, ≥30% measured savings, ≥$20k/mo routed spend, ≥2 sponsor LOIs. Then sign the first reserved block.

## Sources
- GPU pricing: [cloudzero.com](https://www.cloudzero.com/blog/h100-gpu-cost/) · [uvik.net GPU price index](https://uvik.net/blog/gpu-price-index/) · [mercatus-ai.com](https://www.mercatus-ai.com/blog/gpu-rental-prices) · [datastorage.com](https://datastorage.com/articles/gpu-rental-prices-mid-2026-what-h100-h200-and-b200-capacity-actually-costs-right-now/) · [economize.cloud](https://www.economize.cloud/resources/gpu/pricing/) · [jarvislabs H200](https://jarvislabs.ai/blog/h200-price) · [getdeploying H200](https://getdeploying.com/gpus/nvidia-h200) · [gpuperhour H200 SXM](https://gpuperhour.com/rent/h200-sxm) · [gloom.sh H200](https://gloom.sh/gpu-prices/h200)
- Neoclouds: [CoreWeave 8-K FY2026](https://www.sec.gov/Archives/edgar/data/0001769628/000176962826000236/ex9912.htm) · [prompt20 neoclouds guide](https://blog.prompt20.com/posts/neoclouds-complete-guide/) · [intuitionlabs data-center GPU pricing](https://intuitionlabs.ai/articles/data-center-gpu-pricing-2026)
- DePIN: [blockeden DePIN revenue pivot (Apr)](https://blockeden.xyz/blog/2026/04/12/depin-revenue-pivot-token-subsidies-ai-compute-akash-render-ionet/) · [blockeden (Mar)](https://blockeden.xyz/blog/2026/03/12/depin-compute-revenue-pivot-akash-ionet-aethir/) · [hoge.gg](https://hoge.gg/akash-vs-io-net-render-depin-gpu-race-2026/) · [cointelegraph](https://cointelegraph.com/features/what-role-is-left-for-decentralized-gpu-networks-in-ai) · [androidheadlines](https://www.androidheadlines.com/2026/09/startups-far-labs-evolving-edge-rent-idle-pc-ai-inference.html)
- Tokens & dev spend: [OpenRouter open-weight June 2026](https://openrouter.ai/blog/insights/the-open-weight-models-that-matter-june-2026/) · [benchlm OpenRouter](https://benchlm.ai/hosts/openrouter) · [tech-insider Copilot metered](https://tech-insider.org/ie/github-copilot-ai-credits-pricing-2026/) · [augmentcode AI coding plans](https://www.augmentcode.com/learn/ai-coding-plans-2026-billing-models)
- Free programs: [gpuperhour free GPUs](https://gpuperhour.com/blog/free-cloud-gpus-and-credits) · [spheron free credits](https://www.spheron.network/blog/free-gpu-cloud-credits-2026/)
