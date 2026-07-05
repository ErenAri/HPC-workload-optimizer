# HPCOpt Competitive Roadmap

Strategy in one line: **stop positioning HPCOpt as a scheduler that competes with Slurm/Flux, and
position it as the referee — the fast, neutral, contract-gated evaluation harness that scheduling
claims (academic or vendor) are measured by, plus a what-if tuning tool for operators.**

Grounding (researched July 2026):

- NVIDIA acquired SchedMD and is shipping AI-driven backfill/priority inside Slurm — the
  "standalone smart scheduler" endgame is closed; the "neutral verifier" niche is open.
- Batsim (research incumbent) has documented limits: no core-level allocation (SimGrid), perfect
  speedup, CPU-only energy, weakly tested ecosystem, painful trace conversion.
- UB CCR Slurm Simulator (operator incumbent) is version-locked (patched Slurm 23.02) and slow
  (~17 simulated days per wall-clock hour). Operators are documented as "in the dark" on Slurm
  parameter tuning. Our Rust engine is ~4 orders of magnitude faster.
- Quantile-regression runtime prediction is published SOTA (UARP, J. Supercomputing 2026) — our
  runtime model alone is not a contribution; the contract/fidelity/gating layer is.
- The field's dominant research wave is energy/power/carbon-aware scheduling; modern open datasets
  with per-job power exist (PM100/Marconi100 with GPUs; F-DATA/Fugaku, 24M jobs).

---

## Phase 0 — Truth and reframe (days)

Goal: make the README's headline claim the one that matters, and remove credibility leaks.

1. Run the full policy matrix on all three reference traces:
   `FIFO_STRICT, EASY_BACKFILL_BASELINE, EASY_BACKFILL_TSAFRIR, SJF_BACKFILL, FAIRSHARE_BACKFILL,
   ML_BACKFILL_P50, ML_BACKFILL_P10, RL_TRAINED` × {CTC-SP2, HPC2N, SDSC-SP2}.
   Publish the table win **or lose** — Tsafrir (prediction-based backfill) is the real baseline,
   not FIFO. A loss is reframed as a harness demonstration (claim correctly blocked).
2. Rewrite README top: harness positioning, the matrix table, and a "comparison with existing
   tools" section (Batsim, Slurm Simulator, RLScheduler, CQSim/Alea). Move ops evidence to docs/.
3. Fix credibility leaks: mint a real Zenodo DOI (badge currently says `zenodo.pending`), rename
   the GitHub repo `HCP-` → `HPC-workload-optimizer` (keep redirect), publish the mkdocs site.

Acceptance: README leads with policy-vs-policy results incl. prediction-based baselines; no
pending/broken badges; ops content demoted below the science.

## Phase 1 — Batsim agreement study (1–2 weeks)

Goal: answer "why trust your simulator?" the way SPARS did — by validating against Batsim.

1. Finish the existing Batsim path (WSL/Docker) for FIFO + EASY on all three traces.
2. Publish `docs/validation/batsim-agreement.md`: per-metric deltas (utilization, mean/p95 wait,
   makespan, BSLD) with explicit tolerance thresholds and explanations of any divergence.
3. Optional: nightly CI job running one small-trace agreement check.

Acceptance: quantified agreement table; every downstream claim can cite it.

## Phase 2 — `hpcopt whatif`: the operator wedge (2–4 weeks)

Goal: a tool HPC centers actually run. Fast what-if analysis for Slurm tuning does not exist
(Slurm Simulator is too slow and version-locked); the need is documented.

1. New CLI verb: `hpcopt whatif --sacct <dump> --change <param=value> [...]`
   Pipeline: ingest sacct window → calibrate capacity from trace → baseline replay + fidelity
   gate → replay with proposed change → delta report (p95 BSLD, wait, utilization) with a
   fidelity-confidence grade attached.
2. Map a bounded subset of Slurm scheduling parameters onto policy configs: scheduler/backfill
   choice, backfill horizon, priority-weight approximations (via FAIRSHARE knobs), job-size and
   partition limits. Document what is and is not modeled.
3. Ship an end-to-end example with synthetic sacct data in `examples/`.

Acceptance: one command from raw sacct dump to a markdown what-if report; runtime seconds, not
hours.

## Phase 3 — Modern era: multi-resource, GPU, energy, modern traces (4–8 weeks)

Goal: escape 1996. This is the largest engineering item and the paper payload.

1. Extend the Rust core (`sim-runner/src/main.rs`, currently a single `capacity_cpus: u32`) to a
   resource vector `{cpus, gpus, mem}`; keep the scalar path as a compatibility mode. Node-level
   allocation is a stretch goal, not a blocker.
2. Ingest PM100 (Marconi100: 231K jobs, GPU allocations, measured node/CPU/mem power) and an
   F-DATA slice (Fugaku, 24M jobs — chunked ingestion).
3. Energy as a first-class objective: per-job joules from measured power, energy metrics in
   reports, energy×BSLD Pareto fronts in the recommendation engine (Pareto mode already exists),
   plus a power-cap stress scenario.

Acceptance: policy comparison on PM100 with a BSLD-vs-energy Pareto front; cross-language parity
tests updated; sim throughput still <1s per 200K jobs.

## Phase 4 — Plug-in policies + public leaderboard (2–3 weeks, parallel with 3)

Goal: make competing research strengthen the platform instead of racing it.

1. Freeze and document the adapter contract as a public plug-in API; provide a template repo for
   external policies.
2. Implement a UARP-style (p50 dispatch / p99 reservation) policy as the first "external" plugin,
   benchmarked head-to-head with Tsafrir and our ML/RL policies.
3. Generate a standing leaderboard page from credibility dossiers (mkdocs/GitHub Pages): policies
   × traces × objectives, every cell backed by a manifest hash.

Acceptance: a third party can add a policy without touching core; leaderboard auto-builds in CI.

## Phase 5 — Distribution and legitimacy (ongoing)

- PyPI release + Spack package (how HPC centers actually install software).
- arXiv preprint; SC'26 workshop paper (deadlines ~Aug–Sep 2026; energy/harness angle fits EESP);
  JSSPP 2027 full framework paper (deadline ~Feb 2027; their "Open Scheduling Problems and
  Workload Traces" track fits the leaderboard/benchmark contribution).
- JOSS submission for a citable software paper; pursue ACM artifact badges (the manifest/dossier
  system is purpose-built for them) and advertise HPCOpt as the tool other authors use to earn
  theirs.
- HPSF (Linux Foundation) sandbox-project application once there are external users.

---

## Sequencing and dependencies

```
Phase 0 ──> Phase 1 ──> Phase 2 (operator wedge)
                └─────> Phase 3 (modern era) ──> Phase 5 papers
                              └── Phase 4 (parallel)
```

## Risks

- **ML/RL policies may lose to Tsafrir/UARP baselines.** Acceptable: the harness framing makes
  honest losses publishable; the research problem (better predictors) becomes roadmap, not shame.
- **Batsim agreement may reveal sim divergences.** That is the point — fix or document them; the
  study is valuable either way.
- **F-DATA scale (24M jobs).** Rust engine throughput is fine; ingestion needs chunked/streaming
  parquet. Start with a monthly slice.
- **Slurm parameter mapping fidelity (Phase 2).** Scope tightly, document unmodeled parameters
  explicitly — honesty is the brand.
