# Artifact Description / Artifact Evaluation Appendix

This appendix follows the SC (Supercomputing) Artifact Description / Artifact
Evaluation (AD/AE) template and the ACM Artifact Review and Badging v2.0
guidance. It documents how to obtain, build, configure, and run HPC Workload
Optimizer (HPCOpt) so an independent reviewer can reproduce every benchmark
number cited in the README and `paper/hpcopt_paper.md`.

We target three ACM badges:

- **Artifacts Available** — code, configs, schemas, and trace pointers are
  archived under a Zenodo DOI (see `CITATION.cff`).
- **Artifacts Evaluated — Reusable** — the artifact installs cleanly, the
  test suite (≥420 tests) passes, and CLI entry points work as documented.
- **Results Reproduced** — running `scripts/reproduce_paper.py` regenerates
  the README "Benchmark Results (Parallel Workloads Archive)" table within
  the tolerances stated below.

---

## A. Artifact Identification

- **Repository:** https://github.com/ErenAri/HPC-workload-optimizer
- **Pinned release:** `v2.1.0` (see `CITATION.cff` and Zenodo DOI).
- **License:** Apache-2.0 (see `LICENSE`).
- **Languages:** Python (≥3.11) and Rust (stable, optional accelerator).
- **Persistent identifier:** Zenodo DOI minted at release tagging
  (`10.5281/zenodo.<id>` — placeholder `pending` in README badge until first
  Zenodo deposit).

## B. Hardware & Software Requirements

Minimum reviewer setup:

- 4 CPU cores, 8 GB RAM, 5 GB free disk.
- OS: Linux (Ubuntu 22.04+), macOS 13+, or Windows 11 with WSL2.
- Python 3.11 or 3.12.
- Optional: Rust ≥1.75 for the accelerated `sim-runner` binary.
- Optional: Docker 24+ for the containerised path.

No GPUs, accelerators, or special interconnects are required.

## C. Datasets

All benchmarks use traces from the public Parallel Workloads Archive (PWA)
maintained by Dror Feitelson at the Hebrew University of Jerusalem
(https://www.cs.huji.ac.il/labs/parallel/workload/).

The reference suite for §"Benchmark Results" uses the cleaned versions:

| Trace id   | File                            | Jobs    | Capacity (CPUs) |
| ---------- | ------------------------------- | ------- | --------------- |
| `ctc_sp2`  | `CTC-SP2-1996-3.1-cln.swf.gz`   | 77,222  | 512             |
| `hpc2n`    | `HPC2N-2002-2.2-cln.swf.gz`     | 202,870 | 240             |
| `sdsc_sp2` | `SDSC-SP2-1998-4.2-cln.swf.gz`  | 54,044  | 128             |

Place the gzip-compressed SWF files under `data/raw/`. Hashes are pinned in
`configs/data/reference_suite.yaml` and validated by
`hpcopt pipeline reference-suite verify`.

## D. Installation

### D.1 From a fresh clone (Linux/macOS/WSL)

```bash
git clone https://github.com/ErenAri/HPC-workload-optimizer.git
cd HPC-workload-optimizer
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### D.2 Optional Rust accelerator

```bash
cd rust && cargo build --release && cd ..
# binary at: rust/target/release/sim-runner(.exe)
```

### D.3 Optional Docker path

```bash
docker compose up --build
```

## E. Smoke Test (≈ 1 minute)

```bash
python -m pytest tests/unit/test_tsafrir_baseline.py \
                 tests/unit/test_ml_backfill_policy.py \
                 tests/unit/test_simulation_properties.py -q
```

Expected: all tests pass. This validates that the simulator core, the
EASY-backfill family of policies, and the new Tsafrir baseline are
operating end-to-end.

## F. Reproducing the README Benchmark Table

```bash
python scripts/reproduce_paper.py
```

Outputs land in `outputs/reproduce_paper/<UTC-timestamp>/`:

- `results.json`   -- machine-readable table (one row per (trace, policy)).
- `results.md`     -- markdown rendering matching the README.
- `environment.json` -- Python/platform/git fingerprint for provenance.

Subset runs:

```bash
python scripts/reproduce_paper.py --traces ctc_sp2
python scripts/reproduce_paper.py --policies FIFO_STRICT,EASY_BACKFILL_BASELINE,EASY_BACKFILL_TSAFRIR
```

### F.1 Expected numbers and tolerances

Determinism is guaranteed for a fixed `(trace, policy, capacity_cpus)`
triple: HPCOpt's discrete-event loop, dispatch tie-breaking, and metric
aggregations are all integer-deterministic. Reproduced numbers should match
the README **exactly** for `FIFO_STRICT` and `EASY_BACKFILL_BASELINE`.

For `EASY_BACKFILL_TSAFRIR`:

- Numbers are deterministic given the same trace.
- They are **not** in the README's existing baseline table because the
  Tsafrir baseline was added in v2.1.0; expected ranges will be added after
  the first cross-reviewer reproduction. The number we expect to publish is
  the p95 BSLD on each PWA trace; reviewers should record their observed
  numbers for archival.

## G. Mapping from Paper Claims to Commands

| Paper / README claim                            | Reproducer command                                                | Output field                            |
| ----------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------- |
| "EASY_BACKFILL vs FIFO p95 BSLD improvement"    | `python scripts/reproduce_paper.py --traces ctc_sp2,hpc2n,sdsc_sp2 --policies FIFO_STRICT,EASY_BACKFILL_BASELINE` | `results.json: rows[].p95_bsld`         |
| Per-trace utilization                           | same as above                                                     | `results.json: rows[].utilization`      |
| Per-trace mean wait                             | same as above                                                     | `results.json: rows[].mean_wait_sec`    |
| Tsafrir vs EASY p95 BSLD                        | `python scripts/reproduce_paper.py --policies EASY_BACKFILL_BASELINE,EASY_BACKFILL_TSAFRIR` | `results.json: rows[].p95_bsld`         |
| Cross-language Python ↔ Rust adapter parity     | `python -m pytest tests/unit/test_cross_language_adapter_parity.py -q` | pytest exit code 0                      |
| Fidelity gate behavior                          | `python -m pytest tests/unit/test_fidelity_gate.py tests/unit/test_fidelity_properties.py -q` | pytest exit code 0                      |
| Recommendation-engine constraint contract       | `python -m pytest tests/unit/test_recommend_engine.py tests/unit/test_objective_contract.py -q` | pytest exit code 0                      |

## H. Repeatability Across Hardware

The simulator uses no floating-point operations in the dispatch path
(integer arithmetic only) and tie-breaks queued jobs by
`(submit_ts, job_id)`. Numerical results are identical across
operating systems and CPU architectures for a given trace + policy +
capacity. Wall-clock simulation time may vary by ~10×; numerical
results may not.

## I. Provenance and Run Manifests

Every CLI invocation under `hpcopt simulate run` and the credibility
protocol writes a JSON-Schema-validated run manifest under
`outputs/manifests/`. Manifests include: input trace SHA-256, policy id,
capacity, runtime guard parameters, library and Rust binary versions, and
the OS / Python fingerprint. Schemas live under `schemas/`.

## J. Known Limitations for Reviewers

- The reference SWF traces are not bundled in the git repository (license).
  Reviewers must fetch them from the Parallel Workloads Archive once;
  hashes are pinned in `configs/data/reference_suite.yaml`.
- The Rust accelerator currently implements `FIFO_STRICT` and
  `EASY_BACKFILL_BASELINE` only. The new `EASY_BACKFILL_TSAFRIR` policy is
  Python-only in v2.1.0 (Rust port tracked as a follow-up).
- Modern (post-PWA) traces — Microsoft Philly, Alibaba GPU, Google 2019 —
  are not yet integrated; their addition is on the v2.x roadmap.

## K. Contact

- Author: Eren Ari (`erenari27@gmail.com`)
- Issue tracker: https://github.com/ErenAri/HPC-workload-optimizer/issues
