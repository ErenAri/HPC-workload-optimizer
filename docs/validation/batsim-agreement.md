# Batsim Cross-Validation Study

**Question answered:** why should anyone trust HPCOpt's simulation engines?
**Method:** replay identical workloads through three independent engines — HPCOpt's Rust
`sim-runner`, HPCOpt's Python reference simulator, and [Batsim](https://batsim.readthedocs.io/)
(SimGrid-based, the research-community standard) — and quantify metric agreement.

Run: July 2026. Batsim 5.0.0-rc1 (Nix, WSL2/Ubuntu), FCFS EDC (`libfcfs.so`).

## Headline result

After fixing a metric-parity defect this study uncovered (see below), **FIFO replay of all three
Parallel Workloads Archive reference traces agrees across all three engines within 0.7–3.5% on
every reported metric**, with makespan agreeing to within 0.04% and several metrics matching
exactly.

## FIFO agreement (after fix)

CTC-SP2 (77,222 jobs, 512 CPUs):

| Metric | Rust | Batsim | divergence |
|---|---|---|---|
| p95 BSLD | 188.05 | 185.64 | 1.3% |
| mean wait (s) | 6,183 | 6,102 | 1.3% |
| p95 wait (s) | 34,361 | 34,117 | 0.7% |
| utilization | 55.5% | 55.5% | ~0 |
| makespan (s) | 29,306,751 | 29,306,682 | 0.0002% |

HPC2N (202,870 jobs, 240 CPUs):

| Metric | Rust | Batsim | divergence |
|---|---|---|---|
| p95 BSLD | 286.98 | 277.06 | 3.5% |
| mean wait (s) | 16,189 | 15,843 | 2.1% |
| p95 wait (s) | 68,219 | 66,120 | 3.1% |
| utilization | 59.6% | 59.5% | ~0 |
| makespan (s) | 109,256,855 | 109,256,855 | exact |

SDSC-SP2 (54,044 jobs, 128 CPUs) — three-way:

| Metric | Rust | Python reference | Batsim | max divergence |
|---|---|---|---|---|
| p95 BSLD | 56,784.93 | 56,784.93 | 56,397.61 | 0.7% |
| mean wait (s) | 1,552,128 | 1,552,128 | 1,536,585 | 1.0% |
| p95 wait (s) | 5,103,933 | 5,103,921 | 5,077,954 | 0.5% |
| utilization | 76.8% | 76.8% | 76.0% | 1.1% |
| makespan (s) | 68,440,542 | 68,440,542 | 68,412,581 | 0.04% |

Rust and Python agree exactly (same event semantics, same metric formulas); Batsim differs by
~1% due to independent event ordering at simultaneous timestamps and delay-profile rounding.

## The defect this study caught (and why cross-validation matters)

The first run of this study showed p95 BSLD divergences of **5–63%** between the Rust engine and
Batsim, while Batsim agreed with the Python reference engine within 0.7%. The Rust engine was the
outlier. Root cause — the Rust engine computed a different quantity under the same metric name:

| | formula | runtime floor | percentile method |
|---|---|---|---|
| Python (contract) | `(wait + runtime) / max(runtime, 60)` | 60 s | linear interpolation |
| Rust (defective) | `max(1, wait / max(runtime, 10))` | 10 s | nearest-rank |

For short jobs with long waits the defective formula inflates BSLD by ~6×, which dominated the
p95 on congested traces (SDSC-SP2 FIFO: 82,865 reported vs. 56,785 true). The fix
(`rust/sim-runner/src/main.rs`, `bsld()` and `percentile_*()`) aligns Rust to the Python metric
contract; all tables above are post-fix. Historical results published from the Rust engine before
this fix overstate FIFO p95 BSLD (and therefore overstated EASY-vs-FIFO improvement on that
metric).

The existing CI cross-language parity test covered adapter *decision* equivalence but not metric
formulas — this is exactly the class of silent error that external cross-validation exists to
catch, and why HPCOpt treats "validated against Batsim" as a standing claim to maintain, not a
one-time exercise.

**Second defect caught by the same discipline (July 2026).** While freezing the public plug-in
API, a Rust-vs-Python semantic diff of the EASY shadow-time computation revealed that the Python
reference engine walked running jobs in *dispatch* order rather than end-time order when
computing head-of-line reservations (the adapter contract requires `(end_ts, job_id)` order;
`core.py` bypassed the sorted constructor). Wrong reservations could block legitimate backfills.
FIFO — everything cross-validated above — is unaffected (it never computes reservations), as are
CONSERVATIVE (sorts internally) and RL (does not consume running-job order). All EASY-family
matrix cells were regenerated after the fix (commit `6cd0101`); EASY-family numbers published
before it are slightly pessimistic (SDSC EASY p95 BSLD: 627.93 → 585.27).

## Scope and limitations

- **Policies:** FIFO/FCFS only. The Batsim FCFS EDC is the only scheduler library available in
  the current environment; EASY backfill cross-validation requires building `batsched` and is the
  next step for this study.
- **Batsim model simplifications:** jobs are replayed as delay profiles (millisecond-rounded), so
  communication/topology effects are absent by construction — appropriate for schedule-level
  agreement, not application-performance claims.
- **Residual ~1–3.5% divergence:** consistent with different tie-breaking at simultaneous
  submit/complete events and Batsim's floating-point clock vs. HPCOpt's integer clock. Not yet
  reduced further; tracked as an open item.

## Reproduction

```bash
# HPCOpt engines
python scripts/benchmark_suite.py                      # Rust engine, FIFO + EASY
python scripts/policy_matrix.py                        # Python reference engine, all policies

# Batsim (WSL/Linux with batsim + libfcfs.so on PATH/nix profile)
hpcopt simulate batsim-config --trace data/curated/SDSC-SP2-1998-4.2-cln.parquet \
  --policy FIFO_STRICT --capacity-cpus 128 --run-id batsim_sdsc_fifo
hpcopt simulate batsim-run --config outputs/simulations/batsim_sdsc_fifo_batsim_run_config.json \
  --use-wsl --no-dry-run
# batsim-run normalizes Batsim CSV output into the standard sim-report format
# and emits a candidate fidelity report vs. the observed trace.
```

Capacities: CTC-SP2 512, HPC2N 240, SDSC-SP2 128 CPUs. Traces are the cleaned SWF versions from
the Parallel Workloads Archive, hash-locked in `configs/data/reference_suite.yaml`.
