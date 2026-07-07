# HPCOpt Policy Leaderboard

Ranking of every evaluated scheduling policy on the reference traces, by **p95 bounded slowdown** (`(wait + runtime) / max(runtime, 60 s)`, lower is better). All rows come from the same referee: the Python reference simulator replaying the full trace under the shared metric contract. Numbers regenerate from `outputs/benchmark/policy_matrix.json` via `python scripts/build_leaderboard.py`; the matrix itself regenerates via `python scripts/policy_matrix.py`.

**Add your policy:** implement the frozen chooser contract, register it through the `hpcopt.policies` entry point, and run the matrix — see [docs/plugin-api.md](plugin-api.md). Plug-in policies compete in the same table as built-ins.

## SDSC-SP2 (54,044 jobs, capacity 128 CPUs)

| # | Policy | Kind | p95 BSLD | Mean wait (s) | Utilization | Starved | Wall (s) |
|---|---|---|---|---|---|---|---|
| 1 | SJF_BACKFILL | built-in | 144.14 | 61,673 | 80.1% | 2.6% | 19 |
| 2 | FAIRSHARE_BACKFILL | built-in | 144.44 | 86,269 | 80.3% | 2.3% | 34 |
| 3 | RL_TRAINED [^1] | built-in | 271.83 | 31,226 | 83.3% | 2.1% | 222 |
| 4 | EASY_BACKFILL_TSAFRIR | built-in | 355.67 | 19,681 | 83.4% | 1.6% | 14 |
| 5 | ML_BACKFILL_P10 | built-in | 374.57 | 17,900 | 83.4% | 1.4% | 910 |
| 6 | ML_BACKFILL_P50 | built-in | 427.30 | 24,123 | 83.4% | 2.2% | 967 |
| 7 | LJF_BACKFILL | built-in | 466.27 | 78,385 | 83.3% | 5.0% | 29 |
| 8 | CONSERVATIVE_BACKFILL_BASELINE | built-in | 550.82 | 28,708 | 83.4% | 2.7% | 904 |
| 9 | EASY_BACKFILL_BASELINE | built-in | 585.27 | 34,010 | 83.3% | 5.2% | 16 |
| 10 | UARP_BACKFILL [^2] | plugin | 631.45 | 58,291 | 82.7% | 9.7% | 24 |
| 11 | FIFO_STRICT | built-in | 56,784.93 | 1,552,128 | 76.8% | 59.6% | 1,088 |

## CTC-SP2 (77,222 jobs, capacity 512 CPUs)

| # | Policy | Kind | p95 BSLD | Mean wait (s) | Utilization | Starved | Wall (s) |
|---|---|---|---|---|---|---|---|
| 1 | RL_TRAINED [^1] | built-in | 3.85 | 714 | 55.5% | 0.0% | 227 |
| 2 | FAIRSHARE_BACKFILL | built-in | 4.15 | 834 | 55.5% | 0.0% | 25 |
| 3 | SJF_BACKFILL | built-in | 4.46 | 868 | 55.5% | 0.0% | 15 |
| 4 | EASY_BACKFILL_TSAFRIR | built-in | 8.74 | 1,666 | 55.5% | 0.0% | 21 |
| 5 | ML_BACKFILL_P10 | built-in | 10.03 | 1,411 | 55.5% | 0.0% | 1,958 |
| 6 | UARP_BACKFILL [^2] | plugin | 12.90 | 2,253 | 55.5% | 0.0% | 20 |
| 7 | LJF_BACKFILL | built-in | 13.22 | 1,907 | 55.5% | 0.0% | 17 |
| 8 | ML_BACKFILL_P50 | built-in | 15.38 | 2,150 | 55.5% | 0.0% | 1,390 |
| 9 | EASY_BACKFILL_BASELINE | built-in | 18.24 | 2,466 | 55.5% | 0.0% | 17 |
| 10 | CONSERVATIVE_BACKFILL_BASELINE | built-in | 20.89 | 2,270 | 55.5% | 0.0% | 133 |
| 11 | FIFO_STRICT | built-in | 188.05 | 6,183 | 55.5% | 0.0% | 21 |

## HPC2N (202,870 jobs, capacity 240 CPUs)

| # | Policy | Kind | p95 BSLD | Mean wait (s) | Utilization | Starved | Wall (s) |
|---|---|---|---|---|---|---|---|
| 1 | FAIRSHARE_BACKFILL | built-in | 17.82 | 7,900 | 59.6% | 0.7% | 122 |
| 2 | SJF_BACKFILL | built-in | 20.46 | 7,148 | 59.6% | 0.5% | 94 |
| 3 | RL_TRAINED [^1] | built-in | 31.09 | 8,885 | 59.6% | 0.9% | 647 |
| 4 | EASY_BACKFILL_TSAFRIR | built-in | 56.01 | 10,748 | 59.6% | 0.8% | 107 |
| 5 | UARP_BACKFILL [^2] | plugin | 79.79 | 11,039 | 59.6% | 0.8% | 104 |
| 6 | LJF_BACKFILL | built-in | 83.65 | 13,630 | 59.6% | 1.2% | 106 |
| 7 | EASY_BACKFILL_BASELINE | built-in | 113.40 | 12,369 | 59.6% | 0.9% | 107 |
| 8 | ML_BACKFILL_P10 | built-in | 114.61 | 11,488 | 59.6% | 0.8% | 3,331 |
| 9 | ML_BACKFILL_P50 | built-in | 130.88 | 12,373 | 59.6% | 0.8% | 3,310 |
| 10 | FIFO_STRICT | built-in | 286.98 | 16,189 | 59.6% | 1.1% | 88 |

Not evaluated on this trace: CONSERVATIVE_BACKFILL_BASELINE.

[^1]: RL_TRAINED: MaskablePPO, single seed, trained in-distribution on windows of the eval trace.
[^2]: UARP_BACKFILL: plug-in policy (`hpcopt.plugins.uarp`) — guard-gated, shortest-guard-first backfill.
