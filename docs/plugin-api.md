# HPCOpt Policy Plug-in API

HPCOpt evaluates scheduling policies as a referee: every policy sees the same
event stream, the same state snapshots, and is scored under the same metric
contract. This document freezes the interface a third-party policy programs
against. Everything here is importable from **`hpcopt.plugins`** — a plugin
must not import from `hpcopt.simulate` internals.

## The contract in one line

```python
def chooser(snapshot: SchedulerStateSnapshot) -> SchedulerDecision: ...
```

A policy is a **pure function of the snapshot**: deterministic (no wall
clock, no unseeded RNG) and stateless across calls. The harness may replay
any snapshot; identical snapshots must produce identical decisions.

## Frozen types

All types are frozen dataclasses re-exported by `hpcopt.plugins`.

`SchedulerStateSnapshot` — what the policy sees at each decision point:

| field | type | meaning |
|---|---|---|
| `clock_ts` | int | current simulation time (epoch seconds) |
| `capacity_cpus` | int | total machine CPUs |
| `free_cpus` | int | CPUs free right now |
| `queued_jobs` | tuple[AdapterQueuedJob] | sorted by (submit_ts, job_id) |
| `running_jobs` | tuple[AdapterRunningJob] | sorted by (end_ts, job_id) |

`AdapterQueuedJob`: `job_id`, `submit_ts`, `requested_cpus`,
`runtime_estimate_sec` (the point estimate the harness provides — the user's
requested walltime by default, an ML/Tsafrir prediction under those
configurations), optional `runtime_p90_sec` / `runtime_guard_sec`
(pessimistic bounds, populated when a quantile predictor is active),
`estimate_source`, optional `priority_score`.

`AdapterRunningJob`: `job_id`, `end_ts` (estimated), `allocated_cpus`.

`SchedulerDecision`: `policy_id`, optional `reservation_ts` (head-of-line
reservation, for audit), `decisions` — a tuple of `DispatchDecision`
(`job_id`, `requested_cpus`, `runtime_estimate_sec`,
`estimated_completion_ts`, `reason`). Every decision must reference a
currently queued job; the engine independently re-validates that each
dispatched job fits in free CPUs (a decision that does not fit is dropped,
never partially applied).

Helper: `earliest_start_for(snapshot, requested_cpus)` returns the earliest
timestamp at which that many CPUs can be free (the EASY shadow time), or
`NEVER_TS` if the request exceeds capacity.

## Registering a policy

```python
from hpcopt.plugins import SchedulerDecision, SchedulerStateSnapshot, register_policy

@register_policy("MY_POLICY", author="you@example.org", description="...", version="1.0")
def choose_my_policy(snapshot: SchedulerStateSnapshot) -> SchedulerDecision:
    ...
```

Policy ids are UPPER_SNAKE_CASE, 3–64 chars, and may not shadow a built-in.
Once registered, the id works everywhere a built-in id does:

```
hpcopt simulate run --policy MY_POLICY --trace data/curated/<trace>.parquet ...
python scripts/policy_matrix.py --policies MY_POLICY
```

To ship a policy as a separate pip package, expose it through the
`hpcopt.policies` entry-point group — HPCOpt discovers it automatically at
first policy lookup:

```toml
[project.entry-points."hpcopt.policies"]
MY_POLICY = "my_package.my_module"   # module registers at import time
```

The bundled UARP plugin (`hpcopt.plugins.uarp`, registered through this
repo's own pyproject entry point) is the reference implementation: an EASY
skeleton whose backfill gate uses the pessimistic guard runtime and packs
shortest-guard-first, written entirely against this public API.

## Metric contract (how you are scored)

The headline metric is **p95 bounded slowdown**:
`BSLD = (wait + runtime) / max(runtime, 60s)`, percentile computed with
numpy linear interpolation. The matrix also reports mean/p95 wait,
utilization, starvation rate (wait > 48h), and Jain fairness. All metrics
are computed by the harness from the completed schedule — a policy cannot
influence its own scoring. Runs with invariant violations are flagged.

Results land in `outputs/benchmark/policy_matrix.json`; regenerate the
public leaderboard with `python scripts/build_leaderboard.py`.
