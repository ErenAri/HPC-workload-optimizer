"""What-if analysis engine: fast, fidelity-graded scheduler change evaluation.

Answers the operator question "what happens to queueing behavior if I change
the scheduler policy (or capacity) on my cluster?" from a window of accounting
data, in seconds instead of production weeks.

The pipeline: replay the observed workload under a baseline policy that
approximates the current configuration, grade how well that replay reproduces
observed behavior (the fidelity gate — this is the confidence attached to
every downstream number), replay the proposed change, and report constraint-
checked deltas. No claim is emitted without its confidence grade.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from hpcopt.simulate.core import SUPPORTED_POLICIES, run_simulation_from_trace
from hpcopt.simulate.fidelity import run_candidate_fidelity_report
from hpcopt.simulate.objective import evaluate_constraint_contract
from hpcopt.utils.io import ensure_dir, write_json

# Mapping from Slurm SchedulerType values to the closest HPCOpt policy
# contract. This is an approximation and is documented as such in every
# report; parameters like PriorityWeight* / bf_window are NOT modeled yet.
SLURM_SCHEDULER_POLICY_MAP = {
    "sched/builtin": "FIFO_STRICT",
    "sched/backfill": "EASY_BACKFILL_BASELINE",
}

# Relative p95 BSLD reduction below which a delta is reported as noise.
_KPI_IMPROVEMENT_EPSILON = 0.01

UNMODELED_CAVEATS = [
    "Node topology and per-node resources are not modeled (single CPU pool).",
    "Memory, GPUs, and licenses are not modeled as schedulable resources.",
    "Preemption, job dependencies, QOS limits, and reservations are not modeled.",
    "Slurm priority weights and backfill tuning parameters (bf_window, bf_interval) "
    "map only approximately onto policy contracts.",
    "Job resubmission behavior under a different scheduler is not modeled (open-loop replay).",
]


@dataclass(frozen=True)
class WhatIfResult:
    run_id: str
    verdict: str
    confidence: str
    report_path: Path
    markdown_path: Path
    payload: dict[str, Any]


def infer_capacity_cpus(trace_df: pd.DataFrame) -> int:
    """Estimate cluster capacity as the peak concurrent allocated CPUs observed."""
    starts = pd.to_numeric(trace_df["start_ts"], errors="coerce")
    ends = pd.to_numeric(trace_df["end_ts"], errors="coerce")
    cpus = pd.to_numeric(trace_df["requested_cpus"], errors="coerce").fillna(1)
    mask = starts.notna() & ends.notna() & (ends > starts)
    events: list[tuple[int, int]] = []
    for s, e, c in zip(starts[mask].astype(int), ends[mask].astype(int), cpus[mask].astype(int), strict=True):
        events.append((s, int(c)))
        events.append((e, -int(c)))
    events.sort()
    peak = current = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return max(1, peak)


def _metric_deltas(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, dict[str, float]]:
    keys = ["p95_bsld", "mean_wait_sec", "p95_wait_sec", "utilization_cpu", "makespan_sec", "starved_rate", "jain"]
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        base = float(baseline.get(key, 0.0))
        cand = float(candidate.get(key, 0.0))
        out[key] = {
            "baseline": base,
            "candidate": cand,
            "delta": cand - base,
            "delta_pct": ((cand - base) / abs(base) * 100.0) if abs(base) > 1e-12 else 0.0,
        }
    return out


def run_whatif(
    trace_df: pd.DataFrame,
    baseline_policy: str,
    candidate_policy: str,
    out_dir: Path,
    run_id: str | None = None,
    capacity_cpus: int | None = None,
    candidate_capacity_cpus: int | None = None,
    runtime_predictor: Any | None = None,
    fidelity_config_path: Path | None = None,
    starvation_wait_cap_sec: int = 172800,
) -> WhatIfResult:
    if baseline_policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported baseline policy: {baseline_policy}")
    if candidate_policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported candidate policy: {candidate_policy}")

    ensure_dir(out_dir)
    resolved_run_id = run_id or f"whatif_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    capacity = int(capacity_cpus) if capacity_cpus else infer_capacity_cpus(trace_df)
    capacity_inferred = capacity_cpus is None
    candidate_capacity = int(candidate_capacity_cpus) if candidate_capacity_cpus else capacity

    baseline_sim = run_simulation_from_trace(
        trace_df=trace_df,
        policy_id=baseline_policy,
        capacity_cpus=capacity,
        run_id=f"{resolved_run_id}_baseline",
        starvation_wait_cap_sec=starvation_wait_cap_sec,
    )
    candidate_sim = run_simulation_from_trace(
        trace_df=trace_df,
        policy_id=candidate_policy,
        capacity_cpus=candidate_capacity,
        run_id=f"{resolved_run_id}_candidate",
        runtime_predictor=runtime_predictor,
        starvation_wait_cap_sec=starvation_wait_cap_sec,
    )

    fidelity = run_candidate_fidelity_report(
        trace_df=trace_df,
        simulated_jobs=baseline_sim.jobs_df,
        simulated_queue=baseline_sim.queue_series_df,
        capacity_cpus=capacity,
        out_path=out_dir / f"{resolved_run_id}_baseline_fidelity.json",
        run_id=resolved_run_id,
        policy_id=baseline_policy,
        config_path=fidelity_config_path,
    )
    confidence = "high" if fidelity.status == "pass" else "low"

    deltas = _metric_deltas(baseline_sim.objective_metrics, candidate_sim.objective_metrics)
    constraints = evaluate_constraint_contract(
        candidate=candidate_sim.objective_metrics,
        baseline=baseline_sim.objective_metrics,
    )

    base_bsld = float(baseline_sim.objective_metrics["p95_bsld"])
    cand_bsld = float(candidate_sim.objective_metrics["p95_bsld"])
    rel_improvement = (base_bsld - cand_bsld) / base_bsld if base_bsld > 1e-12 else 0.0

    if not constraints["constraints_passed"]:
        verdict = "blocked_constraints"
    elif rel_improvement > _KPI_IMPROVEMENT_EPSILON:
        verdict = "improvement"
    elif rel_improvement < -_KPI_IMPROVEMENT_EPSILON:
        verdict = "regression"
    else:
        verdict = "no_material_change"

    payload: dict[str, Any] = {
        "run_id": resolved_run_id,
        "timestamp_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "verdict": verdict,
        "confidence": confidence,
        "config": {
            "baseline_policy": baseline_policy,
            "candidate_policy": candidate_policy,
            "capacity_cpus": capacity,
            "capacity_inferred_from_trace": capacity_inferred,
            "candidate_capacity_cpus": candidate_capacity,
            "job_count": int(len(baseline_sim.jobs_df)),
        },
        "primary_kpi": {
            "metric": "p95_bsld",
            "baseline": base_bsld,
            "candidate": cand_bsld,
            "relative_improvement": rel_improvement,
            "epsilon": _KPI_IMPROVEMENT_EPSILON,
        },
        "metric_deltas": deltas,
        "constraint_contract": constraints,
        "baseline_fidelity": {
            "status": fidelity.status,
            "report": str(fidelity.report_path),
            "core_metric_divergence": fidelity.report.get("aggregate_metrics", {}),
        },
        "invariants": {
            "baseline_violations": int(baseline_sim.invariant_report.get("violation_count", 0)),
            "candidate_violations": int(candidate_sim.invariant_report.get("violation_count", 0)),
        },
        "unmodeled_caveats": UNMODELED_CAVEATS,
    }

    report_path = out_dir / f"{resolved_run_id}_whatif_report.json"
    write_json(report_path, payload)
    markdown_path = out_dir / f"{resolved_run_id}_whatif_report.md"
    markdown_path.write_text(render_whatif_markdown(payload), encoding="utf-8")

    return WhatIfResult(
        run_id=resolved_run_id,
        verdict=verdict,
        confidence=confidence,
        report_path=report_path,
        markdown_path=markdown_path,
        payload=payload,
    )


def render_whatif_markdown(payload: dict[str, Any]) -> str:
    config = payload["config"]
    kpi = payload["primary_kpi"]
    verdict = payload["verdict"]
    confidence = payload["confidence"]

    verdict_line = {
        "improvement": "**IMPROVEMENT** — the candidate improves the primary KPI under all constraints.",
        "regression": "**REGRESSION** — the candidate degrades the primary KPI; do not apply.",
        "no_material_change": "**NO MATERIAL CHANGE** — the delta is within noise.",
        "blocked_constraints": "**BLOCKED** — fairness/starvation constraints failed; the KPI delta is not claimable.",
    }[verdict]

    lines = [
        f"# What-If Report: `{payload['run_id']}`",
        "",
        verdict_line,
        "",
        f"Simulation confidence: **{confidence}** "
        f"(baseline replay fidelity vs. observed trace: `{payload['baseline_fidelity']['status']}`).",
        "",
        "## Scenario",
        "",
        f"- Baseline policy: `{config['baseline_policy']}` @ {config['capacity_cpus']} CPUs"
        + (" (capacity inferred from trace)" if config["capacity_inferred_from_trace"] else ""),
        f"- Candidate policy: `{config['candidate_policy']}` @ {config['candidate_capacity_cpus']} CPUs",
        f"- Jobs replayed: {config['job_count']:,}",
        "",
        "## Primary KPI (p95 bounded slowdown)",
        "",
        f"- Baseline: {kpi['baseline']:,.2f}",
        f"- Candidate: {kpi['candidate']:,.2f}",
        f"- Relative improvement: {kpi['relative_improvement']:+.1%}",
        "",
        "## Metric Deltas",
        "",
        "| Metric | Baseline | Candidate | Delta | Delta % |",
        "|---|---|---|---|---|",
    ]
    for key, row in payload["metric_deltas"].items():
        lines.append(
            f"| {key} | {row['baseline']:,.3f} | {row['candidate']:,.3f} "
            f"| {row['delta']:+,.3f} | {row['delta_pct']:+.1f}% |"
        )
    constraints = payload["constraint_contract"]
    lines += [
        "",
        "## Constraint Contract",
        "",
        f"- Passed: {constraints['constraints_passed']}",
        f"- Violations: {constraints['violations'] or 'none'}",
        "",
        "## Not Modeled (read before acting)",
        "",
    ]
    lines += [f"- {caveat}" for caveat in payload["unmodeled_caveats"]]
    lines += [
        "",
        f"Invariant violations: baseline={payload['invariants']['baseline_violations']}, "
        f"candidate={payload['invariants']['candidate_violations']}. "
        f"Full JSON: `{payload['run_id']}_whatif_report.json`.",
        "",
    ]
    return "\n".join(lines)
