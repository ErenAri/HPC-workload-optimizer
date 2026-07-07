from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpcopt.simulate.objective import (
    compute_weighted_analysis_score,
    evaluate_constraint_contract,
)
from hpcopt.utils.io import write_json


def _load_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object json: {path}")
    return payload


def is_dominated(point: list[float], other: list[float]) -> bool:
    """Check if `point` is dominated by `other` (all objectives to maximize).

    Public: benchmark studies (e.g. scripts/pm100_cap_pareto.py) reuse this
    dominance test so every Pareto frontier in the project shares one
    definition.
    """
    at_least_one_better = False
    for p, o in zip(point, other):
        if o < p:
            return False
        if o > p:
            at_least_one_better = True
    return at_least_one_better


def generate_pareto_recommendation(
    baseline_report_path: Path,
    candidate_report_paths: list[Path],
    out_path: Path,
    objectives: list[str] | None = None,
) -> RecommendationResult:
    """Generate Pareto frontier over multiple objectives.

    Default objectives: maximize delta_p95_bsld, maximize delta_utilization, minimize fairness_dev.
    """
    if objectives is None:
        objectives = ["delta_p95_bsld", "delta_utilization", "neg_fairness_dev"]

    baseline = _load_json(baseline_report_path)
    baseline_obj = _extract_objective(baseline)

    candidates: list[dict[str, Any]] = []
    for path in candidate_report_paths:
        cand = _load_json(path)
        cand_obj = _extract_objective(cand)

        delta_p95 = float(baseline_obj["p95_bsld"] - cand_obj["p95_bsld"])
        delta_util = float(cand_obj["utilization_cpu"] - baseline_obj["utilization_cpu"])
        neg_fairness = -float(cand_obj["fairness_dev"])

        obj_values = {
            "delta_p95_bsld": delta_p95,
            "delta_utilization": delta_util,
            "neg_fairness_dev": neg_fairness,
        }

        candidates.append(
            {
                "candidate_report_path": str(path),
                "policy_id": cand.get("policy_id"),
                "run_id": cand.get("run_id"),
                "objective_metrics": cand_obj,
                "pareto_objectives": obj_values,
            }
        )

    # Compute Pareto frontier
    n = len(candidates)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        point_i = [candidates[i]["pareto_objectives"].get(o, 0) for o in objectives]
        for j in range(n):
            if i == j or dominated[j]:
                continue
            point_j = [candidates[j]["pareto_objectives"].get(o, 0) for o in objectives]
            if is_dominated(point_i, point_j):
                dominated[i] = True
                break

    for i, cand in enumerate(candidates):
        cand["pareto_dominated"] = dominated[i]
        cand["pareto_frontier"] = not dominated[i]

    pareto_front = [c for c in candidates if c["pareto_frontier"]]
    dominated_set = [c for c in candidates if c["pareto_dominated"]]

    payload = {
        "mode": "pareto",
        "objectives": objectives,
        "baseline": {
            "policy_id": baseline.get("policy_id"),
            "objective_metrics": baseline_obj,
        },
        "pareto_frontier": pareto_front,
        "dominated": dominated_set,
        "frontier_size": len(pareto_front),
        "total_candidates": n,
    }

    write_json(out_path, payload)
    return RecommendationResult(report_path=out_path, payload=payload)


@dataclass(frozen=True)
class RecommendationResult:
    report_path: Path
    payload: dict[str, Any]


def _extract_objective(report: dict[str, Any]) -> dict[str, float]:
    obj = report.get("objective_metrics")
    if not isinstance(obj, dict):
        raise ValueError("simulation report missing objective_metrics")
    required = {"p95_bsld", "utilization_cpu", "fairness_dev", "jain", "starved_rate"}
    missing = [key for key in required if key not in obj]
    if missing:
        raise ValueError(f"objective_metrics missing fields: {missing}")
    return {key: float(obj[key]) for key in obj.keys()}


def workload_regime_analysis(
    baseline_objective: dict[str, float],
    candidate_objective: dict[str, float],
    trace_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify workload regime and produce regime-specific explanations."""
    regime = "balanced"
    regime_explanation = "Standard workload with moderate congestion and runtime distribution."
    sensitivity_hint = "Consider adjusting runtime_guard_k to balance prediction confidence vs backfill opportunity."
    suggested_next_step = "Run sensitivity sweep with hpcopt analysis sensitivity-sweep."

    if trace_profile is not None:
        heavy_tail = trace_profile.get("runtime_heavy_tail", {})
        tail_ratio = heavy_tail.get("tail_ratio_p99_over_p50")
        congestion = trace_profile.get("congestion_regime", {})
        queue_mean = congestion.get("queue_len_mean", 0)
        user_skew = trace_profile.get("user_skew", {})
        top_user_share = user_skew.get("top_user_share", 0)

        if tail_ratio is not None and tail_ratio > 50.0:
            regime = "heavy_tail"
            regime_explanation = (
                f"Workload is heavily tail-distributed (p99/p50 ratio={tail_ratio:.1f}). "
                "ML predictions may struggle with extreme outliers, limiting backfill gains."
            )
            sensitivity_hint = "Higher guard_k values provide safety margin for tail jobs."
            suggested_next_step = "Review feature importance for tail-correlated features."
        elif queue_mean is not None and queue_mean < 2.0:
            regime = "low_congestion"
            regime_explanation = (
                f"Low queue congestion (mean queue length={queue_mean:.1f}). "
                "Little room for backfill improvement when queue is rarely deep."
            )
            sensitivity_hint = (
                "Benefits emerge only under higher load; consider testing with more concurrent submissions."
            )
            suggested_next_step = "Validate with stress scenarios: hpcopt stress gen --scenario low_congestion."
        elif top_user_share > 0.5:
            regime = "user_skew"
            regime_explanation = (
                f"Dominant user controls {top_user_share:.0%} of jobs. "
                "Fairness constraints may block improvement for the majority."
            )
            sensitivity_hint = "Consider relaxing fairness_dev_delta_max if dominant user pattern is expected."
            suggested_next_step = "Run user-skew stress scenario for detailed fairness analysis."

    # Compute deltas
    delta_p95 = baseline_objective.get("p95_bsld", 0) - candidate_objective.get("p95_bsld", 0)
    delta_util = candidate_objective.get("utilization_cpu", 0) - baseline_objective.get("utilization_cpu", 0)

    return {
        "workload_regime": regime,
        "regime_explanation": regime_explanation,
        "sensitivity_hint": sensitivity_hint,
        "suggested_next_step": suggested_next_step,
        "delta_p95_bsld": float(delta_p95),
        "delta_utilization": float(delta_util),
    }


def _fidelity_gate_ok(fidelity_report: dict[str, Any] | None) -> tuple[bool, str | None]:
    if fidelity_report is None:
        return True, None
    status = str(fidelity_report.get("status", "")).lower()
    if status == "pass":
        return True, None
    return False, "fidelity_failed"


def generate_recommendation_report(
    baseline_report_path: Path,
    candidate_report_paths: list[Path],
    out_path: Path,
    fidelity_report_path: Path | None = None,
    w1: float = 1.0,
    w2: float = 0.3,
    w3: float = 2.0,
    trace_profile_path: Path | None = None,
) -> RecommendationResult:
    baseline = _load_json(baseline_report_path)
    fidelity = _load_json(fidelity_report_path) if fidelity_report_path else None
    trace_profile = _load_json(trace_profile_path) if trace_profile_path else None
    fidelity_ok, fidelity_reason = _fidelity_gate_ok(fidelity)
    baseline_objective = _extract_objective(baseline)
    baseline_policy = str(baseline.get("policy_id", "baseline"))

    candidates: list[dict[str, Any]] = []
    for path in candidate_report_paths:
        cand = _load_json(path)
        cand_obj = _extract_objective(cand)
        score = compute_weighted_analysis_score(
            candidate=cand_obj,
            baseline=baseline_objective,
            w1=w1,
            w2=w2,
            w3=w3,
        )
        constraints = evaluate_constraint_contract(
            candidate=cand_obj,
            baseline=baseline_objective,
        )
        primary_improved = score["delta_p95_bsld"] > 0.0
        accepted = fidelity_ok and constraints["constraints_passed"] and primary_improved
        rejection_reasons: list[str] = []
        if not fidelity_ok and fidelity_reason:
            rejection_reasons.append(fidelity_reason)
        if not constraints["constraints_passed"]:
            rejection_reasons.extend(constraints["violations"])
        if not primary_improved:
            rejection_reasons.append("primary_kpi_not_improved")

        candidates.append(
            {
                "candidate_report_path": str(path),
                "policy_id": cand.get("policy_id"),
                "run_id": cand.get("run_id"),
                "objective_metrics": cand_obj,
                "fallback_accounting": cand.get("fallback_accounting"),
                "score": score,
                "constraints": constraints,
                "primary_improved": primary_improved,
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
            }
        )

    candidates_sorted = sorted(
        candidates,
        key=lambda item: (item["accepted"], item["score"]["score"]),
        reverse=True,
    )
    winner = candidates_sorted[0] if candidates_sorted else None
    accepted_winner = winner if winner and winner["accepted"] else None

    no_improvement_narrative = None
    if accepted_winner is None and winner is not None:
        cand_obj = winner.get("objective_metrics", {})
        regime_analysis = workload_regime_analysis(
            baseline_objective=baseline_objective,
            candidate_objective=cand_obj,
            trace_profile=trace_profile,
        )
        no_improvement_narrative = {
            "summary": "No candidate passed guardrails and primary objective improvement criteria.",
            "likely_causes": winner["rejection_reasons"],
            "workload_regime": regime_analysis["workload_regime"],
            "regime_explanation": regime_analysis["regime_explanation"],
            "sensitivity_hint": regime_analysis["sensitivity_hint"],
            "suggested_next_step": regime_analysis["suggested_next_step"],
        }

    payload = {
        "status": "accepted" if accepted_winner else "blocked",
        "baseline": {
            "policy_id": baseline_policy,
            "run_id": baseline.get("run_id"),
            "objective_metrics": baseline_objective,
            "report_path": str(baseline_report_path),
        },
        "fidelity_status": fidelity.get("status") if fidelity else "not_provided",
        "candidates": candidates_sorted,
        "selected_recommendation": accepted_winner,
        "failure_modes": [
            {
                "policy_id": item["policy_id"],
                "run_id": item["run_id"],
                "rejection_reasons": item["rejection_reasons"],
            }
            for item in candidates_sorted
            if not item["accepted"]
        ],
        "no_improvement_narrative": no_improvement_narrative,
    }
    write_json(out_path, payload)
    return RecommendationResult(report_path=out_path, payload=payload)
