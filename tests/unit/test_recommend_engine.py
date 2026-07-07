"""Tests for the recommendation engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hpcopt.recommend.engine import (
    RecommendationResult,
    _extract_objective,
    _fidelity_gate_ok,
    generate_pareto_recommendation,
    generate_recommendation_report,
    is_dominated,
    workload_regime_analysis,
)


def _make_sim_report(
    path: Path, *, policy_id: str, p95_bsld: float, utilization_cpu: float, fairness_dev: float
) -> Path:
    payload = {
        "policy_id": policy_id,
        "run_id": f"run_{policy_id}",
        "objective_metrics": {
            "p95_bsld": p95_bsld,
            "utilization_cpu": utilization_cpu,
            "fairness_dev": fairness_dev,
            "jain": 0.95,
            "starved_rate": 0.01,
            "p95_wait_sec": 300,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_is_dominated() -> None:
    assert is_dominated([1, 1], [2, 2]) is True
    assert is_dominated([2, 2], [1, 1]) is False
    assert is_dominated([1, 2], [2, 1]) is False


def test_extract_objective_valid() -> None:
    report = {
        "objective_metrics": {
            "p95_bsld": 2.0,
            "utilization_cpu": 0.7,
            "fairness_dev": 0.1,
            "jain": 0.95,
            "starved_rate": 0.01,
        }
    }
    obj = _extract_objective(report)
    assert obj["p95_bsld"] == 2.0


def test_extract_objective_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing"):
        _extract_objective({"objective_metrics": {"p95_bsld": 1.0}})


def test_extract_objective_missing_key() -> None:
    with pytest.raises(ValueError, match="missing objective_metrics"):
        _extract_objective({"other": {}})


def test_fidelity_gate_ok_none() -> None:
    assert _fidelity_gate_ok(None) == (True, None)


def test_fidelity_gate_ok_pass() -> None:
    assert _fidelity_gate_ok({"status": "pass"}) == (True, None)


def test_fidelity_gate_ok_fail() -> None:
    ok, reason = _fidelity_gate_ok({"status": "fail"})
    assert ok is False
    assert reason == "fidelity_failed"


def test_regime_balanced() -> None:
    result = workload_regime_analysis(
        {"p95_bsld": 2.0, "utilization_cpu": 0.7},
        {"p95_bsld": 1.8, "utilization_cpu": 0.72},
    )
    assert result["workload_regime"] == "balanced"


def test_regime_heavy_tail() -> None:
    result = workload_regime_analysis(
        {"p95_bsld": 2.0, "utilization_cpu": 0.7},
        {"p95_bsld": 2.1, "utilization_cpu": 0.69},
        trace_profile={"runtime_heavy_tail": {"tail_ratio_p99_over_p50": 100.0}},
    )
    assert result["workload_regime"] == "heavy_tail"


def test_regime_low_congestion() -> None:
    result = workload_regime_analysis(
        {"p95_bsld": 2.0, "utilization_cpu": 0.7},
        {"p95_bsld": 2.0, "utilization_cpu": 0.7},
        trace_profile={"congestion_regime": {"queue_len_mean": 1.0}},
    )
    assert result["workload_regime"] == "low_congestion"


def test_regime_user_skew() -> None:
    result = workload_regime_analysis(
        {"p95_bsld": 2.0, "utilization_cpu": 0.7},
        {"p95_bsld": 2.0, "utilization_cpu": 0.7},
        trace_profile={
            "user_skew": {"top_user_share": 0.8},
            "congestion_regime": {"queue_len_mean": 10.0},
        },
    )
    assert result["workload_regime"] == "user_skew"


def test_generate_recommendation_accepted(tmp_path: Path) -> None:
    baseline = _make_sim_report(
        tmp_path / "baseline.json",
        policy_id="FIFO",
        p95_bsld=3.0,
        utilization_cpu=0.65,
        fairness_dev=0.1,
    )
    candidate = _make_sim_report(
        tmp_path / "candidate.json",
        policy_id="ML",
        p95_bsld=2.0,
        utilization_cpu=0.70,
        fairness_dev=0.1,
    )
    out = tmp_path / "rec.json"
    result = generate_recommendation_report(
        baseline_report_path=baseline,
        candidate_report_paths=[candidate],
        out_path=out,
    )
    assert isinstance(result, RecommendationResult)
    assert result.payload["status"] == "accepted"
    assert result.payload["selected_recommendation"] is not None


def test_generate_recommendation_blocked(tmp_path: Path) -> None:
    baseline = _make_sim_report(
        tmp_path / "baseline.json",
        policy_id="FIFO",
        p95_bsld=1.0,
        utilization_cpu=0.90,
        fairness_dev=0.05,
    )
    candidate = _make_sim_report(
        tmp_path / "candidate.json",
        policy_id="ML",
        p95_bsld=2.0,
        utilization_cpu=0.60,
        fairness_dev=0.3,
    )
    out = tmp_path / "rec.json"
    result = generate_recommendation_report(
        baseline_report_path=baseline,
        candidate_report_paths=[candidate],
        out_path=out,
    )
    assert result.payload["status"] == "blocked"
    assert result.payload["no_improvement_narrative"] is not None


def test_generate_pareto_recommendation(tmp_path: Path) -> None:
    baseline = _make_sim_report(
        tmp_path / "baseline.json",
        policy_id="FIFO",
        p95_bsld=3.0,
        utilization_cpu=0.65,
        fairness_dev=0.1,
    )
    c1 = _make_sim_report(
        tmp_path / "c1.json",
        policy_id="ML_A",
        p95_bsld=2.0,
        utilization_cpu=0.70,
        fairness_dev=0.12,
    )
    c2 = _make_sim_report(
        tmp_path / "c2.json",
        policy_id="ML_B",
        p95_bsld=2.5,
        utilization_cpu=0.75,
        fairness_dev=0.08,
    )
    out = tmp_path / "pareto.json"
    result = generate_pareto_recommendation(
        baseline_report_path=baseline,
        candidate_report_paths=[c1, c2],
        out_path=out,
    )
    assert result.payload["mode"] == "pareto"
    assert result.payload["total_candidates"] == 2
    assert result.payload["frontier_size"] >= 1
