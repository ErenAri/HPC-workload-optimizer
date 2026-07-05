"""Tests for the what-if analysis engine and CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from hpcopt.cli.main import app
from hpcopt.whatif import (
    SLURM_SCHEDULER_POLICY_MAP,
    infer_capacity_cpus,
    render_whatif_markdown,
    run_whatif,
)


def _congested_trace(n_jobs: int = 60) -> pd.DataFrame:
    """Synthetic congested workload where scheduling order matters.

    A burst of same-time submissions with mixed runtimes/widths, plus observed
    start/end timestamps so fidelity comparison against 'observed' works.
    """
    rows = []
    clock = 1_000_000
    for i in range(n_jobs):
        runtime = 300 if i % 3 else 7200
        cpus = 4 if i % 2 else 16
        submit = clock + (i // 6) * 120
        observed_start = submit + 600 + (i % 7) * 90
        rows.append(
            {
                "job_id": i + 1,
                "submit_ts": submit,
                "start_ts": observed_start,
                "end_ts": observed_start + runtime,
                "runtime_actual_sec": runtime,
                "runtime_requested_sec": runtime * 2,
                "requested_cpus": cpus,
                "allocated_cpus": cpus,
                "user_id": i % 5,
                "group_id": 1,
                "queue_id": 1,
                "partition_id": 1,
                "requested_mem": None,
            }
        )
    return pd.DataFrame(rows)


def test_infer_capacity_cpus_peak_concurrency() -> None:
    df = pd.DataFrame(
        {
            "start_ts": [0, 0, 50, 200],
            "end_ts": [100, 100, 150, 300],
            "requested_cpus": [8, 8, 4, 2],
        }
    )
    # Jobs 1+2+3 overlap in [50, 100): 8 + 8 + 4 = 20.
    assert infer_capacity_cpus(df) == 20


def test_run_whatif_produces_graded_report(tmp_path: Path) -> None:
    trace_df = _congested_trace()
    result = run_whatif(
        trace_df=trace_df,
        baseline_policy="FIFO_STRICT",
        candidate_policy="EASY_BACKFILL_BASELINE",
        out_dir=tmp_path,
        run_id="whatif_test",
        capacity_cpus=32,
    )
    assert result.verdict in {"improvement", "regression", "no_material_change", "blocked_constraints"}
    assert result.confidence in {"high", "low"}
    assert result.report_path.exists()
    assert result.markdown_path.exists()

    payload = result.payload
    assert payload["config"]["capacity_cpus"] == 32
    assert payload["config"]["capacity_inferred_from_trace"] is False
    assert payload["primary_kpi"]["metric"] == "p95_bsld"
    assert payload["unmodeled_caveats"]
    assert payload["invariants"]["baseline_violations"] == 0
    assert payload["invariants"]["candidate_violations"] == 0

    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "What-If Report" in markdown
    assert "Not Modeled" in markdown
    assert "p95 bounded slowdown" in markdown


def test_run_whatif_infers_capacity(tmp_path: Path) -> None:
    trace_df = _congested_trace()
    result = run_whatif(
        trace_df=trace_df,
        baseline_policy="EASY_BACKFILL_BASELINE",
        candidate_policy="SJF_BACKFILL",
        out_dir=tmp_path,
    )
    assert result.payload["config"]["capacity_inferred_from_trace"] is True
    assert result.payload["config"]["capacity_cpus"] >= 1


def test_run_whatif_rejects_unknown_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported candidate policy"):
        run_whatif(
            trace_df=_congested_trace(),
            baseline_policy="EASY_BACKFILL_BASELINE",
            candidate_policy="NOT_A_POLICY",
            out_dir=tmp_path,
        )


def test_render_markdown_covers_all_verdicts() -> None:
    base_payload = {
        "run_id": "r",
        "confidence": "high",
        "config": {
            "baseline_policy": "EASY_BACKFILL_BASELINE",
            "candidate_policy": "SJF_BACKFILL",
            "capacity_cpus": 64,
            "capacity_inferred_from_trace": True,
            "candidate_capacity_cpus": 64,
            "job_count": 10,
        },
        "primary_kpi": {"baseline": 10.0, "candidate": 5.0, "relative_improvement": 0.5},
        "metric_deltas": {},
        "constraint_contract": {"constraints_passed": True, "violations": []},
        "baseline_fidelity": {"status": "pass"},
        "invariants": {"baseline_violations": 0, "candidate_violations": 0},
        "unmodeled_caveats": ["x"],
    }
    for verdict in ("improvement", "regression", "no_material_change", "blocked_constraints"):
        text = render_whatif_markdown({**base_payload, "verdict": verdict})
        assert "What-If Report" in text


def test_whatif_cli_with_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.parquet"
    _congested_trace().to_parquet(trace_path, index=False)
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "whatif",
            "run",
            "--trace",
            str(trace_path),
            "--candidate-policy",
            "SJF_BACKFILL",
            "--capacity-cpus",
            "32",
            "--out",
            str(out_dir),
            "--run-id",
            "cli_whatif",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Verdict:" in result.output
    assert (out_dir / "cli_whatif_whatif_report.md").exists()
    assert (out_dir / "cli_whatif_whatif_manifest.json").exists()


def test_whatif_cli_slurm_scheduler_mapping(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.parquet"
    _congested_trace().to_parquet(trace_path, index=False)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "whatif",
            "run",
            "--trace",
            str(trace_path),
            "--slurm-scheduler-type",
            "sched/builtin",
            "--capacity-cpus",
            "32",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert SLURM_SCHEDULER_POLICY_MAP["sched/builtin"] == "FIFO_STRICT"


def test_whatif_cli_requires_exactly_one_input(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["whatif", "run", "--candidate-policy", "SJF_BACKFILL"])
    assert result.exit_code != 0
