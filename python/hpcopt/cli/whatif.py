"""CLI for what-if analysis: evaluate a scheduler change from accounting data."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import typer

from hpcopt.artifacts.manifest import build_manifest, write_manifest
from hpcopt.models.runtime_quantile import RuntimeQuantilePredictor, resolve_runtime_model_dir
from hpcopt.simulate.core import SUPPORTED_POLICIES
from hpcopt.utils.io import ensure_dir
from hpcopt.whatif import SLURM_SCHEDULER_POLICY_MAP, run_whatif

whatif_app = typer.Typer(help="What-if analysis: evaluate scheduler changes before applying them")

_ML_POLICIES = {"ML_BACKFILL_P50", "ML_BACKFILL_P10"}


@whatif_app.command("run")
def whatif_run_cmd(
    sacct: Path | None = typer.Option(
        None,
        exists=True,
        readable=True,
        help="Raw `sacct --parsable2` dump covering the analysis window",
    ),
    trace: Path | None = typer.Option(
        None,
        exists=True,
        readable=True,
        help="Canonical parquet dataset (alternative to --sacct)",
    ),
    candidate_policy: str | None = typer.Option(
        None,
        help=f"Proposed policy; one of {sorted(SUPPORTED_POLICIES)}",
    ),
    slurm_scheduler_type: str | None = typer.Option(
        None,
        help=f"Alternative to --candidate-policy: a Slurm SchedulerType value {sorted(SLURM_SCHEDULER_POLICY_MAP)}",
    ),
    baseline_policy: str = typer.Option(
        "EASY_BACKFILL_BASELINE",
        help="Policy approximating the current cluster configuration",
    ),
    capacity_cpus: int | None = typer.Option(
        None, min=1, help="Cluster CPU capacity; inferred from peak concurrency when omitted"
    ),
    candidate_capacity_cpus: int | None = typer.Option(
        None, min=1, help="Capacity what-if: candidate replays with this capacity"
    ),
    runtime_model_dir: Path | None = typer.Option(None, help="Runtime quantile model dir (ML policies)"),
    fidelity_config: Path | None = typer.Option(
        Path("configs/simulation/fidelity_gate.yaml"),
        help="Fidelity gate threshold config",
    ),
    out: Path = typer.Option(Path("outputs/whatif"), help="Output directory"),
    run_id: str | None = typer.Option(None, help="Run identifier override"),
) -> None:
    if (sacct is None) == (trace is None):
        raise typer.BadParameter("Provide exactly one of --sacct or --trace")
    if candidate_policy is None and slurm_scheduler_type is None:
        raise typer.BadParameter("Provide --candidate-policy or --slurm-scheduler-type")
    if candidate_policy is None:
        mapped = SLURM_SCHEDULER_POLICY_MAP.get(str(slurm_scheduler_type))
        if mapped is None:
            raise typer.BadParameter(
                f"Unknown SchedulerType '{slurm_scheduler_type}'. Known: {sorted(SLURM_SCHEDULER_POLICY_MAP)}"
            )
        candidate_policy = mapped

    ensure_dir(out)
    resolved_run_id = run_id or f"whatif_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"

    if sacct is not None:
        from hpcopt.ingest.slurm import ingest_slurm

        ingest_result = ingest_slurm(
            input_path=sacct,
            out_dir=out / "curated",
            dataset_id=resolved_run_id,
            report_dir=out / "reports",
        )
        trace = Path(ingest_result.dataset_path)
        typer.echo(f"Ingested sacct dump: {trace} ({ingest_result.row_count} jobs)")

    assert trace is not None
    trace_df = pd.read_parquet(trace)

    runtime_predictor = None
    if candidate_policy in _ML_POLICIES:
        resolved_model_dir = resolve_runtime_model_dir(runtime_model_dir)
        if resolved_model_dir is None:
            raise typer.BadParameter(
                f"{candidate_policy} requires a trained runtime model (--runtime-model-dir)"
            )
        runtime_predictor = RuntimeQuantilePredictor(resolved_model_dir)

    resolved_fidelity_config = fidelity_config if fidelity_config and fidelity_config.exists() else None
    result = run_whatif(
        trace_df=trace_df,
        baseline_policy=baseline_policy,
        candidate_policy=candidate_policy,
        out_dir=out,
        run_id=resolved_run_id,
        capacity_cpus=capacity_cpus,
        candidate_capacity_cpus=candidate_capacity_cpus,
        runtime_predictor=runtime_predictor,
        fidelity_config_path=resolved_fidelity_config,
    )

    manifest = build_manifest(
        command="hpcopt whatif run",
        inputs=[trace] + ([resolved_fidelity_config] if resolved_fidelity_config else []),
        outputs=[result.report_path, result.markdown_path],
        params={
            "run_id": result.run_id,
            "baseline_policy": baseline_policy,
            "candidate_policy": candidate_policy,
            "capacity_cpus": capacity_cpus,
            "candidate_capacity_cpus": candidate_capacity_cpus,
        },
        config_paths=[resolved_fidelity_config] if resolved_fidelity_config else [],
        seeds=[],
    )
    write_manifest(out / f"{result.run_id}_whatif_manifest.json", manifest)

    typer.echo(f"Verdict: {result.verdict} (confidence: {result.confidence})")
    typer.echo(f"Report: {result.markdown_path}")
