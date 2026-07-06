"""CLI commands for data ingestion (SWF, Slurm, PBS, shadow)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hpcopt.artifacts.manifest import build_manifest, write_manifest
from hpcopt.data.reference_suite import assert_reference_trace_hash_match
from hpcopt.ingest.swf import ingest_swf
from hpcopt.utils.io import write_json

ingest_app = typer.Typer(help="Ingestion commands")


@ingest_app.command("swf")
def ingest_swf_cmd(
    input: Path = typer.Option(..., exists=True, readable=True, help="Input SWF or SWF.GZ path"),
    out: Path = typer.Option(Path("data/curated"), help="Output curated dataset directory"),
    dataset_id: str | None = typer.Option(None, help="Dataset ID override"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Report output directory"),
    reference_suite_config: Path = typer.Option(
        Path("configs/data/reference_suite.yaml"),
        exists=True,
        readable=True,
        help="Reference suite config for hash contract checks",
    ),
) -> None:
    ds_id = dataset_id or input.stem.replace(".swf", "")
    result = ingest_swf(input_path=input, out_dir=out, dataset_id=ds_id, report_dir=report_out)

    reference_suite_match = assert_reference_trace_hash_match(
        trace_path=input,
        config_path=reference_suite_config,
    )
    if reference_suite_match is not None:
        metadata = json.loads(result.dataset_metadata_path.read_text(encoding="utf-8"))
        metadata["reference_suite"] = reference_suite_match
        write_json(result.dataset_metadata_path, metadata)

    manifest = build_manifest(
        command="hpcopt ingest swf",
        inputs=[input, reference_suite_config],
        outputs=[result.dataset_path, result.quality_report_path, result.dataset_metadata_path],
        params={
            "dataset_id": ds_id,
            "out_dir": str(out),
            "report_out": str(report_out),
            "reference_suite_match": reference_suite_match,
        },
        config_paths=[reference_suite_config],
        seeds=[],
    )
    manifest_path = report_out / f"{ds_id}_run_manifest.json"
    write_manifest(manifest_path, manifest)

    typer.echo(f"Dataset: {result.dataset_path}")
    typer.echo(f"Dataset metadata: {result.dataset_metadata_path}")
    typer.echo(f"Quality report: {result.quality_report_path}")
    typer.echo(f"Run manifest: {manifest_path}")
    typer.echo(f"Rows: {result.row_count}")


@ingest_app.command("slurm")
def ingest_slurm_cmd(
    input: Path = typer.Option(..., exists=True, readable=True, help="sacct --parsable2 output file"),
    out: Path = typer.Option(Path("data/curated"), help="Output curated dataset directory"),
    dataset_id: str | None = typer.Option(None, help="Dataset ID override"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Report output directory"),
) -> None:
    from hpcopt.ingest.slurm import ingest_slurm

    ds_id = dataset_id or input.stem
    result = ingest_slurm(input_path=input, out_dir=out, dataset_id=ds_id, report_dir=report_out)
    typer.echo(f"Dataset: {result.dataset_path}")
    typer.echo(f"Rows: {result.row_count}")


@ingest_app.command("pbs")
def ingest_pbs_cmd(
    input: Path = typer.Option(..., exists=True, readable=True, help="PBS/Torque accounting log"),
    out: Path = typer.Option(Path("data/curated"), help="Output curated dataset directory"),
    dataset_id: str | None = typer.Option(None, help="Dataset ID override"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Report output directory"),
) -> None:
    from hpcopt.ingest.pbs import ingest_pbs

    ds_id = dataset_id or input.stem
    result = ingest_pbs(input_path=input, out_dir=out, dataset_id=ds_id, report_dir=report_out)
    typer.echo(f"Dataset: {result.dataset_path}")
    typer.echo(f"Rows: {result.row_count}")


@ingest_app.command("pm100")
def ingest_pm100_cmd(
    input: Path = typer.Option(..., exists=True, readable=True, help="PM100 job_table.parquet path"),
    out: Path = typer.Option(Path("data/curated"), help="Output curated dataset directory"),
    dataset_id: str | None = typer.Option(None, help="Dataset ID override"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Report output directory"),
) -> None:
    """Ingest the PM100 (Marconi100) job table: GPU requests + measured power."""
    from hpcopt.ingest.pm100 import ingest_pm100

    ds_id = dataset_id or "PM100"
    result = ingest_pm100(input_path=input, out_dir=out, dataset_id=ds_id, report_dir=report_out)
    typer.echo(f"Dataset: {result.dataset_path}")
    typer.echo(f"Dataset metadata: {result.dataset_metadata_path}")
    typer.echo(f"Quality report: {result.quality_report_path}")
    typer.echo(f"Rows: {result.row_count}")


@ingest_app.command("fdata")
def ingest_fdata_cmd(
    input: Path = typer.Option(..., exists=True, readable=True, help="F-DATA monthly parquet (YY_MM.parquet)"),
    out: Path = typer.Option(Path("data/curated"), help="Output curated dataset directory"),
    dataset_id: str | None = typer.Option(None, help="Dataset ID override"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Report output directory"),
    batch_size: int = typer.Option(131_072, min=1024, help="Streaming batch size (rows)"),
) -> None:
    """Ingest an F-DATA (Fugaku) monthly file: 24M-job dataset, streamed in batches."""
    from hpcopt.ingest.fdata import ingest_fdata

    ds_id = dataset_id or f"FDATA_{input.stem}"
    result = ingest_fdata(
        input_path=input, out_dir=out, dataset_id=ds_id, report_dir=report_out, batch_size=batch_size
    )
    typer.echo(f"Dataset: {result.dataset_path}")
    typer.echo(f"Dataset metadata: {result.dataset_metadata_path}")
    typer.echo(f"Quality report: {result.quality_report_path}")
    typer.echo(f"Rows: {result.row_count}")


@ingest_app.command("shadow-start")
def ingest_shadow_start_cmd(
    source_type: str = typer.Option("slurm", help="slurm|pbs"),
    source_path: Path = typer.Option(..., help="sacct output or accounting log path"),
    out: Path = typer.Option(Path("data/curated"), help="Output directory"),
    interval_sec: int = typer.Option(300, min=10, help="Polling interval in seconds"),
    watermark_path: Path = typer.Option(Path("outputs/shadow_watermark.json"), help="Watermark file"),
) -> None:
    from hpcopt.ingest.shadow import ShadowIngestionDaemon

    daemon = ShadowIngestionDaemon(
        out_dir=out,
        report_dir=out / "reports",
        watermark_path=watermark_path,
    )
    daemon.start(
        interval_sec=interval_sec,
        source_type=source_type,
        source_path=source_path,
        blocking=True,
    )
