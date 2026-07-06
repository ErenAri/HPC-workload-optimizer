"""Ingest F-DATA (Fugaku) monthly job files into canonical parquet.

F-DATA (Antici et al., Scientific Data 2025; https://doi.org/10.5281/zenodo.11467483)
covers ~24M jobs from Supercomputer Fugaku (Mar 2021 - Apr 2024) in 38 monthly
parquet chunks (up to ~1.3 GB / ~1M jobs each). Jobs carry timestamps, core/node
counts, memory limits, and measured energy/power (econ, avgpcon/maxpcon).

Scale note: files are read in streaming batches (pyarrow ``iter_batches``) with
a column subset — the Sentence-BERT ``embedding`` column alone is most of each
file's bytes and is never loaded. Peak memory stays bounded by the batch size,
not the file size.

Unit notes (verified empirically on 23_06; see quality report fields):

- ``duration`` and ``elpl`` (elapsed-time limit) are seconds.
- ``mszl`` (memory size limit) is BYTES per node, with 2^64 as an
  "unlimited" sentinel (mapped to null). Canonical ``requested_mem`` is
  stated as GiB *per job* (mszl x allocated nodes) to be comparable with
  the multi-resource engine's per-job accounting.
- ``econ`` is WATT-HOURS for the whole job; ``avgpcon``/``maxpcon`` are
  whole-job watts (not per-node): on multi-node jobs the identity
  econ x 3600 == avgpcon x duration holds with median ratio 1.0004
  (p25 1.0001 / p75 1.0005), while the per-node reading is off by the
  node count. The quality report carries the measured ratio so this
  claim stays auditable.
- Fugaku has no GPUs; ``requested_gpus`` is 0 for engine compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hpcopt.ingest import finalize_ingest
from hpcopt.ingest.swf import IngestResult
from hpcopt.utils.io import ensure_dir

logger = logging.getLogger(__name__)

MAX_INPUT_FILE_BYTES = 8 * 1024**3  # 8 GB
DEFAULT_BATCH_SIZE = 131_072

_SOURCE_COLUMNS = [
    "jid",
    "usr",
    "adt",
    "sdt",
    "edt",
    "duration",
    "elpl",
    "cnumr",
    "cnumat",
    "nnumr",
    "nnuma",
    "mszl",
    "pri",
    "econ",
    "avgpcon",
    "maxpcon",
    "exit state",
    "jobenv_req",
]


def _epoch_seconds(series: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(series, utc=True, errors="coerce")
    delta = stamps - pd.Timestamp("1970-01-01", tz="UTC")
    return (delta // pd.Timedelta(seconds=1)).astype("Int64")


def _transform_batch(raw: pd.DataFrame) -> pd.DataFrame:
    submit_ts = _epoch_seconds(raw["adt"])
    start_ts = _epoch_seconds(raw["sdt"])
    end_ts = _epoch_seconds(raw["edt"])

    duration = pd.to_numeric(raw["duration"], errors="coerce")
    derived = (end_ts - start_ts).astype("Float64")
    runtime = duration.fillna(derived.astype("float64")).fillna(0).clip(lower=0).astype("int64")

    start_ts = start_ts.fillna(submit_ts)
    end_ts = end_ts.fillna(start_ts + runtime)

    requested_cpus = pd.to_numeric(raw["cnumr"], errors="coerce")
    allocated_cpus = pd.to_numeric(raw["cnumat"], errors="coerce")
    requested_cpus = requested_cpus.fillna(allocated_cpus).fillna(1).clip(lower=1).astype("int64")
    allocated_cpus = allocated_cpus.fillna(requested_cpus).astype("int64")

    allocated_nodes = pd.to_numeric(raw["nnuma"], errors="coerce").fillna(1).clip(lower=1)
    # mszl: bytes per node; values at/above 1 EiB are the "unlimited" sentinel
    # (observed as 2^64) and carry no scheduling information.
    mszl_bytes = pd.to_numeric(raw["mszl"], errors="coerce")
    mszl_bytes = mszl_bytes.where(mszl_bytes < 2.0**60)
    mszl_gib_per_node = mszl_bytes / 2.0**30

    elpl = pd.to_numeric(raw["elpl"], errors="coerce")
    runtime_requested_sec = elpl.round().astype("Int64")

    econ_wh = pd.to_numeric(raw["econ"], errors="coerce")
    avgpcon = pd.to_numeric(raw["avgpcon"], errors="coerce")
    maxpcon = pd.to_numeric(raw["maxpcon"], errors="coerce")

    df = pd.DataFrame(
        {
            "job_id": raw["jid"].astype("string"),
            "submit_ts": submit_ts,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "wait_sec": (start_ts - submit_ts).clip(lower=0),
            "runtime_actual_sec": runtime,
            "runtime_requested_sec": runtime_requested_sec,
            "allocated_cpus": allocated_cpus,
            "requested_cpus": requested_cpus,
            # GiB per job (per-node limit x allocated nodes); see module docstring.
            "requested_mem": mszl_gib_per_node * allocated_nodes,
            "status": raw["exit state"].astype("string").str.upper(),
            "user_id": raw["usr"].astype("string"),
            "group_id": None,
            "queue_id": raw["jobenv_req"].astype("string"),
            "partition_id": None,
            "priority": pd.to_numeric(raw["pri"], errors="coerce"),
            "requested_gpus": 0,
            "allocated_gpus": 0,
            "allocated_nodes": allocated_nodes.astype("int64"),
            "runtime_overrequest_ratio": None,
            # avgpcon/maxpcon are whole-job watts (verified; module docstring).
            "node_power_mean_watts": avgpcon,
            "node_power_max_watts": maxpcon,
            "node_energy_joules": econ_wh * 3600.0,
        }
    )

    positive = (df["runtime_actual_sec"] > 0) & df["runtime_requested_sec"].notna()
    ratio = df.loc[positive, "runtime_requested_sec"].astype("float64") / df.loc[
        positive, "runtime_actual_sec"
    ]
    df["runtime_overrequest_ratio"] = pd.to_numeric(df["runtime_overrequest_ratio"], errors="coerce")
    df.loc[positive, "runtime_overrequest_ratio"] = ratio
    return df


def ingest_fdata(
    input_path: Path | str,
    out_dir: Path | str,
    dataset_id: str,
    report_dir: Path | str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IngestResult:
    """Ingest one F-DATA monthly parquet into canonical parquet, streaming."""
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    report_dir = Path(report_dir)
    ensure_dir(out_dir)
    ensure_dir(report_dir)

    file_size = input_path.stat().st_size
    if file_size > MAX_INPUT_FILE_BYTES:
        raise ValueError(
            f"Input file too large ({file_size / (1024**3):.1f} GB). "
            f"Maximum allowed: {MAX_INPUT_FILE_BYTES / (1024**3):.0f} GB."
        )

    parquet = pq.ParquetFile(input_path)
    available = set(parquet.schema_arrow.names)
    missing = [c for c in _SOURCE_COLUMNS if c not in available]
    if missing:
        raise ValueError(f"Not an F-DATA job file (missing columns {missing}): {input_path}")

    chunks: list[pd.DataFrame] = []
    total_rows = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=_SOURCE_COLUMNS):
        raw = batch.to_pandas()
        total_rows += len(raw)
        chunks.append(_transform_batch(raw))
    if not chunks:
        raise ValueError(f"No rows in F-DATA file: {input_path}")

    df = pd.concat(chunks, ignore_index=True)
    del chunks

    dropped_no_submit = int(df["submit_ts"].isna().sum())
    df = df[df["submit_ts"].notna()].reset_index(drop=True)
    for col in ("submit_ts", "start_ts", "end_ts", "wait_sec"):
        df[col] = df[col].astype("int64")

    # Auditable unit claim: econ x 3600 (J) == avgpcon (whole-job W) x duration.
    consistent = df[
        (df["runtime_actual_sec"] > 0)
        & df["node_energy_joules"].notna()
        & df["node_power_mean_watts"].notna()
        & (df["node_power_mean_watts"] > 0)
    ]
    power_consistency = float(
        (
            consistent["node_energy_joules"]
            / (consistent["node_power_mean_watts"] * consistent["runtime_actual_sec"])
        ).median()
    ) if len(consistent) else float("nan")

    parse_stats: dict[str, Any] = {
        "total_rows": total_rows,
        "parsed_rows": int(len(df)),
        "dropped_no_submit_ts": dropped_no_submit,
        "batches": int(np.ceil(total_rows / batch_size)) if total_rows else 0,
    }
    extra_quality = {
        "energy_missing_rows": int(df["node_energy_joules"].isna().sum()),
        "node_energy_joules_total": float(df["node_energy_joules"].sum()),
        "econ_vs_avgpcon_median_ratio": power_consistency,
        "requested_mem_unlimited_sentinel_rows": int(df["requested_mem"].isna().sum()),
        "mem_unit": "GiB_per_job",
        "power_unit": "watts_whole_job",
        "energy_unit_source": "econ_watt_hours_x3600",
    }

    dataset_path, quality_report_path, dataset_metadata_path = finalize_ingest(
        df=df,
        dataset_id=dataset_id,
        input_path=input_path,
        out_dir=out_dir,
        report_dir=report_dir,
        parse_stats=parse_stats,
        source_format="fdata_fugaku",
        extra_quality_fields=extra_quality,
    )

    logger.info("F-DATA ingest complete: %d rows written to %s", len(df), dataset_path)
    return IngestResult(
        dataset_path=dataset_path,
        quality_report_path=quality_report_path,
        dataset_metadata_path=dataset_metadata_path,
        row_count=int(len(df)),
    )
