"""Ingest the PM100 job table (Marconi100, CINECA) into canonical parquet.

PM100 (Antici et al., SC'23 workshops; https://doi.org/10.5281/zenodo.10127767)
contains 231K jobs from the Marconi100 GPU supercomputer (May-Oct 2020) with
per-job power consumption sampled at node, CPU, and memory level roughly every
20 seconds. It is the project's first modern trace: GPU requests, memory
requests, and measured power become first-class scheduling inputs.

Unit notes (verified against the published system configuration):

- ``time_limit`` is in minutes (Slurm convention).
- ``mem_req`` / ``mem_alloc`` are in GB (a full 256 GB node shows ~240 GB
  usable; the observed maximum, 61500, is ~256 nodes worth).
- ``*_power_consumption`` are watt samples summed over the job's allocated
  nodes, taken at ~20 s intervals. We keep summary statistics, not the raw
  series: mean power is robust to the exact sampling interval, and per-job
  energy is estimated as ``mean_power * runtime``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hpcopt.ingest import finalize_ingest
from hpcopt.ingest.swf import IngestResult
from hpcopt.utils.io import ensure_dir

logger = logging.getLogger(__name__)

MAX_INPUT_FILE_BYTES = 4 * 1024**3  # 4 GB

_SOURCE_COLUMNS = [
    "job_id",
    "submit_time",
    "start_time",
    "end_time",
    "run_time",
    "time_limit",
    "num_cores_req",
    "num_cores_alloc",
    "num_gpus_req",
    "num_gpus_alloc",
    "num_nodes_req",
    "num_nodes_alloc",
    "mem_req",
    "mem_alloc",
    "user_id",
    "group_id",
    "partition",
    "qos",
    "job_state",
    "node_power_consumption",
    "cpu_power_consumption",
    "mem_power_consumption",
]


def _epoch_seconds(series: pd.Series) -> pd.Series:
    """tz-aware timestamps -> integer Unix seconds (nullable Int64).

    Timedelta arithmetic is resolution-independent (PM100 stores
    microsecond-resolution timestamps, not pandas' default nanoseconds).
    """
    stamps = pd.to_datetime(series, utc=True)
    delta = stamps - pd.Timestamp("1970-01-01", tz="UTC")
    return (delta // pd.Timedelta(seconds=1)).astype("Int64")


def _sample_len(samples: Any) -> int:
    if samples is None:
        return 0
    try:
        return len(samples)
    except TypeError:  # scalar NaN for a missing list
        return 0


def _power_stats(series: pd.Series, prefix: str) -> dict[str, pd.Series]:
    """Mean/max watt summaries for a list-of-samples column."""

    def mean_of(samples: Any) -> float | None:
        if _sample_len(samples) == 0:
            return None
        return float(np.mean(samples))

    def max_of(samples: Any) -> float | None:
        if _sample_len(samples) == 0:
            return None
        return float(np.max(samples))

    return {
        f"{prefix}_power_mean_watts": series.map(mean_of),
        f"{prefix}_power_max_watts": series.map(max_of),
    }


def ingest_pm100(
    input_path: Path | str,
    out_dir: Path | str,
    dataset_id: str,
    report_dir: Path | str,
) -> IngestResult:
    """Ingest the PM100 ``job_table.parquet`` into canonical parquet.

    Canonical columns match the SWF/Slurm ingesters; PM100 additionally
    provides ``requested_gpus``/``allocated_gpus``, ``allocated_nodes``,
    per-job power summaries (watts), and ``node_energy_joules``.
    """
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

    raw = pd.read_parquet(input_path, columns=_SOURCE_COLUMNS)
    total_rows = int(len(raw))
    if total_rows == 0:
        raise ValueError(f"No rows in PM100 job table: {input_path}")

    submit_ts = _epoch_seconds(raw["submit_time"])
    start_ts = _epoch_seconds(raw["start_time"])
    end_ts = _epoch_seconds(raw["end_time"])

    dropped_no_submit = int(submit_ts.isna().sum())

    runtime = pd.to_numeric(raw["run_time"], errors="coerce").fillna(0).clip(lower=0).astype("int64")
    # Cancelled-before-start jobs have no start; fall back to submit (0 wait),
    # mirroring the Slurm ingester.
    start_ts = start_ts.fillna(submit_ts)
    end_ts = end_ts.fillna(start_ts + runtime)

    requested_cpus = pd.to_numeric(raw["num_cores_req"], errors="coerce")
    allocated_cpus = pd.to_numeric(raw["num_cores_alloc"], errors="coerce")
    requested_cpus = requested_cpus.fillna(allocated_cpus).fillna(1).astype("int64")
    allocated_cpus = allocated_cpus.fillna(requested_cpus).astype("int64")

    requested_gpus = pd.to_numeric(raw["num_gpus_req"], errors="coerce").fillna(0).astype("int64")
    allocated_gpus = (
        pd.to_numeric(raw["num_gpus_alloc"], errors="coerce").fillna(requested_gpus).astype("int64")
    )

    time_limit_min = pd.to_numeric(raw["time_limit"], errors="coerce")
    runtime_requested_sec = (time_limit_min * 60).astype("Int64")

    df = pd.DataFrame(
        {
            "job_id": pd.to_numeric(raw["job_id"], errors="coerce").astype("int64"),
            "submit_ts": submit_ts,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "wait_sec": (start_ts - submit_ts).clip(lower=0),
            "runtime_actual_sec": runtime,
            "runtime_requested_sec": runtime_requested_sec,
            "allocated_cpus": allocated_cpus,
            "requested_cpus": requested_cpus,
            # Canonical requested_mem is "as stated by the source"; PM100
            # states it in GB (see module docstring).
            "requested_mem": pd.to_numeric(raw["mem_req"], errors="coerce"),
            "status": raw["job_state"].astype("string").str.upper(),
            "user_id": pd.to_numeric(raw["user_id"], errors="coerce"),
            "group_id": pd.to_numeric(raw["group_id"], errors="coerce"),
            "queue_id": raw["qos"].astype("string"),
            "partition_id": raw["partition"].astype("string"),
            "requested_gpus": requested_gpus,
            "allocated_gpus": allocated_gpus,
            "allocated_nodes": pd.to_numeric(raw["num_nodes_alloc"], errors="coerce"),
            "runtime_overrequest_ratio": None,
        }
    )

    positive_runtime = df["runtime_actual_sec"] > 0
    has_request = df["runtime_requested_sec"].notna()
    df.loc[positive_runtime & has_request, "runtime_overrequest_ratio"] = (
        df.loc[positive_runtime & has_request, "runtime_requested_sec"].astype("float64")
        / df.loc[positive_runtime & has_request, "runtime_actual_sec"]
    )
    df["runtime_overrequest_ratio"] = pd.to_numeric(df["runtime_overrequest_ratio"], errors="coerce")

    # Power summaries + energy (watts summed across allocated nodes).
    for prefix in ("node", "cpu", "mem"):
        for name, col in _power_stats(raw[f"{prefix}_power_consumption"], prefix).items():
            df[name] = col
    df["power_sample_count"] = raw["node_power_consumption"].map(_sample_len)
    df["node_energy_joules"] = df["node_power_mean_watts"] * df["runtime_actual_sec"]

    # Drop rows without a submit timestamp: they cannot be replayed.
    df = df[df["submit_ts"].notna()].reset_index(drop=True)
    df["submit_ts"] = df["submit_ts"].astype("int64")
    df["start_ts"] = df["start_ts"].astype("int64")
    df["end_ts"] = df["end_ts"].astype("int64")
    df["wait_sec"] = df["wait_sec"].astype("int64")

    parse_stats: dict[str, Any] = {
        "total_rows": total_rows,
        "parsed_rows": int(len(df)),
        "dropped_no_submit_ts": dropped_no_submit,
    }
    extra_quality = {
        "zero_cpu_request_rows": int((df["requested_cpus"] <= 0).sum()),
        "gpu_job_rows": int((df["requested_gpus"] > 0).sum()),
        "gpu_job_rate": float((df["requested_gpus"] > 0).mean()),
        "power_series_missing_rows": int((df["power_sample_count"] == 0).sum()),
        "node_energy_joules_total": float(df["node_energy_joules"].sum()),
        "mem_unit": "GB",
        "power_unit": "watts_summed_over_allocated_nodes",
    }

    dataset_path, quality_report_path, dataset_metadata_path = finalize_ingest(
        df=df,
        dataset_id=dataset_id,
        input_path=input_path,
        out_dir=out_dir,
        report_dir=report_dir,
        parse_stats=parse_stats,
        source_format="pm100_job_table",
        extra_quality_fields=extra_quality,
    )

    logger.info("PM100 ingest complete: %d rows written to %s", len(df), dataset_path)
    return IngestResult(
        dataset_path=dataset_path,
        quality_report_path=quality_report_path,
        dataset_metadata_path=dataset_metadata_path,
        row_count=int(len(df)),
    )
