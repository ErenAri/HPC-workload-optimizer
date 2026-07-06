from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hpcopt.ingest.pm100 import ingest_pm100


def _write_fixture(path: Path) -> None:
    base = pd.Timestamp("2020-05-05 12:00:00", tz="UTC")
    df = pd.DataFrame(
        {
            "job_id": [101, 102, 103],
            "submit_time": [base, base + pd.Timedelta(minutes=1), base + pd.Timedelta(minutes=2)],
            "start_time": [base + pd.Timedelta(minutes=5), pd.NaT, base + pd.Timedelta(minutes=8)],
            "end_time": [base + pd.Timedelta(minutes=25), pd.NaT, base + pd.Timedelta(minutes=9)],
            "run_time": [1200, 0, 60],
            "time_limit": [30, 60, 10],  # minutes (Slurm convention)
            "num_cores_req": [128, 32, 4],
            "num_cores_alloc": [128, 0, 4],
            "num_gpus_req": [4, 0, 1],
            "num_gpus_alloc": [4, 0, 1],
            "num_nodes_req": [1, 1, 1],
            "num_nodes_alloc": [1, 1, 1],
            "mem_req": [240, 8, 16],  # GB
            "mem_alloc": [240, 8, 16],
            "user_id": [7, 7, 9],
            "group_id": [1, 1, 2],
            "partition": ["1", "1", "0"],
            "qos": ["normal", "normal", "debug"],
            "job_state": ["COMPLETED", "CANCELLED", "Failed"],
            "node_power_consumption": [np.array([500, 700]), None, np.array([300])],
            "cpu_power_consumption": [np.array([100, 200]), None, np.array([50])],
            "mem_power_consumption": [np.array([40, 60]), None, np.array([20])],
        }
    )
    df.to_parquet(path, index=False)


def test_ingest_pm100_maps_canonical_schema(tmp_path: Path) -> None:
    fixture = tmp_path / "job_table.parquet"
    _write_fixture(fixture)

    result = ingest_pm100(
        input_path=fixture,
        out_dir=tmp_path / "curated",
        dataset_id="pm100_test",
        report_dir=tmp_path / "reports",
    )

    assert result.row_count == 3
    df = pd.read_parquet(result.dataset_path)

    # Canonical columns shared with the SWF/Slurm ingesters.
    assert set(df.columns) >= {
        "job_id",
        "submit_ts",
        "start_ts",
        "end_ts",
        "wait_sec",
        "runtime_actual_sec",
        "runtime_requested_sec",
        "allocated_cpus",
        "requested_cpus",
        "requested_mem",
        "status",
        "user_id",
        "queue_id",
        "partition_id",
        "runtime_overrequest_ratio",
    }

    j1 = df[df.job_id == 101].iloc[0]
    assert j1["wait_sec"] == 300
    assert j1["runtime_actual_sec"] == 1200
    assert j1["runtime_requested_sec"] == 30 * 60  # minutes -> seconds
    assert j1["requested_gpus"] == 4
    assert j1["requested_mem"] == 240
    assert j1["status"] == "COMPLETED"
    assert j1["node_power_mean_watts"] == pytest.approx(600.0)
    assert j1["node_power_max_watts"] == pytest.approx(700.0)
    assert j1["node_energy_joules"] == pytest.approx(600.0 * 1200)
    assert j1["power_sample_count"] == 2
    assert j1["runtime_overrequest_ratio"] == pytest.approx(1800 / 1200)

    # Cancelled-before-start: start falls back to submit, zero wait/runtime,
    # power summaries stay null.
    j2 = df[df.job_id == 102].iloc[0]
    assert j2["wait_sec"] == 0
    assert j2["start_ts"] == j2["submit_ts"]
    assert j2["end_ts"] == j2["start_ts"]
    assert pd.isna(j2["node_power_mean_watts"])
    assert j2["power_sample_count"] == 0
    assert j2["requested_gpus"] == 0

    # State is normalised to upper case.
    j3 = df[df.job_id == 103].iloc[0]
    assert j3["status"] == "FAILED"
    assert j3["node_energy_joules"] == pytest.approx(300.0 * 60)


def test_ingest_pm100_quality_report_counts_gpu_jobs(tmp_path: Path) -> None:
    import json

    fixture = tmp_path / "job_table.parquet"
    _write_fixture(fixture)

    result = ingest_pm100(
        input_path=fixture,
        out_dir=tmp_path / "curated",
        dataset_id="pm100_test",
        report_dir=tmp_path / "reports",
    )

    quality = json.loads(result.quality_report_path.read_text(encoding="utf-8"))
    assert quality["gpu_job_rows"] == 2
    assert quality["power_series_missing_rows"] == 1
    assert quality["mem_unit"] == "GB"
    assert quality["total_rows"] == 3
