from pathlib import Path

import pandas as pd
import pytest
from hpcopt.ingest.fdata import ingest_fdata


def _write_fixture(path: Path) -> None:
    base = pd.Timestamp("2023-06-01 09:00:00", tz="UTC")
    df = pd.DataFrame(
        {
            "jid": ["a1", "b2", "c3"],
            "usr": ["u01", "u01", "u02"],
            "jnam": ["j1", "j2", "j3"],
            "adt": [base, base + pd.Timedelta(minutes=1), base + pd.Timedelta(minutes=2)],
            "sdt": [base + pd.Timedelta(minutes=10), pd.NaT, base + pd.Timedelta(minutes=4)],
            "edt": [base + pd.Timedelta(minutes=40), pd.NaT, base + pd.Timedelta(minutes=5)],
            "duration": [1800.0, 0.0, 60.0],
            "elpl": [3600.0, 600.0, 120.0],  # seconds
            "cnumr": [96, 48, 48],
            "cnumat": [96, 0, 48],
            "nnumr": [2, 1, 1],
            "nnuma": [2, 1, 1],
            # bytes per node; 2^64 is the "unlimited" sentinel
            "mszl": [28 * 2.0**30, 2.0**64, 14 * 2.0**30],
            "pri": [1, 1, 2],
            "econ": [100.0, None, 5.0 / 3.0],  # watt-hours, whole job
            "avgpcon": [200.0, None, 100.0],  # whole-job watts
            "minpcon": [180.0, None, 95.0],
            "maxpcon": [240.0, None, 105.0],
            "exit state": ["completed", "failed", "completed"],
            "jobenv_req": ["env1", "env1", "env2"],
        }
    )
    df.to_parquet(path, index=False)


def test_ingest_fdata_maps_canonical_schema(tmp_path: Path) -> None:
    fixture = tmp_path / "23_06.parquet"
    _write_fixture(fixture)

    result = ingest_fdata(
        input_path=fixture,
        out_dir=tmp_path / "curated",
        dataset_id="fdata_test",
        report_dir=tmp_path / "reports",
        batch_size=2,  # force multiple streaming batches
    )

    assert result.row_count == 3
    df = pd.read_parquet(result.dataset_path)

    j1 = df[df.job_id == "a1"].iloc[0]
    assert j1["wait_sec"] == 600
    assert j1["runtime_actual_sec"] == 1800
    assert j1["runtime_requested_sec"] == 3600
    assert j1["requested_cpus"] == 96
    assert j1["requested_mem"] == pytest.approx(56.0)  # 28 GiB/node x 2 nodes
    assert j1["requested_gpus"] == 0
    assert j1["status"] == "COMPLETED"
    # econ is watt-hours: 100 Wh -> 360 kJ.
    assert j1["node_energy_joules"] == pytest.approx(360_000.0)
    assert j1["node_power_mean_watts"] == pytest.approx(200.0)  # whole-job watts
    assert j1["runtime_overrequest_ratio"] == pytest.approx(2.0)

    # Never-started job: start falls back to submit; power fields stay null;
    # the 2^64 memory sentinel maps to null, not to an absurd request.
    j2 = df[df.job_id == "b2"].iloc[0]
    assert j2["start_ts"] == j2["submit_ts"]
    assert j2["wait_sec"] == 0
    assert pd.isna(j2["node_energy_joules"])
    assert pd.isna(j2["requested_mem"])
    assert j2["requested_cpus"] == 48  # cnumr kept even though cnumat is 0


def test_ingest_fdata_quality_report_carries_unit_evidence(tmp_path: Path) -> None:
    import json

    fixture = tmp_path / "23_06.parquet"
    _write_fixture(fixture)

    result = ingest_fdata(
        input_path=fixture,
        out_dir=tmp_path / "curated",
        dataset_id="fdata_test",
        report_dir=tmp_path / "reports",
    )

    quality = json.loads(result.quality_report_path.read_text(encoding="utf-8"))
    # econ x 3600 / (avgpcon x duration) == 1.0 when econ is Wh and avgpcon
    # is whole-job watts.
    assert quality["econ_vs_avgpcon_median_ratio"] == pytest.approx(1.0)
    assert quality["mem_unit"] == "GiB_per_job"
    assert quality["energy_missing_rows"] == 1
    assert quality["requested_mem_unlimited_sentinel_rows"] == 1
    assert quality["total_rows"] == 3


def test_ingest_fdata_rejects_non_fdata_files(tmp_path: Path) -> None:
    bogus = tmp_path / "other.parquet"
    pd.DataFrame({"job_id": [1]}).to_parquet(bogus, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        ingest_fdata(
            input_path=bogus,
            out_dir=tmp_path / "curated",
            dataset_id="x",
            report_dir=tmp_path / "reports",
        )
