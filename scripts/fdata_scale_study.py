"""F-DATA (Fugaku) scale study: engine throughput at ~1M-job monthly scale.

Replays one curated F-DATA month through the Rust engine on Fugaku's real
capacity and records wall time — the scale proof for the "200K+ jobs in
under a second" class of claims, one order of magnitude up.

Prerequisites:
    hpcopt ingest fdata --input data/raw/fdata/23_06.parquet
    cd rust && cargo build --release

Usage:
    python scripts/fdata_scale_study.py [--dataset FDATA_23_06]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "benchmark"

# Supercomputer Fugaku: 158,976 nodes x (48 compute cores, 32 GiB HBM2).
CAPACITY = {"cpus": 158_976 * 48, "mem": 158_976 * 32}

POLICIES = ["FIFO_STRICT", "EASY_BACKFILL_BASELINE"]


def find_binary() -> Path:
    for rel in ("rust/target/release/sim-runner.exe", "rust/target/release/sim-runner"):
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return candidate
    sys.exit("sim-runner binary not found; build with: cd rust && cargo build --release")


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FDATA_23_06")
    args = parser.parse_args()

    curated = PROJECT_ROOT / "data" / "curated" / f"{args.dataset}.parquet"
    if not curated.exists():
        sys.exit(f"{curated} missing; run: hpcopt ingest fdata --input data/raw/fdata/<YY_MM>.parquet")

    binary = find_binary()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(
        curated,
        columns=[
            "job_id",
            "submit_ts",
            "runtime_actual_sec",
            "requested_cpus",
            "requested_mem",
            "node_power_mean_watts",
        ],
    )
    jobs = pd.DataFrame(
        {
            # Engine wants integer ids; F-DATA ids are anonymized strings.
            "job_id": range(1, len(df) + 1),
            "submit_ts": df.submit_ts,
            "runtime_actual_sec": df.runtime_actual_sec,
            "requested_cpus": df.requested_cpus,
            "requested_mem": df.requested_mem.fillna(0).round().astype("int64"),
            "power_mean_watts": df.node_power_mean_watts.fillna(0.0).round(2),
        }
    ).to_dict("records")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        trace_json = Path(tmp) / "fdata.json"
        trace_json.write_text(json.dumps(jobs), encoding="utf-8")

        for policy in POLICIES:
            report_path = Path(tmp) / f"{policy}.json"
            started = time.perf_counter()
            subprocess.run(
                [
                    str(binary),
                    "--input", str(trace_json),
                    "--policy", policy,
                    "--capacity-cpus", str(CAPACITY["cpus"]),
                    "--capacity-mem", str(CAPACITY["mem"]),
                    "--output", str(report_path),
                ],
                check=True,
                capture_output=True,
            )
            wall = time.perf_counter() - started
            metrics = json.loads(report_path.read_text(encoding="utf-8"))["metrics"]
            results.append({"policy": policy, "wall_sec_incl_io": round(wall, 2), "metrics": metrics})
            print(
                f"[fdata] {policy}: {metrics['jobs_total']} jobs in {wall:.2f}s "
                f"(incl. JSON parse) | p95_bsld={metrics['p95_bsld']:.3f} "
                f"util={metrics['utilization_mean']:.3f} "
                f"energy_GWh={(metrics.get('energy_joules_total') or 0) / 3.6e12:.2f}",
                flush=True,
            )

    out = OUT_DIR / "fdata_scale.json"
    out.write_text(
        json.dumps({"dataset": args.dataset, "capacity": CAPACITY, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"[fdata] done: {out}")


if __name__ == "__main__":
    main()
