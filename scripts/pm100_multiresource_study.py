"""PM100 multi-resource study: what GPU-blind simulation gets wrong.

Replays the PM100 trace (Marconi100, 231K jobs, 88% GPU) through the Rust
engine twice per policy: once CPU-only (the classic SWF-era model) and once
with the full {cpus, gpus, mem} resource vector. The delta is the error a
CPU-only simulator makes on a modern GPU machine.

Prerequisites:
    hpcopt ingest pm100 --input data/raw/pm100/job_table.parquet
    cd rust && cargo build --release

Usage:
    python scripts/pm100_multiresource_study.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED = PROJECT_ROOT / "data" / "curated" / "PM100.parquet"
OUT_DIR = PROJECT_ROOT / "outputs" / "benchmark"

# Marconi100 (CINECA), the machine PM100 was recorded on: 980 nodes, each
# 2x POWER9 (128 hardware threads), 4x V100 GPUs, ~240 GB allocatable RAM.
CAPACITY = {"cpus": 980 * 128, "gpus": 980 * 4, "mem": 980 * 240}

POLICIES = ["FIFO_STRICT", "EASY_BACKFILL_BASELINE"]


def find_binary() -> Path:
    for rel in ("rust/target/release/sim-runner.exe", "rust/target/release/sim-runner"):
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return candidate
    sys.exit("sim-runner binary not found; build with: cd rust && cargo build --release")


def main() -> None:
    import pandas as pd

    if not CURATED.exists():
        sys.exit(f"{CURATED} missing; run: hpcopt ingest pm100 --input data/raw/pm100/job_table.parquet")

    binary = find_binary()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CURATED)
    jobs = pd.DataFrame(
        {
            "job_id": df.job_id,
            "submit_ts": df.submit_ts,
            "runtime_actual_sec": df.runtime_actual_sec,
            "requested_cpus": df.requested_cpus,
            "requested_gpus": df.requested_gpus,
            "requested_mem": df.requested_mem.fillna(0).astype("int64"),
        }
    ).to_dict("records")

    results: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        trace_json = Path(tmp) / "pm100.json"
        trace_json.write_text(json.dumps(jobs), encoding="utf-8")

        for policy in POLICIES:
            for mode in ("cpu_only", "multi_resource"):
                report_path = Path(tmp) / f"{policy}_{mode}.json"
                cmd = [
                    str(binary),
                    "--input", str(trace_json),
                    "--policy", policy,
                    "--capacity-cpus", str(CAPACITY["cpus"]),
                    "--output", str(report_path),
                ]
                if mode == "multi_resource":
                    cmd += [
                        "--capacity-gpus", str(CAPACITY["gpus"]),
                        "--capacity-mem", str(CAPACITY["mem"]),
                    ]
                subprocess.run(cmd, check=True, capture_output=True)
                metrics = json.loads(report_path.read_text(encoding="utf-8"))["metrics"]
                results.append({"policy": policy, "mode": mode, "metrics": metrics})
                print(
                    f"[pm100] {policy} {mode}: p95_bsld={metrics['p95_bsld']:.3f} "
                    f"mean_wait={metrics['mean_wait_sec']:.1f}s "
                    f"util_gpu={metrics.get('utilization_gpu_mean', float('nan'))}",
                    flush=True,
                )

    json_path = OUT_DIR / "pm100_multiresource.json"
    json_path.write_text(json.dumps({"capacity": CAPACITY, "results": results}, indent=2), encoding="utf-8")

    lines = [
        "# PM100 (Marconi100): CPU-only vs multi-resource simulation",
        "",
        f"Capacity: {CAPACITY['cpus']} hw threads, {CAPACITY['gpus']} GPUs, {CAPACITY['mem']} GB.",
        "",
        "| Policy | Resource model | p95 BSLD | Mean Wait (s) | p95 Wait (s) | CPU Util | GPU Util |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in results:
        m = row["metrics"]
        gpu = m.get("utilization_gpu_mean")
        lines.append(
            "| {p} | {mode} | {bsld:.3f} | {mw:,.1f} | {pw:,.1f} | {cu:.1%} | {gu} |".format(
                p=row["policy"],
                mode=row["mode"].replace("_", " "),
                bsld=m["p95_bsld"],
                mw=m["mean_wait_sec"],
                pw=m["p95_wait_sec"],
                cu=m["utilization_mean"],
                gu=f"{gpu:.1%}" if gpu is not None else "—",
            )
        )
    lines += [
        "",
        "PM100 contains only Marconi100's exclusive-resource jobs (May–Oct 2020), so absolute",
        "congestion is lower than the machine actually experienced; the comparison between the",
        "two resource models on identical input is the point, not the absolute waits.",
    ]
    md_path = OUT_DIR / "pm100_multiresource.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[pm100] done: {json_path}")
    print(f"[pm100] done: {md_path}")


if __name__ == "__main__":
    main()
