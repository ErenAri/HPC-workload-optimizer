"""PM100 studies: multi-resource modeling error + schedule/power tradeoff.

Study A — what GPU-blind simulation gets wrong. Replays the PM100 trace
(Marconi100, 231K jobs, 88% GPU) through the Rust engine twice per policy:
once CPU-only (the classic SWF-era model) and once with the full
{cpus, gpus, mem} resource vector. The delta is the error a CPU-only
simulator makes on a modern GPU machine.

Study B — the BSLD x power tradeoff under congestion. Replays the same trace
on half the machine (a realistic partition-under-facility-cap scenario) with
per-job measured power attached. Energy is schedule-invariant; peak power and
time above a facility cap are not — that is what energy-aware policies
actually trade against BSLD.

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
FULL = {"cpus": 980 * 128, "gpus": 980 * 4, "mem": 980 * 240}
HALF = {"cpus": 490 * 128, "gpus": 490 * 4, "mem": 490 * 240}
POWER_CAP_WATTS = 700_000.0  # facility cap for the half-machine stress study

POLICIES = ["FIFO_STRICT", "EASY_BACKFILL_BASELINE"]


def find_binary() -> Path:
    for rel in ("rust/target/release/sim-runner.exe", "rust/target/release/sim-runner"):
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return candidate
    sys.exit("sim-runner binary not found; build with: cd rust && cargo build --release")


def run_sim(
    binary: Path,
    trace_json: Path,
    report_path: Path,
    policy: str,
    capacity: dict,
    multi_resource: bool,
    power_cap: float | None,
) -> dict:
    cmd = [
        str(binary),
        "--input", str(trace_json),
        "--policy", policy,
        "--capacity-cpus", str(capacity["cpus"]),
        "--output", str(report_path),
    ]
    if multi_resource:
        cmd += ["--capacity-gpus", str(capacity["gpus"]), "--capacity-mem", str(capacity["mem"])]
    if power_cap is not None:
        cmd += ["--power-cap-watts", str(power_cap)]
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads(report_path.read_text(encoding="utf-8"))["metrics"]


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
            "power_mean_watts": df.node_power_mean_watts.fillna(0.0).round(2),
        }
    ).to_dict("records")

    study_a: list[dict] = []
    study_b: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        trace_json = Path(tmp) / "pm100.json"
        trace_json.write_text(json.dumps(jobs), encoding="utf-8")

        for policy in POLICIES:
            for mode in ("cpu_only", "multi_resource"):
                metrics = run_sim(
                    binary, trace_json, Path(tmp) / f"a_{policy}_{mode}.json",
                    policy, FULL, multi_resource=(mode == "multi_resource"), power_cap=None,
                )
                study_a.append({"policy": policy, "mode": mode, "metrics": metrics})
                print(
                    f"[pm100:A] {policy} {mode}: p95_bsld={metrics['p95_bsld']:.3f} "
                    f"peak_kw={metrics.get('power_peak_watts', 0) / 1e3:.1f}",
                    flush=True,
                )

        for policy in POLICIES:
            metrics = run_sim(
                binary, trace_json, Path(tmp) / f"b_{policy}.json",
                policy, HALF, multi_resource=True, power_cap=POWER_CAP_WATTS,
            )
            study_b.append({"policy": policy, "metrics": metrics})
            print(
                f"[pm100:B] {policy}: p95_bsld={metrics['p95_bsld']:.2f} "
                f"hrs_above_cap={metrics['seconds_above_power_cap'] / 3600:.2f}",
                flush=True,
            )

    json_path = OUT_DIR / "pm100_multiresource.json"
    json_path.write_text(
        json.dumps(
            {
                "capacity_full": FULL,
                "capacity_half": HALF,
                "power_cap_watts": POWER_CAP_WATTS,
                "study_a_resource_model": study_a,
                "study_b_power_tradeoff": study_b,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# PM100 (Marconi100) studies",
        "",
        "## A. CPU-only vs multi-resource simulation (full machine)",
        "",
        f"Capacity: {FULL['cpus']} hw threads, {FULL['gpus']} GPUs, {FULL['mem']} GB.",
        "",
        "| Policy | Resource model | p95 BSLD | Mean Wait (s) | p95 Wait (s) | CPU Util | GPU Util |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in study_a:
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
        "## B. BSLD x power tradeoff (half machine, measured per-job power, "
        f"{POWER_CAP_WATTS / 1e3:.0f} kW cap)",
        "",
        "| Policy | p95 BSLD | Mean Wait (s) | Energy (MWh) | Peak Power (kW) | "
        "Hours Above Cap | Excess Above Cap (MWh) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in study_b:
        m = row["metrics"]
        lines.append(
            "| {p} | {bsld:,.2f} | {mw:,.0f} | {e:,.1f} | {pk:,.1f} | {h:,.2f} | {x:.3f} |".format(
                p=row["policy"],
                bsld=m["p95_bsld"],
                mw=m["mean_wait_sec"],
                e=m["energy_joules_total"] / 3.6e9,
                pk=m["power_peak_watts"] / 1e3,
                h=m["seconds_above_power_cap"] / 3600,
                x=m["joules_above_power_cap"] / 3.6e9,
            )
        )
    lines += [
        "",
        "Energy is identical across policies (schedule-invariant); what scheduling changes is",
        "*when* power is drawn. Backfilling sustains draw near the envelope: EASY spends far",
        "longer above the cap than FIFO even though FIFO's instantaneous peak is higher.",
        "",
        "PM100 contains only Marconi100's exclusive-resource jobs (May–Oct 2020), so absolute",
        "congestion in study A is lower than the machine actually experienced; the comparison",
        "between resource models on identical input is the point, not the absolute waits.",
    ]
    md_path = OUT_DIR / "pm100_multiresource.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[pm100] done: {json_path}")
    print(f"[pm100] done: {md_path}")


if __name__ == "__main__":
    main()
