"""PM100 power-cap Pareto sweep: how low can the facility cap go for free?

Part 4 established that enforcing a 700 kW cap on the half-Marconi100 PM100
scenario costs nothing (p95 BSLD 152.24 vs 152.33 uncapped). This study asks
the follow-up question an operator actually has: *what is the lowest cap
that is still (nearly) free, and where does capping start to hurt?*

Method: replay PM100 on half of Marconi100 (490 nodes) with measured per-job
power, sweeping --enforce-power-cap from 700 kW down past the feasibility
floor. The floor is physical: the largest single job draws ~313 kW summed
over its nodes, so any cap below that strands jobs forever (the engine
reports them as jobs_completed < jobs_total rather than deadlocking).

Outputs (outputs/benchmark/pm100_cap_pareto.{json,md}):
- per-cap metrics for EASY and FIFO,
- Pareto-efficient caps under (minimize cap, minimize p95 BSLD), using the
  same dominance test as the recommendation engine,
- the knee: the lowest cap whose p95 BSLD penalty vs uncapped is <= 1%.

Prerequisites:
    hpcopt ingest pm100 --input data/raw/pm100/job_table.parquet
    cd rust && cargo build --release

Usage:
    python scripts/pm100_cap_pareto.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from hpcopt.recommend.engine import is_dominated  # noqa: E402

CURATED = PROJECT_ROOT / "data" / "curated" / "PM100.parquet"
OUT_DIR = PROJECT_ROOT / "outputs" / "benchmark"

# Half of Marconi100: 490 nodes x (128 hw threads, 4 V100, 240 GB) — the
# congested partition-under-facility-cap scenario from the Part 4 study.
HALF = {"cpus": 490 * 128, "gpus": 490 * 4, "mem": 490 * 240}

# kW grid: from the known-free 700 kW down through the feasibility floor
# (max single-job draw ~313.3 kW). 300 kW is deliberately below the floor to
# demonstrate stranded-job reporting.
CAP_GRID_KW = [700, 650, 600, 550, 500, 450, 400, 375, 350, 340, 330, 320, 315, 300]
KNEE_TOLERANCE = 0.01  # "free" = p95 BSLD within 1% of uncapped

POLICIES = ["EASY_BACKFILL_BASELINE", "FIFO_STRICT"]


def find_binary() -> Path:
    for rel in ("rust/target/release/sim-runner.exe", "rust/target/release/sim-runner"):
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return candidate
    sys.exit("sim-runner binary not found; build with: cd rust && cargo build --release")


def run_sim(binary: Path, trace_json: Path, report: Path, policy: str, cap_watts: float | None) -> dict:
    cmd = [
        str(binary),
        "--input", str(trace_json),
        "--policy", policy,
        "--capacity-cpus", str(HALF["cpus"]),
        "--capacity-gpus", str(HALF["gpus"]),
        "--capacity-mem", str(HALF["mem"]),
        "--output", str(report),
    ]
    if cap_watts is not None:
        cmd += ["--power-cap-watts", str(cap_watts), "--enforce-power-cap"]
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads(report.read_text(encoding="utf-8"))["metrics"]


def mark_pareto(rows: list[dict]) -> None:
    """Efficient caps under (minimize cap_kw, minimize p95_bsld), feasible only."""
    feasible = [r for r in rows if r["feasible"] and r["cap_kw"] is not None]
    for r in rows:
        r["pareto_efficient"] = False
    # is_dominated() maximizes every objective, so negate both.
    for r in feasible:
        point = [-r["cap_kw"], -r["metrics"]["p95_bsld"]]
        r["pareto_efficient"] = not any(
            is_dominated(point, [-o["cap_kw"], -o["metrics"]["p95_bsld"]])
            for o in feasible
            if o is not r
        )


def main() -> None:
    import pandas as pd

    if not CURATED.exists():
        sys.exit(f"{CURATED} missing; run: hpcopt ingest pm100 --input data/raw/pm100/job_table.parquet")
    binary = find_binary()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CURATED)
    max_job_kw = float(df.node_power_mean_watts.fillna(0).max()) / 1e3
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

    results: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        trace_json = Path(tmp) / "pm100.json"
        trace_json.write_text(json.dumps(jobs), encoding="utf-8")

        for policy in POLICIES:
            rows: list[dict] = []
            for cap_kw in [None, *CAP_GRID_KW]:
                cap_watts = None if cap_kw is None else cap_kw * 1e3
                report = Path(tmp) / f"{policy}_{cap_kw or 'uncapped'}.json"
                metrics = run_sim(binary, trace_json, report, policy, cap_watts)
                feasible = metrics["jobs_completed"] == metrics["jobs_total"]
                rows.append(
                    {
                        "cap_kw": cap_kw,
                        "metrics": metrics,
                        "feasible": feasible,
                        "stranded_jobs": int(metrics["jobs_total"] - metrics["jobs_completed"]),
                    }
                )
                print(
                    f"[cap-pareto] {policy} cap={cap_kw or 'none'} kW: "
                    f"p95_bsld={metrics['p95_bsld']:.2f} "
                    f"peak_kw={metrics.get('power_peak_watts', 0.0) / 1e3:.1f} "
                    f"completed={metrics['jobs_completed']}/{metrics['jobs_total']}",
                    flush=True,
                )
            mark_pareto(rows)

            uncapped_bsld = rows[0]["metrics"]["p95_bsld"]
            knee_kw = None
            for r in rows[1:]:
                if not r["feasible"]:
                    continue
                penalty = r["metrics"]["p95_bsld"] / uncapped_bsld - 1.0
                r["bsld_penalty_vs_uncapped"] = penalty
                if penalty <= KNEE_TOLERANCE:
                    knee_kw = r["cap_kw"] if knee_kw is None else min(knee_kw, r["cap_kw"])
            results[policy] = rows
            results[f"{policy}__knee_kw"] = knee_kw  # type: ignore[assignment]

    payload = {
        "scenario": {
            "capacity": HALF,
            "trace": "PM100 (Marconi100, 231,238 jobs, measured per-job power)",
            "max_single_job_kw": max_job_kw,
            "knee_tolerance": KNEE_TOLERANCE,
        },
        "results": results,
    }
    json_path = OUT_DIR / "pm100_cap_pareto.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# PM100 power-cap Pareto sweep (half Marconi100, enforced caps)",
        "",
        f"Feasibility floor: largest single job draws {max_job_kw:.1f} kW — caps below strand it.",
        "",
    ]
    for policy in POLICIES:
        rows = results[policy]
        uncapped_bsld = rows[0]["metrics"]["p95_bsld"]
        knee = results[f"{policy}__knee_kw"]
        lines += [
            f"## {policy} (uncapped p95 BSLD {uncapped_bsld:,.2f}; "
            f"knee = {knee} kW at <={KNEE_TOLERANCE:.0%} penalty)",
            "",
            "| Cap (kW) | p95 BSLD | vs uncapped | Mean Wait (s) | Peak (kW) | Makespan (h) | "
            "Stranded | Efficient |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            m = r["metrics"]
            cap_txt = "uncapped" if r["cap_kw"] is None else f"{r['cap_kw']}"
            penalty = (
                f"{m['p95_bsld'] / uncapped_bsld - 1.0:+.1%}" if r["cap_kw"] is not None else "—"
            )
            lines.append(
                "| {cap} | {bsld:,.2f} | {pen} | {mw:,.0f} | {pk:,.1f} | {mk:,.1f} | {s} | {eff} |".format(
                    cap=cap_txt,
                    bsld=m["p95_bsld"],
                    pen=penalty if r["feasible"] else "infeasible",
                    mw=m["mean_wait_sec"],
                    pk=m.get("power_peak_watts", 0.0) / 1e3,
                    mk=m["makespan_sec"] / 3600,
                    s=r["stranded_jobs"] or "",
                    eff="x" if r["pareto_efficient"] else "",
                )
            )
        lines.append("")
    md_path = OUT_DIR / "pm100_cap_pareto.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[cap-pareto] done: {json_path}")
    print(f"[cap-pareto] done: {md_path}")


if __name__ == "__main__":
    main()
