"""Full policy matrix: every supported policy against every reference trace.

Produces the headline comparison table (JSON + markdown) used in the README.
Unlike scripts/benchmark_suite.py (Rust engine, FIFO/EASY only), this drives
the Python reference simulator so prediction-based policies (Tsafrir, ML) are
included. Results are appended incrementally so partial runs remain usable.

Usage:
    python scripts/policy_matrix.py [--traces ctc_sp2,hpc2n,sdsc_sp2] [--policies ...]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
OUT_DIR = PROJECT_ROOT / "outputs" / "benchmark"

TRACES = [
    {
        "name": "SDSC-SP2",
        "key": "sdsc_sp2",
        "dataset": "SDSC-SP2-1998-4.2-cln.parquet",
        "model_dir": "runtime_SDSC-SP2-1998-4.2-cln",
        "capacity_cpus": 128,
    },
    {
        "name": "CTC-SP2",
        "key": "ctc_sp2",
        "dataset": "CTC-SP2-1996-3.1-cln.parquet",
        "model_dir": "runtime_CTC-SP2-1996-3.1-cln",
        "capacity_cpus": 512,
    },
    {
        "name": "HPC2N",
        "key": "hpc2n",
        "dataset": "HPC2N-2002-2.2-cln.parquet",
        "model_dir": "runtime_HPC2N-2002-2.2-cln",
        "capacity_cpus": 240,
    },
]

# Ordered: cheap baselines first, prediction-based candidates next.
# CONSERVATIVE last: full-queue reservations make it by far the slowest cell
# on congested traces, and it must not block the rest of the sweep.
POLICIES = [
    "FIFO_STRICT",
    "EASY_BACKFILL_BASELINE",
    "SJF_BACKFILL",
    "LJF_BACKFILL",
    "FAIRSHARE_BACKFILL",
    "EASY_BACKFILL_TSAFRIR",
    "ML_BACKFILL_P50",
    "ML_BACKFILL_P10",
    "CONSERVATIVE_BACKFILL_BASELINE",
]

ML_POLICIES = {"ML_BACKFILL_P50", "ML_BACKFILL_P10"}


def run_cell(trace: dict, policy: str) -> dict:
    import pandas as pd
    from hpcopt.simulate.core import run_simulation_from_trace

    trace_df = pd.read_parquet(CURATED_DIR / trace["dataset"])

    runtime_predictor = None
    if policy in ML_POLICIES:
        from hpcopt.models.runtime_quantile import RuntimeQuantilePredictor

        model_dir = MODELS_DIR / trace["model_dir"]
        if not model_dir.exists():
            return {"status": "skipped", "reason": f"missing model dir {model_dir.name}"}
        runtime_predictor = RuntimeQuantilePredictor(model_dir)

    started = time.perf_counter()
    result = run_simulation_from_trace(
        trace_df=trace_df,
        policy_id=policy,
        capacity_cpus=trace["capacity_cpus"],
        run_id=f"matrix_{trace['key']}_{policy.lower()}",
        strict_invariants=False,
        runtime_predictor=runtime_predictor,
    )
    elapsed = time.perf_counter() - started

    invariants = result.invariant_report
    return {
        "status": "ok",
        "jobs": int(len(result.jobs_df)),
        "wall_time_sec": round(elapsed, 1),
        "metrics": result.metrics,
        "objective_metrics": result.objective_metrics,
        "fallback_accounting": result.fallback_accounting,
        "invariant_violations": int(invariants.get("violation_count", 0)),
    }


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# Policy Matrix (reference traces, Python reference simulator)",
        "",
        "| Trace | Policy | p95 BSLD | Utilization | Mean Wait (s) | p95 Wait (s) | Violations | Wall (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        if row["result"]["status"] != "ok":
            lines.append(
                f"| {row['trace']} | {row['policy']} | — | — | — | — | — | {row['result']['reason']} |"
            )
            continue
        obj = row["result"]["objective_metrics"]
        met = row["result"]["metrics"]
        p95_bsld = obj.get("p95_bsld", met.get("p95_bsld"))
        util = obj.get("utilization_cpu", met.get("utilization_cpu"))
        lines.append(
            "| {trace} | {policy} | {bsld:,.2f} | {util:.1%} | {mw:,.0f} | {pw:,.0f} | {viol} | {wall:,.1f} |".format(
                trace=row["trace"],
                policy=row["policy"],
                bsld=float(p95_bsld),
                util=float(util),
                mw=float(met.get("mean_wait_sec", float("nan"))),
                pw=float(met.get("p95_wait_sec", float("nan"))),
                viol=row["result"]["invariant_violations"],
                wall=row["result"]["wall_time_sec"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default=",".join(t["key"] for t in TRACES))
    parser.add_argument("--policies", default=",".join(POLICIES))
    args = parser.parse_args()

    trace_keys = set(args.traces.split(","))
    policies = [p for p in args.policies.split(",") if p]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "policy_matrix.json"
    md_path = OUT_DIR / "policy_matrix.md"

    # Resume: keep completed cells from a previous (possibly interrupted) sweep.
    results: list[dict] = []
    if json_path.exists():
        prior = json.loads(json_path.read_text(encoding="utf-8"))
        results = [row for row in prior if row["result"]["status"] == "ok"]
    done = {(row["trace"], row["policy"]) for row in results}

    for trace in TRACES:
        if trace["key"] not in trace_keys:
            continue
        for policy in policies:
            label = f"{trace['name']} x {policy}"
            if (trace["name"], policy) in done:
                print(f"[matrix] skipping {label} (already complete)", flush=True)
                continue
            print(f"[matrix] running {label} ...", flush=True)
            try:
                cell = run_cell(trace, policy)
            except Exception as exc:  # keep the sweep alive; record the failure
                cell = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            results.append({"trace": trace["name"], "policy": policy, "result": cell})
            status = cell["status"]
            wall = cell.get("wall_time_sec", "-")
            print(f"[matrix] {label}: {status} (wall={wall}s)", flush=True)
            # Persist incrementally so partial sweeps are usable.
            json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            md_path.write_text(render_markdown(results), encoding="utf-8")

    print(f"[matrix] done: {json_path}")
    print(f"[matrix] done: {md_path}")


if __name__ == "__main__":
    main()
