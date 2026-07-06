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
    "RL_TRAINED",
    "CONSERVATIVE_BACKFILL_BASELINE",
]

ML_POLICIES = {"ML_BACKFILL_P50", "ML_BACKFILL_P10"}

# Rough relative wall-time weights used to schedule the longest cells first
# when running with --workers > 1 (better core packing; the sweep's wall time
# approaches the single longest cell instead of an unlucky tail).
_POLICY_COST = {
    "ML_BACKFILL_P50": 100,
    "ML_BACKFILL_P10": 100,
    "CONSERVATIVE_BACKFILL_BASELINE": 80,
    "FIFO_STRICT": 20,
}
_TRACE_COST = {"hpc2n": 4, "sdsc_sp2": 2, "ctc_sp2": 1}


def _cell_cost(trace: dict, policy: str) -> int:
    return _TRACE_COST.get(trace["key"], 1) * _POLICY_COST.get(policy, 5)


def run_cell(trace: dict, policy: str) -> dict:
    if policy == "RL_TRAINED":
        # torch must load before pandas/pyarrow on Windows: if pyarrow's DLLs
        # load first, torch's c10.dll fails to initialize (WinError 1114).
        try:
            import torch  # noqa: F401
        except ImportError:
            pass

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

    policy_context = None
    if policy == "RL_TRAINED":
        from hpcopt.rl.inference import RLPolicy

        checkpoint = MODELS_DIR / "rl" / f"ppo_{trace['key']}.zip"
        if not checkpoint.exists():
            return {"status": "skipped", "reason": f"missing RL checkpoint {checkpoint.name}"}
        policy_context = {"rl_policy": RLPolicy.load(checkpoint)}

    started = time.perf_counter()
    result = run_simulation_from_trace(
        trace_df=trace_df,
        policy_id=policy,
        capacity_cpus=trace["capacity_cpus"],
        run_id=f"matrix_{trace['key']}_{policy.lower()}",
        strict_invariants=False,
        runtime_predictor=runtime_predictor,
        policy_context=policy_context,
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


def _canonical_sort(results: list[dict]) -> list[dict]:
    trace_order = {t["name"]: i for i, t in enumerate(TRACES)}
    policy_order = {p: i for i, p in enumerate(POLICIES)}
    return sorted(
        results,
        key=lambda row: (
            trace_order.get(row["trace"], 99),
            policy_order.get(row["policy"], 99),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default=",".join(t["key"] for t in TRACES))
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Run up to N cells as parallel processes (cells are independent).",
    )
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

    pending: list[tuple[dict, str]] = []
    for trace in TRACES:
        if trace["key"] not in trace_keys:
            continue
        for policy in policies:
            if (trace["name"], policy) in done:
                print(f"[matrix] skipping {trace['name']} x {policy} (already complete)", flush=True)
            else:
                pending.append((trace, policy))

    def record(trace_name: str, policy: str, cell: dict) -> None:
        results.append({"trace": trace_name, "policy": policy, "result": cell})
        wall = cell.get("wall_time_sec", "-")
        print(f"[matrix] {trace_name} x {policy}: {cell['status']} (wall={wall}s)", flush=True)
        # Persist incrementally so partial sweeps are usable.
        ordered = _canonical_sort(results)
        json_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(ordered), encoding="utf-8")

    if args.workers <= 1:
        for trace, policy in pending:
            print(f"[matrix] running {trace['name']} x {policy} ...", flush=True)
            try:
                cell = run_cell(trace, policy)
            except Exception as exc:  # keep the sweep alive; record the failure
                cell = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            record(trace["name"], policy, cell)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Longest-expected cells first so they don't trail the sweep alone.
        pending.sort(key=lambda tp: _cell_cost(*tp), reverse=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for trace, policy in pending:
                print(f"[matrix] queueing {trace['name']} x {policy} ...", flush=True)
                futures[pool.submit(run_cell, trace, policy)] = (trace["name"], policy)
            for future in as_completed(futures):
                trace_name, policy = futures[future]
                try:
                    cell = future.result()
                except Exception as exc:  # keep the sweep alive; record the failure
                    cell = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
                record(trace_name, policy, cell)

    print(f"[matrix] done: {json_path}")
    print(f"[matrix] done: {md_path}")


if __name__ == "__main__":
    main()
