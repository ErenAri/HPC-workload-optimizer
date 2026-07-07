"""Shared helpers for the simulation core."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from hpcopt.models.baseline_tsafrir import compute_tsafrir_estimates
from hpcopt.simulate.adapter import (
    SchedulerStateSnapshot,
    choose_conservative_backfill,
    choose_easy_backfill,
    choose_fairshare_backfill,
    choose_fifo_strict,
    choose_ljf_backfill,
    choose_ml_backfill_p50,
    choose_sjf_backfill,
)
from hpcopt.simulate.fairshare import compute_fairshare_priorities

logger = logging.getLogger(__name__)


def coerce_trace_df(trace_df: pd.DataFrame) -> pd.DataFrame:
    required = {"job_id", "submit_ts", "runtime_actual_sec"}
    missing = required - set(trace_df.columns)
    if missing:
        raise ValueError(f"Trace dataframe missing required columns: {sorted(missing)}")

    df = trace_df.copy()
    if "requested_cpus" not in df.columns:
        if "allocated_cpus" in df.columns:
            df["requested_cpus"] = df["allocated_cpus"]
        else:
            raise ValueError("Trace requires requested_cpus (or allocated_cpus fallback)")

    if "runtime_requested_sec" not in df.columns:
        df["runtime_requested_sec"] = None
    for col in ["user_id", "group_id", "queue_id", "partition_id", "requested_mem"]:
        if col not in df.columns:
            df[col] = None

    for col in ["job_id", "submit_ts", "runtime_actual_sec", "requested_cpus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["job_id", "submit_ts", "runtime_actual_sec", "requested_cpus"])
    if df.empty:
        raise ValueError("Trace dataframe has no valid rows after coercion.")

    df["job_id"] = df["job_id"].astype(int)
    df["submit_ts"] = df["submit_ts"].astype(int)
    df["runtime_actual_sec"] = df["runtime_actual_sec"].clip(lower=0).astype(int)
    df["requested_cpus"] = df["requested_cpus"].clip(lower=1).astype(int)

    df["runtime_requested_sec"] = pd.to_numeric(df["runtime_requested_sec"], errors="coerce")
    df["runtime_requested_sec"] = df["runtime_requested_sec"].where(df["runtime_requested_sec"] > 0)
    df["runtime_estimate_sec"] = df["runtime_requested_sec"].fillna(df["runtime_actual_sec"]).astype(int)

    df = df.sort_values(["submit_ts", "job_id"]).reset_index(drop=True)
    return df


def choose_decisions(
    snapshot: SchedulerStateSnapshot,
    policy_id: str,
    strict_uncertainty_mode: bool = False,
    policy_context: dict[str, Any] | None = None,
) -> Any:
    if policy_id == "FIFO_STRICT":
        return choose_fifo_strict(snapshot)
    if policy_id == "EASY_BACKFILL_BASELINE":
        return choose_easy_backfill(snapshot)
    if policy_id == "EASY_BACKFILL_TSAFRIR":
        # Same dispatch logic as EASY_BACKFILL_BASELINE; only the per-job
        # runtime_estimate_sec carried in the snapshot differs (set to the
        # Tsafrir prediction by attach_runtime_estimates).
        decision = choose_easy_backfill(snapshot)
        # Re-tag the policy id so downstream metrics/manifests attribute results correctly.
        return type(decision)(
            policy_id="EASY_BACKFILL_TSAFRIR",
            reservation_ts=decision.reservation_ts,
            decisions=decision.decisions,
        )
    if policy_id == "CONSERVATIVE_BACKFILL_BASELINE":
        return choose_conservative_backfill(snapshot)
    if policy_id == "SJF_BACKFILL":
        return choose_sjf_backfill(snapshot)
    if policy_id == "LJF_BACKFILL":
        return choose_ljf_backfill(snapshot)
    if policy_id == "FAIRSHARE_BACKFILL":
        return choose_fairshare_backfill(snapshot)
    if policy_id == "RL_TRAINED":
        # Lazy import: keeps gymnasium/torch optional.
        from hpcopt.rl.inference import choose_rl_trained
        rl_policy = (policy_context or {}).get("rl_policy")
        return choose_rl_trained(snapshot, rl_policy)
    if policy_id in ("ML_BACKFILL_P50", "ML_BACKFILL_P10"):
        # Both ML policies use the same dispatcher; the difference is that
        # ML_BACKFILL_P10 sets runtime_estimate_sec to p10 (conservative) in
        # attach_runtime_estimates, so the backfill window is tighter.
        return choose_ml_backfill_p50(
            snapshot=snapshot,
            strict_uncertainty_mode=strict_uncertainty_mode,
        )
    # Registered plug-in policies (hpcopt.plugins) get the same snapshot the
    # built-ins do; lazy import keeps plugin discovery off the hot path for
    # built-in-only runs.
    from hpcopt import plugins

    spec = plugins.get_policy(policy_id)
    if spec is not None:
        return spec.chooser(snapshot)
    raise ValueError(f"Unsupported policy_id '{policy_id}'")


def build_prediction_features(job: dict[str, Any]) -> dict[str, Any]:
    requested_runtime = int(job["runtime_requested_sec"]) if pd.notna(job.get("runtime_requested_sec")) else None
    return {
        "submit_ts": int(job["submit_ts"]),
        "requested_cpus": int(job["requested_cpus"]),
        "runtime_requested_sec": requested_runtime,
        "requested_mem": (int(job["requested_mem"]) if pd.notna(job.get("requested_mem")) else None),
        "queue_id": int(job["queue_id"]) if pd.notna(job.get("queue_id")) else None,
        "partition_id": (int(job["partition_id"]) if pd.notna(job.get("partition_id")) else None),
        "user_id": int(job["user_id"]) if pd.notna(job.get("user_id")) else None,
        "group_id": int(job["group_id"]) if pd.notna(job.get("group_id")) else None,
        # Populate lookback-style features when unavailable in online inference.
        "user_overrequest_mean_lookback": (
            float(job["user_overrequest_mean_lookback"]) if pd.notna(job.get("user_overrequest_mean_lookback")) else 1.0
        ),
        "user_runtime_median_lookback": (
            int(job["user_runtime_median_lookback"])
            if pd.notna(job.get("user_runtime_median_lookback"))
            else requested_runtime
        ),
        "queue_congestion_at_submit_jobs": (
            int(job["queue_congestion_at_submit_jobs"]) if pd.notna(job.get("queue_congestion_at_submit_jobs")) else 0
        ),
    }


def attach_runtime_estimates(
    jobs_df: pd.DataFrame,
    policy_id: str,
    runtime_predictor: Any | None,
    runtime_guard_k: float,
) -> pd.DataFrame:
    df = jobs_df.copy()

    if policy_id == "EASY_BACKFILL_TSAFRIR":
        # Tsafrir/Etsion/Feitelson 2007 user-history predictor. Per-job
        # estimate is precomputed by a chronological scan with per-user
        # completion history. Falls back to the user-supplied wall-time
        # request on cold start (zero completed jobs) via compute_tsafrir_estimates.
        requested = pd.to_numeric(df["runtime_requested_sec"], errors="coerce")
        has_requested = requested.notna() & (requested > 0)
        actual = df["runtime_actual_sec"].astype(int)
        user_estimate = actual.copy()
        user_estimate[has_requested] = requested[has_requested].astype(int)
        # Ensure the column used by compute_tsafrir_estimates is populated.
        df["runtime_estimate_sec"] = user_estimate
        df = compute_tsafrir_estimates(df)
        estimate = df["tsafrir_runtime_sec"].astype(int)
        source = pd.Series("tsafrir_cold_start", index=df.index)
        source[df["tsafrir_history_count"] >= 1] = "tsafrir_history_1"
        source[df["tsafrir_history_count"] >= 2] = "tsafrir_history_2"
        df["runtime_p50_sec"] = estimate
        df["runtime_p90_sec"] = estimate
        df["runtime_guard_sec"] = estimate
        df["runtime_estimate_sec"] = estimate
        df["estimate_source"] = source
        return df

    if policy_id not in ("ML_BACKFILL_P50", "ML_BACKFILL_P10"):
        # Vectorized path for non-ML policies (FIFO, EASY, CBF, SJF, LJF, FAIRSHARE).
        requested = pd.to_numeric(df["runtime_requested_sec"], errors="coerce")
        has_requested = requested.notna() & (requested > 0)
        actual = df["runtime_actual_sec"].astype(int)

        estimate = actual.copy()
        estimate[has_requested] = requested[has_requested].astype(int)

        source = pd.Series("actual_fallback", index=df.index)
        source[has_requested] = "requested_fallback"

        df["runtime_p50_sec"] = estimate
        df["runtime_p90_sec"] = estimate
        df["runtime_guard_sec"] = estimate
        df["runtime_estimate_sec"] = estimate
        df["estimate_source"] = source

        if policy_id == "FAIRSHARE_BACKFILL":
            df["priority_score"] = compute_fairshare_priorities(df)
        return df

    # ML policies: per-row prediction needed, but collect into lists
    # for bulk column assignment instead of slow per-cell df.at[].
    p50_list: list[int] = []
    p90_list: list[int] = []
    guard_list: list[int] = []
    estimate_list: list[int] = []
    source_list: list[str] = []

    for _idx, row in df.iterrows():
        requested_runtime = row.get("runtime_requested_sec")
        actual_runtime = int(row["runtime_actual_sec"])

        predicted = None
        if runtime_predictor is not None:
            try:
                predicted = runtime_predictor.predict_one(build_prediction_features(row.to_dict()))
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Runtime prediction failed for job %s: %s", row.get("job_id"), exc)

        if predicted is not None:
            p50 = int(max(1, round(predicted["p50"])))
            p90 = int(max(p50, round(predicted["p90"])))
            if policy_id == "ML_BACKFILL_P10":
                p10 = int(max(1, round(predicted["p10"])))
                estimate = p10
                guard = int(round(p10 + runtime_guard_k * (p50 - p10)))
            else:
                estimate = p50
                guard = int(round(p50 + runtime_guard_k * (p90 - p50)))
            source = "prediction"
        elif pd.notna(requested_runtime) and float(requested_runtime) > 0:
            p50 = int(requested_runtime)
            p90 = int(requested_runtime)
            estimate = p50
            guard = int(requested_runtime)
            source = "requested_fallback"
        else:
            p50 = actual_runtime
            p90 = actual_runtime
            estimate = p50
            guard = actual_runtime
            source = "actual_fallback"

        p50_list.append(p50)
        p90_list.append(p90)
        guard_list.append(max(1, guard))
        estimate_list.append(estimate)
        source_list.append(source)

    df["runtime_p50_sec"] = p50_list
    df["runtime_p90_sec"] = p90_list
    df["runtime_guard_sec"] = guard_list
    df["runtime_estimate_sec"] = estimate_list
    df["estimate_source"] = source_list
    return df


def check_invariants(
    clock_ts: int,
    capacity_cpus: int,
    free_cpus: int,
    queued_jobs: list[dict[str, Any]],
    running_jobs: list[dict[str, Any]],
) -> list[str]:
    failed: list[str] = []
    if free_cpus < 0:
        failed.append("free_cpus_negative")
    if free_cpus > capacity_cpus:
        failed.append("free_cpus_exceeds_capacity")

    running_cpu = sum(int(job["requested_cpus"]) for job in running_jobs)
    if running_cpu + free_cpus != capacity_cpus:
        failed.append("cpu_conservation_broken")

    queued_ids = {int(job["job_id"]) for job in queued_jobs}
    running_ids = {int(job["job_id"]) for job in running_jobs}
    overlap = queued_ids & running_ids
    if overlap:
        failed.append("job_exists_in_queue_and_running")

    for running in running_jobs:
        if int(running["start_ts"]) < int(running["submit_ts"]):
            failed.append(f"job_start_before_submit:{running['job_id']}")
        if int(running["end_ts"]) < int(running["start_ts"]):
            failed.append(f"job_end_before_start:{running['job_id']}")
        if int(running["requested_cpus"]) <= 0:
            failed.append(f"job_nonpositive_cpu:{running['job_id']}")

    for queued in queued_jobs:
        if int(queued["submit_ts"]) > clock_ts:
            failed.append(f"queued_job_submit_in_future:{queued['job_id']}")

    return failed


def invariant_report(
    run_id: str,
    strict_mode: bool,
    step_count: int,
    violations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "strict_mode": bool(strict_mode),
        "step_count": int(step_count),
        "violations": violations,
    }
