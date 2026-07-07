from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from hpcopt.simulate.adapter import (
    AdapterQueuedJob,
    AdapterRunningJob,
    SchedulerStateSnapshot,
    snapshot_state_hash,
)
from hpcopt.simulate.core_helpers import (
    attach_runtime_estimates as _attach_runtime_estimates,
)
from hpcopt.simulate.core_helpers import (
    check_invariants as _check_invariants,
)
from hpcopt.simulate.core_helpers import (
    choose_decisions as _choose_decisions,
)
from hpcopt.simulate.core_helpers import (
    coerce_trace_df as _coerce_trace_df,
)
from hpcopt.simulate.core_helpers import (
    invariant_report as _invariant_report,
)
from hpcopt.simulate.metrics import compute_job_metrics
from hpcopt.simulate.objective import compute_objective_contract_metrics

SUPPORTED_POLICIES = {
    "FIFO_STRICT",
    "EASY_BACKFILL_BASELINE",
    "EASY_BACKFILL_TSAFRIR",
    "CONSERVATIVE_BACKFILL_BASELINE",
    "SJF_BACKFILL",
    "LJF_BACKFILL",
    "FAIRSHARE_BACKFILL",
    "ML_BACKFILL_P50",
    "ML_BACKFILL_P10",
    "RL_TRAINED",
}


@dataclass
class SimulationResult:
    policy_id: str
    jobs_df: pd.DataFrame
    queue_series_df: pd.DataFrame
    metrics: dict[str, float]
    objective_metrics: dict[str, float]
    invariant_report: dict[str, Any]
    fallback_accounting: dict[str, Any]


def run_simulation_from_trace(
    trace_df: pd.DataFrame,
    policy_id: str,
    capacity_cpus: int,
    run_id: str,
    strict_invariants: bool = False,
    runtime_predictor: Any | None = None,
    runtime_guard_k: float = 0.5,
    strict_uncertainty_mode: bool = False,
    starvation_wait_cap_sec: int = 172800,
    policy_context: dict[str, Any] | None = None,
) -> SimulationResult:
    if policy_id not in SUPPORTED_POLICIES:
        from hpcopt import plugins

        if not plugins.is_registered(policy_id):
            raise ValueError(
                f"Unsupported policy: {policy_id}. Available: {', '.join(plugins.all_policy_ids())}"
            )
    if capacity_cpus <= 0:
        raise ValueError("capacity_cpus must be > 0")

    jobs_df = _coerce_trace_df(trace_df)
    jobs_df = _attach_runtime_estimates(
        jobs_df=jobs_df,
        policy_id=policy_id,
        runtime_predictor=runtime_predictor,
        runtime_guard_k=runtime_guard_k,
    )
    jobs = jobs_df.to_dict(orient="records")
    total_jobs = len(jobs)

    submit_idx = 0
    queue: list[dict[str, Any]] = []
    running: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    queue_series: list[dict[str, int]] = []
    free_cpus = int(capacity_cpus)
    clock_ts = int(jobs[0]["submit_ts"])

    step_index = 0
    violations: list[dict[str, Any]] = []
    fallback_counts = {
        "prediction_used_count": 0,
        "requested_fallback_count": 0,
        "actual_fallback_count": 0,
        "tsafrir_history_count": 0,
        "tsafrir_cold_start_count": 0,
    }

    while len(completed) < total_jobs:
        next_submit_ts = int(jobs[submit_idx]["submit_ts"]) if submit_idx < total_jobs else 10**18
        next_complete_ts = min(int(job["end_ts"]) for job in running) if running else 10**18
        next_ts = min(next_submit_ts, next_complete_ts)
        if next_ts >= 10**18:
            raise RuntimeError("Simulation deadlock: no next event; check resource sizing/policy.")
        clock_ts = next_ts

        completed_now = sorted(
            [job for job in running if int(job["end_ts"]) == clock_ts],
            key=lambda job: int(job["job_id"]),
        )
        if completed_now:
            completed_ids = {int(job["job_id"]) for job in completed_now}
            running = [job for job in running if int(job["job_id"]) not in completed_ids]
            for job in completed_now:
                free_cpus += int(job["requested_cpus"])
                completed.append(job)

        while submit_idx < total_jobs and int(jobs[submit_idx]["submit_ts"]) == clock_ts:
            queue.append(jobs[submit_idx])
            submit_idx += 1
        queue.sort(key=lambda job: (int(job["submit_ts"]), int(job["job_id"])))

        snapshot = SchedulerStateSnapshot(
            clock_ts=clock_ts,
            capacity_cpus=capacity_cpus,
            free_cpus=free_cpus,
            queued_jobs=tuple(
                AdapterQueuedJob(
                    job_id=int(job["job_id"]),
                    submit_ts=int(job["submit_ts"]),
                    requested_cpus=int(job["requested_cpus"]),
                    runtime_estimate_sec=int(job["runtime_estimate_sec"]),
                    runtime_p90_sec=int(job["runtime_p90_sec"]),
                    runtime_guard_sec=int(job["runtime_guard_sec"]),
                    estimate_source=str(job["estimate_source"]),
                    priority_score=(
                        float(job["priority_score"])
                        if job.get("priority_score") is not None and pd.notna(job.get("priority_score"))
                        else None
                    ),
                )
                for job in queue
            ),
            # Contract (parse_state_snapshot): running_jobs sorted by
            # (end_ts, job_id). The EASY-family reservation accumulates freed
            # CPUs in iteration order, so dispatch order here would yield a
            # wrong shadow time.
            running_jobs=tuple(
                sorted(
                    (
                        AdapterRunningJob(
                            job_id=int(job["job_id"]),
                            end_ts=int(job["end_ts"]),
                            allocated_cpus=int(job["requested_cpus"]),
                        )
                        for job in running
                    ),
                    key=lambda rj: (rj.end_ts, rj.job_id),
                )
            ),
        )
        decision = _choose_decisions(
            snapshot=snapshot,
            policy_id=policy_id,
            strict_uncertainty_mode=strict_uncertainty_mode,
            policy_context=policy_context,
        )
        queue_by_id: dict[int, dict[str, Any]] = {int(j["job_id"]): j for j in queue}
        dispatched_ids: set[int] = set()
        for dispatch in decision.decisions:
            dispatch_id = int(dispatch.job_id)
            if dispatch_id not in queue_by_id:
                continue
            job = queue_by_id[dispatch_id]
            requested = int(job["requested_cpus"])
            if requested > free_cpus:
                continue

            dispatched_ids.add(dispatch_id)
            start_ts = clock_ts
            runtime = int(job["runtime_actual_sec"])
            end_ts = start_ts + runtime
            estimate_source = str(job.get("estimate_source", "unknown"))
            running.append(
                {
                    "job_id": int(job["job_id"]),
                    "submit_ts": int(job["submit_ts"]),
                    "start_ts": int(start_ts),
                    "end_ts": int(end_ts),
                    "runtime_actual_sec": int(runtime),
                    "requested_cpus": int(requested),
                    "runtime_estimate_sec": int(job["runtime_estimate_sec"]),
                    "runtime_p90_sec": int(job["runtime_p90_sec"]),
                    "runtime_guard_sec": int(job["runtime_guard_sec"]),
                    "estimate_source": estimate_source,
                    "user_id": job.get("user_id"),
                    "group_id": job.get("group_id"),
                    "queue_id": job.get("queue_id"),
                    "partition_id": job.get("partition_id"),
                }
            )
            free_cpus -= requested
            if estimate_source == "prediction":
                fallback_counts["prediction_used_count"] += 1
            elif estimate_source == "requested_fallback":
                fallback_counts["requested_fallback_count"] += 1
            elif estimate_source == "tsafrir_cold_start":
                fallback_counts["tsafrir_cold_start_count"] += 1
            elif estimate_source in ("tsafrir_history_1", "tsafrir_history_2"):
                fallback_counts["tsafrir_history_count"] += 1
            else:
                fallback_counts["actual_fallback_count"] += 1

        if dispatched_ids:
            queue = [job for job in queue if int(job["job_id"]) not in dispatched_ids]

        failed = _check_invariants(
            clock_ts=clock_ts,
            capacity_cpus=capacity_cpus,
            free_cpus=free_cpus,
            queued_jobs=queue,
            running_jobs=running,
        )
        if failed:
            violation = {
                "step_index": step_index,
                "event_type": "tick",
                "clock_ts": clock_ts,
                "failed_invariants": failed,
                "severity": "error",
                "state_hash": snapshot_state_hash(snapshot),
            }
            violations.append(violation)
            if strict_invariants:
                raise RuntimeError(f"Strict invariant violation: {failed}")

        queue_series.append(
            {
                "ts": int(clock_ts),
                "queue_len_jobs": int(len(queue)),
                "queue_len_cpu_demand": int(sum(int(job["requested_cpus"]) for job in queue)),
            }
        )
        step_index += 1

        if not running and submit_idx >= total_jobs and queue:
            raise RuntimeError(
                "Simulation cannot progress: queued jobs remain but none can be dispatched. "
                "Likely requested_cpus > capacity_cpus."
            )

    completed_df = pd.DataFrame(completed)
    if completed_df.empty:
        completed_df = pd.DataFrame(
            columns=[
                "job_id",
                "submit_ts",
                "start_ts",
                "end_ts",
                "runtime_actual_sec",
                "requested_cpus",
            ]
        )
    completed_df = completed_df.sort_values(["job_id"]).reset_index(drop=True)
    metrics = compute_job_metrics(completed_df, capacity_cpus=capacity_cpus)
    objective_metrics = compute_objective_contract_metrics(
        jobs_df=completed_df,
        capacity_cpus=capacity_cpus,
        starvation_wait_cap_sec=starvation_wait_cap_sec,
    )
    queue_series_df = pd.DataFrame(queue_series).sort_values("ts").drop_duplicates("ts", keep="last")
    queue_series_df = queue_series_df.reset_index(drop=True)

    invariant_report = _invariant_report(
        run_id=run_id,
        strict_mode=strict_invariants,
        step_count=step_index,
        violations=violations,
    )
    total_scheduled = int(len(completed_df))
    denominator = total_scheduled if total_scheduled > 0 else 1
    fallback_accounting = {
        **fallback_counts,
        "prediction_used_rate": float(fallback_counts["prediction_used_count"] / denominator),
        "requested_fallback_rate": float(fallback_counts["requested_fallback_count"] / denominator),
        "actual_fallback_rate": float(fallback_counts["actual_fallback_count"] / denominator),
        "tsafrir_history_rate": float(fallback_counts["tsafrir_history_count"] / denominator),
        "tsafrir_cold_start_rate": float(fallback_counts["tsafrir_cold_start_count"] / denominator),
        "total_scheduled_jobs": total_scheduled,
        "runtime_guard_k": float(runtime_guard_k),
        "strict_uncertainty_mode": bool(strict_uncertainty_mode),
    }
    return SimulationResult(
        policy_id=policy_id,
        jobs_df=completed_df,
        queue_series_df=queue_series_df,
        metrics=metrics,
        objective_metrics=objective_metrics,
        invariant_report=invariant_report,
        fallback_accounting=fallback_accounting,
    )


def build_observed_jobs_df(trace_df: pd.DataFrame) -> pd.DataFrame:
    required = {"job_id", "submit_ts", "start_ts", "end_ts"}
    missing = required - set(trace_df.columns)
    if missing:
        raise ValueError(f"Trace missing required observed columns: {sorted(missing)}")

    df = trace_df.copy()
    if "requested_cpus" not in df.columns:
        if "allocated_cpus" in df.columns:
            df["requested_cpus"] = df["allocated_cpus"]
        else:
            raise ValueError("Trace requires requested_cpus (or allocated_cpus fallback)")

    for col in ["job_id", "submit_ts", "start_ts", "end_ts", "requested_cpus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["job_id", "submit_ts", "start_ts", "end_ts", "requested_cpus"])
    df["job_id"] = df["job_id"].astype(int)
    df["submit_ts"] = df["submit_ts"].astype(int)
    df["start_ts"] = df["start_ts"].astype(int)
    df["end_ts"] = df["end_ts"].astype(int)
    df["requested_cpus"] = df["requested_cpus"].clip(lower=1).astype(int)
    return df.sort_values(["submit_ts", "job_id"]).reset_index(drop=True)


def build_observed_queue_series(trace_df: pd.DataFrame) -> pd.DataFrame:
    observed_jobs = build_observed_jobs_df(trace_df)
    events: list[tuple[int, int, int]] = []
    for row in observed_jobs.itertuples(index=False):
        events.append((int(row.submit_ts), 0, int(row.requested_cpus)))
        events.append((int(row.start_ts), 1, int(row.requested_cpus)))

    events.sort(key=lambda item: (item[0], item[1]))
    queue_len = 0
    queue_cpu = 0
    out_rows: list[dict[str, int]] = []

    i = 0
    while i < len(events):
        ts = events[i][0]
        while i < len(events) and events[i][0] == ts and events[i][1] == 0:
            queue_len += 1
            queue_cpu += events[i][2]
            i += 1
        while i < len(events) and events[i][0] == ts and events[i][1] == 1:
            queue_len -= 1
            queue_cpu -= events[i][2]
            i += 1

        queue_len = max(queue_len, 0)
        queue_cpu = max(queue_cpu, 0)
        out_rows.append(
            {
                "ts": int(ts),
                "queue_len_jobs": int(queue_len),
                "queue_len_cpu_demand": int(queue_cpu),
            }
        )
    if not out_rows:
        out_rows.append({"ts": 0, "queue_len_jobs": 0, "queue_len_cpu_demand": 0})
    return pd.DataFrame(out_rows).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
