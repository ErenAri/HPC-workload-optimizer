"""Tests for the public policy plug-in API (hpcopt.plugins) and the bundled
UARP reference plugin."""

from __future__ import annotations

import pandas as pd
import pytest
from hpcopt import plugins
from hpcopt.plugins import (
    NEVER_TS,
    AdapterQueuedJob,
    AdapterRunningJob,
    SchedulerDecision,
    SchedulerStateSnapshot,
    earliest_start_for,
    register_policy,
)
from hpcopt.plugins.uarp import choose_uarp_backfill
from hpcopt.simulate.adapter import choose_easy_backfill
from hpcopt.simulate.core import run_simulation_from_trace


@pytest.fixture()
def scratch_registry():
    """Track policy ids registered during a test and unregister afterwards."""
    added: list[str] = []
    yield added
    for policy_id in added:
        plugins._registry.pop(policy_id, None)


def _queued(job_id: int, submit_ts: int, cpus: int, estimate: int, guard: int | None = None) -> AdapterQueuedJob:
    return AdapterQueuedJob(
        job_id=job_id,
        submit_ts=submit_ts,
        requested_cpus=cpus,
        runtime_estimate_sec=estimate,
        runtime_p90_sec=guard,
        runtime_guard_sec=guard,
        estimate_source="test",
    )


# ── registry ─────────────────────────────────────────────────────────


def test_uarp_is_discovered_with_metadata() -> None:
    assert plugins.is_registered("UARP_BACKFILL")
    spec = plugins.get_policy("UARP_BACKFILL")
    assert spec is not None
    assert spec.chooser is choose_uarp_backfill
    assert spec.author is not None
    assert "UARP_BACKFILL" in plugins.all_policy_ids()
    # Built-ins are part of the combined id list too.
    assert "EASY_BACKFILL_BASELINE" in plugins.all_policy_ids()


def test_register_rejects_builtin_shadowing() -> None:
    with pytest.raises(ValueError, match="shadows a built-in"):
        register_policy("EASY_BACKFILL_BASELINE")(lambda s: None)


def test_register_rejects_taken_id_but_is_idempotent_for_same_callable(scratch_registry) -> None:
    def chooser(snapshot):  # pragma: no cover - never invoked
        raise NotImplementedError

    register_policy("TEST_TAKEN_ID")(chooser)
    scratch_registry.append("TEST_TAKEN_ID")

    # Same callable again: no-op (safe under repeated imports).
    register_policy("TEST_TAKEN_ID")(chooser)

    with pytest.raises(ValueError, match="already registered"):
        register_policy("TEST_TAKEN_ID")(lambda s: None)


@pytest.mark.parametrize("bad_id", ["lower_case", "X", "1STARTS_WITH_DIGIT", "HAS-DASH"])
def test_register_rejects_malformed_policy_ids(bad_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid policy_id"):
        register_policy(bad_id)


def test_unknown_policy_error_lists_available_ids() -> None:
    trace_df = pd.DataFrame(
        {"job_id": [1], "submit_ts": [0], "runtime_actual_sec": [10], "requested_cpus": [1]}
    )
    with pytest.raises(ValueError, match="UARP_BACKFILL"):
        run_simulation_from_trace(
            trace_df=trace_df,
            policy_id="NO_SUCH_POLICY",
            capacity_cpus=4,
            run_id="unknown_policy",
        )


# ── end-to-end through the harness ───────────────────────────────────


def test_registered_plugin_runs_through_the_harness(scratch_registry) -> None:
    from hpcopt.plugins import DispatchDecision

    @register_policy("TEST_GREEDY_FIFO")
    def choose_test_greedy(snapshot: SchedulerStateSnapshot) -> SchedulerDecision:
        decisions = []
        available = snapshot.free_cpus
        for job in snapshot.queued_jobs:
            if 0 < job.requested_cpus <= available:
                decisions.append(
                    DispatchDecision(
                        job_id=job.job_id,
                        requested_cpus=job.requested_cpus,
                        runtime_estimate_sec=job.runtime_estimate_sec,
                        estimated_completion_ts=snapshot.clock_ts + job.runtime_estimate_sec,
                        reason="test_greedy",
                    )
                )
                available -= job.requested_cpus
        return SchedulerDecision(policy_id="TEST_GREEDY_FIFO", reservation_ts=None, decisions=tuple(decisions))

    scratch_registry.append("TEST_GREEDY_FIFO")

    trace_df = pd.DataFrame(
        {
            "job_id": [1, 2, 3, 4],
            "submit_ts": [0, 0, 5, 5],
            "runtime_actual_sec": [30, 20, 10, 40],
            "runtime_requested_sec": [60, 40, 20, 80],
            "requested_cpus": [2, 2, 3, 1],
        }
    )
    result = run_simulation_from_trace(
        trace_df=trace_df,
        policy_id="TEST_GREEDY_FIFO",
        capacity_cpus=4,
        run_id="plugin_e2e",
        strict_invariants=True,
    )
    assert len(result.jobs_df) == 4
    assert result.invariant_report["violations"] == []


def test_uarp_completes_all_jobs_with_strict_invariants() -> None:
    trace_df = pd.DataFrame(
        {
            "job_id": [1, 2, 3, 4],
            "submit_ts": [0, 1, 2, 2],
            "runtime_actual_sec": [100, 50, 10, 60],
            "runtime_requested_sec": [100, 50, 10, 60],
            "requested_cpus": [4, 4, 8, 2],
        }
    )
    result = run_simulation_from_trace(
        trace_df=trace_df,
        policy_id="UARP_BACKFILL",
        capacity_cpus=10,
        run_id="uarp_e2e",
        strict_invariants=True,
    )
    jobs = result.jobs_df.set_index("job_id")
    assert len(jobs) == 4
    # Guard == estimate here, so UARP's gate reduces to EASY: J4 backfills.
    assert int(jobs.loc[4, "start_ts"]) == 2
    assert int(jobs.loc[3, "start_ts"]) == 100


# ── UARP dispatch semantics ──────────────────────────────────────────


def test_uarp_pessimistic_gate_blocks_what_easy_backfills() -> None:
    # Head needs 10; only 2 free until t=100. Candidate estimate 50 fits the
    # window (EASY backfills) but guard 200 does not (UARP refuses).
    snapshot = SchedulerStateSnapshot(
        clock_ts=0,
        capacity_cpus=10,
        free_cpus=2,
        queued_jobs=(
            _queued(1, 0, 10, 30, guard=30),
            _queued(2, 1, 2, 50, guard=200),
        ),
        running_jobs=(AdapterRunningJob(job_id=9, end_ts=100, allocated_cpus=8),),
    )
    easy = choose_easy_backfill(snapshot)
    assert [d.job_id for d in easy.decisions] == [2]

    uarp = choose_uarp_backfill(snapshot)
    assert uarp.reservation_ts == 100
    assert uarp.decisions == tuple()


def test_uarp_backfills_shortest_guard_first() -> None:
    # Two candidates both fit the reservation window, but only 3 cpus are
    # free: shortest-guard job 3 wins even though job 2 arrived earlier.
    snapshot = SchedulerStateSnapshot(
        clock_ts=0,
        capacity_cpus=10,
        free_cpus=3,
        queued_jobs=(
            _queued(1, 0, 10, 30, guard=30),
            _queued(2, 1, 3, 80, guard=80),
            _queued(3, 2, 3, 20, guard=20),
        ),
        running_jobs=(AdapterRunningJob(job_id=9, end_ts=100, allocated_cpus=7),),
    )
    uarp = choose_uarp_backfill(snapshot)
    assert [d.job_id for d in uarp.decisions] == [3]
    assert uarp.decisions[0].reason == "uarp_backfill"


# ── public helper ────────────────────────────────────────────────────


def test_earliest_start_for_accumulates_in_end_time_order() -> None:
    snapshot = SchedulerStateSnapshot(
        clock_ts=2,
        capacity_cpus=10,
        free_cpus=2,
        queued_jobs=tuple(),
        # Deliberately NOT in end-time order: the helper must sort.
        running_jobs=(
            AdapterRunningJob(job_id=1, end_ts=100, allocated_cpus=4),
            AdapterRunningJob(job_id=2, end_ts=51, allocated_cpus=4),
        ),
    )
    assert earliest_start_for(snapshot, 2) == 2  # fits now
    assert earliest_start_for(snapshot, 6) == 51  # after the earlier end
    assert earliest_start_for(snapshot, 8) == 100  # needs both to finish
    assert earliest_start_for(snapshot, 11) == NEVER_TS  # exceeds capacity
