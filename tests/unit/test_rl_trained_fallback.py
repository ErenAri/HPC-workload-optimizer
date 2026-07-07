"""RL_TRAINED policy fallback path — runs without gymnasium / sb3.

Validates that requesting ``RL_TRAINED`` with no loaded model degrades
gracefully to FIFO behaviour rather than crashing the simulator.  This
keeps the policy registered as 'safe to enumerate' even on systems
without the ``[rl]`` extras.
"""

from __future__ import annotations

import pandas as pd
from hpcopt.simulate.core import SUPPORTED_POLICIES, run_simulation_from_trace


def _make_trace(n: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "job_id": i,
                "submit_ts": i * 5,
                "runtime_actual_sec": 50 + (i % 4) * 25,
                "requested_cpus": 1 + (i % 4),
            }
        )
    return pd.DataFrame(rows)


def test_rl_trained_registered_in_supported_policies():
    assert "RL_TRAINED" in SUPPORTED_POLICIES


def test_rl_trained_falls_back_to_fifo_without_policy():
    trace = _make_trace(8)
    result = run_simulation_from_trace(
        trace_df=trace,
        policy_id="RL_TRAINED",
        capacity_cpus=8,
        run_id="rl_fallback",
        strict_invariants=True,
        policy_context=None,
    )
    assert len(result.jobs_df) == len(trace)
    assert result.invariant_report["violations"] == []


def test_choose_rl_trained_dispatch_loop_with_stub_policy() -> None:
    """The RL dispatch loop re-encodes state after each pick, honors the
    fit mask, and stops when the model points at a masked slot."""
    from hpcopt.rl.inference import choose_rl_trained
    from hpcopt.simulate.adapter import AdapterQueuedJob, SchedulerStateSnapshot

    def queued(job_id: int, cpus: int) -> AdapterQueuedJob:
        return AdapterQueuedJob(
            job_id=job_id,
            submit_ts=job_id,
            requested_cpus=cpus,
            runtime_estimate_sec=100,
            runtime_p90_sec=100,
            runtime_guard_sec=100,
            estimate_source="test",
        )

    snapshot = SchedulerStateSnapshot(
        clock_ts=0,
        capacity_cpus=8,
        free_cpus=8,
        queued_jobs=(queued(1, 4), queued(2, 4), queued(3, 4)),
        running_jobs=tuple(),
    )

    class StubPolicy:
        """Always picks the first still-dispatchable queue slot."""

        def predict_action(self, obs, action_masks):
            import numpy as np

            return int(np.argmax(action_masks))

    decision = choose_rl_trained(snapshot, StubPolicy())
    # Two 4-cpu jobs fit in 8 cpus; the third no longer fits, mask goes
    # empty, and the loop stops.
    assert [d.job_id for d in decision.decisions] == [1, 2]
    assert all(d.reason == "rl_trained_pick" for d in decision.decisions)
    assert decision.policy_id == "RL_TRAINED"
