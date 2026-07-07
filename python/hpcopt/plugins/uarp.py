"""UARP: Uncertainty-Aware Reservation Policy (bundled reference plugin).

This is the harness's proof-of-concept "external" policy: it is written
entirely against the public plug-in API (``hpcopt.plugins``) — no imports
from ``hpcopt.simulate`` internals — and is discovered through the same
``hpcopt.policies`` entry-point mechanism a third-party package would use.

The policy is an EASY-backfill skeleton with two changes:

1. **Pessimistic backfill gate.** A candidate may backfill only if its
   *guard* runtime (``runtime_guard_sec``, falling back to p90, falling back
   to the point estimate) finishes by the head-of-line reservation. With ML
   quantile estimates this gates on the upper tail of the runtime
   distribution instead of the median, so underprediction cannot delay the
   reserved head. With plain walltime requests (guard == estimate) the gate
   reduces to standard EASY.

2. **Shortest-guard-first backfill order.** Candidates are packed shortest
   (pessimistic) runtime first — the EASY-SJBF variant shown by Srinivasan
   et al. (IPDPS 2002) to improve slowdown at equal utilisation.
"""

from __future__ import annotations

from hpcopt.plugins import (
    AdapterQueuedJob,
    DispatchDecision,
    SchedulerDecision,
    SchedulerStateSnapshot,
    earliest_start_for,
    register_policy,
)

POLICY_ID = "UARP_BACKFILL"


def _guard_runtime(job: AdapterQueuedJob) -> int:
    """Pessimistic runtime bound used for the backfill gate."""
    for value in (job.runtime_guard_sec, job.runtime_p90_sec, job.runtime_estimate_sec):
        if value is not None:
            return max(0, int(value))
    return 0


def _dispatch(job: AdapterQueuedJob, clock_ts: int, reason: str) -> DispatchDecision:
    return DispatchDecision(
        job_id=job.job_id,
        requested_cpus=job.requested_cpus,
        runtime_estimate_sec=job.runtime_estimate_sec,
        estimated_completion_ts=clock_ts + job.runtime_estimate_sec,
        reason=reason,
    )


@register_policy(
    POLICY_ID,
    name="Uncertainty-Aware Reservation Policy",
    author="hpcopt (bundled reference plugin)",
    description="EASY skeleton; backfill gated on guard/p90 runtime, packed shortest-guard-first",
    version="1.0",
)
def choose_uarp_backfill(snapshot: SchedulerStateSnapshot) -> SchedulerDecision:
    if not snapshot.queued_jobs:
        return SchedulerDecision(policy_id=POLICY_ID, reservation_ts=None, decisions=tuple())

    queue = sorted(snapshot.queued_jobs, key=lambda j: (j.submit_ts, j.job_id))
    hol = queue[0]
    reservation_ts = earliest_start_for(snapshot, hol.requested_cpus)
    available = snapshot.free_cpus
    decisions: list[DispatchDecision] = []

    if 0 < hol.requested_cpus <= available:
        # Head runs now; pack the rest shortest-guard-first.
        decisions.append(_dispatch(hol, snapshot.clock_ts, "uarp_head_dispatch"))
        available -= hol.requested_cpus
        tail = sorted(queue[1:], key=lambda j: (_guard_runtime(j), j.submit_ts, j.job_id))
        for job in tail:
            if job.requested_cpus <= 0 or job.requested_cpus > available:
                continue
            decisions.append(_dispatch(job, snapshot.clock_ts, "uarp_follow_dispatch"))
            available -= job.requested_cpus
        return SchedulerDecision(
            policy_id=POLICY_ID, reservation_ts=reservation_ts, decisions=tuple(decisions)
        )

    # Head blocked: backfill only jobs whose PESSIMISTIC completion respects
    # the head's reservation, shortest guard first.
    candidates = sorted(queue[1:], key=lambda j: (_guard_runtime(j), j.submit_ts, j.job_id))
    for job in candidates:
        if job.requested_cpus <= 0 or job.requested_cpus > available:
            continue
        if snapshot.clock_ts + _guard_runtime(job) <= reservation_ts:
            decisions.append(_dispatch(job, snapshot.clock_ts, "uarp_backfill"))
            available -= job.requested_cpus

    return SchedulerDecision(
        policy_id=POLICY_ID, reservation_ts=reservation_ts, decisions=tuple(decisions)
    )
