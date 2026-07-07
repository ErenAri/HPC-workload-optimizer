"""RLScheduler-style Gymnasium environment for HPC batch scheduling.

Mirrors the action / observation protocol of Zhang, Dai, Bose, Li, Xu, Park,
"RLScheduler: An Automated HPC Batch Job Scheduler Using Reinforcement
Learning" (SC'20, https://arxiv.org/abs/1910.08925) and the reference
implementation at https://github.com/DIR-LAB/RLScheduler.

Protocol
--------
* **Observation**: ``Box(0, 1, shape=(MAX_QUEUE_SIZE, JOB_FEATURES),
  dtype=float32)`` — the front ``MAX_QUEUE_SIZE`` waiting jobs, each
  encoded as a fixed-length normalised feature vector.  Padding rows are
  zero, with the validity flag (last feature) set to 0.
* **Action**: ``Discrete(MAX_QUEUE_SIZE)`` — pick which waiting job to
  dispatch next.  Invalid actions (padding rows or jobs that don't fit
  the current free CPUs) are masked via ``action_masks()``; combine with
  ``sb3_contrib.MaskablePPO``.
* **Reward**: shaped per-step.  Each step the dispatched job contributes
  ``-bsld(job)`` once it completes, accumulated into the next step's
  reward.  Sum over an episode ≈ ``-mean_bsld``.
* **Episode**: one trace = one episode; configurable ``max_jobs`` (default
  ``len(trace)``) lets you train on random windows for stochasticity.

Determinism: when the agent picks a feasible job, that job dispatches at
``clock_ts``; the simulator then advances to the next event (submit or
completion) and the next observation is built.  CPU conservation and
no-start-before-submit invariants are enforced just like the main
``simulate.core`` loop.

This module hard-imports ``gymnasium``; install ``hpc-workload-optimizer[rl]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - surfaced at import time
    raise ImportError(
        "hpcopt.rl.env requires gymnasium. Install with: pip install 'hpc-workload-optimizer[rl]'"
    ) from exc


MAX_QUEUE_SIZE: int = 128
JOB_FEATURES: int = 8
# Per-job feature layout (all in [0,1]):
#   0: requested_cpus / capacity_cpus
#   1: runtime_estimate_sec / WALL_NORM      (WALL_NORM = 12h)
#   2: wait_so_far_sec / WALL_NORM
#   3: requested_cpus / free_cpus_now (clipped to [0,1]; 1 means won't fit)
#   4: queue_position / MAX_QUEUE_SIZE
#   5: free_cpus_now / capacity_cpus  (broadcast; same for every row)
#   6: log10(runtime_estimate_sec+1)/6  (compact magnitude feature)
#   7: validity flag (1 if real job, 0 if padding)

WALL_NORM_SEC: float = 12.0 * 3600.0  # 12-hour normalisation reference


@dataclass
class _RunningJob:
    job_id: int
    start_ts: int
    end_ts: int
    cpus: int


def _encode_jobs(
    queued: list[dict[str, Any]],
    clock_ts: int,
    capacity_cpus: int,
    free_cpus: int,
) -> np.ndarray:
    """Build the (MAX_QUEUE_SIZE, JOB_FEATURES) observation matrix."""
    obs = np.zeros((MAX_QUEUE_SIZE, JOB_FEATURES), dtype=np.float32)
    cap = float(max(capacity_cpus, 1))
    free = float(max(free_cpus, 1))
    for i, job in enumerate(queued[:MAX_QUEUE_SIZE]):
        cpus = int(job["requested_cpus"])
        rt = max(1, int(job.get("runtime_estimate_sec", job["runtime_actual_sec"])))
        wait = max(0, clock_ts - int(job["submit_ts"]))
        obs[i, 0] = min(cpus / cap, 1.0)
        obs[i, 1] = min(rt / WALL_NORM_SEC, 1.0)
        obs[i, 2] = min(wait / WALL_NORM_SEC, 1.0)
        obs[i, 3] = min(cpus / free, 1.0)
        obs[i, 4] = i / MAX_QUEUE_SIZE
        obs[i, 5] = free_cpus / cap
        obs[i, 6] = min(np.log10(rt + 1.0) / 6.0, 1.0)
        obs[i, 7] = 1.0  # valid
    return obs


class RLSchedulerEnv(gym.Env):
    """Gymnasium env exposing the RLScheduler-style scheduling MDP."""

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self,
        trace_df: pd.DataFrame,
        capacity_cpus: int,
        max_jobs: int | None = None,
        window_random: bool = False,
        seed: int | None = None,
    ):
        super().__init__()
        required = {"job_id", "submit_ts", "runtime_actual_sec", "requested_cpus"}
        missing = required - set(trace_df.columns)
        if missing:
            raise ValueError(f"trace_df missing required columns: {sorted(missing)}")
        if capacity_cpus <= 0:
            raise ValueError("capacity_cpus must be > 0")

        df = trace_df.copy()
        df["job_id"] = df["job_id"].astype(int)
        df["submit_ts"] = df["submit_ts"].astype(int)
        df["runtime_actual_sec"] = df["runtime_actual_sec"].clip(lower=0).astype(int)
        df["requested_cpus"] = df["requested_cpus"].clip(lower=1).astype(int)
        if "runtime_requested_sec" not in df.columns:
            df["runtime_requested_sec"] = df["runtime_actual_sec"]
        df["runtime_estimate_sec"] = (
            pd.to_numeric(df["runtime_requested_sec"], errors="coerce")
            .where(lambda s: s > 0, df["runtime_actual_sec"])
            .astype(int)
        )
        df = df.sort_values(["submit_ts", "job_id"]).reset_index(drop=True)
        self._all_jobs = df.to_dict("records")
        self.capacity_cpus = int(capacity_cpus)
        self.max_jobs = int(max_jobs) if max_jobs is not None else len(self._all_jobs)
        self.window_random = bool(window_random)
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(MAX_QUEUE_SIZE, JOB_FEATURES), dtype=np.float32
        )
        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)

        self._reset_state()

    # ── Gymnasium API ───────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.window_random and len(self._all_jobs) > self.max_jobs:
            offset = int(self._rng.integers(0, len(self._all_jobs) - self.max_jobs + 1))
        else:
            offset = 0
        self._jobs = [j.copy() for j in self._all_jobs[offset : offset + self.max_jobs]]
        # Re-anchor submit_ts at the window start so the agent's state is
        # episode-relative.
        if self._jobs:
            base = int(self._jobs[0]["submit_ts"])
            for j in self._jobs:
                j["submit_ts"] = int(j["submit_ts"]) - base
        self._reset_state()
        self._advance_to_decision_point()
        obs = self._observe()
        return obs, self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not isinstance(action, (int, np.integer)):
            action = int(action)
        action = int(action)
        if action < 0 or action >= MAX_QUEUE_SIZE:
            raise ValueError(f"action {action} out of range [0,{MAX_QUEUE_SIZE})")

        masks = self.action_masks()
        # Reward accumulator: jobs that completed since the last step contribute -bsld.
        reward = 0.0

        if masks[action]:
            job = self._queued.pop(action)
            self._dispatch(job)
        else:
            # Illegal action: small penalty, then advance time without dispatch.
            reward -= 1.0

        # Advance the clock until either (a) a new decision is needed (queue
        # non-empty AND at least one queued job fits) or (b) the episode ends.
        completed_now = self._advance_to_decision_point()
        for cj in completed_now:
            wait = cj["start_ts"] - cj["submit_ts"]
            runtime = max(cj["runtime_actual_sec"], 10)
            bsld = max(1.0, wait / runtime)
            reward -= float(bsld)

        terminated = (
            self._submit_idx >= len(self._jobs)
            and not self._queued
            and not self._running
        )
        truncated = False
        return self._observe(), reward, terminated, truncated, self._info()

    # ── action mask for MaskablePPO ─────────────────────────────────

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(MAX_QUEUE_SIZE, dtype=bool)
        for i, job in enumerate(self._queued[:MAX_QUEUE_SIZE]):
            if int(job["requested_cpus"]) <= self._free_cpus:
                mask[i] = True
        return mask

    # ── internals ───────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self._clock_ts: int = 0
        self._free_cpus: int = self.capacity_cpus
        self._submit_idx: int = 0
        self._queued: list[dict[str, Any]] = []
        self._running: list[_RunningJob] = []
        self._completed: list[dict[str, Any]] = []
        self._n_steps: int = 0

    def _admit_arrivals(self) -> None:
        while (
            self._submit_idx < len(self._jobs)
            and int(self._jobs[self._submit_idx]["submit_ts"]) <= self._clock_ts
        ):
            j = self._jobs[self._submit_idx]
            self._queued.append(j)
            self._submit_idx += 1

    def _next_event_ts(self) -> int | None:
        candidates: list[int] = []
        if self._submit_idx < len(self._jobs):
            candidates.append(int(self._jobs[self._submit_idx]["submit_ts"]))
        if self._running:
            candidates.append(min(rj.end_ts for rj in self._running))
        return min(candidates) if candidates else None

    def _complete_due_jobs(self) -> list[dict[str, Any]]:
        finished: list[_RunningJob] = [rj for rj in self._running if rj.end_ts <= self._clock_ts]
        if not finished:
            return []
        finished_ids = {rj.job_id for rj in finished}
        self._running = [rj for rj in self._running if rj.job_id not in finished_ids]
        for rj in finished:
            self._free_cpus += rj.cpus
        completed_records: list[dict[str, Any]] = []
        for rj in finished:
            rec = {
                "job_id": rj.job_id,
                "start_ts": rj.start_ts,
                "end_ts": rj.end_ts,
                "submit_ts": next(
                    (j["submit_ts"] for j in self._jobs if int(j["job_id"]) == rj.job_id), rj.start_ts
                ),
                "runtime_actual_sec": rj.end_ts - rj.start_ts,
                "requested_cpus": rj.cpus,
            }
            self._completed.append(rec)
            completed_records.append(rec)
        return completed_records

    def _advance_to_decision_point(self) -> list[dict[str, Any]]:
        """Advance simulation clock until a decision is required or the episode ends.

        Returns the list of jobs that completed during the advance.
        """
        completed_during_advance: list[dict[str, Any]] = []
        max_steps = 10_000  # safety net against any pathological loop
        for _ in range(max_steps):
            self._admit_arrivals()
            # If any queued job fits, the agent must decide.
            if self._queued and any(int(j["requested_cpus"]) <= self._free_cpus for j in self._queued):
                return completed_during_advance
            nxt = self._next_event_ts()
            if nxt is None:
                return completed_during_advance
            self._clock_ts = max(self._clock_ts, nxt)
            completed_during_advance.extend(self._complete_due_jobs())
            self._admit_arrivals()
            if not self._queued and self._submit_idx >= len(self._jobs) and not self._running:
                return completed_during_advance
        return completed_during_advance

    def _dispatch(self, job: dict[str, Any]) -> None:
        cpus = int(job["requested_cpus"])
        runtime = int(job["runtime_actual_sec"])
        rj = _RunningJob(
            job_id=int(job["job_id"]),
            start_ts=self._clock_ts,
            end_ts=self._clock_ts + runtime,
            cpus=cpus,
        )
        self._running.append(rj)
        self._free_cpus -= cpus
        self._n_steps += 1

    def _observe(self) -> np.ndarray:
        return _encode_jobs(self._queued, self._clock_ts, self.capacity_cpus, self._free_cpus)

    def _info(self) -> dict[str, Any]:
        return {
            "clock_ts": int(self._clock_ts),
            "queue_len": int(len(self._queued)),
            "running_len": int(len(self._running)),
            "completed": int(len(self._completed)),
            "free_cpus": int(self._free_cpus),
            "n_steps": int(self._n_steps),
        }
