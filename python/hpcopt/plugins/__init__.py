"""Public policy plug-in API for the HPCOpt evaluation harness.

This module is the ONLY supported import surface for third-party scheduling
policies. Everything re-exported here is a frozen contract (see
``docs/plugin-api.md``): a policy is a pure function

    chooser(snapshot: SchedulerStateSnapshot) -> SchedulerDecision

that must be deterministic in the snapshot alone — no wall clock, no RNG
without a fixed seed, no hidden state across calls. The harness replays the
same event stream to every policy and scores the resulting schedule under the
shared metric contract (p95 BSLD = (wait + runtime) / max(runtime, 60s)).

Registering a policy
--------------------

Decorate the chooser and the policy id becomes usable everywhere a built-in
id is (``hpcopt simulate run --policy MY_POLICY``, ``scripts/policy_matrix.py``):

    from hpcopt.plugins import SchedulerDecision, SchedulerStateSnapshot, register_policy

    @register_policy("MY_POLICY", author="you", description="...")
    def choose_my_policy(snapshot: SchedulerStateSnapshot) -> SchedulerDecision:
        ...

Third-party packages are discovered via the ``hpcopt.policies`` entry-point
group; each entry point must resolve to a module that registers its policies
at import time (or directly to a decorated chooser). See
``[project.entry-points."hpcopt.policies"]`` in this repo's pyproject.toml —
the built-in UARP plugin is wired through the same mechanism as a packaging
reference.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from typing import Callable

from hpcopt.simulate.adapter import (
    AdapterQueuedJob,
    AdapterRunningJob,
    DispatchDecision,
    SchedulerDecision,
    SchedulerStateSnapshot,
)

__all__ = [
    "AdapterQueuedJob",
    "AdapterRunningJob",
    "DispatchDecision",
    "SchedulerDecision",
    "SchedulerStateSnapshot",
    "PolicyChooser",
    "PolicySpec",
    "ENTRY_POINT_GROUP",
    "register_policy",
    "get_policy",
    "is_registered",
    "registered_policy_ids",
    "all_policy_ids",
    "ensure_discovered",
    "earliest_start_for",
]

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "hpcopt.policies"

PolicyChooser = Callable[[SchedulerStateSnapshot], SchedulerDecision]

_POLICY_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")

# Reservation sentinel shared with the reference EASY implementation: a job
# that can never fit gets a reservation in the unreachable future.
NEVER_TS = 10**18


@dataclass(frozen=True)
class PolicySpec:
    """A registered policy and its provenance (shown on the leaderboard)."""

    policy_id: str
    chooser: PolicyChooser
    name: str | None = None
    author: str | None = None
    description: str | None = None
    version: str | None = None
    source: str = field(default="decorator")


_registry: dict[str, PolicySpec] = {}
_discovered = False


def _builtin_policy_ids() -> frozenset[str]:
    # Lazy import: hpcopt.simulate.core imports helpers that consult this
    # module, so a top-level import would be circular.
    from hpcopt.simulate.core import SUPPORTED_POLICIES

    return frozenset(SUPPORTED_POLICIES)


def register_policy(
    policy_id: str,
    *,
    name: str | None = None,
    author: str | None = None,
    description: str | None = None,
    version: str | None = None,
    source: str = "decorator",
) -> Callable[[PolicyChooser], PolicyChooser]:
    """Register ``policy_id`` -> chooser. Returns the chooser unchanged.

    Re-registering the *same* callable under the same id is a no-op (safe
    under repeated imports); registering a different callable under a taken
    id, or shadowing a built-in policy id, raises ``ValueError``.
    """
    if not _POLICY_ID_PATTERN.match(policy_id):
        raise ValueError(
            f"Invalid policy_id '{policy_id}': must match {_POLICY_ID_PATTERN.pattern} "
            "(UPPER_SNAKE_CASE, 3-64 chars)"
        )

    def _decorator(chooser: PolicyChooser) -> PolicyChooser:
        if policy_id in _builtin_policy_ids():
            raise ValueError(f"policy_id '{policy_id}' shadows a built-in policy")
        existing = _registry.get(policy_id)
        if existing is not None and existing.chooser is not chooser:
            raise ValueError(f"policy_id '{policy_id}' is already registered by {existing.chooser!r}")
        _registry[policy_id] = PolicySpec(
            policy_id=policy_id,
            chooser=chooser,
            name=name,
            author=author,
            description=description,
            version=version,
            source=source,
        )
        return chooser

    return _decorator


def ensure_discovered() -> None:
    """Load bundled plugins and ``hpcopt.policies`` entry points, once.

    A broken third-party plugin logs a warning and is skipped — it must not
    take the harness down with it.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    # Bundled reference plugin(s) register themselves at import.
    from hpcopt.plugins import uarp as _uarp  # noqa: F401

    try:
        entry_points = importlib_metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # pragma: no cover - importlib.metadata quirk
        logger.warning("Could not enumerate %s entry points: %s", ENTRY_POINT_GROUP, exc)
        return

    for entry_point in entry_points:
        try:
            loaded = entry_point.load()
        except Exception as exc:
            logger.warning(
                "Skipping plugin entry point '%s' (%s): %s",
                entry_point.name,
                entry_point.value,
                exc,
            )
            continue
        # A module entry point registers via decorators on import; a callable
        # entry point is registered under the entry point's name.
        if callable(loaded):
            policy_id = entry_point.name
            existing = _registry.get(policy_id)
            if existing is not None and existing.chooser is loaded:
                continue
            try:
                register_policy(policy_id, source=f"entry_point:{entry_point.value}")(loaded)
            except ValueError as exc:
                logger.warning("Skipping plugin entry point '%s': %s", entry_point.name, exc)


def get_policy(policy_id: str) -> PolicySpec | None:
    ensure_discovered()
    return _registry.get(policy_id)


def is_registered(policy_id: str) -> bool:
    ensure_discovered()
    return policy_id in _registry


def registered_policy_ids() -> tuple[str, ...]:
    ensure_discovered()
    return tuple(sorted(_registry))


def all_policy_ids() -> tuple[str, ...]:
    """Built-in + plugin policy ids, for validation and error messages."""
    ensure_discovered()
    return tuple(sorted(_builtin_policy_ids() | set(_registry)))


def earliest_start_for(snapshot: SchedulerStateSnapshot, requested_cpus: int) -> int:
    """Earliest timestamp at which ``requested_cpus`` can be free.

    Shadow-time helper for EASY-style plugins: walks running jobs in end-time
    order accumulating freed CPUs. Returns ``clock_ts`` if the job fits now
    and ``NEVER_TS`` if it can never fit. Identical semantics to the
    reference EASY reservation.
    """
    if requested_cpus > snapshot.capacity_cpus:
        return NEVER_TS
    if requested_cpus <= snapshot.free_cpus:
        return snapshot.clock_ts

    free = snapshot.free_cpus
    # The harness provides running_jobs sorted by (end_ts, job_id); sort
    # defensively so the helper is correct for any snapshot producer.
    for running in sorted(snapshot.running_jobs, key=lambda rj: (rj.end_ts, rj.job_id)):
        free += running.allocated_cpus
        if free >= requested_cpus:
            return max(snapshot.clock_ts, running.end_ts)
    return NEVER_TS
