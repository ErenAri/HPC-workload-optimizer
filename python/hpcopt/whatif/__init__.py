"""What-if analysis: fidelity-graded scheduler change evaluation for operators."""

from hpcopt.whatif.engine import (
    SLURM_SCHEDULER_POLICY_MAP,
    UNMODELED_CAVEATS,
    WhatIfResult,
    infer_capacity_cpus,
    render_whatif_markdown,
    run_whatif,
)

__all__ = [
    "SLURM_SCHEDULER_POLICY_MAP",
    "UNMODELED_CAVEATS",
    "WhatIfResult",
    "infer_capacity_cpus",
    "render_whatif_markdown",
    "run_whatif",
]
