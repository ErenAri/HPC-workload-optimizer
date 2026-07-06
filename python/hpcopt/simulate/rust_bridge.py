"""Python wrapper for the Rust sim-runner binary.

Falls back to the pure-Python simulator if the Rust binary is not available.

Usage:
    from hpcopt.simulate.rust_bridge import run_rust_simulation

    report = run_rust_simulation(
        trace_json_path="data/trace.json",
        policy="EASY_BACKFILL_BASELINE",
        capacity_cpus=64,
    )
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Search paths for the Rust binary (relative to project root).
_BINARY_SEARCH_PATHS = [
    Path("rust/target/release/sim-runner"),
    Path("rust/target/release/sim-runner.exe"),
    Path("rust/target/debug/sim-runner"),
    Path("rust/target/debug/sim-runner.exe"),
]


def find_rust_binary() -> Path | None:
    """Locate the sim-runner binary, returning None if not found."""
    # Check PATH first
    which = shutil.which("sim-runner")
    if which:
        return Path(which)

    # Check project-relative paths
    for candidate in _BINARY_SEARCH_PATHS:
        if candidate.exists():
            return candidate

    return None


def rust_available() -> bool:
    """Check if the Rust sim-runner is available."""
    return find_rust_binary() is not None


def run_rust_simulation(
    trace_json_path: str | Path,
    policy: str = "FIFO_STRICT",
    capacity_cpus: int = 64,
    capacity_gpus: int = 0,
    capacity_mem: int = 0,
    strict_invariants: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a simulation using the Rust sim-runner binary.

    Args:
        trace_json_path: Path to JSON file with job records.
        policy: FIFO_STRICT or EASY_BACKFILL_BASELINE.
        capacity_cpus: Cluster CPU capacity.
        capacity_gpus: Cluster GPU capacity. 0 disables the GPU dimension
            (job GPU requests are ignored; identical to the CPU-only engine).
        capacity_mem: Cluster memory capacity, in the trace's requested_mem
            unit. 0 disables the dimension.
        strict_invariants: Fail on invariant violations.
        output_path: Optional path for the report JSON.

    Returns:
        Parsed simulation report dict.

    Raises:
        FileNotFoundError: If the Rust binary is not found.
        RuntimeError: If the simulation process fails.
    """
    binary = find_rust_binary()
    if binary is None:
        raise FileNotFoundError("Rust sim-runner binary not found. Build with: cd rust && cargo build --release")

    trace_path = Path(trace_json_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    use_temp = output_path is None
    if use_temp:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        resolved_output = Path(tmp.name)
        tmp.close()
    else:
        resolved_output = Path(output_path)  # type: ignore[arg-type]

    cmd = [
        str(binary),
        "--input",
        str(trace_path),
        "--policy",
        policy,
        "--capacity-cpus",
        str(capacity_cpus),
        "--output",
        str(resolved_output),
    ]
    if capacity_gpus:
        cmd += ["--capacity-gpus", str(capacity_gpus)]
    if capacity_mem:
        cmd += ["--capacity-mem", str(capacity_mem)]
    if strict_invariants:
        cmd.append("--strict-invariants")

    logger.info("Running Rust sim-runner: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(f"sim-runner failed (exit {result.returncode}): {result.stderr}")

    with open(resolved_output) as f:
        report: dict[str, Any] = json.load(f)

    if use_temp:
        resolved_output.unlink(missing_ok=True)

    return report
