"""End-to-end what-if demo on synthetic Slurm accounting data.

Generates a synthetic `sacct --parsable2` dump for a congested cluster window,
then asks: "what happens to p95 bounded slowdown if we switch from plain
backfill to SJF-ordered backfill?" — the exact workflow an operator would run
against a real accounting dump:

    hpcopt whatif run --sacct <dump> --candidate-policy SJF_BACKFILL

Run: python examples/whatif_demo.py
Outputs land in outputs/whatif_demo/.
"""

from __future__ import annotations

import datetime as dt
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "whatif_demo"
SACCT_PATH = OUT_DIR / "synthetic_sacct.txt"

N_JOBS = 1500
CAPACITY_CPUS = 96
BASE = dt.datetime(2026, 6, 1, 8, 0, 0)


def _fmt(ts: dt.datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def _elapsed(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_sacct_dump(path: Path, seed: int = 7) -> None:
    rng = random.Random(seed)
    lines = ["JobID|Submit|Start|End|Elapsed|AllocCPUS|ReqCPUS|ReqMem|User|Group|Partition|State"]
    # Rolling in-use model so observed Start times reflect a congested queue.
    free_at: list[dt.datetime] = [BASE] * CAPACITY_CPUS
    for i in range(N_JOBS):
        submit = BASE + dt.timedelta(seconds=i * rng.randint(60, 180))
        cpus = rng.choice([1, 2, 4, 4, 8, 8, 16, 32])
        runtime = rng.choice([120, 300, 900, 900, 1800, 3600, 7200])
        free_at.sort()
        earliest = max(submit, free_at[cpus - 1])
        start = earliest + dt.timedelta(seconds=rng.randint(0, 120))
        end = start + dt.timedelta(seconds=runtime)
        for slot in range(cpus):
            free_at[slot] = end
        user = f"user{rng.randint(1, 24)}"
        lines.append(
            f"{1000 + i}|{_fmt(submit)}|{_fmt(start)}|{_fmt(end)}|{_elapsed(runtime)}"
            f"|{cpus}|{cpus}|{cpus * 2000}M|{user}|grp{rng.randint(1, 4)}|normal|COMPLETED"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    generate_sacct_dump(SACCT_PATH)
    print(f"Synthetic sacct dump: {SACCT_PATH} ({N_JOBS} jobs)")

    sys.argv = [
        "hpcopt",
        "whatif",
        "run",
        "--sacct",
        str(SACCT_PATH),
        "--candidate-policy",
        "SJF_BACKFILL",
        "--out",
        str(OUT_DIR),
        "--run-id",
        "whatif_demo",
    ]
    from hpcopt.cli.main import run

    try:
        run()
    except SystemExit as exc:  # typer exits 0 on success
        if exc.code not in (0, None):
            return int(exc.code)
    print(f"\nRead the report: {OUT_DIR / 'whatif_demo_whatif_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
