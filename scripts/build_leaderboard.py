"""Generate the public policy leaderboard from the benchmark matrix.

Reads outputs/benchmark/policy_matrix.json (written by scripts/policy_matrix.py,
Python reference simulator) and renders docs/leaderboard.md: one ranking table
per reference trace, ordered by the headline metric p95 BSLD.

Usage:
    python scripts/build_leaderboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_JSON = PROJECT_ROOT / "outputs" / "benchmark" / "policy_matrix.json"
OUT_MD = PROJECT_ROOT / "docs" / "leaderboard.md"

TRACE_CAPS = {"SDSC-SP2": 128, "CTC-SP2": 512, "HPC2N": 240}

POLICY_NOTES = {
    "RL_TRAINED": "MaskablePPO, single seed, trained in-distribution on windows of the eval trace",
    "UARP_BACKFILL": "plug-in policy (`hpcopt.plugins.uarp`) — guard-gated, shortest-guard-first backfill",
}


def _policy_kind(policy_id: str) -> str:
    from hpcopt import plugins  # late import: registry needs the package installed

    return "plugin" if plugins.is_registered(policy_id) else "built-in"


def render(rows: list[dict]) -> str:
    lines = [
        "# HPCOpt Policy Leaderboard",
        "",
        "Ranking of every evaluated scheduling policy on the reference traces, "
        "by **p95 bounded slowdown** (`(wait + runtime) / max(runtime, 60 s)`, "
        "lower is better). All rows come from the same referee: the Python "
        "reference simulator replaying the full trace under the shared metric "
        "contract. Numbers regenerate from `outputs/benchmark/policy_matrix.json` "
        "via `python scripts/build_leaderboard.py`; the matrix itself regenerates "
        "via `python scripts/policy_matrix.py`.",
        "",
        "**Add your policy:** implement the frozen chooser contract, register it "
        "through the `hpcopt.policies` entry point, and run the matrix — see "
        "[docs/plugin-api.md](plugin-api.md). Plug-in policies compete in the "
        "same table as built-ins.",
        "",
    ]

    by_trace: dict[str, list[dict]] = {}
    for row in rows:
        if row["result"].get("status") != "ok":
            continue
        by_trace.setdefault(row["trace"], []).append(row)

    notes_used: dict[str, int] = {}

    for trace, cells in by_trace.items():
        cap = TRACE_CAPS.get(trace)
        cap_txt = f", capacity {cap} CPUs" if cap else ""
        n_jobs = max(c["result"].get("jobs", 0) for c in cells)
        lines += [
            f"## {trace} ({n_jobs:,} jobs{cap_txt})",
            "",
            "| # | Policy | Kind | p95 BSLD | Mean wait (s) | Utilization | Starved | Wall (s) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        all_policies = {r["policy"] for r in rows}
        cells.sort(key=lambda c: float(c["result"]["objective_metrics"]["p95_bsld"]))
        for rank, cell in enumerate(cells, start=1):
            obj = cell["result"]["objective_metrics"]
            met = cell["result"]["metrics"]
            policy = cell["policy"]
            marker = ""
            if policy in POLICY_NOTES:
                idx = notes_used.setdefault(policy, len(notes_used) + 1)
                marker = f" [^{idx}]"
            row_fmt = (
                "| {rank} | {policy}{marker} | {kind} | {bsld:,.2f} "
                "| {wait:,.0f} | {util:.1%} | {starved:.1%} | {wall:,.0f} |"
            )
            lines.append(
                row_fmt.format(
                    rank=rank,
                    policy=policy,
                    marker=marker,
                    kind=_policy_kind(policy),
                    bsld=float(obj["p95_bsld"]),
                    wait=float(met.get("mean_wait_sec", float("nan"))),
                    util=float(obj.get("utilization_cpu", float("nan"))),
                    starved=float(obj.get("starved_rate", float("nan"))),
                    wall=float(cell["result"].get("wall_time_sec", float("nan"))),
                )
            )
        # Surface gaps (e.g. CONSERVATIVE on HPC2N: computationally
        # prohibitive, >3.4 h) instead of letting them read as omissions.
        missing = all_policies - {c["policy"] for c in cells}
        if missing:
            lines.append("")
            lines.append(f"Not evaluated on this trace: {', '.join(sorted(missing))}.")
        lines.append("")

    for policy, idx in sorted(notes_used.items(), key=lambda kv: kv[1]):
        lines.append(f"[^{idx}]: {policy}: {POLICY_NOTES[policy]}.")
    if notes_used:
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    rows = json.loads(MATRIX_JSON.read_text(encoding="utf-8"))
    OUT_MD.write_text(render(rows), encoding="utf-8")
    print(f"[leaderboard] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
