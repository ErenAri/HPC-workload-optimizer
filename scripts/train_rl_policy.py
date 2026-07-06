"""Train a MaskablePPO scheduling policy on a workload trace.

Mirrors the protocol from Zhang et al. SC'20 (RLScheduler).  Reads a
canonical parquet trace produced by ``hpcopt.ingest.swf.ingest_swf``,
trains for ``--total-timesteps`` steps, and writes the model to
``--save-path`` for use by the ``RL_TRAINED`` simulator policy.

Requires the ``[rl]`` optional dependency group:

    pip install -e ".[rl]"

Example
-------
    python scripts/train_rl_policy.py \\
        --trace data/parquet/SDSC-SP2.parquet \\
        --capacity-cpus 128 \\
        --total-timesteps 200000 \\
        --save-path models/rl/sdsc_ppo.zip
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# torch must be imported before pandas/pyarrow on Windows: if pyarrow's DLLs
# load first, torch's c10.dll fails to initialize (WinError 1114).
try:
    import torch  # noqa: F401
except ImportError:
    pass

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_rl_policy")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True, help="Path to parquet trace.")
    p.add_argument("--capacity-cpus", type=int, required=True, help="Cluster CPU capacity.")
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--max-jobs-per-episode", type=int, default=256)
    p.add_argument("--save-path", default="models/rl/ppo_policy.zip")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=0.97)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--no-window-random", action="store_true",
                   help="Disable random window sampling (use the trace head every episode).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        logger.error("trace not found: %s", trace_path)
        return 1

    logger.info("loading trace %s", trace_path)
    df = pd.read_parquet(trace_path)
    logger.info("loaded %d jobs", len(df))

    try:
        from hpcopt.rl.train import train_ppo
    except ImportError as exc:
        logger.error("RL extras not installed: %s", exc)
        logger.error("Install with: pip install -e \".[rl]\"")
        return 2

    train_ppo(
        trace_df=df,
        capacity_cpus=args.capacity_cpus,
        total_timesteps=args.total_timesteps,
        save_path=args.save_path,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        clip_range=args.clip_range,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        max_jobs_per_episode=args.max_jobs_per_episode,
        window_random=not args.no_window_random,
        seed=args.seed,
    )
    logger.info("training complete; model at %s", args.save_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
