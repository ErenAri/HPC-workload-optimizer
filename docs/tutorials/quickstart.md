# Quickstart

Get from zero to benchmark results in under 5 minutes.

## Prerequisites

- Python 3.11+
- Rust 1.70+ (optional, for Rust sim-runner)

## Installation

```bash
# Clone the repository
git clone https://github.com/ErenAri/HPC-workload-optimizer.git
cd HPC-workload-optimizer

# Install with pip (editable mode)
pip install -e ".[dev]"

# Optional: LightGBM for 50× faster training
pip install -e ".[lightgbm]"
```

## Download Sample Data

The project uses traces from the [Parallel Workloads Archive](https://www.cs.huji.ac.il/labs/parallel/workload/):

```bash
mkdir -p data/raw
# CTC-SP2 (77K jobs)
curl -o data/raw/CTC-SP2-1996-3.1-cln.swf.gz \
    https://www.cs.huji.ac.il/labs/parallel/workload/l_ctc_sp2/CTC-SP2-1996-3.1-cln.swf.gz
```

## Step 1: Ingest Trace

```bash
hpcopt ingest swf \
    --input data/raw/CTC-SP2-1996-3.1-cln.swf.gz \
    --dataset-id ctc_sp2 \
    --out outputs/curated \
    --report-out outputs/reports
```

This produces:

- `outputs/curated/ctc_sp2.parquet` — canonical job records
- `outputs/curated/ctc_sp2_features.parquet` — engineered features
- `outputs/reports/ctc_sp2_quality.json` — ingestion quality report

## Step 2: Train Model

```bash
hpcopt train runtime \
    --input outputs/curated/ctc_sp2_features.parquet \
    --model-id ctc_sp2_model \
    --out outputs/models
```

!!! tip "Use LightGBM for 50× faster training"
    If LightGBM is installed, it's automatically used as the default backend.
    Training drops from ~6 minutes to ~8 seconds.

## Step 3: Run Simulation

=== "Python"

    ```bash
    hpcopt simulate \
        --input outputs/curated/ctc_sp2.parquet \
        --policy EASY_BACKFILL_BASELINE \
        --capacity-cpus 512 \
        --out outputs/reports
    ```

=== "Rust (16,000× faster)"

    ```bash
    cd rust && cargo build --release
    ./target/release/sim-runner \
        --input ../outputs/curated/ctc_sp2.json \
        --policy EASY_BACKFILL_BASELINE \
        --capacity-cpus 512 \
        --output ../outputs/reports/rust_report.json
    ```

## Step 4: View Results

```bash
python scripts/benchmark_suite.py
```

## What's Next?

- [Rust Simulation Engine](rust-sim-runner.md) — Build and use the Rust sim-runner
- [LightGBM Training](lightgbm-training.md) — GPU-accelerated model training
- [RL Policy Search](rl-policy-search.md) — Find optimal scheduling parameters
- [Live Integration](live-integration.md) — Connect to Slurm/PBS clusters
