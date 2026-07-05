# Installation

## Requirements

- **Python 3.11+** — Required for all Python components
- **Rust 1.70+** — Optional, for the Rust simulation engine
- **NVIDIA GPU** — Optional, for GPU-accelerated LightGBM training

## From PyPI

```bash
pip install hpc-workload-optimizer
```

### Optional Extras

```bash
# GPU-accelerated model training
pip install hpc-workload-optimizer[lightgbm]

# OpenTelemetry tracing
pip install hpc-workload-optimizer[tracing]

# Development tools
pip install hpc-workload-optimizer[dev]
```

## From Source

```bash
git clone https://github.com/ErenAri/HPC-workload-optimizer.git
cd HPC-workload-optimizer
pip install -e ".[dev,lightgbm]"
```

## Rust Simulation Engine

```bash
cd rust/sim-runner
cargo build --release
```

The binary will be at `rust/target/release/sim-runner` (or `sim-runner.exe` on Windows).

## Verify Installation

```bash
# Python
python -c "import hpcopt; print('hpcopt OK')"

# CLI
hpcopt --help

# Rust  (optional)
./rust/target/release/sim-runner --help
```

## Docker

```bash
docker build -t hpcopt .
docker run -p 8080:8080 hpcopt
```
