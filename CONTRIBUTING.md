# Contributing to HPC Workload Optimizer

## Development Setup

```bash
# Clone and install
git clone <repo-url>
cd HPC-workload-optimizer
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Quick Start with Make

```bash
make help          # Show all available targets
make lint          # Run ruff linter
make typecheck     # Run mypy type checker
make test          # Run full test suite
make coverage      # Run tests with coverage (82% gate)
make serve         # Start local API server
make docker-build  # Build Docker image
make rust-check    # Run Rust checks + tests
make verify        # Run full CI-equivalent verification
```

## Code Quality

All code must pass:

- **Lint**: `ruff check python/`
- **Type-check**: `mypy python/hpcopt/ --ignore-missing-imports`
- **Tests**: `pytest tests/ -v` (324+ tests, 82% coverage gate)
- **Security**: `bandit -r python/hpcopt/ -ll -ii`
- **Version consistency**: `python scripts/verify_version_consistency.py --check-unreleased-link`

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Run without slow tests (Rust cross-language parity)
pytest tests/ -v -m "not slow"

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/load/ -v -m load
```

## Rust Components

```bash
cd rust
cargo check --workspace
cargo clippy --workspace -- -D warnings
cargo test --workspace
cargo build --release
```

## Dependency Management

Dependencies are specified with version ranges in `requirements.txt` and pinned
to exact versions in `requirements.lock` (generated via `pip-tools`).

```bash
# Update the lockfile after changing requirements.txt
pip install pip-tools
pip-compile --output-file=requirements.lock requirements.txt
```

> **Note**: Docker builds use `requirements.lock` for reproducible installs.
> Always regenerate the lockfile before cutting a release.

## Changelog Process

The `CHANGELOG.md` is maintained manually. When preparing a release:

1. Add entries under an `## [Unreleased]` section as you merge PRs
2. At release time, rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`
3. The CI release workflow auto-generates a commit-log changelog for the GitHub release page
4. `pyproject.toml` version, release tag (`vX.Y.Z`), and changelog sections must stay aligned.

## Branch Strategy

- `main` is the primary integration branch
- Feature branches should target `main`
- CI must pass before merge

## Commit Messages

Use concise, imperative-mood commit messages:
- `Add retry decorator for model loading`
- `Fix race condition in registry writes`
- `Harden adapter schema with enum constraints`

## Architecture

See [docs/02-system-architecture.md](docs/02-system-architecture.md) for the full architecture overview.

## Schema Changes

All schemas in `schemas/` enforce `additionalProperties: false`. When modifying schemas:

1. Update the schema JSON file
2. Verify `pytest tests/unit/test_schema_validation.py -v` passes
3. Update any corresponding documentation in `docs/08-reproducibility-and-artifacts.md`

## Security

- Never commit secrets or API keys
- Use file-based secrets loading (see `docs/security/secrets-management.md`)
- All new code is scanned by Bandit SAST in CI
