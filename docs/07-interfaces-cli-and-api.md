# Interfaces: CLI and API

## 1. CLI Overview

Primary interface:
- `hpcopt` (Typer-based command surface).

### Modular Architecture

The CLI is implemented as a modular assembler pattern:

- `python/hpcopt/cli/main.py` -- assembler (41 lines), imports and mounts sub-apps from domain modules.
- `python/hpcopt/cli/ingest.py` -- `ingest swf`, `ingest slurm`, `ingest pbs`, `ingest shadow-start`
- `python/hpcopt/cli/train.py` -- `train runtime`, `train tune`, `train resource-fit`
- `python/hpcopt/cli/simulate.py` -- `simulate run`, `simulate replay-baselines`, `simulate fidelity-gate`, `simulate batsim-config`, `simulate batsim-run`, `stress gen`, `stress run`
- `python/hpcopt/cli/pipeline.py` -- `profile trace`, `features build`, `analysis sensitivity-sweep`, `analysis feature-importance`, `credibility run-suite`, `credibility dossier`
- `python/hpcopt/cli/model.py` -- `model list`, `model promote`, `model archive`, `model drift-check`, `data lock-reference-suite`, `serve api`
- `python/hpcopt/cli/report.py` -- `report export`, `report benchmark`, `recommend generate`, `artifacts cleanup`

Top-level command groups:

- `ingest` -- multi-format trace ingestion and shadow polling
- `profile` -- trace profiling
- `features` -- feature engineering pipeline
- `train` -- model training, tuning, resource-fit
- `simulate` -- policy replay, fidelity, Batsim path
- `stress` -- stress scenario generation and testing
- `recommend` -- recommendation generation
- `report` -- export and benchmark
- `serve` -- API service
- `data` -- dataset contract management
- `credibility` -- credibility protocol and dossier
- `analysis` -- sensitivity sweeps and feature importance
- `model` -- model registry management
- `artifacts` -- artifact retention management

## 2. Implemented Commands

### Ingestion

#### SWF Format

```bash
hpcopt ingest swf --input <trace.swf|trace.swf.gz> --out data/curated
```

Outputs:
- canonical parquet,
- dataset metadata file,
- quality report,
- run manifest.

#### Slurm (sacct --parsable2)

```bash
hpcopt ingest slurm --input <sacct_dump.txt> --out data/curated
```

Parses pipe-delimited `sacct --parsable2` output. Handles:
- array jobs (`12345_0`),
- job steps (skipped by default),
- `Elapsed` field (`[DD-]HH:MM:SS`) to seconds conversion,
- `ReqMem` field parsing (`4000Mc`, `8Gn`) to megabytes.

Outputs: canonical parquet, quality report, dataset metadata.

#### PBS/Torque Accounting Log

```bash
hpcopt ingest pbs --input <accounting_log> --out data/curated
```

Parses PBS/Torque accounting log format (`timestamp;type;id;attrs`). Extracts only `E` (exit) records. Handles:
- `nodes=1:ppn=8` style CPU specifications,
- memory units (`kb`, `mb`, `gb`, `tb`),
- walltime in `HH:MM:SS` or `DD:HH:MM:SS`.

Outputs: canonical parquet, quality report, dataset metadata.

#### Shadow Ingestion Daemon

```bash
hpcopt ingest shadow-start \
  --source-type slurm \
  --source-path /var/log/slurm/sacct_dump.txt \
  --interval-sec 300
```

Polls a scheduler data source periodically for incremental ingestion:
- watermark-based deduplication (only new rows since last poll),
- persistent watermark state across restarts,
- supports `slurm`, `pbs`, and `swf` source types.

### Trace Profiling

```bash
hpcopt profile trace --dataset <dataset.parquet> --out outputs/reports
```

Outputs:
- trace profile JSON,
- profile manifest.

### Feature Pipeline

```bash
hpcopt features build \
  --dataset <dataset.parquet> \
  --out data/curated \
  --report-out outputs/reports \
  --n-folds 3
```

Outputs:
- feature dataset parquet,
- chronological split manifest,
- feature quality report,
- features manifest.

### Model Training

#### Runtime Quantile Training

```bash
hpcopt train runtime --dataset <dataset.parquet> --out outputs/models --backend sklearn
```

Options:
- `--hyperparams-config`: optional YAML with hyperparameter overrides.
- `--backend`: `sklearn` or `lightgbm`.

Outputs:
- quantile model artifacts (`p10/p50/p90`),
- metrics and metadata,
- training manifest.

#### Hyperparameter Tuning

```bash
hpcopt train tune \
  --dataset <dataset.parquet> \
  --out outputs/reports \
  --quantile 0.5 \
  --n-trials 20 \
  --n-folds 3 \
  --backend sklearn
```

Outputs: tuning report with best parameters, best score, and trial history.

#### Resource-Fit Training

```bash
hpcopt train resource-fit \
  --dataset <dataset.parquet> \
  --out outputs/models \
  --backend sklearn
```

Trains two models:
- fragmentation risk classifier (low/medium/high),
- optimal node size regressor.

Outputs: model artifacts, metrics, metadata.

### Stress Scenario Generation and Execution

```bash
hpcopt stress gen --scenario heavy_tail --out data/curated --n-jobs 5000
hpcopt stress run \
  --scenario heavy_tail \
  --policy configs/simulation/policy_ml_backfill.yaml \
  --model runtime_latest \
  --capacity-cpus 64
```

Supported scenarios: `heavy_tail`, `low_congestion`, `user_skew`, `burst_shock`.

Outputs:
- generated stress dataset and metadata (`stress gen`),
- baseline/candidate stress simulation artifacts (`stress run`),
- stress report with constraint pass/fail, degrade signatures, and baseline policy comparison,
- stress run manifest.

### Simulation

```bash
hpcopt simulate run --trace <dataset.parquet> --policy FIFO_STRICT --capacity-cpus 64
hpcopt simulate run --trace <dataset.parquet> --policy EASY_BACKFILL_BASELINE --capacity-cpus 64
hpcopt simulate run --trace <dataset.parquet> --policy ML_BACKFILL_P50 --capacity-cpus 64
hpcopt simulate run --trace <dataset.parquet> --policy ML_BACKFILL_P10 --capacity-cpus 64
```

Supported `--policy` values: `FIFO_STRICT`, `EASY_BACKFILL_BASELINE`, `EASY_BACKFILL_TSAFRIR`
(Tsafrir/Etsion/Feitelson 2007 user-history predictor), `CONSERVATIVE_BACKFILL_BASELINE`
(reservations for all queued jobs), `SJF_BACKFILL`, `LJF_BACKFILL`, `FAIRSHARE_BACKFILL`
(decayed-usage multifactor priority), `ML_BACKFILL_P50`, `ML_BACKFILL_P10`, and `RL_TRAINED`
(MaskablePPO agent; requires the `[rl]` extras and a trained checkpoint).

Key options:
- `--strict-invariants`
- `--runtime-model-dir`
- `--runtime-guard-k`
- `--strict-uncertainty-mode`

Outputs:
- jobs artifact parquet,
- queue artifact parquet,
- simulation report,
- invariant report,
- manifest.

### What-If Analysis (Operator Mode)

```bash
hpcopt whatif run --sacct /var/log/slurm/sacct_dump.txt --candidate-policy SJF_BACKFILL
hpcopt whatif run --trace <dataset.parquet> --slurm-scheduler-type sched/builtin --capacity-cpus 512
hpcopt whatif run --trace <dataset.parquet> --candidate-policy EASY_BACKFILL_BASELINE --candidate-capacity-cpus 640
```

Key options:
- `--sacct` / `--trace` (exactly one; sacct dumps are ingested automatically)
- `--candidate-policy` or `--slurm-scheduler-type` (`sched/builtin` → `FIFO_STRICT`, `sched/backfill` → `EASY_BACKFILL_BASELINE`)
- `--baseline-policy` (default `EASY_BACKFILL_BASELINE`)
- `--capacity-cpus` (inferred from peak observed concurrency when omitted)
- `--candidate-capacity-cpus` (capacity what-if)
- `--runtime-model-dir` (required for ML policies)

Outputs:
- what-if report (JSON + markdown) with verdict (`improvement` / `regression` /
  `no_material_change` / `blocked_constraints`), fidelity-graded confidence, metric deltas,
  constraint contract result, and unmodeled-caveat list,
- baseline fidelity report,
- manifest.

### Baseline Replay Bundle

```bash
hpcopt simulate replay-baselines --trace <dataset.parquet> --capacity-cpus 64
```

Outputs:
- baseline replay report,
- per-policy artifacts,
- replay manifest.

### Fidelity Gate

```bash
hpcopt simulate fidelity-gate --trace <dataset.parquet> --capacity-cpus 64
```

Outputs:
- fidelity report,
- fidelity manifest.

### Batsim Path

```bash
hpcopt simulate batsim-config --trace <dataset.parquet> --policy FIFO_STRICT --run-id batsim_demo
hpcopt simulate batsim-run --config outputs/simulations/batsim_demo_batsim_run_config.json --dry-run
```

Optional live run:

```bash
hpcopt simulate batsim-run \
  --config outputs/simulations/batsim_demo_batsim_run_config.json \
  --use-wsl \
  --no-dry-run
```

Post-run behaviors:
- normalize Batsim output into standard simulation artifacts,
- emit candidate fidelity report (if source trace parquet is available).

### Recommendation

```bash
hpcopt recommend generate \
  --baseline-report <baseline_sim_report.json> \
  --candidate-report <candidate_sim_report.json> \
  --fidelity-report <optional_fidelity_report.json>
```

Pareto multi-objective mode:

```bash
hpcopt recommend generate \
  --baseline-report <baseline.json> \
  --candidate-report <candidate1.json> \
  --candidate-report <candidate2.json> \
  --pareto
```

Outputs:
- recommendation report,
- recommendation manifest.

### Report Export

```bash
hpcopt report export --run-id <run_id> --format both
```

Outputs:
- run export JSON,
- run export markdown.

### Benchmark Suite

```bash
hpcopt report benchmark \
  --trace <dataset.parquet> \
  --raw-trace <optional_trace.swf.gz> \
  --policy FIFO_STRICT \
  --capacity-cpus 64 \
  --samples 3
```

Outputs:
- benchmark report JSON,
- benchmark manifest,
- benchmark history ledger (`benchmark_history.jsonl`).

### Credibility Protocol

```bash
hpcopt credibility run-suite \
  --config configs/credibility/default_sweep.yaml \
  --raw-dir data/raw \
  --out outputs/credibility

hpcopt credibility dossier \
  --input-dir outputs/credibility \
  --out outputs/credibility/dossier
```

Outputs:
- per-trace results (ingestion, profiling, training, simulation, fidelity, recommendation),
- optional predictor ensemble summary (when both sklearn and LightGBM models are available),
- aggregate credibility dossier (JSON + markdown).

### Analysis

```bash
hpcopt analysis sensitivity-sweep \
  --trace <dataset.parquet> \
  --capacity-cpus 64 \
  --k-values "0.0,0.25,0.5,0.75,1.0,1.5"

hpcopt analysis feature-importance \
  --model-dir outputs/models/runtime_ctc_v1 \
  --dataset <dataset.parquet>
```

Outputs:
- sensitivity report with optimal k identification,
- feature importance report with per-quantile rankings.

### Model Management

```bash
hpcopt model list
hpcopt model promote --model-id <model_id>
hpcopt model archive --model-id <model_id>
hpcopt model drift-check --eval-dataset <new_data.parquet>
```

Drift check outputs:
- per-feature PSI (Population Stability Index) values,
- per-quantile pinball loss degradation ratios,
- overall drift detection status.

### Artifact Retention

```bash
hpcopt artifacts cleanup --outputs-dir outputs --max-age-days 90
hpcopt artifacts cleanup --outputs-dir outputs --max-age-days 90 --no-dry-run
```

Protected from cleanup:
- current production model directory,
- artifacts referenced by credibility dossiers,
- model registry file.

### Reference Suite Lock

```bash
hpcopt data lock-reference-suite --config configs/data/reference_suite.yaml --raw-dir data/raw
```

## 3. API Overview

Service entrypoint:

```bash
hpcopt serve api --host 0.0.0.0 --port 8080
```

Implementation modules:
- `python/hpcopt/api/app.py` -- FastAPI application assembler, lifespan management, startup validation
- `python/hpcopt/api/models.py` -- Pydantic request/response schemas (`extra="forbid"`)
- `python/hpcopt/api/errors.py` -- RFC 7807 error helpers and exception handlers
- `python/hpcopt/api/middleware.py` -- Body size limit, auth, rate limiting, timeout, deprecation headers
- `python/hpcopt/api/endpoints.py` -- All route handlers (`register_routes()`)
- `python/hpcopt/api/auth.py` -- API key authentication and `EXEMPT_PATHS` constant
- `python/hpcopt/api/rate_limit.py` -- Token-bucket rate limiter (per-endpoint, keyed by API key)
- `python/hpcopt/api/model_cache.py` -- Thread-safe runtime predictor cache with startup pre-warming
- `python/hpcopt/api/deprecation.py` -- Deprecation config loading (RFC 8594/9745)
- `python/hpcopt/api/metrics.py` -- Prometheus metrics integration
- `python/hpcopt/api/tracing.py` -- OpenTelemetry distributed tracing
- `python/hpcopt/utils/secrets.py` -- File-based API key loading

## 4. API Endpoints

### Health

- `GET /health`

Response: service status and package version.

### Readiness

- `GET /ready`

Response: readiness status (`ok` when model is loaded, `degraded` otherwise).

### Runtime Prediction

- `POST /v1/runtime/predict`

Request body:
- `requested_cpus` (int, required, >= 1)
- `requested_runtime_sec` (float, optional)
- `requested_mem` (float, optional)
- `queue_id`, `partition_id`, `user_id`, `group_id` (string, optional)

Behavior:
- uses trained quantile model when available,
- else deterministic heuristic fallback,
- always returns `runtime_p50_sec`, `runtime_p90_sec`, and `runtime_guard_sec`.

### Resource Fit

- `POST /v1/resource-fit/predict`

Request body:
- `requested_cpus` (int, required, >= 1)
- `candidate_node_cpus` (list of int, optional)

Behavior:
- uses trained resource-fit model when available,
- else deterministic capacity-fit baseline,
- returns fragmentation risk category (`low`, `medium`, `high`) and recommended node size.

### Recommendation Retrieval

- `GET /v1/recommendations/{run_id}`

Response: stored recommendation report JSON for the given run ID.

Errors:
- `400` -- invalid `run_id` (forbidden characters),
- `404` -- recommendation not found,
- `500` -- internal error.

### Admin Log Level

- `POST /v1/admin/log-level`

Request body: `{"level": "DEBUG|INFO|WARNING|ERROR"}`.

Requires admin RBAC (API key with `admin-` prefix). Changes are audit-logged (who, when, old→new level). Returns `403 FORBIDDEN` for non-admin keys.

### Prometheus Metrics

- `GET /metrics`

Response: Prometheus text exposition format with:
- `hpcopt_requests_total` (counter by method/endpoint/status),
- `hpcopt_request_duration_seconds` (histogram by endpoint),
- `hpcopt_fallback_total` (counter),
- `hpcopt_model_loaded` (gauge),
- `hpcopt_model_staleness_seconds` (gauge),
- `hpcopt_rate_limit_rejections_total` (counter),
- `hpcopt_auth_failures_total` (counter),
- `hpcopt_cache_hits_total` (counter),
- `hpcopt_model_load_duration_seconds` (histogram).

Requires `prometheus_client` package. Returns empty response if unavailable.

## 5. Authentication

API key authentication is enabled when keys are configured via any of these sources (checked in priority order):

1. **`HPCOPT_API_KEYS_FILE`** env var -- points to a file with one key per line. Supports comments (`#`) and blank lines.
2. **`/run/secrets/hpcopt_api_keys`** -- Docker/Kubernetes secret mount (auto-detected).
3. **`HPCOPT_API_KEYS`** env var -- comma-separated list (legacy, logs deprecation warning).

Implementation:
- Auth check: `python/hpcopt/api/auth.py` (`check_api_key_auth()`, `EXEMPT_PATHS`)
- Key loading: `python/hpcopt/utils/secrets.py` (`load_api_keys()`)

Behavior:
- Keys are **re-read on every request**, enabling rotation without restart.
- Requests must include `X-API-Key` header with a valid key.
- `GET /health`, `GET /ready`, `GET /metrics`, `GET /docs`, `GET /openapi.json`, and `GET /v1/system/status` are always exempt.
- If no keys are configured, all requests pass through without authentication.

## 5a. Admin Role-Based Access Control (RBAC)

API endpoints under `/v1/admin/*` require admin-level access controlled via API key prefix:

- **Admin key prefix**: API keys with the `admin-` prefix (e.g., `admin-production-key-12345`) are granted admin access.
- **Non-admin keys**: rejected on admin paths with `403 FORBIDDEN`.
- **Current admin endpoints**: `POST /v1/admin/log-level`.
- **Audit logging**: admin operations (e.g., log-level changes) are audit-logged with the API key, timestamp, and old→new value.
- **Development mode**: when no API keys are configured, all paths are unrestricted.

Implementation: `python/hpcopt/api/auth.py` (`check_admin_auth()`, `ADMIN_KEY_PREFIX = "admin-"`).

## 5b. Request Body Size Limit

All request bodies are limited to **1 MB**. Requests exceeding this limit receive `413 PAYLOAD_TOO_LARGE` with an RFC 7807 error response.

Implementation: `api/middleware.py:body_size_limit_middleware()`, `_MAX_BODY_BYTES` (configurable via `max_body_bytes` in environment config).

## 5c. Input Validation Bounds

All Pydantic request models enforce strict input bounds:

- `requested_cpus`: ≤ 100,000
- `queue_depth_jobs`: ≤ 1,000,000
- `requested_runtime_sec`: ≤ 31,536,000 (1 year)
- `candidate_node_cpus`: max length 1,000
- All models use `extra="forbid"` to reject unknown fields.

Ingestion file size guards: 2GB max file size, 1M max line length, 50M row cap (SWF, Slurm, PBS parsers).

## 6. Request Timeout

All requests are subject to a configurable timeout. If the handler does not produce a response within the allowed time, the middleware returns `504 GATEWAY_TIMEOUT`.

- Default: `30` seconds.
- Override: set `HPCOPT_REQUEST_TIMEOUT_SEC` environment variable.
- Scope: covers the endpoint handler phase (header production); streaming response bodies are not covered.

## 7. Model Cache Pre-Warming

On startup, the API lifespan handler calls `model_cache.warm_cache()` to eagerly load the runtime predictor into cache. This eliminates cold-start latency on the first prediction request.

- If `HPCOPT_RUNTIME_MODEL_DIR` is set and the model directory exists, the predictor is loaded with retry (3 attempts, exponential backoff).
- If no model is found, the API starts normally and uses the fallback heuristic.

## 8. Docker Deployment

```bash
docker compose up --build
```

The `docker-compose.yaml` configures:
- port mapping (`8080:8080`),
- volume mounts for `data/` and `outputs/models/`,
- Docker secrets mount for API keys (`secrets/api_keys.txt` -> `/run/secrets/hpcopt_api_keys`),
- `HPCOPT_API_KEYS_FILE` env var pointing to the secret mount,
- health check against `/health`.

## 9. API Documentation

When API is running:

- OpenAPI UI: `http://localhost:8080/docs`

## 10. Error Response Format (RFC 7807 Problem Details)

All error responses follow the [RFC 7807](https://datatracker.ietf.org/doc/html/rfc7807) Problem Details specification:

```json
{
  "type": "urn:hpcopt:error:VALIDATION_ERROR",
  "title": "VALIDATION_ERROR",
  "status": 422,
  "detail": "Human-readable error message",
  "instance": "<trace-id>",
  "errors": ["<validation-details>"]
}
```

Status codes returned:

| Status | Error Code | Description |
|--------|------------|-------------|
| 401 | UNAUTHORIZED | Invalid or missing API key |
| 403 | FORBIDDEN | Valid key but lacks admin prefix for `/v1/admin/*` paths |
| 413 | PAYLOAD_TOO_LARGE | Request body exceeds 1 MB |
| 422 | VALIDATION_ERROR | Request validation failed (includes `errors` array) |
| 429 | RATE_LIMITED | Rate limit exceeded (includes `Retry-After` header) |
| 504 | GATEWAY_TIMEOUT | Request exceeded timeout (default 30s) |
| 500 | INTERNAL_ERROR | Unhandled exception |

Implementation: `api/errors.py:error_content()`, exception handlers in `api/errors.py`.

## 11. Circuit Breaker (Prediction Path)

The prediction path is protected by a circuit breaker that fails fast if model I/O repeatedly fails:

- **Failure threshold**: 5 consecutive errors.
- **Reset timeout**: 60 seconds.
- **When open**: returns deterministic heuristic fallback response (no model I/O attempted).
- **Half-open**: allows one trial request to test recovery.

Implementation: `api/endpoints.py:_prediction_circuit`, `utils/resilience.py:CircuitBreaker`.

## 12. Interface Stability Notes

- CLI commands and report schemas are treated as contract-bearing interfaces.
- Artifact keys used by evaluation/recommendation pipelines should be considered stable unless versioned migration is introduced.
- API endpoints under `/v1/` are versioned and will maintain backward compatibility within a major version.
