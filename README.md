# Sigma-cast-DL

## Bitcoin Volatility Forecasting API

Production-minded FastAPI service that serves a Keras model predicting the next 7 days of Bitcoin realized volatility from a recent window of daily observations.

Beyond the baseline scope (REST endpoint + Docker + CI), this service ships with two capabilities typically missing from student projects:

1. **Statistical validation of the model against a naive baseline.** A Diebold-Mariano test on squared-error loss differentials — reported at `GET /info` — lets any consumer verify that the deployed model beats persistence with statistical significance, not just anecdotally.
2. **Runtime drift monitoring.** Every request's input is checked against the training distribution; the fraction of out-of-distribution requests flips `GET /health` to `degraded` when it crosses a configurable threshold. Designed to integrate with Kubernetes liveness probes or any HTTP-based monitor.

---

## Architecture

```
┌─────────────┐   POST /predict          ┌──────────────────────┐
│   Client    │ ───────────────────────▶ │  FastAPI (app.main)  │
└─────────────┘                          │                      │
                                         │  ┌────────────────┐  │
                                         │  │ VolatilityModel │  │
                                         │  │  • scaler_x    │  │
                                         │  │  • scaler_y    │  │
                                         │  │  • Keras .keras│  │
                                         │  └────────────────┘  │
                                         │  ┌────────────────┐  │
                                         │  │ TelemetryStore │  │  ring buffer
                                         │  │  (drift)       │  │  (thread-safe)
                                         │  └────────────────┘  │
                                         └──────────────────────┘
                                                 │
                                                 ▼
                                    app/baselines.json  (Diebold-Mariano, offline)
                                    app/training_stats.json  (p1/p99, offline)
```

The API is stateless across requests except for the in-process telemetry buffer. Moving to multi-replica deployment only requires replacing `TelemetryStore` with a Prometheus counter or a shared store.

---

## Project layout

```
.
├── app/                         # Serving layer
│   ├── __init__.py
│   ├── config.py                #  env-driven settings
│   ├── schemas.py               #  Pydantic request/response models
│   ├── model.py                 #  Keras wrapper + preprocessing + drift flags
│   ├── monitoring.py            #  in-memory ring buffer + status evaluator
│   ├── baselines.py             #  loader for the offline DM comparison
│   ├── main.py                  #  FastAPI app + endpoints
│   ├── model_keras.keras        #  (produced by training notebook)
│   ├── scalers.joblib           #  (produced by training notebook)
│   ├── baselines.json           #  (produced by scripts/compute_baselines.py)
│   └── training_stats.json      #  (produced by scripts/compute_training_stats.py)
├── scripts/                     # Offline tools
│   ├── compute_baselines.py     #  naive baseline + Diebold-Mariano
│   └── compute_training_stats.py
├── tests/                       # Pytest suite (mocks TensorFlow)
│   ├── conftest.py              #  FakeVolatilityModel fixture
│   ├── test_api.py
│   ├── test_model.py
│   ├── test_monitoring.py
│   └── test_baselines.py
├── .github/workflows/ci.yml     # Lint + tests on push / PR
├── Dockerfile                   # Non-root, healthcheck, OCI labels
├── Makefile                     # setup · test · run · docker-build · ...
├── pyproject.toml               # pytest, coverage, ruff
├── requirements.txt             # Runtime
└── requirements-dev.txt         # + pytest, httpx, ruff
```

---

## Quick start

### Prerequisites

* Python 3.11+
* A trained model: `app/model_keras.keras` and `app/scalers.joblib` produced by the training notebook (`3_model_training.ipynb`).

### Local development

```bash
# One-time setup
make setup

# Generate the offline artifacts (baselines + drift stats)
make artifacts

# Run the API with auto-reload
make run
# → http://localhost:8000/docs
```

### Tests

```bash
make test          # Quiet
make cov           # With coverage report
```

TensorFlow is **not** loaded during the test suite. Tests run against a `FakeVolatilityModel` fixture, which keeps CI under a minute.

### Docker

```bash
make docker-build
make docker-run
# → http://localhost:8000/health
```

The container runs as a non-root user (UID 1001) and registers an internal `HEALTHCHECK` that hits `/health` every 30 seconds.

---

## API

### `POST /predict`

```jsonc
// Request — length of `lags` must match the model's training lag
{
  "lags": [0.043, 0.038, 0.051, 0.047, 0.055, 0.062, 0.058,
           0.049, 0.041, 0.046, 0.053, 0.057, 0.064, 0.061]
}
```

```jsonc
// 200 OK
{
  "volatility": [0.059, 0.058, 0.057, 0.057, 0.056, 0.056, 0.055],
  "horizons": [
    {"horizon_day": 1, "volatility": 0.059},
    {"horizon_day": 2, "volatility": 0.058}
    // ...
  ],
  "model_version": "v1",
  "model_type": "MLP",
  "lag": 14,
  "prediction_timestamp": "2026-04-21T14:23:11.482Z",
  "drift": {
    "input_out_of_distribution": false,
    "fraction_out_of_distribution": 0.0
  }
}
```

Errors: `400` for wrong `lags` length; `422` for schema violations (missing, non-numeric, negative); `503` if the model has not finished loading.

### `POST /predict/compare`

Same contract as `/predict`, plus a `naive_baseline` field carrying the persistence forecast (`[lags[-1]] * 7`). Useful for side-by-side visualization without a second round-trip.

### `GET /health`

Cheap, side-effect-free. Suitable as a container liveness/readiness probe.

```jsonc
{
  "status": "healthy",          // "healthy" | "degraded" | "unhealthy"
  "uptime_seconds": 1523.4,
  "model_loaded": true,
  "model_version": "v1",
  "model_type": "MLP",
  "predictions_total": 847,
  "predictions_last_window": 847,
  "drift_pct_out_of_distribution": 3.18,
  "drift_threshold_pct": 10.0,
  "degraded_reasons": []
}
```

Status transitions:

| Condition                                                      | `status`     |
| -------------------------------------------------------------- | ------------ |
| Model not loaded                                               | `unhealthy`  |
| `drift_pct_out_of_distribution > drift_threshold_pct`          | `degraded`   |
| Otherwise                                                      | `healthy`    |

### `GET /info`

Model metadata and the offline baseline comparison:

```jsonc
{
  "app_name": "Bitcoin Volatility Forecasting API",
  "app_version": "1.0.0",
  "model_version": "v1",
  "model_type": "MLP",
  "lag": 14,
  "forecast_horizon": 7,
  "training_stats_available": true,
  "baseline_comparison": {
    "naive_rmse": 0.2450,
    "model_rmse": 0.2040,
    "rmse_improvement_pct": 16.73,
    "diebold_mariano_statistic": -2.413,
    "diebold_mariano_pvalue": 0.018,
    "significance_alpha": 0.05,
    "beats_naive_significantly": true
  }
}
```

---

## Statistical validation

We compare the deployed model against a naive persistence baseline (`ŷ(t+h) = y(t)` for every horizon) using the Diebold-Mariano test on squared-error loss differentials:

* **Null hypothesis**: the two forecasts have equal expected loss.
* **Test statistic**: Harvey-Leybourne-Newbold-corrected DM statistic on `d_t = e_model² - e_naive²`.
* **Reference distribution**: Student's t with `n-1` degrees of freedom (HLN, 1997).

A p-value below 0.05 combined with `model_rmse < naive_rmse` is recorded as `beats_naive_significantly = true`. The computation is deterministic — given the same `results.pkl`, the resulting `baselines.json` is byte-identical.

Regenerate after retraining:

```bash
make baselines   # writes app/baselines.json
```

---

## Drift monitoring

At startup the API loads `app/training_stats.json` (produced by `scripts/compute_training_stats.py` from `features.pkl`). This file contains the 1st and 99th percentile of training-set features. Every request computes:

* `input_out_of_distribution` — `true` if any lag value falls outside `[p1, p99]`.
* `fraction_out_of_distribution` — fraction of lag values outside the range.

A thread-safe ring buffer tracks the last `TELEMETRY_BUFFER_SIZE` predictions (default 1000). `GET /health` exposes the fraction with any OOD input across that window and flips to `degraded` when the fraction strictly exceeds `DRIFT_DEGRADED_THRESHOLD_PCT` (default 10%).

If `training_stats.json` is missing at startup, the API logs a warning and reports `fraction_out_of_distribution = 0` on every request — it remains fully functional but loses the drift signal.

---

## Configuration

All knobs are environment variables (see `app/config.py`):

| Variable                        | Default | Purpose                                              |
| ------------------------------- | ------- | ---------------------------------------------------- |
| `APP_VERSION`                   | `1.0.0` | Surfaced at `/info`                                  |
| `MODEL_VERSION`                 | `v1`    | Logical identifier of the deployed model artifacts   |
| `TELEMETRY_BUFFER_SIZE`         | `1000`  | Recent-predictions window for drift computation      |
| `DRIFT_DEGRADED_THRESHOLD_PCT`  | `10.0`  | Percent of OOD requests that flips `/health` status  |

---

## CI

`.github/workflows/ci.yml` runs on every push and PR to `main`:

1. **Lint** — `ruff check` and `ruff format --check`.
2. **Test** — `pytest --cov` against Python 3.11.

Pip is cached keyed on `requirements-dev.txt`. Typical run time: under 90 seconds.

---

## Deployment notes

* **Process model.** One Uvicorn worker per container replica. TensorFlow's graph-mode inference is thread-safe, so a single process with the default worker settings handles the request load for a research-scale API.
* **Startup time.** First request takes ~2 seconds because the Keras model warms up its inference graph during the `lifespan` event. Subsequent requests are sub-millisecond on CPU.
* **Rolling updates.** The `/health` endpoint is the canonical readiness probe. Set `start_period ≥ 40s` on your orchestrator — TensorFlow's import alone takes longer than the default 10 seconds.

---

## License

MIT.
