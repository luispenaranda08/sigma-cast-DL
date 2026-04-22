"""FastAPI application — serving layer.

Endpoints:
  GET  /health            operational status + drift telemetry
  GET  /info              model metadata + baseline comparison
  POST /predict           point forecast, H horizons ahead
  POST /predict/compare   same as /predict, plus naive-baseline forecast

UI:
  GET  /                  Palantir-style operations dashboard (static)
  GET  /docs              Scalar API Reference (dark-themed, replaces Swagger)
  GET  /openapi.json      OpenAPI 3.1 spec (FastAPI default, not overridden)

Design notes
------------
* The Keras model is loaded once at startup and reused. All preprocessing
  lives in `VolatilityModel`; this module only wires HTTP to that wrapper.
* Errors carry a stable `code` string so that clients can react
  programmatically without parsing human messages.
* The `/health` endpoint is cheap and side-effect-free — suitable as a
  liveness/readiness probe for container orchestrators.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.baselines import load_baseline_comparison
from app.config import settings
from app.model import (
    InvalidInputError,
    ModelNotReadyError,
    model,
)
from app.monitoring import TelemetryStore, evaluate_status
from app.schemas import (
    ComparePredictionResponse,
    DriftFlags,
    ErrorResponse,
    HealthResponse,
    HorizonPoint,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("app.main")

telemetry = TelemetryStore(buffer_size=settings.telemetry_buffer_size)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — load the model once.
    try:
        model.load()
        logger.info("Startup complete — model loaded.")
    except Exception as e:  # pragma: no cover — if this fails, /health returns unhealthy
        logger.exception("Model failed to load at startup: %s", e)
    yield
    # Shutdown hook reserved for future use (flush metrics, close files, etc.)


# ── Tag metadata renders as section descriptions in Scalar ──────────────────
tags_metadata = [
    {
        "name": "operations",
        "description": (
            "Liveness, readiness and deployment metadata. "
            "Safe to poll as a container liveness probe — side-effect-free."
        ),
    },
    {
        "name": "inference",
        "description": (
            "Point forecasts at horizon H, optionally benchmarked against "
            "a naive persistence baseline."
        ),
    },
]


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Forecasts 7-day-ahead Bitcoin realized volatility from a recent "
        "window of daily observations. Ships with drift monitoring and a "
        "Diebold-Mariano comparison against a naive persistence baseline.\n\n"
        "### Pipeline\n"
        "`data → features → cross-validated Keras model → FastAPI → Docker → CI/CD`\n\n"
        "### Model selection\n"
        "Best global candidate is selected across **MLP / RNN / LSTM** with "
        "grid search over lag windows; the active model and lag are reported "
        "by `GET /info`."
    ),
    openapi_tags=tags_metadata,
    # ── Desactivamos Swagger & ReDoc nativos; servimos Scalar en /docs. ────
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

# CORS abierto para que el dashboard pueda consumir los endpoints desde el mismo origen
# (y también desde herramientas externas si se prueban).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static assets (dashboard + scalar theme) ───────────────────────────────
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
# Si prefieres la carpeta static dentro de app/, cambia por:
#   _STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
else:
    logger.warning("Static dir not found at %s — dashboard disabled.", _STATIC_DIR)


# ═══════════════════════════════════════════════════════════════════════════
#   UI
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/", include_in_schema=False)
async def dashboard():
    """Operations dashboard (live metrics + forecast chart + endpoint index)."""
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse(
            {"detail": "Dashboard not deployed — static/index.html missing."},
            status_code=404,
        )
    return FileResponse(index)


@app.get("/docs", include_in_schema=False)
async def scalar_docs():
    """Scalar API Reference — dark-themed replacement for Swagger UI."""
    return HTMLResponse(
        """<!doctype html>
<html>
  <head>
    <title>BTC Volatility API — Reference</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="stylesheet" href="/static/scalar-theme.css" />
  </head>
  <body>
    <script
      id="api-reference"
      data-url="/openapi.json"
      data-configuration='{"theme":"none","layout":"modern","darkMode":true,"hideDownloadButton":false}'
    ></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
  </body>
</html>"""
    )


# ═══════════════════════════════════════════════════════════════════════════
#   /health
# ═══════════════════════════════════════════════════════════════════════════
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["operations"],
    summary="Liveness and operational status.",
)
async def health() -> HealthResponse:
    snap = telemetry.snapshot()
    model_loaded = model.is_ready
    status_str, reasons = evaluate_status(
        drift_pct=snap["drift_pct_out_of_distribution"],
        threshold_pct=settings.drift_degraded_threshold_pct,
        model_loaded=model_loaded,
    )
    return HealthResponse(
        status=status_str,  # type: ignore[arg-type]
        uptime_seconds=snap["uptime_seconds"],
        model_loaded=model_loaded,
        model_version=settings.model_version,
        model_type=(model.model_type if model_loaded else None),
        predictions_total=snap["predictions_total"],
        predictions_last_window=snap["predictions_last_window"],
        drift_pct_out_of_distribution=snap["drift_pct_out_of_distribution"],
        drift_threshold_pct=settings.drift_degraded_threshold_pct,
        degraded_reasons=reasons,
    )


# ═══════════════════════════════════════════════════════════════════════════
#   /info
# ═══════════════════════════════════════════════════════════════════════════
@app.get(
    "/info",
    response_model=ModelInfoResponse,
    tags=["operations"],
    summary="Deployed model metadata and baseline comparison.",
)
async def info() -> ModelInfoResponse:
    if not model.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading.",
        )
    return ModelInfoResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        model_version=settings.model_version,
        model_type=model.model_type,
        lag=model.lag,
        forecast_horizon=settings.forecast_horizon,
        training_stats_available=(model.training_stats is not None),
        baseline_comparison=load_baseline_comparison(),
    )


# ═══════════════════════════════════════════════════════════════════════════
#   /predict
# ═══════════════════════════════════════════════════════════════════════════
@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["inference"],
    summary="Point forecast for the next H days.",
)
async def predict(req: PredictionRequest) -> PredictionResponse:
    try:
        yhat = model.predict(req.lags)
    except ModelNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except InvalidInputError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    ood_any, ood_frac = model.drift_flags_for(req.lags)
    telemetry.record(any_ood=ood_any, fraction_ood=ood_frac, lag_count=len(req.lags))

    return PredictionResponse(
        volatility=[float(v) for v in yhat],
        horizons=[
            HorizonPoint(horizon_day=h + 1, volatility=float(v))
            for h, v in enumerate(yhat)
        ],
        model_version=settings.model_version,
        model_type=model.model_type,
        lag=model.lag,
        prediction_timestamp=datetime.now(timezone.utc),
        drift=DriftFlags(
            input_out_of_distribution=ood_any,
            fraction_out_of_distribution=ood_frac,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
#   /predict/compare  (plus — naive baseline attached)
# ═══════════════════════════════════════════════════════════════════════════
@app.post(
    "/predict/compare",
    response_model=ComparePredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["inference"],
    summary="Point forecast plus the naive persistence baseline.",
)
async def predict_compare(req: PredictionRequest) -> ComparePredictionResponse:
    base_resp = await predict(req)
    naive = model.naive_baseline(req.lags, horizon=settings.forecast_horizon)
    return ComparePredictionResponse(
        **base_resp.model_dump(),
        naive_baseline=naive,
    )


# ═══════════════════════════════════════════════════════════════════════════
#   Global error handlers
# ═══════════════════════════════════════════════════════════════════════════
@app.exception_handler(HTTPException)
async def _http_exc_handler(_, exc: HTTPException):
    code_map = {
        400: "invalid_input",
        422: "validation_error",
        503: "service_unavailable",
    }
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=str(exc.detail),
            code=code_map.get(exc.status_code, "error"),
        ).model_dump(),
    )
