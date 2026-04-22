"""Pydantic request/response schemas.

Strict typing across the API boundary. Every response carries enough
metadata for a caller to audit which model produced it and when.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# We use `model_version`, `model_type`, etc. as fields. These collide with
# Pydantic's default protected namespace "model_", which is only relevant
# for subclasses customizing Pydantic internals. Opting out is the
# idiomatic fix and removes boot-time warnings without risk.
_BASE_CONFIG = ConfigDict(protected_namespaces=())


# ═══════════════════════════════════════════════════════════════════════════
#   /predict
# ═══════════════════════════════════════════════════════════════════════════
class PredictionRequest(BaseModel):
    """Client payload for POST /predict.

    `lags` must be the last N realized-volatility observations, oldest first,
    newest last. N equals the `lag` the model was trained with (exposed at
    GET /info).
    """

    model_config = _BASE_CONFIG

    lags: list[float] = Field(
        ...,
        description="Historical daily realized-volatility values, oldest to newest.",
        min_length=1,
        max_length=128,
    )

    @field_validator("lags")
    @classmethod
    def _finite_and_non_negative(cls, v: list[float]) -> list[float]:
        for x in v:
            if x is None or not _is_finite(x):
                raise ValueError("lags must contain finite float values")
            if x < 0:
                raise ValueError(
                    "lags represent realized volatility and must be non-negative"
                )
        return v


class HorizonPoint(BaseModel):
    """A single point forecast at a given horizon."""

    model_config = _BASE_CONFIG

    horizon_day: int = Field(..., ge=1, description="Horizon index, h=1..H.")
    volatility: float = Field(..., description="Forecasted realized volatility.")


class DriftFlags(BaseModel):
    """Per-request drift diagnostics attached to a prediction response."""

    model_config = _BASE_CONFIG

    input_out_of_distribution: bool = Field(
        ...,
        description=(
            "True if any input lag falls outside the [p1, p99] range of the "
            "training distribution."
        ),
    )
    fraction_out_of_distribution: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of input values outside the training range.",
    )


class PredictionResponse(BaseModel):
    """Response from POST /predict."""

    model_config = _BASE_CONFIG

    volatility: list[float] = Field(
        ...,
        description="Point forecast, one value per horizon day (length H).",
    )
    horizons: list[HorizonPoint] = Field(
        ...,
        description="Structured forecast with horizon index.",
    )
    model_version: str
    model_type: Literal["MLP", "RNN", "LSTM"]
    lag: int = Field(..., description="Window length the model was trained on.")
    prediction_timestamp: datetime
    drift: DriftFlags


# ═══════════════════════════════════════════════════════════════════════════
#   /health
# ═══════════════════════════════════════════════════════════════════════════
class HealthResponse(BaseModel):
    """Operational status of the service."""

    model_config = _BASE_CONFIG

    status: Literal["healthy", "degraded", "unhealthy"]
    uptime_seconds: float
    model_loaded: bool
    model_version: str
    model_type: Literal["MLP", "RNN", "LSTM"] | None = None
    predictions_total: int
    predictions_last_window: int = Field(
        ...,
        description="Predictions in the most recent telemetry window.",
    )
    drift_pct_out_of_distribution: float = Field(
        ...,
        description="Percentage of recent requests with any OOD input feature.",
    )
    drift_threshold_pct: float
    degraded_reasons: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#   /info
# ═══════════════════════════════════════════════════════════════════════════
class BaselineComparison(BaseModel):
    """Statistical comparison of the model against the naive baseline."""

    model_config = _BASE_CONFIG

    naive_rmse: float = Field(..., description="RMSE of persistence baseline.")
    model_rmse: float = Field(..., description="RMSE of the deployed model.")
    rmse_improvement_pct: float = Field(
        ...,
        description="(naive_rmse - model_rmse) / naive_rmse * 100.",
    )
    diebold_mariano_statistic: float
    diebold_mariano_pvalue: float = Field(
        ...,
        description=(
            "Two-sided p-value. Values < 0.05 indicate the model's forecast "
            "accuracy differs from the naive baseline with statistical "
            "significance."
        ),
    )
    significance_alpha: float = 0.05
    beats_naive_significantly: bool


class ModelInfoResponse(BaseModel):
    """Deployed model metadata, exposed for auditability."""

    model_config = _BASE_CONFIG

    app_name: str
    app_version: str
    model_version: str
    model_type: Literal["MLP", "RNN", "LSTM"]
    lag: int
    forecast_horizon: int
    training_stats_available: bool
    baseline_comparison: BaselineComparison | None = None


# ═══════════════════════════════════════════════════════════════════════════
#   /predict/compare  (plus — includes naive baseline in response)
# ═══════════════════════════════════════════════════════════════════════════
class ComparePredictionResponse(PredictionResponse):
    """Prediction response extended with the naive-baseline point forecast.

    The naive baseline is persistence: ŷ(t+h) = y(t) for every h.
    Useful for downstream consumers to see the model's delta over the
    trivial predictor in real time.
    """

    model_config = _BASE_CONFIG

    naive_baseline: list[float] = Field(
        ...,
        description="Persistence baseline prediction (last observed value "
        "repeated H times).",
    )


# ═══════════════════════════════════════════════════════════════════════════
#   Errors
# ═══════════════════════════════════════════════════════════════════════════
class ErrorResponse(BaseModel):
    model_config = _BASE_CONFIG

    detail: str
    code: str
    hint: str | None = None


# ── helpers ─────────────────────────────────────────────────────────────────
def _is_finite(x: float) -> bool:
    import math

    return not (math.isnan(x) or math.isinf(x))
