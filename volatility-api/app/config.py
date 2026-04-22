"""Centralized runtime configuration.

All tunable values live here so that the rest of the code never reads
environment variables directly. This keeps the app testable and makes
operational parameters explicit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # ── Paths ─────────────────────────────────────────────────────────────
    app_dir: Path = Path(__file__).resolve().parent
    model_path: Path = Path(__file__).resolve().parent / "model_keras.keras"
    scalers_path: Path = Path(__file__).resolve().parent / "scalers.joblib"
    training_stats_path: Path = Path(__file__).resolve().parent / "training_stats.json"
    baselines_path: Path = Path(__file__).resolve().parent / "baselines.json"

    # ── API metadata ──────────────────────────────────────────────────────
    app_name: str = "Bitcoin Volatility Forecasting API"
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    model_version: str = os.getenv("MODEL_VERSION", "v1")

    # ── Drift monitoring ──────────────────────────────────────────────────
    # Ring buffer size for recent predictions
    telemetry_buffer_size: int = int(os.getenv("TELEMETRY_BUFFER_SIZE", "1000"))
    # % of requests out-of-distribution that flips status to "degraded"
    drift_degraded_threshold_pct: float = float(
        os.getenv("DRIFT_DEGRADED_THRESHOLD_PCT", "10.0")
    )
    # Percentile range considered "in-distribution"
    ood_lower_percentile: float = 1.0
    ood_upper_percentile: float = 99.0

    # ── Prediction semantics ──────────────────────────────────────────────
    forecast_horizon: int = 7  # days predicted ahead


settings = Settings()
