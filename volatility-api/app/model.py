"""Model wrapper — load once, predict on demand.

Encapsulates the Keras model + scalers + lag + model_type into a single
object. All preprocessing (reshape, scaling, inverse-scaling) happens here
so that the API layer stays thin.

Thread-safety: TensorFlow is thread-safe for `predict` when the model is
built in graph mode, which is the case here. We serialize loading with a
lock to avoid double-loading in a multi-worker Uvicorn setup.
"""
from __future__ import annotations

import contextlib
import json
import logging
import threading
from pathlib import Path
from typing import Literal

import joblib
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_MODEL_TYPES = ("MLP", "RNN", "LSTM")


class ModelNotReadyError(RuntimeError):
    """Raised when inference is attempted before the model is loaded."""


class InvalidInputError(ValueError):
    """Raised when the input payload violates the model's expected shape."""


class VolatilityModel:
    """Singleton-ish wrapper around the trained Keras model."""

    def __init__(self) -> None:
        self._model = None  # lazy
        self._scaler_x = None
        self._scaler_y = None
        self._lag: int | None = None
        self._model_type: Literal["MLP", "RNN", "LSTM"] | None = None
        self._training_stats: dict | None = None
        self._lock = threading.Lock()

    # ── public state ─────────────────────────────────────────────────────
    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def lag(self) -> int:
        if self._lag is None:
            raise ModelNotReadyError("Model is not loaded yet.")
        return self._lag

    @property
    def model_type(self) -> Literal["MLP", "RNN", "LSTM"]:
        if self._model_type is None:
            raise ModelNotReadyError("Model is not loaded yet.")
        return self._model_type

    @property
    def training_stats(self) -> dict | None:
        """Per-feature percentiles used for drift detection."""
        return self._training_stats

    # ── helpers ──────────────────────────────────────────────────────────
    def _device_ctx(self):
        """CPU context for LSTM (DirectML no soporta CudnnRNN)."""
        import tensorflow as tf
        if self._model_type == "LSTM":
            return tf.device("/CPU:0")
        return contextlib.nullcontext()

    # ── lifecycle ────────────────────────────────────────────────────────
    def load(
        self,
        model_path: Path | None = None,
        scalers_path: Path | None = None,
        training_stats_path: Path | None = None,
    ) -> None:
        """Load model, scalers, and training stats from disk.

        Idempotent. Safe to call from an async startup event.
        """
        with self._lock:
            if self.is_ready:
                return

            model_path = model_path or settings.model_path
            scalers_path = scalers_path or settings.scalers_path
            training_stats_path = (
                training_stats_path or settings.training_stats_path
            )

            import tensorflow as tf

            # ── Scalers PRIMERO — necesitamos lag y model_type
            #    antes de llamar _dummy_input() en el warmup ──────────────
            logger.info("Loading scalers from %s", scalers_path)
            bundle = joblib.load(scalers_path)
            self._scaler_x = bundle["scaler_x"]
            self._scaler_y = bundle["scaler_y"]
            self._lag = int(bundle["lag"])
            mtype = bundle["model_type"]
            if mtype not in _MODEL_TYPES:
                raise ValueError(
                    f"Unexpected model_type={mtype!r} in scalers.joblib"
                )
            self._model_type = mtype  # type: ignore[assignment]

            # ── Modelo DESPUÉS — _dummy_input ya conoce lag y model_type ─
            logger.info("Loading Keras model from %s", model_path)
            self._model = tf.keras.models.load_model(
                model_path, compile=False
            )
            # Warm-up: LSTM forzado a CPU por incompatibilidad DirectML/CudnnRNN
            with self._device_ctx():
                _ = self._model.predict(self._dummy_input(), verbose=0)

            if training_stats_path.exists():
                with open(training_stats_path) as f:
                    self._training_stats = json.load(f)
                logger.info("Loaded training stats from %s", training_stats_path)
            else:
                logger.warning(
                    "training_stats.json not found at %s — drift detection "
                    "will report fraction_out_of_distribution=0 always.",
                    training_stats_path,
                )

            logger.info(
                "Model ready — type=%s lag=%d horizon=%d",
                self._model_type,
                self._lag,
                settings.forecast_horizon,
            )

    # ── inference ────────────────────────────────────────────────────────
    def predict(self, lags: list[float]) -> np.ndarray:
        """Return the H-step forecast for the given lag vector.

        Raises InvalidInputError if the length of `lags` does not match the
        lag the model was trained on.
        """
        if not self.is_ready:
            raise ModelNotReadyError("Model is not loaded.")
        self._validate_input(lags)

        x = np.asarray(lags, dtype=np.float32).reshape(1, -1)
        x_scaled = self._scaler_x.transform(x)
        x_in = self._reshape_for_model(x_scaled)

        # LSTM forzado a CPU por incompatibilidad DirectML/CudnnRNN
        with self._device_ctx():
            y_scaled = self._model.predict(x_in, verbose=0)

        y = self._scaler_y.inverse_transform(y_scaled)
        forecast = np.asarray(y, dtype=np.float64).reshape(-1)
        return np.clip(forecast, a_min=0.0, a_max=None)

    # ── baseline ─────────────────────────────────────────────────────────
    @staticmethod
    def naive_baseline(lags: list[float], horizon: int) -> list[float]:
        """Persistence forecast: ŷ(t+h) = y(t) for all h."""
        return [float(lags[-1])] * int(horizon)

    # ── drift diagnostics ────────────────────────────────────────────────
    def drift_flags_for(self, lags: list[float]) -> tuple[bool, float]:
        """Return (input_out_of_distribution, fraction_out_of_distribution)."""
        if not self._training_stats:
            return False, 0.0
        lo = self._training_stats.get("feature_p1")
        hi = self._training_stats.get("feature_p99")
        if lo is None or hi is None:
            return False, 0.0
        arr = np.asarray(lags, dtype=np.float64)
        ood = (arr < lo) | (arr > hi)
        frac = float(ood.mean())
        return bool(ood.any()), frac

    # ── helpers ──────────────────────────────────────────────────────────
    def _validate_input(self, lags: list[float]) -> None:
        if self._lag is None:
            raise ModelNotReadyError("Model is not loaded.")
        if len(lags) != self._lag:
            raise InvalidInputError(
                f"Expected {self._lag} lag values; got {len(lags)}."
            )

    def _reshape_for_model(self, x_scaled: np.ndarray) -> np.ndarray:
        if self._model_type in ("RNN", "LSTM"):
            return x_scaled.reshape(1, self._lag, 1)
        return x_scaled

    def _dummy_input(self) -> np.ndarray:
        """Zero-valued tensor matching the model's expected input shape."""
        lag = int(self._lag) if self._lag else 14
        mtype = self._model_type or "MLP"
        if mtype in ("RNN", "LSTM"):
            return np.zeros((1, lag, 1), dtype=np.float32)
        return np.zeros((1, lag), dtype=np.float32)


# Singleton exported to the rest of the app.
model = VolatilityModel()