"""Shared pytest fixtures.

We avoid loading TensorFlow in the test suite by patching the
`VolatilityModel` singleton with a lightweight fake. Tests that want to
exercise real inference can opt into a `real_model` fixture (skipped in
CI if the artifacts are missing).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fake model — no TF dependency
# ─────────────────────────────────────────────────────────────────────────────
class FakeVolatilityModel:
    """In-memory drop-in for the real VolatilityModel used by tests."""

    def __init__(
        self,
        lag: int = 14,
        model_type: Literal["MLP", "RNN", "LSTM"] = "MLP",
        training_stats: dict | None = None,
    ) -> None:
        self._lag = lag
        self._model_type = model_type
        self._training_stats = training_stats or {
            "feature_p1": 0.01,
            "feature_p99": 0.20,
        }
        self._is_ready = True

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def lag(self) -> int:
        return self._lag

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def training_stats(self) -> dict | None:
        return self._training_stats

    def load(self, *_, **__) -> None:  # no-op
        self._is_ready = True

    def predict(self, lags: list[float]) -> np.ndarray:
        from app.model import InvalidInputError

        if len(lags) != self._lag:
            raise InvalidInputError(
                f"Expected {self._lag} lag values; got {len(lags)}."
            )
        # Deterministic fake: decaying forecast centered on the last obs.
        last = float(lags[-1])
        decay = np.linspace(1.0, 0.9, 7)
        return np.clip(np.asarray([last * d for d in decay]), 0.0, None)

    @staticmethod
    def naive_baseline(lags: list[float], horizon: int) -> list[float]:
        return [float(lags[-1])] * int(horizon)

    def drift_flags_for(self, lags: list[float]) -> tuple[bool, float]:
        lo = self._training_stats["feature_p1"]
        hi = self._training_stats["feature_p99"]
        arr = np.asarray(lags, dtype=np.float64)
        ood = (arr < lo) | (arr > hi)
        return bool(ood.any()), float(ood.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def fake_model() -> FakeVolatilityModel:
    return FakeVolatilityModel(lag=14, model_type="MLP")


@pytest.fixture
def client(monkeypatch, fake_model):
    """TestClient with the real model replaced by a fake."""
    from fastapi.testclient import TestClient

    from app import main as main_module
    from app import model as model_module

    # Patch the module-level singleton so main.py sees the fake.
    monkeypatch.setattr(model_module, "model", fake_model)
    monkeypatch.setattr(main_module, "model", fake_model)

    # Reset telemetry so each test starts clean.
    main_module.telemetry.reset()

    with TestClient(main_module.app) as c:
        yield c


@pytest.fixture
def valid_lags() -> list[float]:
    return [0.05] * 14


@pytest.fixture
def ood_lags() -> list[float]:
    # Well above feature_p99=0.20 to force OOD flags.
    return [0.95] * 14


@pytest.fixture
def baselines_payload(tmp_path: Path) -> Path:
    payload = {
        "naive_rmse": 0.2450,
        "model_rmse": 0.2040,
        "rmse_improvement_pct": 16.73,
        "diebold_mariano_statistic": -2.413,
        "diebold_mariano_pvalue": 0.018,
        "significance_alpha": 0.05,
        "beats_naive_significantly": True,
    }
    p = tmp_path / "baselines.json"
    p.write_text(json.dumps(payload))
    return p
