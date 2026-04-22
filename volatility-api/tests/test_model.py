"""Unit tests for the model wrapper."""
from __future__ import annotations

import numpy as np
import pytest


def test_fake_model_predict_returns_length_7(fake_model, valid_lags):
    y = fake_model.predict(valid_lags)
    assert isinstance(y, np.ndarray)
    assert y.shape == (7,)
    assert all(isinstance(float(v), float) for v in y), "must be JSON-serializable floats"


def test_fake_model_predict_rejects_wrong_length(fake_model):
    from app.model import InvalidInputError

    with pytest.raises(InvalidInputError) as exc:
        fake_model.predict([0.05] * 10)  # lag=14 expected
    assert "Expected 14 lag values" in str(exc.value)


def test_fake_model_predict_rejects_empty_input(fake_model):
    from app.model import InvalidInputError

    with pytest.raises(InvalidInputError):
        fake_model.predict([])


def test_naive_baseline_is_persistence(fake_model):
    lags = [0.01, 0.02, 0.03, 0.05]
    out = fake_model.naive_baseline(lags, horizon=7)
    assert out == [0.05] * 7


def test_drift_flags_for_in_distribution(fake_model, valid_lags):
    any_ood, frac = fake_model.drift_flags_for(valid_lags)
    assert any_ood is False
    assert frac == 0.0


def test_drift_flags_for_out_of_distribution(fake_model, ood_lags):
    any_ood, frac = fake_model.drift_flags_for(ood_lags)
    assert any_ood is True
    assert frac == 1.0  # all 14 values above p99


def test_drift_flags_mixed_distribution(fake_model):
    lags = [0.05] * 12 + [0.95, 0.99]  # 2 OOD out of 14
    any_ood, frac = fake_model.drift_flags_for(lags)
    assert any_ood is True
    assert 0.0 < frac < 1.0
    assert frac == pytest.approx(2 / 14, rel=1e-6)


def test_predictions_are_non_negative(fake_model, valid_lags):
    """Volatility forecasts must never be negative (physical constraint)."""
    y = fake_model.predict(valid_lags)
    assert (y >= 0).all()
