"""Tests for the Diebold-Mariano implementation."""
from __future__ import annotations

import json

import numpy as np
import pytest


def test_dm_statistic_on_identical_forecasts_is_zero_with_high_pvalue():
    from scripts.compute_baselines import diebold_mariano

    rng = np.random.default_rng(42)
    errs = rng.normal(0, 0.1, size=200)
    stat, pval = diebold_mariano(errs, errs, horizon=1)
    assert stat == pytest.approx(0.0, abs=1e-8)
    assert pval == pytest.approx(1.0, abs=1e-8)


def test_dm_rejects_equality_when_model_is_clearly_better():
    from scripts.compute_baselines import diebold_mariano

    rng = np.random.default_rng(0)
    # Model errors centered on 0 with std 0.05; naive errors with std 0.20.
    err_model = rng.normal(0.0, 0.05, size=500)
    err_naive = rng.normal(0.0, 0.20, size=500)
    stat, pval = diebold_mariano(err_model, err_naive, horizon=1)
    assert stat < 0  # model loss < naive loss → negative mean d
    assert pval < 0.01


def test_dm_returns_nan_on_very_short_series():
    from scripts.compute_baselines import diebold_mariano

    stat, pval = diebold_mariano(
        np.array([0.1, 0.2]), np.array([0.3, 0.4]), horizon=1
    )
    assert np.isnan(stat)
    assert np.isnan(pval)


def test_baselines_schema_loads_expected_payload(baselines_payload):
    from app.baselines import load_baseline_comparison

    comparison = load_baseline_comparison(path=baselines_payload)
    assert comparison is not None
    assert comparison.beats_naive_significantly is True
    assert comparison.diebold_mariano_pvalue < 0.05
    assert comparison.model_rmse < comparison.naive_rmse


def test_baselines_returns_none_when_file_missing(tmp_path):
    from app.baselines import load_baseline_comparison

    missing = tmp_path / "does_not_exist.json"
    assert load_baseline_comparison(path=missing) is None
