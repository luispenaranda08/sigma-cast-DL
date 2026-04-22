"""Tests for the telemetry/drift module in isolation."""
from __future__ import annotations

from app.monitoring import TelemetryStore, evaluate_status


def test_snapshot_on_empty_buffer():
    store = TelemetryStore(buffer_size=10)
    snap = store.snapshot()
    assert snap["predictions_total"] == 0
    assert snap["predictions_last_window"] == 0
    assert snap["drift_pct_out_of_distribution"] == 0.0
    assert snap["uptime_seconds"] >= 0.0


def test_record_accumulates_and_reports_ood_pct():
    store = TelemetryStore(buffer_size=10)
    # 3 out of 5 OOD.
    for ood in [True, True, False, False, True]:
        store.record(any_ood=ood, fraction_ood=1.0 if ood else 0.0, lag_count=14)
    snap = store.snapshot()
    assert snap["predictions_total"] == 5
    assert snap["predictions_last_window"] == 5
    assert snap["drift_pct_out_of_distribution"] == 60.0


def test_ring_buffer_forgets_old_events():
    store = TelemetryStore(buffer_size=3)
    # Record 5 events — only last 3 remain in the window.
    for i in range(5):
        store.record(any_ood=(i >= 3), fraction_ood=1.0, lag_count=14)
    snap = store.snapshot()
    # Total count is cumulative; window-based OOD% uses only the buffer.
    assert snap["predictions_total"] == 5
    assert snap["predictions_last_window"] == 3
    # Buffer contents: indices 2,3,4 → ood = [False, True, True] → 66.67%
    assert snap["drift_pct_out_of_distribution"] == 66.667


def test_reset_clears_state():
    store = TelemetryStore(buffer_size=10)
    store.record(any_ood=True, fraction_ood=1.0, lag_count=14)
    store.reset()
    snap = store.snapshot()
    assert snap["predictions_total"] == 0
    assert snap["predictions_last_window"] == 0


def test_evaluate_status_healthy():
    status, reasons = evaluate_status(
        drift_pct=5.0, threshold_pct=10.0, model_loaded=True
    )
    assert status == "healthy"
    assert reasons == []


def test_evaluate_status_degraded_on_drift():
    status, reasons = evaluate_status(
        drift_pct=15.0, threshold_pct=10.0, model_loaded=True
    )
    assert status == "degraded"
    assert any("input_drift" in r for r in reasons)


def test_evaluate_status_unhealthy_when_model_missing():
    status, reasons = evaluate_status(
        drift_pct=0.0, threshold_pct=10.0, model_loaded=False
    )
    assert status == "unhealthy"
    assert "model_not_loaded" in reasons


def test_threshold_is_strict_greater_than():
    """Drift exactly at threshold is NOT degraded — only strictly above is."""
    status, _ = evaluate_status(
        drift_pct=10.0, threshold_pct=10.0, model_loaded=True
    )
    assert status == "healthy"
