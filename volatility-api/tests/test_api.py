"""End-to-end tests for the FastAPI layer.

Every test in this module runs against the FastAPI TestClient with the
real Keras model replaced by a fake (see conftest.py). This keeps the
suite fast and independent of heavyweight dependencies.
"""
from __future__ import annotations


def test_health_returns_healthy_with_no_predictions(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_type"] == "MLP"
    assert body["predictions_total"] == 0
    assert body["predictions_last_window"] == 0
    assert body["drift_pct_out_of_distribution"] == 0.0
    assert body["degraded_reasons"] == []


def test_info_returns_model_metadata(client):
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] == "MLP"
    assert body["lag"] == 14
    assert body["forecast_horizon"] == 7
    assert "training_stats_available" in body


def test_predict_happy_path(client, valid_lags):
    r = client.post("/predict", json={"lags": valid_lags})
    assert r.status_code == 200
    body = r.json()
    assert len(body["volatility"]) == 7
    assert len(body["horizons"]) == 7
    assert body["horizons"][0]["horizon_day"] == 1
    assert body["horizons"][-1]["horizon_day"] == 7
    assert body["model_type"] == "MLP"
    assert body["lag"] == 14
    assert body["drift"]["input_out_of_distribution"] is False
    assert body["drift"]["fraction_out_of_distribution"] == 0.0
    assert "prediction_timestamp" in body


def test_predict_rejects_wrong_length(client):
    r = client.post("/predict", json={"lags": [0.05] * 10})  # lag=14 expected
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "invalid_input"
    assert "Expected 14" in body["detail"]


def test_predict_rejects_negative_values(client):
    lags = [0.05] * 13 + [-0.1]
    r = client.post("/predict", json={"lags": lags})
    # Pydantic validation rejects before the handler runs.
    assert r.status_code == 422


def test_predict_rejects_nan(client):
    r = client.post("/predict", json={"lags": [0.05] * 13 + ["nan"]})
    # FastAPI rejects the literal "nan" string as not a float.
    assert r.status_code == 422


def test_predict_rejects_missing_field(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_flags_drift_on_ood_input(client, ood_lags):
    r = client.post("/predict", json={"lags": ood_lags})
    assert r.status_code == 200
    body = r.json()
    assert body["drift"]["input_out_of_distribution"] is True
    assert body["drift"]["fraction_out_of_distribution"] == 1.0


def test_health_turns_degraded_after_repeated_ood(client, ood_lags):
    # Send enough OOD requests to exceed the 10% threshold.
    for _ in range(5):
        r = client.post("/predict", json={"lags": ood_lags})
        assert r.status_code == 200

    r = client.get("/health")
    body = r.json()
    assert body["predictions_total"] == 5
    assert body["drift_pct_out_of_distribution"] == 100.0
    assert body["status"] == "degraded"
    assert any("input_drift" in reason for reason in body["degraded_reasons"])


def test_health_stays_healthy_when_inputs_are_mostly_in_distribution(
    client, valid_lags, ood_lags
):
    # 9 in-distribution + 1 OOD = 10% OOD — under the (>10%) threshold.
    for _ in range(9):
        client.post("/predict", json={"lags": valid_lags})
    client.post("/predict", json={"lags": ood_lags})

    r = client.get("/health")
    body = r.json()
    assert body["predictions_total"] == 10
    assert body["drift_pct_out_of_distribution"] == 10.0
    assert body["status"] == "healthy"  # strictly > threshold to degrade


def test_predict_compare_includes_naive_baseline(client, valid_lags):
    r = client.post("/predict/compare", json={"lags": valid_lags})
    assert r.status_code == 200
    body = r.json()
    assert len(body["volatility"]) == 7
    assert len(body["naive_baseline"]) == 7
    # Naive baseline is persistence of the last observation.
    assert all(v == valid_lags[-1] for v in body["naive_baseline"])


def test_predictions_counter_increments(client, valid_lags):
    assert client.get("/health").json()["predictions_total"] == 0
    client.post("/predict", json={"lags": valid_lags})
    client.post("/predict", json={"lags": valid_lags})
    assert client.get("/health").json()["predictions_total"] == 2


def test_invalid_input_does_not_bump_telemetry_counter(client):
    # Bad inputs should NOT inflate the prediction counter.
    client.post("/predict", json={"lags": [0.05] * 10})
    assert client.get("/health").json()["predictions_total"] == 0
