from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():
    response = client.post(
        "/api/v1/predict",
        json={
            "transaction_id": "tx-1",
            "amount": 1200,
            "merchant_risk": 0.8,
            "velocity_1h": 9,
            "distance_from_home": 110,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "tx-1"
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_tier"] in {"low", "medium", "high", "critical"}


def test_model_info():
    response = client.get("/api/v1/model_info")
    assert response.status_code == 200
    payload = response.json()
    assert "classical" in payload
    assert "lstm" in payload
    assert "anomaly" in payload
