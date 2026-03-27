"""API endpoint tests."""

from datetime import datetime

import pytest
from httpx import AsyncClient

from app.models.schemas import TransactionRequest


@pytest.mark.asyncio
async def test_predict_success(
    async_client: AsyncClient,
    sample_transaction: TransactionRequest,
):
    """Test successful prediction endpoint."""
    response = await async_client.post(
        "/api/v1/predict",
        json=sample_transaction.model_dump(),
        headers={"X-API-Key": "test-key-dev"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == sample_transaction.transaction_id
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert "model_version" in data
    assert "latency_ms" in data
    assert isinstance(data["top_risk_factors"], list)


@pytest.mark.asyncio
async def test_predict_missing_api_key(
    async_client: AsyncClient,
    sample_transaction: TransactionRequest,
):
    """Test prediction without API key."""
    response = await async_client.post(
        "/api/v1/predict",
        json=sample_transaction.model_dump(),
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_predict_invalid_api_key(
    async_client: AsyncClient,
    sample_transaction: TransactionRequest,
):
    """Test prediction with invalid API key."""
    response = await async_client.post(
        "/api/v1/predict",
        json=sample_transaction.model_dump(),
        headers={"X-API-Key": "invalid-key"},
    )

    assert response.status_code == 403
    assert "Invalid API key" in response.json()["detail"]


@pytest.mark.asyncio
async def test_batch_predict_success(
    async_client: AsyncClient,
    sample_transactions: list[TransactionRequest],
):
    """Test successful batch prediction endpoint."""
    request_data = {
        "transactions": [t.model_dump() for t in sample_transactions],
    }

    response = await async_client.post(
        "/api/v1/batch_predict",
        json=request_data,
        headers={"X-API-Key": "test-key-dev"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total_processed"] == len(sample_transactions)
    assert len(data["predictions"]) == len(sample_transactions)
    assert "processing_time_ms" in data

    # Check each prediction
    for pred, txn in zip(data["predictions"], sample_transactions):
        assert pred["transaction_id"] == txn.transaction_id
        assert 0.0 <= pred["fraud_probability"] <= 1.0
        assert pred["risk_tier"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@pytest.mark.asyncio
async def test_batch_predict_too_many_transactions(
    async_client: AsyncClient,
):
    """Test batch prediction with too many transactions."""
    # Create more than 1000 transactions
    transactions = [
        {
            "transaction_id": f"txn_{i}",
            "user_id": f"user_{i}",
            "merchant_id": f"merchant_{i}",
            "amount": 100.0,
            "timestamp": datetime.utcnow().isoformat(),
        }
        for i in range(1001)
    ]

    response = await async_client.post(
        "/api/v1/batch_predict",
        json={"transactions": transactions},
        headers={"X-API-Key": "test-key-dev"},
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_model_info(async_client: AsyncClient):
    """Test model info endpoint."""
    response = await async_client.get(
        "/api/v1/model_info",
        headers={"X-API-Key": "test-key-dev"},
    )

    assert response.status_code == 200
    data = response.json()

    assert "model_version" in data
    assert "model_type" in data
    assert "features" in data
    assert isinstance(data["features"], list)
    assert "metrics" in data
    assert "last_updated" in data


@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    """Test health check endpoint."""
    response = await async_client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert data["status"] in ["healthy", "unhealthy"]
    assert "model_loaded" in data
    assert "redis_connected" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient):
    """Test root endpoint."""
    response = await async_client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert "name" in data
    assert "version" in data
    assert "status" in data
    assert "docs" in data


@pytest.mark.asyncio
async def test_predict_invalid_schema(async_client: AsyncClient):
    """Test prediction with invalid request schema."""
    invalid_data = {
        "transaction_id": "txn_001",
        # Missing required fields
    }

    response = await async_client.post(
        "/api/v1/predict",
        json=invalid_data,
        headers={"X-API-Key": "test-key-dev"},
    )

    assert response.status_code == 422
