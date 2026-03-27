"""Pytest fixtures and configuration."""

import asyncio
from datetime import datetime
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, Mock, patch

import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.models.predictor import FraudPredictor
from app.models.schemas import TransactionRequest
from app.services.cache import RedisCache


@pytest.fixture
def sample_transaction() -> TransactionRequest:
    """Sample transaction for testing."""
    return TransactionRequest(
        transaction_id="txn_test_001",
        user_id="user_test_001",
        merchant_id="merchant_test_001",
        amount=150.00,
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def sample_transactions() -> list[TransactionRequest]:
    """Sample transactions for batch testing."""
    return [
        TransactionRequest(
            transaction_id=f"txn_test_{i:03d}",
            user_id=f"user_test_{i:03d}",
            merchant_id=f"merchant_test_{i:03d}",
            amount=100.0 + i * 10,
            timestamp=datetime.utcnow(),
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def mock_model() -> Mock:
    """Mock XGBoost model."""
    model = Mock()
    model.predict_proba = Mock(return_value=np.array([[0.2, 0.8], [0.9, 0.1]]))
    model.predict = Mock(return_value=np.array([1, 0]))
    model.__class__.__name__ = "XGBClassifier"
    return model


@pytest.fixture
def mock_pipeline() -> Mock:
    """Mock feature pipeline."""
    pipeline = Mock()

    # Mock transform to return feature array
    pipeline.transform = Mock(
        return_value=np.array([
            [0.5, 0.3, 0.8, 0.1, 0.2, 0.9],
            [0.1, 0.9, 0.2, 0.8, 0.7, 0.3],
        ])
    )

    # Mock feature names
    pipeline.get_feature_names_out = Mock(
        return_value=np.array([
            "amount",
            "hour",
            "velocity_count_1h",
            "deviation_amount_zscore",
            "merchant_fraud_rate",
            "user_transaction_count",
        ])
    )

    return pipeline


@pytest.fixture
def mock_predictor(mock_model: Mock, mock_pipeline: Mock) -> FraudPredictor:
    """Mock fraud predictor."""
    predictor = Mock(spec=FraudPredictor)

    # Mock is_model_loaded
    predictor.is_model_loaded = Mock(return_value=True)

    # Mock predict_single
    predictor.predict_single = Mock(
        return_value={
            "fraud_probability": 0.85,
            "risk_tier": "HIGH",
            "risk_factors": ["velocity_count_1h", "deviation_amount_zscore"],
            "latency_ms": 45.0,
        }
    )

    # Mock predict_batch
    def mock_predict_batch(transactions):
        return [
            {
                "fraud_probability": 0.15 + i * 0.1,
                "risk_tier": "LOW" if i < 3 else "MEDIUM",
                "risk_factors": ["amount"] if i % 2 == 0 else ["hour"],
                "latency_ms": 40.0 + i,
            }
            for i in range(len(transactions))
        ]

    predictor.predict_batch = Mock(side_effect=mock_predict_batch)

    # Mock get_model_info
    predictor.get_model_info = Mock(
        return_value={
            "model_version": "1.0.0",
            "model_type": "XGBClassifier",
            "features": [
                "amount",
                "hour",
                "velocity_count_1h",
                "deviation_amount_zscore",
                "merchant_fraud_rate",
            ],
            "metrics": {
                "precision": 0.85,
                "recall": 0.78,
                "f1_score": 0.81,
                "auc_roc": 0.92,
            },
            "last_updated": datetime.utcnow().isoformat(),
        }
    )

    return predictor


@pytest.fixture
def mock_redis() -> Mock:
    """Mock Redis client."""
    redis = Mock()
    redis.get = Mock(return_value=None)
    redis.set = Mock(return_value=True)
    redis.delete = Mock(return_value=True)
    redis.ping = Mock(return_value=True)
    redis.from_url = Mock(return_value=redis)
    return redis


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
