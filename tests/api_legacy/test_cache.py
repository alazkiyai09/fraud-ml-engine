"""Cache service tests."""

import pytest

from app.services.cache import RedisCache, RedisCacheError


@pytest.mark.asyncio
async def test_cache_generate_key():
    """Test cache key generation."""
    cache = RedisCache()

    features = {
        "transaction_id": "txn_001",
        "user_id": "user_001",
        "amount": 100.0,
    }

    key = cache.generate_prediction_cache_key("txn_001", features)

    assert "prediction:txn_001:" in key
    assert len(key.split(":")[-1]) == 16  # Hash length


@pytest.mark.asyncio
async def test_cache_features_hash_deterministic():
    """Test that feature hash is deterministic."""
    cache = RedisCache()

    features = {
        "transaction_id": "txn_001",
        "user_id": "user_001",
        "amount": 100.0,
    }

    hash1 = cache._compute_features_hash(features)
    hash2 = cache._compute_features_hash(features)

    assert hash1 == hash2


@pytest.mark.asyncio
async def test_cache_features_hash_different():
    """Test that different features produce different hashes."""
    cache = RedisCache()

    features1 = {"amount": 100.0, "user_id": "user_001"}
    features2 = {"amount": 200.0, "user_id": "user_001"}

    hash1 = cache._compute_features_hash(features1)
    hash2 = cache._compute_features_hash(features2)

    assert hash1 != hash2


@pytest.mark.asyncio
async def test_cache_get_set_prediction(monkeypatch):
    """Test caching and retrieving predictions."""
    # Mock Redis client
    mock_redis = pytest.Mock()
    mock_redis.get = pytest.Mock(return_value=None)
    mock_redis.set = pytest.Mock(return_value=True)

    monkeypatch.setattr("redis.asyncio.from_url", pytest.Mock(return_value=mock_redis))

    cache = RedisCache()
    await cache.connect()

    features = {"amount": 100.0, "user_id": "user_001"}
    prediction = {"fraud_probability": 0.85, "risk_tier": "HIGH"}

    # Set prediction
    result = await cache.set_prediction("txn_001", features, prediction, ttl=60)
    assert result is True

    # Get prediction (will return None as we mocked get)
    result = await cache.get_prediction("txn_001", features)
    assert result is None


@pytest.mark.asyncio
async def test_cache_is_connected(monkeypatch):
    """Test Redis connection check."""
    mock_redis = pytest.Mock()
    mock_redis.ping = pytest.Mock(return_value=True)

    monkeypatch.setattr("redis.asyncio.from_url", pytest.Mock(return_value=mock_redis))

    cache = RedisCache()
    await cache.connect()

    is_connected = await cache.is_connected()
    assert is_connected is True
