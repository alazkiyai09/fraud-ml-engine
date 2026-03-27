"""Redis caching service for prediction responses."""

import hashlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis

from src.api.legacy_app.core.config import settings


logger = logging.getLogger(__name__)


class RedisCacheError(Exception):
    """Exception raised for Redis cache errors."""

    pass


class RedisCache:
    """
    Async Redis cache service.

    Provides caching for prediction responses with configurable TTL.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
        ttl: int | None = None,
    ) -> None:
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            ttl: Default time-to-live for cached values (seconds)
        """
        self.host = host or settings.redis_host
        self.port = port or settings.redis_port
        self.db = db or settings.redis_db
        self.ttl = ttl or settings.cache_ttl_seconds

        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            self._client = await aioredis.from_url(
                f"redis://{self.host}:{self.port}/{self.db}",
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info(f"Connected to Redis at {self.host}:{self.port}/{self.db}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise RedisCacheError(f"Redis connection failed: {str(e)}") from e

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            logger.info("Redis connection closed")

    async def get(self, key: str) -> str | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value as string, or None if not found

        Raises:
            RedisCacheError: If Redis operation fails
        """
        if not self._client:
            await self.connect()

        try:
            value = await self._client.get(key)
            if value:
                logger.debug(f"Cache hit for key: {key}")
            else:
                logger.debug(f"Cache miss for key: {key}")
            return value
        except Exception as e:
            logger.error(f"Redis GET failed: {e}")
            raise RedisCacheError(f"Cache get failed: {str(e)}") from e

    async def set(
        self, key: str, value: str, ttl: int | None = None
    ) -> bool:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized if dict)
            ttl: Time-to-live in seconds (uses default if None)

        Returns:
            True if successful, False otherwise

        Raises:
            RedisCacheError: If Redis operation fails
        """
        if not self._client:
            await self.connect()

        try:
            # Serialize dict to JSON
            if isinstance(value, dict):
                value = json.dumps(value)

            # Use default TTL if not specified
            cache_ttl = ttl if ttl is not None else self.ttl

            # Set with expiration
            await self._client.setex(key, cache_ttl, value)
            logger.debug(f"Cached key: {key} (TTL: {cache_ttl}s)")
            return True

        except Exception as e:
            logger.error(f"Redis SET failed: {e}")
            raise RedisCacheError(f"Cache set failed: {str(e)}") from e

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False otherwise

        Raises:
            RedisCacheError: If Redis operation fails
        """
        if not self._client:
            await self.connect()

        try:
            result = await self._client.delete(key)
            deleted = result > 0
            if deleted:
                logger.debug(f"Deleted cache key: {key}")
            return deleted

        except Exception as e:
            logger.error(f"Redis DELETE failed: {e}")
            raise RedisCacheError(f"Cache delete failed: {str(e)}") from e

    async def is_connected(self) -> bool:
        """
        Check Redis connection status.

        Returns:
            True if connected, False otherwise
        """
        if not self._client:
            return False

        try:
            await self._client.ping()
            return True
        except Exception:
            return False

    def generate_prediction_cache_key(
        self, transaction_id: str, features: dict[str, Any]
    ) -> str:
        """
        Generate cache key for prediction.

        Args:
            transaction_id: Transaction identifier
            features: Feature dictionary

        Returns:
            Cache key string
        """
        # Compute hash of features
        features_hash = self._compute_features_hash(features)

        # Generate key: prediction:{transaction_id}:{features_hash}
        return f"prediction:{transaction_id}:{features_hash}"

    def _compute_features_hash(self, features: dict[str, Any]) -> str:
        """
        Compute hash of feature dict.

        Args:
            features: Feature dictionary

        Returns:
            Hexadecimal hash string (first 16 chars)
        """
        # Sort keys for deterministic ordering
        sorted_features = dict(sorted(features.items()))

        # Convert to JSON string
        feature_json = json.dumps(sorted_features, sort_keys=True, default=str)

        # Compute SHA256 hash
        hash_object = hashlib.sha256(feature_json.encode())
        return hash_object.hexdigest()[:16]

    async def get_prediction(
        self, transaction_id: str, features: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Get cached prediction for transaction.

        Args:
            transaction_id: Transaction identifier
            features: Feature dictionary

        Returns:
            Cached prediction dict, or None if not found
        """
        key = self.generate_prediction_cache_key(transaction_id, features)
        value = await self.get(key)

        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in cache for key: {key}")
                return None

        return None

    async def set_prediction(
        self,
        transaction_id: str,
        features: dict[str, Any],
        prediction: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """
        Cache prediction for transaction.

        Args:
            transaction_id: Transaction identifier
            features: Feature dictionary
            prediction: Prediction response to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful, False otherwise
        """
        key = self.generate_prediction_cache_key(transaction_id, features)
        return await self.set(key, prediction, ttl)
