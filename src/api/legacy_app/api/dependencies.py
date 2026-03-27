"""FastAPI dependency injection for auth, rate limiting, and services."""

import logging
from typing import AsyncGenerator

from fastapi import Depends, HTTPException, status

from src.api.legacy_app.core.config import settings
from src.api.legacy_app.core.security import verify_api_key
from src.api.legacy_app.models.predictor import FraudPredictor
from src.api.legacy_app.services.cache import RedisCache
from src.api.legacy_app.services.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)


# Global instances
_predictor: FraudPredictor | None = None
_cache: RedisCache | None = None
_rate_limiter: RateLimiter | None = None


def get_predictor() -> FraudPredictor:
    """
    Get or create global predictor instance.

    Returns:
        FraudPredictor instance
    """
    global _predictor

    if _predictor is None:
        from src.api.legacy_app.models.predictor import FraudPredictor

        _predictor = FraudPredictor(
            model_path=settings.model_path,
            pipeline_path=settings.pipeline_path,
            model_version=settings.app_version,
        )
        logger.info("FraudPredictor initialized")

    return _predictor


def get_redis_cache() -> RedisCache:
    """
    Get or create global Redis cache instance.

    Returns:
        RedisCache instance
    """
    global _cache

    if _cache is None:
        _cache = RedisCache()
        logger.info("RedisCache initialized")

    return _cache


def get_rate_limiter() -> RateLimiter:
    """
    Get or create global rate limiter instance.

    Returns:
        RateLimiter instance
    """
    global _rate_limiter

    if _rate_limiter is None:
        # Get cache instance (will be created by get_redis_cache)
        # Note: This will be initialized on first use
        _rate_limiter = RateLimiter(
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        # We'll set the cache when first dependency is resolved
        logger.info("Rate limiter initialized")

    return _rate_limiter


async def check_rate_limit(
    api_key: str = Depends(verify_api_key),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    cache: RedisCache = Depends(get_redis_cache),
) -> bool:
    """
    Check if request is within rate limit.

    Args:
        api_key: Validated API key from verify_api_key
        rate_limiter: Rate limiter instance
        cache: Redis cache instance

    Returns:
        True if request is allowed

    Raises:
        HTTPException: If rate limit exceeded (429 Too Many Requests)
    """
    # Set cache on rate limiter if not already set
    if rate_limiter.cache is None:
        rate_limiter.cache = cache

    # Check if allowed
    allowed = await rate_limiter.is_allowed(api_key)

    if not allowed:
        # Get retry-after time
        reset_time = await rate_limiter.get_reset_time(api_key)
        retry_after = max(1, reset_time - int(__import__("time").time()))

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(settings.rate_limit_requests),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_time),
            },
        )

    # Add rate limit headers to response
    remaining = await rate_limiter.get_remaining_count(api_key)
    reset_time = await rate_limiter.get_reset_time(api_key)

    # Store in request state for middleware to add to response
    # Note: In a real app, you'd use middleware for this
    logger.debug(
        f"Rate limit check passed for {api_key}: {remaining} requests remaining"
    )

    return True


async def get_predictor_with_check(
    predictor: FraudPredictor = Depends(get_predictor),
) -> FraudPredictor:
    """
    Get predictor with model loaded check.

    Args:
        predictor: Predictor instance from dependency

    Returns:
        FraudPredictor instance

    Raises:
        HTTPException: If model is not loaded (503 Service Unavailable)
    """
    if not predictor.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later.",
        )

    return predictor


# RateLimiter monkey-patch to set cache
def _set_cache_on_rate_limiter(cache: RedisCache) -> RateLimiter:
    """Set cache on rate limiter instance."""
    rate_limiter = get_rate_limiter()
    rate_limiter.cache = cache
    return rate_limiter
