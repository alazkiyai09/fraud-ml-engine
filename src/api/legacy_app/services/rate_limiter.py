"""Token bucket rate limiting using Redis."""

import logging
import time
from typing import Any

from src.api.legacy_app.services.cache import RedisCache


logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter using Redis.

    Implements a token bucket algorithm for distributed rate limiting.
    Each API key has a bucket of tokens that refill over time.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests per window
            window_seconds: Time window in seconds
        """
        self.cache: RedisCache | None = None
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def _get_window_start(self) -> int:
        """
        Get the start of the current time window.

        Returns:
            Unix timestamp of window start
        """
        current_time = int(time.time())
        window_start = (current_time // self.window_seconds) * self.window_seconds
        return window_start

    def _generate_rate_limit_key(self, api_key: str, window_start: int) -> str:
        """
        Generate Redis key for rate limit counter.

        Args:
            api_key: API key identifier
            window_start: Window start timestamp

        Returns:
            Redis key string
        """
        return f"rate_limit:{api_key}:{window_start}"

    async def is_allowed(self, api_key: str) -> bool:
        """
        Check if request is allowed using token bucket algorithm.

        This method implements the token bucket algorithm:
        - Each API key has a bucket of tokens
        - Tokens refill at a constant rate (max_requests / window_seconds)
        - Each request consumes one token
        - Request is allowed if bucket has tokens

        Args:
            api_key: API key to check

        Returns:
            True if under rate limit, False otherwise

        Raises:
            RedisCacheError: If Redis operation fails
        """
        try:
            window_start = self._get_window_start()
            key = self._generate_rate_limit_key(api_key, window_start)

            # Get current count
            current_str = await self.cache.get(key)

            if current_str is None:
                # First request in window
                await self.cache.set(
                    key, "1", ttl=self.window_seconds
                )
                logger.debug(
                    f"Rate limit: First request for {api_key} in window"
                )
                return True

            current = int(current_str)

            if current >= self.max_requests:
                logger.warning(
                    f"Rate limit exceeded for {api_key}: "
                    f"{current}/{self.max_requests}"
                )
                return False

            # Increment counter
            await self.cache.set(
                key, str(current + 1), ttl=self.window_seconds
            )

            logger.debug(
                f"Rate limit: {current + 1}/{self.max_requests} for {api_key}"
            )
            return True

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if rate limiter fails
            return True

    async def get_remaining_count(self, api_key: str) -> int:
        """
        Get remaining request count for current window.

        Args:
            api_key: API key to check

        Returns:
            Number of remaining requests, or max_requests if not yet tracked

        Raises:
            RedisCacheError: If Redis operation fails
        """
        try:
            window_start = self._get_window_start()
            key = self._generate_rate_limit_key(api_key, window_start)

            current_str = await self.cache.get(key)

            if current_str is None:
                return self.max_requests

            current = int(current_str)
            remaining = max(0, self.max_requests - current)
            return remaining

        except Exception as e:
            logger.error(f"Failed to get remaining count: {e}")
            return self.max_requests

    async def get_reset_time(self, api_key: str) -> int:
        """
        Get Unix timestamp when rate limit window resets.

        Args:
            api_key: API key to check

        Returns:
            Unix timestamp of window reset

        Raises:
            RedisCacheError: If Redis operation fails
        """
        window_start = self._get_window_start()
        return window_start + self.window_seconds

    async def reset(self, api_key: str) -> bool:
        """
        Reset rate limit counter for API key (admin operation).

        Args:
            api_key: API key to reset

        Returns:
            True if successful, False otherwise

        Raises:
            RedisCacheError: If Redis operation fails
        """
        try:
            window_start = self._get_window_start()
            key = self._generate_rate_limit_key(api_key, window_start)

            await self.cache.delete(key)
            logger.info(f"Rate limit reset for {api_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")
            return False
