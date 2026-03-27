"""API security and authentication utilities."""

from typing import Literal

from fastapi import Header, HTTPException, status
from pydantic import ValidationError

from src.api.legacy_app.core.config import settings


class SecurityError(Exception):
    """Custom security error."""

    def __init__(self, message: str, status_code: int = 401):
        """
        Initialize security error.

        Args:
            message: Error message
            status_code: HTTP status code
        """
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


async def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> str:
    """
    Verify API key from request headers.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The validated API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing. Please provide X-API-Key header.",
        )

    if x_api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key. Access denied.",
        )

    return x_api_key


def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key.

    Args:
        api_key: API key to validate

    Returns:
        True if valid, False otherwise
    """
    return api_key in settings.api_keys


def get_security_headers() -> dict[str, str]:
    """
    Get security headers for responses.

    Returns:
        Dictionary of security headers
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    }
