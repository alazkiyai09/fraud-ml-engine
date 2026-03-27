"""Authentication tests."""

import pytest
from fastapi import HTTPException

from app.core.security import verify_api_key, validate_api_key, get_security_headers


def test_validate_api_key_success():
    """Test API key validation with valid key."""
    from app.core.config import settings

    # Use first key from settings
    valid_key = settings.api_keys[0]
    assert validate_api_key(valid_key) is True


def test_validate_api_key_invalid():
    """Test API key validation with invalid key."""
    assert validate_api_key("invalid-key") is False


def test_get_security_headers():
    """Test security headers generation."""
    headers = get_security_headers()

    assert "X-Content-Type-Options" in headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "X-Frame-Options" in headers
    assert headers["X-Frame-Options"] == "DENY"
    assert "X-XSS-Protection" in headers
    assert "Strict-Transport-Security" in headers


@pytest.mark.asyncio
async def test_verify_api_key_missing():
    """Test API key verification with missing key."""
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(None)

    assert exc_info.value.status_code == 401
    assert "API key is missing" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_api_key_invalid():
    """Test API key verification with invalid key."""
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key("invalid-key")

    assert exc_info.value.status_code == 403
    assert "Invalid API key" in exc_info.value.detail
