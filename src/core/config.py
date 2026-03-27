"""Lightweight runtime settings."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    api_title: str = os.getenv("FRAUD_API_TITLE", "fraud-ml-engine")
    api_version: str = os.getenv("FRAUD_API_VERSION", "0.1.0")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    default_threshold: float = float(os.getenv("DEFAULT_THRESHOLD", "0.65"))


settings = Settings()
