"""Structured JSON logging configuration."""

import json
import logging
import sys
import time
import uuid
from datetime import datetime
from typing import Any

from pythonjsonlogger import jsonlogger


class RequestIdFilter(logging.Filter):
    """Add request_id to log records if not present."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id to log record."""
        if not hasattr(record, "request_id"):
            record.request_id = "N/A"
        return True


class JsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)

        # Add timestamp if not already present
        if not log_record.get("timestamp"):
            log_record["timestamp"] = datetime.utcnow().isoformat()

        # Add level
        log_record["level"] = record.levelname

        # Add request_id
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id

        # Add logger name
        log_record["logger"] = record.name


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structured JSON logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Create console handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            timestamp=True,
        )
    )

    # Add request ID filter
    handler.addFilter(RequestIdFilter())

    root_logger.addHandler(handler)

    # Configure uvicorn loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class RequestContext:
    """Context manager for request-scoped logging."""

    def __init__(self, request_id: str | None = None):
        """
        Initialize request context.

        Args:
            request_id: Optional request ID (generated if not provided)
        """
        self.request_id = request_id or str(uuid.uuid4())
        self.logger = logging.getLogger("api")

    def __enter__(self) -> "RequestContext":
        """Enter context and set request_id."""
        self.old_factory = logging.getLogRecordFactory()

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = self.old_factory(*args, **kwargs)
            record.request_id = self.request_id
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and restore old factory."""
        logging.setLogRecordFactory(self.old_factory)
