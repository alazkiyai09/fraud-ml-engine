"""Shared loading helpers for fraud datasets."""

from __future__ import annotations

from typing import Iterable


def normalize_records(records: Iterable[dict]) -> list[dict]:
    """Return plain dictionaries with stable fraud fields."""
    normalized = []
    for record in records:
        normalized.append(
            {
                "transaction_id": record.get("transaction_id"),
                "amount": float(record.get("amount", 0.0)),
                "merchant_risk": float(record.get("merchant_risk", 0.0)),
                "velocity_1h": float(record.get("velocity_1h", 0.0)),
                "distance_from_home": float(record.get("distance_from_home", 0.0)),
            }
        )
    return normalized
