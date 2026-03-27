"""Shared scoring helpers for the unified API."""

from __future__ import annotations


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def classify_risk_tier(probability: float) -> str:
    if probability >= 0.85:
        return "critical"
    if probability >= 0.65:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"
