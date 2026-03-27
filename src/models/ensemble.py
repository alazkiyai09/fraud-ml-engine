"""Lightweight model-ensemble utilities for the unified fraud API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EnsemblePrediction:
    score: float
    label: int
    contributors: dict[str, float]


def blend_scores(scores: dict[str, float], weights: dict[str, float] | None = None) -> EnsemblePrediction:
    """Blend model scores into a single risk estimate using normalized weights."""
    if not scores:
        return EnsemblePrediction(score=0.0, label=0, contributors={})

    if weights is None:
        weights = {name: 1.0 for name in scores}

    denom = sum(max(weights.get(name, 0.0), 0.0) for name in scores) or 1.0
    contributors: dict[str, float] = {}
    blended = 0.0
    for name, value in scores.items():
        w = max(weights.get(name, 0.0), 0.0) / denom
        contributors[name] = round(value * w, 6)
        blended += contributors[name]

    return EnsemblePrediction(score=round(blended, 6), label=int(blended >= 0.5), contributors=contributors)
