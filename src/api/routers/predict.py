"""Prediction routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.core.metrics import clamp_probability, classify_risk_tier


router = APIRouter(prefix="/api/v1", tags=["predict"])


class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = None
    amount: float = Field(default=0.0, ge=0)
    merchant_risk: float = Field(default=0.0, ge=0, le=1)
    velocity_1h: float = Field(default=0.0, ge=0)
    distance_from_home: float = Field(default=0.0, ge=0)
    model: str = "ensemble"


class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    model: str
    fraud_probability: float
    prediction: str
    risk_tier: str
    top_factors: list[str]


class BatchPredictRequest(BaseModel):
    transactions: list[TransactionRequest]


class BatchPredictResponse(BaseModel):
    results: list[PredictionResponse]


def _score_transaction(request: TransactionRequest) -> tuple[float, list[str]]:
    weighted = (
        min(request.amount / 1500.0, 1.0) * 0.35
        + request.merchant_risk * 0.3
        + min(request.velocity_1h / 20.0, 1.0) * 0.2
        + min(request.distance_from_home / 250.0, 1.0) * 0.15
    )
    factors = []
    if request.amount >= 500:
        factors.append("high_amount")
    if request.merchant_risk >= 0.6:
        factors.append("merchant_risk")
    if request.velocity_1h >= 6:
        factors.append("high_velocity")
    if request.distance_from_home >= 80:
        factors.append("distance_from_home")
    return clamp_probability(weighted), factors or ["baseline_pattern"]


def _build_response(request: TransactionRequest) -> PredictionResponse:
    probability, factors = _score_transaction(request)
    return PredictionResponse(
        transaction_id=request.transaction_id,
        model=request.model,
        fraud_probability=probability,
        prediction="fraud" if probability >= 0.5 else "legitimate",
        risk_tier=classify_risk_tier(probability),
        top_factors=factors,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict_transaction(request: TransactionRequest) -> PredictionResponse:
    return _build_response(request)


@router.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(request: BatchPredictRequest) -> BatchPredictResponse:
    return BatchPredictResponse(results=[_build_response(item) for item in request.transactions])
