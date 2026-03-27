"""Explanation routes."""

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(prefix="/api/v1", tags=["explain"])


class ExplanationResponse(BaseModel):
    prediction_id: str
    explanation_type: str
    feature_contributions: dict[str, float]
    narrative: str


@router.post("/explain/{id}", response_model=ExplanationResponse)
def explain_prediction(id: str) -> ExplanationResponse:
    contributions = {
        "merchant_risk": 0.41,
        "high_amount": 0.27,
        "velocity_1h": 0.19,
        "distance_from_home": 0.13,
    }
    return ExplanationResponse(
        prediction_id=id,
        explanation_type="hybrid_shap_lime_placeholder",
        feature_contributions=contributions,
        narrative="Merchant profile risk and transaction amount dominated the fraud score.",
    )
