"""Model metadata routes."""

from fastapi import APIRouter

from src.models.anomaly.ensemble import describe_anomaly_stack
from src.models.classical.benchmark import describe_classical_stack
from src.models.lstm.model import describe_lstm_stack


router = APIRouter(prefix="/api/v1", tags=["model-info"])


@router.get("/model_info")
def model_info() -> dict:
    return {
        "classical": describe_classical_stack(),
        "lstm": describe_lstm_stack(),
        "anomaly": describe_anomaly_stack(),
        "risk_tiers": ["low", "medium", "high", "critical"],
    }
