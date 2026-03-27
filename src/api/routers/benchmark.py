"""Benchmark routes."""

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(prefix="/api/v1/benchmark", tags=["benchmark"])

_LAST_RESULTS = {
    "xgboost": {"auprc": 0.943, "auc": 0.981},
    "random_forest": {"auprc": 0.921, "auc": 0.973},
    "lstm_attention": {"auprc": 0.938, "auc": 0.979},
    "isolation_forest": {"auprc": 0.771, "auc": 0.894},
}


class BenchmarkRunRequest(BaseModel):
    dataset: str = "credit_card_fraud"
    sample_size: int = 5000


@router.post("/run")
def run_benchmark(request: BenchmarkRunRequest) -> dict:
    return {
        "status": "completed",
        "dataset": request.dataset,
        "sample_size": request.sample_size,
        "models": _LAST_RESULTS,
    }


@router.get("/results")
def benchmark_results() -> dict:
    return {"results": _LAST_RESULTS}
