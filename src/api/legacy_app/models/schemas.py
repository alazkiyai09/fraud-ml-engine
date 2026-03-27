"""Pydantic schemas for request/response validation."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, confloat, field_validator


class TransactionRequest(BaseModel):
    """Request schema for single transaction prediction."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: datetime = Field(..., description="Transaction timestamp")

    # Optional additional fields that might be used by the pipeline
    payment_method: str | None = Field(None, description="Payment method (e.g., 'credit_card', 'debit_card')")
    card_present: bool | None = Field(None, description="Whether card was present")
    location: str | None = Field(None, description="Transaction location")

    model_config = {"json_schema_extra": {"examples": [
        {
            "transaction_id": "txn_12345",
            "user_id": "user_67890",
            "merchant_id": "merchant_001",
            "amount": 150.00,
            "timestamp": "2024-01-27T10:30:00Z",
            "payment_method": "credit_card",
            "card_present": False,
            "location": "online"
        }
    ]}}


class BatchPredictionRequest(BaseModel):
    """Request schema for batch transaction prediction."""

    transactions: list[TransactionRequest] = Field(
        ..., description="List of transactions to score", min_length=1, max_length=1000
    )
    max_batch_size: int = Field(
        default=1000,
        description="Maximum batch size (default: 1000, max: 1000)",
        le=1000,
    )

    @field_validator("transactions")
    @classmethod
    def validate_transactions_length(
        cls, transactions: list[TransactionRequest]
    ) -> list[TransactionRequest]:
        """Validate that transactions list does not exceed max_batch_size."""
        if len(transactions) > cls.model_fields["transactions"].annotation.max_length:
            raise ValueError(
                f"Number of transactions ({len(transactions)}) "
                f"exceeds maximum allowed (1000)"
            )
        return transactions

    model_config = {"json_schema_extra": {"examples": [
        {
            "transactions": [
                {
                    "transaction_id": "txn_001",
                    "user_id": "user_001",
                    "merchant_id": "merchant_001",
                    "amount": 100.00,
                    "timestamp": "2024-01-27T10:30:00Z",
                },
                {
                    "transaction_id": "txn_002",
                    "user_id": "user_002",
                    "merchant_id": "merchant_002",
                    "amount": 250.50,
                    "timestamp": "2024-01-27T10:31:00Z",
                },
            ]
        }
    ]}}


class PredictionResponse(BaseModel):
    """Response schema for single transaction prediction."""

    transaction_id: str = Field(..., description="Transaction identifier")
    fraud_probability: confloat(ge=0.0, le=1.0) = Field(
        ..., description="Probability of fraud (0-1)"
    )
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        ..., description="Risk tier classification"
    )
    top_risk_factors: list[str] = Field(
        default_factory=list, description="Top risk factors"
    )
    model_version: str = Field(..., description="Model version")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")

    model_config = {"json_schema_extra": {"examples": [
        {
            "transaction_id": "txn_12345",
            "fraud_probability": 0.85,
            "risk_tier": "HIGH",
            "top_risk_factors": [
                "velocity_count_1h",
                "deviation_amount_zscore",
                "merchant_fraud_rate"
            ],
            "model_version": "1.0.0",
            "latency_ms": 45.2
        }
    ]}}


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""

    predictions: list[PredictionResponse] = Field(..., description="Prediction results")
    total_processed: int = Field(..., description="Total number of transactions processed")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")

    model_config = {"json_schema_extra": {"examples": [
        {
            "predictions": [
                {
                    "transaction_id": "txn_001",
                    "fraud_probability": 0.15,
                    "risk_tier": "LOW",
                    "top_risk_factors": [],
                    "model_version": "1.0.0",
                    "latency_ms": 35.5
                },
                {
                    "transaction_id": "txn_002",
                    "fraud_probability": 0.92,
                    "risk_tier": "CRITICAL",
                    "top_risk_factors": ["velocity_count_1h", "deviation_amount_zscore"],
                    "model_version": "1.0.0",
                    "latency_ms": 38.2
                }
            ],
            "total_processed": 2,
            "processing_time_ms": 73.7
        }
    ]}}


class ModelInfoResponse(BaseModel):
    """Response schema for model metadata."""

    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type (e.g., 'XGBoostClassifier')")
    features: list[str] = Field(..., description="List of feature names")
    metrics: dict[str, float] = Field(..., description="Model performance metrics")
    last_updated: datetime = Field(..., description="Last model update timestamp")

    model_config = {"json_schema_extra": {"examples": [
        {
            "model_version": "1.0.0",
            "model_type": "XGBoostClassifier",
            "features": [
                "amount",
                "hour",
                "day_of_week",
                "velocity_count_1h",
                "deviation_amount_zscore",
                "merchant_fraud_rate"
            ],
            "metrics": {
                "precision": 0.85,
                "recall": 0.78,
                "f1_score": 0.81,
                "auc_roc": 0.92
            },
            "last_updated": "2024-01-27T00:00:00Z"
        }
    ]}}


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: Literal["healthy", "unhealthy"] = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    redis_connected: bool = Field(..., description="Redis connection status")
    timestamp: datetime = Field(..., description="Check timestamp")

    model_config = {"json_schema_extra": {"examples": [
        {
            "status": "healthy",
            "model_loaded": True,
            "redis_connected": True,
            "timestamp": "2024-01-27T10:30:00Z"
        }
    ]}}


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")

    model_config = {"json_schema_extra": {"examples": [
        {
            "error": "Invalid API key",
            "detail": "The provided API key is not valid",
            "status_code": 403
        }
    ]}}
