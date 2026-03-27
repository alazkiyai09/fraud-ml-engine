"""FastAPI route definitions for fraud scoring API."""

import logging
import time
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.legacy_app.api.dependencies import (
    check_rate_limit,
    get_predictor,
    get_predictor_with_check,
    get_redis_cache,
    _set_cache_on_rate_limiter,
)
from src.api.legacy_app.core.config import settings
from src.api.legacy_app.core.security import verify_api_key, get_security_headers
from src.api.legacy_app.models.predictor import FraudPredictor
from src.api.legacy_app.models.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    TransactionRequest,
    ModelInfoResponse,
    HealthResponse,
)
from src.api.legacy_app.services.cache import RedisCache
from src.api.legacy_app.utils.helpers import compute_feature_hash


logger = logging.getLogger(__name__)


# Create router
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Score a single transaction for fraud probability",
    responses={
        200: {"description": "Prediction successful"},
        401: {"description": "Invalid or missing API key"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Model not loaded"},
    },
)
async def predict(
    request: TransactionRequest,
    api_key: str = Depends(verify_api_key),
    rate_limit_ok: bool = Depends(check_rate_limit),
    predictor: FraudPredictor = Depends(get_predictor_with_check),
    cache: RedisCache = Depends(get_redis_cache),
) -> PredictionResponse:
    """
    Score a single transaction for fraud probability.

    This endpoint analyzes a transaction and returns:
    - Fraud probability (0-1)
    - Risk tier classification (LOW, MEDIUM, HIGH, CRITICAL)
    - Top risk factors
    - Prediction latency

    Results are cached for 5 minutes based on transaction features.
    """
    # Prepare features dict for cache key
    features_dict = {
        "transaction_id": request.transaction_id,
        "user_id": request.user_id,
        "merchant_id": request.merchant_id,
        "amount": request.amount,
        "timestamp": request.timestamp.isoformat(),
    }

    # Check cache
    try:
        cached_prediction = await cache.get_prediction(
            request.transaction_id, features_dict
        )
        if cached_prediction:
            logger.info(f"Cache hit for transaction {request.transaction_id}")
            return PredictionResponse(**cached_prediction)
    except Exception as e:
        logger.warning(f"Cache check failed: {e}")

    # Generate prediction
    try:
        result = predictor.predict_single(request)

        # Build response
        response = PredictionResponse(
            transaction_id=request.transaction_id,
            fraud_probability=result["fraud_probability"],
            risk_tier=result["risk_tier"],
            top_risk_factors=result["risk_factors"],
            model_version=settings.app_version,
            latency_ms=result["latency_ms"],
        )

        # Cache response
        try:
            await cache.set_prediction(
                request.transaction_id,
                features_dict,
                response.model_dump(),
            )
        except Exception as e:
            logger.warning(f"Failed to cache prediction: {e}")

        logger.info(
            f"Prediction for {request.transaction_id}: "
            f"probability={result['fraud_probability']:.4f}, "
            f"tier={result['risk_tier']}"
        )

        return response

    except Exception as e:
        logger.error(f"Prediction failed for {request.transaction_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/batch_predict",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Score multiple transactions for fraud probability",
    responses={
        200: {"description": "Batch prediction successful"},
        401: {"description": "Invalid or missing API key"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Model not loaded"},
    },
)
async def batch_predict(
    request: BatchPredictionRequest,
    api_key: str = Depends(verify_api_key),
    rate_limit_ok: bool = Depends(check_rate_limit),
    predictor: FraudPredictor = Depends(get_predictor_with_check),
) -> BatchPredictionResponse:
    """
    Score multiple transactions for fraud probability (max 1000).

    This endpoint processes multiple transactions in a single request.
    Batch predictions are not cached.
    """
    start_time_ns = time.time_ns()

    try:
        # Generate predictions
        results = predictor.predict_batch(request.transactions)

        # Build response
        predictions = [
            PredictionResponse(
                transaction_id=txn.transaction_id,
                fraud_probability=result["fraud_probability"],
                risk_tier=result["risk_tier"],
                top_risk_factors=result["risk_factors"],
                model_version=settings.app_version,
                latency_ms=result["latency_ms"],
            )
            for txn, result in zip(request.transactions, results)
        ]

        # Compute total processing time
        end_time_ns = time.time_ns()
        processing_time_ms = (end_time_ns - start_time_ns) / 1_000_000

        response = BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(request.transactions),
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            f"Batch prediction: {len(request.transactions)} transactions "
            f"in {processing_time_ms:.2f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@router.get(
    "/model_info",
    response_model=ModelInfoResponse,
    tags=["Metadata"],
    summary="Get model metadata and feature information",
    responses={
        200: {"description": "Model info retrieved successfully"},
        503: {"description": "Model not loaded"},
    },
)
async def model_info(
    predictor: FraudPredictor = Depends(get_predictor_with_check),
) -> ModelInfoResponse:
    """
    Return model metadata and feature information.

    This endpoint provides information about the loaded model including:
    - Model version and type
    - Feature names
    - Performance metrics
    - Last update timestamp
    """
    try:
        info = predictor.get_model_info()

        response = ModelInfoResponse(
            model_version=info["model_version"],
            model_type=info["model_type"],
            features=info.get("features", []),
            metrics=info["metrics"],
            last_updated=datetime.fromisoformat(info["last_updated"]),
        )

        logger.info("Model info requested")
        return response

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model info: {str(e)}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
    responses={
        200: {"description": "Health check successful"},
    },
)
async def health_check(
    predictor: FraudPredictor = Depends(get_predictor),
    cache: RedisCache = Depends(get_redis_cache),
) -> HealthResponse:
    """
    Health check endpoint.

    Returns the health status of the API service including:
    - Overall status (healthy/unhealthy)
    - Model loaded status
    - Redis connection status
    - Check timestamp
    """
    # Check model status
    model_loaded = predictor.is_model_loaded()

    # Check Redis status
    redis_connected = await cache.is_connected()

    # Determine overall health
    status = "healthy" if (model_loaded and redis_connected) else "unhealthy"

    response = HealthResponse(
        status=status,
        model_loaded=model_loaded,
        redis_connected=redis_connected,
        timestamp=datetime.utcnow(),
    )

    logger.debug(f"Health check: status={status}, model={model_loaded}, redis={redis_connected}")

    return response


# Startup and shutdown events
async def startup_events() -> None:
    """Initialize services on startup."""
    logger.info("Starting up services...")

    # Initialize rate limiter
    rate_limiter = _set_cache_on_rate_limiter(get_redis_cache())

    # Connect to Redis
    try:
        await get_redis_cache().connect()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")

    # Load model
    try:
        get_predictor().load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    logger.info("Startup complete")


async def shutdown_events() -> None:
    """Cleanup on shutdown."""
    logger.info("Shutting down services...")

    # Disconnect Redis
    cache = get_redis_cache()
    try:
        await cache.disconnect()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {e}")

    logger.info("Shutdown complete")
