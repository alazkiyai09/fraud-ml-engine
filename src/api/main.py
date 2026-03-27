"""Unified FastAPI app for fraud-ml-engine."""

from fastapi import FastAPI

from src.api.routers import benchmark, explain, model_info, predict
from src.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.api_title, version=settings.api_version)
    app.include_router(predict.router)
    app.include_router(explain.router)
    app.include_router(benchmark.router)
    app.include_router(model_info.router)

    @app.get("/api/v1/health")
    def health() -> dict:
        return {"status": "ok", "service": settings.api_title, "version": settings.api_version}

    @app.get("/metrics")
    def metrics() -> dict:
        return {"service": settings.api_title, "families": ["classical", "lstm", "anomaly", "xai"]}

    return app


app = create_app()
