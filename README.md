# Fraud Detection ML Engine (`fraud-ml-engine`)

End-to-end **fraud detection machine learning system** with feature engineering, model benchmarking, explainable AI, and production API serving. Built for teams shipping real-time transaction risk scoring and model observability.

## Why This Repository

Fraud stacks usually fragment EDA, feature pipelines, model training, explainability, and inference APIs. `fraud-ml-engine` brings them into one operational repository.

## Core Features

- Feature engineering for velocity, deviation, and merchant risk
- Classical model benchmark surfaces (XGBoost, LightGBM, sklearn family)
- LSTM and anomaly-model integrations
- Explainability surfaces (SHAP/LIME utility layers)
- Unified prediction API with benchmark and explanation routes
- Config, notebook, and test scaffolds for repeatable ML workflows

## Project Structure

- `src/features/`: feature extraction and transformation modules
- `src/models/`: classical, LSTM, anomaly, and ensemble interfaces
- `src/explainability/`: explanation utilities and narrative helpers
- `src/eda/`: dashboard app and visualization/callback surfaces
- `src/api/`: unified FastAPI service + legacy compatibility app
- `src/core/`: shared config, data-loading, metrics, model base classes

## API Endpoints

- `POST /api/v1/predict`
- `POST /api/v1/batch_predict`
- `POST /api/v1/explain/{id}`
- `POST /api/v1/benchmark/run`
- `GET /api/v1/benchmark/results`
- `GET /api/v1/model_info`
- `GET /api/v1/health`
- `GET /metrics`

## Quick Start

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## Tests

```bash
pytest -q tests/test_api.py
```

## SEO Keywords

fraud detection machine learning, fraud scoring api, xgboost fraud model, feature engineering for fraud, anomaly detection fraud, explainable fraud ai, fastapi fraud detection
