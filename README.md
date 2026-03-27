# fraud-ml-engine

End-to-end fraud detection ML engine split out of `fl-security-research`.

This repo consolidates the original fraud-focused projects into one structure:

- `src/eda/` for the interactive dashboard
- `src/features/` for fraud feature engineering
- `src/models/` for classical, LSTM, and anomaly models
- `src/explainability/` for SHAP/LIME tooling
- `src/api/` for the unified FastAPI surface

Preserved project implementations remain in place under the new layout, especially in `legacy/` folders where the original source tree needed to be embedded with minimal disturbance.

## Legacy Sources Merged

- `fraud_detection_eda_dashboard`
- `imbalanced_classification_benchmark`
- `fraud_feature_engineering`
- `fraud_scoring_api`
- `lstm_fraud_detection`
- `anomaly_detection_benchmark`
- `fraud_model_explainability`

## Quick Start

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## Smoke Test

```bash
pytest -q tests/test_api.py
```
