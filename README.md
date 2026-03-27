<div align="center">

# 🛡️ Fraud ML Engine

### Feature Engineering • Model Benchmarking • Explainability • Real-Time Scoring

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EC6A00?style=flat)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-9ACD32?style=flat)](https://lightgbm.readthedocs.io/)

[Overview](#-overview) • [About](#-about) • [Topics](#-topics) • [API](#-api-surfaces) • [Quick Start](#-quick-start)

---

End-to-end fraud detection ML system combining **feature pipelines**, **classical/deep/anomaly models**, **explainability**, and **production API delivery**.

</div>

---

## 🎯 Overview

`fraud-ml-engine` supports full model lifecycle:

- Feature engineering for transaction risk signals
- Benchmarking across multiple model families
- Explainability with SHAP/LIME-ready outputs
- API-based scoring and benchmark endpoints

## 📌 About

- Built to centralize fraud model experimentation and serving
- Suitable for iterative model development and production integration
- Includes notebooks, configs, and tests for reproducibility

## 🏷️ Topics

`fraud-detection` `machine-learning` `xgboost` `lightgbm` `feature-engineering` `explainable-ai` `fastapi` `risk-scoring`

## 🧩 Architecture

- `src/features/`: velocity, deviation, merchant risk transformations
- `src/models/`: classical, LSTM, anomaly, ensemble modules
- `src/explainability/`: SHAP/LIME tooling
- `src/eda/`: dashboard and visualization logic
- `src/api/`: prediction and benchmark APIs

## 🌐 API Surfaces

- `POST /api/v1/predict`
- `POST /api/v1/batch_predict`
- `POST /api/v1/explain/{id}`
- `POST /api/v1/benchmark/run`
- `GET /api/v1/benchmark/results`
- `GET /api/v1/model_info`
- `GET /api/v1/health`
- `GET /metrics`

## ⚡ Quick Start

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## 🧪 Test

```bash
pytest -q tests/test_api.py
```

## 🛠️ Tech Stack

**ML:** scikit-learn, XGBoost, LightGBM, PyTorch  
**API:** FastAPI, Pydantic, Uvicorn  
**XAI:** SHAP, LIME  
**Visualization:** Plotly, Streamlit
