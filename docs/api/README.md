# Fraud Scoring API

Production-ready FastAPI application for real-time fraud detection scoring.

## Overview

This API serves a machine learning model trained to detect fraudulent transactions with the following features:

- **Real-time scoring**: Single and batch transaction prediction endpoints
- **Redis caching**: Response caching with configurable TTL
- **Rate limiting**: Token bucket rate limiting per API key
- **Structured logging**: JSON logging with request ID tracing
- **Docker deployment**: Multi-stage Docker build with docker-compose
- **Health checks**: Endpoint for monitoring model and service status

## Architecture

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Client    │─────▶│  FastAPI    │─────▶│    Redis    │
└─────────────┘      └─────────────┘      └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Model +   │
                    │  Pipeline   │
                    └─────────────┘
```

## Features

- **Prediction Endpoints**:
  - `POST /api/v1/predict` - Score single transaction
  - `POST /api/v1/batch_predict` - Score multiple transactions (max 1000)
  - `GET /api/v1/model_info` - Get model metadata
  - `GET /api/v1/health` - Health check

- **Security**:
  - API key authentication via `X-API-Key` header
  - Rate limiting (100 req/min per key)
  - Security headers (CORS, XSS protection, etc.)

- **Performance**:
  - Response caching (5 min TTL)
  - Target: p95 latency < 100ms
  - Async I/O for Redis operations

## Quick Start

### Prerequisites

- Python 3.11+
- Redis 7+
- Docker (optional)

### Local Development

1. **Clone and install dependencies**:
```bash
cd fraud_scoring_api
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start Redis**:
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

4. **Run the server**:
```bash
python run.py
```

The API will be available at `http://localhost:8000`

### Docker Deployment

1. **Build and start services**:
```bash
docker-compose up -d
```

2. **View logs**:
```bash
docker-compose logs -f api
```

3. **Stop services**:
```bash
docker-compose down
```

## API Usage

### Authentication

All endpoints require an API key in the `X-API-Key` header:

```bash
export API_KEY="test-key-dev"
```

### Score Single Transaction

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_12345",
    "user_id": "user_67890",
    "merchant_id": "merchant_001",
    "amount": 150.00,
    "timestamp": "2024-01-27T10:30:00Z"
  }'
```

**Response**:
```json
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
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/batch_predict" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "transaction_id": "txn_001",
        "user_id": "user_001",
        "merchant_id": "merchant_001",
        "amount": 100.00,
        "timestamp": "2024-01-27T10:30:00Z"
      },
      {
        "transaction_id": "txn_002",
        "user_id": "user_002",
        "merchant_id": "merchant_002",
        "amount": 250.50,
        "timestamp": "2024-01-27T10:31:00Z"
      }
    ]
  }'
```

**Response**:
```json
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
```

### Model Information

```bash
curl -X GET "http://localhost:8000/api/v1/model_info" \
  -H "X-API-Key: $API_KEY"
```

**Response**:
```json
{
  "model_version": "1.0.0",
  "model_type": "XGBClassifier",
  "features": [
    "amount",
    "hour",
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
```

### Health Check

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "redis_connected": true,
  "timestamp": "2024-01-27T10:30:00Z"
}
```

## Risk Tiers

The API classifies transactions into four risk tiers based on fraud probability:

| Tier | Probability Range | Action |
|------|------------------|--------|
| LOW | < 0.1 | Approve |
| MEDIUM | 0.1 - 0.5 | Review |
| HIGH | 0.5 - 0.9 | Strong review |
| CRITICAL | ≥ 0.9 | Block |

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | Fraud Scoring API | Application name |
| `API_PORT` | 8000 | API server port |
| `MODEL_PATH` | ./model_artifacts/model.pkl | Path to trained model |
| `PIPELINE_PATH` | ./model_artifacts/pipeline.pkl | Path to feature pipeline |
| `REDIS_HOST` | localhost | Redis server host |
| `REDIS_PORT` | 6379 | Redis server port |
| `CACHE_TTL_SECONDS` | 300 | Cache TTL in seconds |
| `API_KEYS` | ["test-key-dev"] | Valid API keys |
| `RATE_LIMIT_REQUESTS` | 100 | Max requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | 60 | Rate limit window |
| `LOG_LEVEL` | INFO | Logging level |

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

## Performance

Target performance metrics:

- **Single prediction**: p95 < 100ms
- **Batch prediction**: Linear scaling
- **Cache hit**: < 10ms
- **Concurrent requests**: 100+ req/s

## Monitoring

### Logs

Structured JSON logs include:

- Request ID for tracing
- Timestamp
- Log level
- API endpoint
- Prediction results
- Error details

Example:
```json
{
  "timestamp": "2024-01-27T10:30:00.123Z",
  "level": "INFO",
  "logger": "api",
  "request_id": "abc-123-def",
  "message": "Prediction for txn_12345: probability=0.8500, tier=HIGH"
}
```

### Health Checks

Monitor the `/api/v1/health` endpoint:

- `status`: Overall health (healthy/unhealthy)
- `model_loaded`: Whether model is loaded
- `redis_connected`: Redis connection status

## Model Artifacts

Place trained model files in `model_artifacts/`:

- `model.pkl`: Trained XGBoost model
- `pipeline.pkl`: Fitted feature pipeline

To train models, see:
- Day 2: `imbalanced_classification_benchmark/`
- Day 3: `fraud_feature_engineering/`

## Troubleshooting

### Model not loaded

**Error**: `Model not loaded. Please try again later.`

**Solution**: Ensure model artifacts exist in `model_artifacts/`:
```bash
ls -la model_artifacts/
# Should show model.pkl and pipeline.pkl
```

### Redis connection failed

**Error**: `Redis connection failed`

**Solution**: Check Redis is running:
```bash
docker ps | grep redis
# or
redis-cli ping
```

### Rate limit exceeded

**Error**: `429 Too Many Requests`

**Solution**: Wait for rate limit window to reset or adjust limits in `.env`.

## Development

### Project Structure

```
fraud_scoring_api/
├── app/
│   ├── api/           # Routes and dependencies
│   ├── core/          # Config, logging, security
│   ├── models/        # Schemas and predictor
│   ├── services/      # Cache, rate limiter, model loader
│   └── utils/         # Helper functions
├── tests/             # Test suite
├── model_artifacts/   # Trained models
└── docs/              # Documentation
```

### Adding New Endpoints

1. Define schema in `app/models/schemas.py`
2. Add route in `app/api/routes.py`
3. Add tests in `tests/test_api.py`
4. Update documentation in README.md

## License

MIT

## Authors

30 Days of Fraud Detection Project

## See Also

- Day 1: [Data Generation](../synthetic_fraud_data/)
- Day 2: [Model Training](../imbalanced_classification_benchmark/)
- Day 3: [Feature Engineering](../fraud_feature_engineering/)
