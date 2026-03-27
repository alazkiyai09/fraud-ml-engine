# Fraud Scoring API - Deployment Checklist

## Pre-Deployment Checklist

### 1. Model Artifacts
- [ ] Train XGBoost model (Day 2: imbalanced_classification_benchmark)
- [ ] Train feature pipeline (Day 3: fraud_feature_engineering)
- [ ] Save model to `model_artifacts/model.pkl`
- [ ] Save pipeline to `model_artifacts/pipeline.pkl`
- [ ] Verify files exist: `ls -la model_artifacts/`

### 2. Configuration
- [ ] Copy `.env.example` to `.env`
- [ ] Update API keys in `API_KEYS` (add production keys)
- [ ] Set `DEBUG=false` for production
- [ ] Configure `LOG_LEVEL` (INFO for prod)
- [ ] Review rate limit settings if needed

### 3. Dependencies
- [ ] Install Python 3.11+
- [ ] Install Docker and Docker Compose
- [ ] Run `pip install -r requirements.txt` (for local dev)

### 4. Infrastructure
- [ ] Ensure Redis is accessible
- [ ] Configure firewall rules for port 8000
- [ ] Set up monitoring for health endpoint
- [ ] Configure log aggregation

## Deployment Steps

### Option A: Docker Compose (Recommended)

```bash
# 1. Start all services
docker-compose up -d

# 2. Check logs
docker-compose logs -f api

# 3. Verify health
curl http://localhost:8000/api/v1/health

# 4. Test prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "X-API-Key: test-key-dev" \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"test","user_id":"u1","merchant_id":"m1","amount":100.0,"timestamp":"2024-01-27T10:00:00Z"}'
```

### Option B: Local Development

```bash
# 1. Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run API
python run.py

# 4. Verify at http://localhost:8000/docs
```

## Post-Deployment Verification

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "redis_connected": true,
  "timestamp": "2024-01-27T..."
}
```

### API Documentation
- [ ] Open http://localhost:8000/docs
- [ ] Verify all 4 endpoints are listed
- [ ] Try example requests from docs

### Authentication Test
```bash
# Test with valid key
curl -X GET "http://localhost:8000/api/v1/model_info" \
  -H "X-API-Key: test-key-dev"

# Test with invalid key (should fail)
curl -X GET "http://localhost:8000/api/v1/model_info" \
  -H "X-API-Key: invalid-key"
```

### Rate Limiting Test
```bash
# Send 101 requests rapidly (should hit rate limit)
for i in {1..101}; do
  curl -X GET "http://localhost:8000/api/v1/health" \
    -H "X-API-Key: test-key-dev"
done
```

### Caching Test
```bash
# Send same request twice (second should be cached)
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "X-API-Key: test-key-dev" \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"cache_test","user_id":"u1","merchant_id":"m1","amount":100.0,"timestamp":"2024-01-27T10:00:00Z"}'
```

Check Redis:
```bash
docker exec -it fraud_redis redis-cli
> KEYS prediction:*
> GET prediction:cache_test:<hash>
```

## Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=app --cov-report=html

# View coverage
open htmlcov/index.html
```

## Monitoring

### Logs
```bash
# Docker logs
docker-compose logs -f api

# Or check log files
tail -f logs/api.log
```

### Metrics to Monitor
- Request latency (p50, p95, p99)
- Error rate (4xx, 5xx)
- Rate limit violations
- Cache hit rate
- Model prediction distribution

### Prometheus (Optional)
Add to `docker-compose.yml`:
```yaml
prometheus:
  image: prom/prometheus
  ports: ["9090:9090"]
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

## Troubleshooting

### Issue: "Model not loaded"
**Solution**: Check model artifacts exist and are readable
```bash
ls -la model_artifacts/
file model_artifacts/model.pkl
file model_artifacts/pipeline.pkl
```

### Issue: "Redis connection failed"
**Solution**: Verify Redis is running
```bash
docker ps | grep redis
redis-cli ping  # Should return PONG
```

### Issue: "Rate limit exceeded"
**Solution**: Wait for window to reset or increase limits in `.env`

### Issue: High latency
**Solution**: 
1. Check cache hit rate
2. Verify model loading isn't repeated
3. Monitor CPU/memory usage
4. Consider scaling horizontally

## Performance Benchmarks

After deployment, verify:

- [ ] Single prediction: < 100ms (p95)
- [ ] Batch prediction: Linear scaling
- [ ] Cache hit: < 10ms
- [ ] Concurrent requests: Handle 100+ req/s

Load test example:
```bash
# Install wrk
brew install wrk  # macOS
apt install wrk   # Ubuntu

# Run load test
wrk -t4 -c100 -d30s \
  -H "X-API-Key: test-key-dev" \
  http://localhost:8000/api/v1/health
```

## Security Checklist

### API Keys
- [ ] Remove default test keys from production
- [ ] Rotate API keys regularly
- [ ] Use environment variables for secrets
- [ ] Never commit `.env` file

### Network
- [ ] Use HTTPS in production
- [ ] Configure firewall rules
- [ ] Restrict Redis access
- [ ] Enable CORS only for trusted domains

### Headers
- [ ] Verify security headers are present
```bash
curl -I http://localhost:8000/api/v1/health
```

Should include:
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Strict-Transport-Security: max-age=31536000

## Rollback Plan

If deployment fails:

1. **Stop services**
   ```bash
   docker-compose down
   ```

2. **Revert to previous version**
   ```bash
   git checkout <previous-tag>
   docker-compose up -d
   ```

3. **Verify health**
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

4. **Check logs for errors**
   ```bash
   docker-compose logs --tail=100 api
   ```

## Success Criteria

Deployment is successful when:

- ✅ All services are running
- ✅ Health check returns "healthy"
- ✅ Model is loaded successfully
- ✅ Redis is connected
- ✅ Test prediction succeeds
- ✅ All tests pass
- ✅ Performance targets met
- ✅ No errors in logs
- ✅ API documentation accessible

## Support

For issues or questions:
- Check logs: `docker-compose logs -f api`
- Review docs: http://localhost:8000/docs
- Run tests: `pytest -v`
- Check health: `curl http://localhost:8000/api/v1/health`
