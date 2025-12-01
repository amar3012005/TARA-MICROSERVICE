# Intent Classification Microservice

**Phase 3** of the Leibniz Agent Cloud Transformation - Two-tier intent classification with pattern matching and LLM fallback.

## Overview

Stateless HTTP REST service that classifies user intent for the Leibniz University Agent using a hybrid approach:

1. **Fast Pattern Matching** (target >80% of requests): Regex + keyword matching for common intents
2. **Gemini LLM Fallback** (target <20% of requests): AI-powered classification for complex cases

### Intent Categories

- `APPOINTMENT_SCHEDULING`: User wants to book/schedule a meeting
- `RAG_QUERY`: User asking for information (most common - routes to RAG system)
- `GREETING`: Standalone social pleasantry (strict - very rare)
- `EXIT`: User wants to end conversation
- `UNCLEAR`: Ambiguous or nonsensical input

## Architecture

```
User Input → FastAPI Endpoint
    ↓
Redis Cache Check (30min TTL)
    ↓ (cache miss)
Fast Pattern Classification
    ↓ (confidence < threshold)
Gemini LLM Fallback
    ↓
Cache Result → Return
```

### Key Features

- **Distributed Caching**: Redis-backed 30-minute TTL cache for repeated queries
- **Two-Tier Classification**: Fast patterns (target 80%+) → Gemini fallback (20%-)
- **Context Extraction**: Enriched classification with user_goal, key_entities, extracted_meaning
- **Performance Tracking**: Metrics endpoint for monitoring fast route percentage and confidence
- **Graceful Degradation**: Works without Redis (no caching) or Gemini (pattern-only)

## API Endpoints

### POST /api/v1/classify

Classify user intent.

**Request:**
```json
{
  "text": "I want to schedule an appointment with admissions",
  "context": {
    "previous_intent": "RAG_QUERY"
  }
}
```

**Response:**
```json
{
  "intent": "APPOINTMENT_SCHEDULING",
  "confidence": 0.95,
  "context": {
    "user_goal": "wants to schedule appointment with admissions",
    "key_entities": {
      "department": "admissions"
    },
    "extracted_meaning": "schedule appointment admissions"
  },
  "reasoning": "Appointment keywords detected: ['appointment', 'schedule']",
  "fast_route": true,
  "response_time": 0.012,
  "cached": false
}
```

### GET /health

Health check with component status.

**Response:**
```json
{
  "status": "healthy",
  "redis": "connected",
  "classifier": "initialized",
  "config": {
    "gemini_model": "gemini-2.0-flash-lite",
    "confidence_threshold": 0.8,
    "gemini_timeout": 5.0,
    "fast_route_target": 0.8,
    "has_gemini_key": true
  }
}
```

### GET /metrics

Performance metrics.

**Response:**
```json
{
  "total_requests": 1523,
  "fast_route_count": 1289,
  "gemini_route_count": 234,
  "fast_route_percentage": 84.63,
  "average_confidence": 0.891
}
```

### POST /admin/clear_cache

Clear all cached classifications.

**Response:**
```json
{
  "message": "Cache cleared successfully",
  "keys_deleted": 42
}
```

## Configuration

Set via environment variables (loaded from `.env.leibniz` or `.env`):

### Required Variables

```bash
# Gemini API key for LLM fallback (REQUIRED)
GEMINI_API_KEY=your_gemini_api_key_here
```

### Optional Variables

```bash
# Gemini model name (default: gemini-2.0-flash-lite)
GEMINI_MODEL=gemini-2.0-flash-lite

# Fast route confidence threshold (default: 0.8)
LEIBNIZ_INTENT_PARSER_CONFIDENCE_THRESHOLD=0.8

# Gemini API timeout in seconds (default: 5.0)
LEIBNIZ_INTENT_PARSER_GEMINI_TIMEOUT=5.0

# Enable context extraction (default: true)
LEIBNIZ_INTENT_PARSER_ENABLE_CONTEXT_EXTRACTION=true

# Log all classifications (default: true)
LEIBNIZ_INTENT_PARSER_LOG_CLASSIFICATIONS=true

# Strip non-essential fields from response (default: false)
LEIBNIZ_INTENT_PARSER_MINIMAL_OUTPUT=false

# Target fast route percentage (default: 0.8)
LEIBNIZ_INTENT_PARSER_FAST_ROUTE_TARGET=0.8

# Redis connection URL (default: redis://localhost:6379)
REDIS_URL=redis://localhost:6379

# Cache TTL in seconds (default: 1800 = 30 minutes)
INTENT_CACHE_TTL=1800
```

## Local Development

### Prerequisites

- Python 3.11+
- Redis (for caching - optional but recommended)
- Gemini API key

### Setup

```bash
# Navigate to service directory
cd leibniz_agent/services/intent

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY=your_key_here
export REDIS_URL=redis://localhost:6379

# Run service
python app.py
# OR with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

Service runs on `http://localhost:8002`

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/test_classifier.py -v

# Run specific test
pytest tests/test_classifier.py::test_appointment_scheduling_clear -v

# Run with coverage
pytest tests/test_classifier.py --cov=. --cov-report=html
```

### Manual Testing

```bash
# Test classification endpoint
curl -X POST http://localhost:8002/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I want to schedule an appointment"}'

# Check health
curl http://localhost:8002/health

# Get metrics
curl http://localhost:8002/metrics

# Clear cache
curl -X POST http://localhost:8002/admin/clear_cache
```

## Docker Deployment

### Build Image

```bash
# From service directory
cd leibniz_agent/services/intent
docker build -t leibniz-intent:latest .

# From repository root
docker build -f leibniz_agent/services/intent/Dockerfile \
  -t leibniz-intent:latest \
  leibniz_agent/services/intent/
```

### Run Container

```bash
docker run -d \
  --name intent-service \
  -p 8002:8002 \
  -e GEMINI_API_KEY=your_key_here \
  -e REDIS_URL=redis://redis:6379 \
  leibniz-intent:latest
```

### Docker Compose

See `docker-compose.leibniz.yml` at repository root:

```bash
# Start all services (Redis + STT/VAD + Intent)
docker-compose -f docker-compose.leibniz.yml up -d

# View logs
docker-compose -f docker-compose.leibniz.yml logs -f intent

# Stop services
docker-compose -f docker-compose.leibniz.yml down
```

## Performance

### Benchmarks

- **Fast Route (Pattern Match)**: <50ms per classification
- **Gemini Fallback**: 500-2000ms per classification (API latency)
- **Cache Hit**: <5ms per classification (Redis read)

### Expected Metrics

- **Fast Route Percentage**: 80-90% (target >80%)
- **Average Confidence**: 0.85-0.95
- **Cache Hit Rate**: 40-60% (depends on query diversity)

### Optimization Tips

1. **Increase Cache TTL** for stable query patterns: `INTENT_CACHE_TTL=3600` (1 hour)
2. **Lower Confidence Threshold** to use fast route more: `CONFIDENCE_THRESHOLD=0.7`
3. **Add Custom Patterns** to `patterns.py` for domain-specific intents
4. **Scale Horizontally** with multiple workers: `uvicorn app:app --workers 4`

## Monitoring

### Key Metrics to Track

```bash
# Fast route percentage (should be >80%)
curl http://localhost:8002/metrics | jq '.fast_route_percentage'

# Average confidence (should be >0.85)
curl http://localhost:8002/metrics | jq '.average_confidence'

# Service health
curl http://localhost:8002/health | jq '.status'

# Redis connection
curl http://localhost:8002/health | jq '.redis'
```

### Logging

Structured JSON logs to stdout/stderr:

```
2025-06-10 14:32:15 - intent.classifier - INFO - ✅ FAST: RAG_QUERY (conf=0.85, time=0.012s)
2025-06-10 14:32:18 - intent.classifier - INFO - 🤖 GEMINI: APPOINTMENT_SCHEDULING (conf=0.93, time=1.234s)
2025-06-10 14:32:20 - intent.app - INFO - ✅ CACHE HIT: What programs do you offer?
```

### Health Checks

Docker health check runs every 30 seconds:

```bash
# Manual health check
docker exec intent-service python -c "import httpx; httpx.get('http://localhost:8002/health', timeout=5.0)"
```

## Troubleshooting

### Service Won't Start

**Symptom**: Container exits immediately or health check fails

**Solutions**:
1. Check GEMINI_API_KEY is set: `docker logs intent-service | grep "Gemini"`
2. Verify Redis is running: `docker ps | grep redis`
3. Check port 8002 is not in use: `netstat -an | grep 8002`

### Low Fast Route Percentage (<60%)

**Symptom**: Too many requests going to Gemini fallback

**Solutions**:
1. Review metrics: `curl http://localhost:8002/metrics`
2. Lower confidence threshold: `CONFIDENCE_THRESHOLD=0.7`
3. Add custom patterns to `patterns.py` for your domain
4. Analyze logs for common low-confidence patterns

### Cache Not Working

**Symptom**: `cached: false` on all requests

**Solutions**:
1. Check Redis connection: `curl http://localhost:8002/health | jq '.redis'`
2. Verify REDIS_URL: `echo $REDIS_URL`
3. Test Redis manually: `redis-cli ping`
4. Check Redis logs: `docker logs redis`

### High Response Times (>100ms on Fast Route)

**Symptom**: Pattern matching taking too long

**Solutions**:
1. Check CPU usage: `docker stats intent-service`
2. Reduce concurrent requests: Lower `--workers` setting
3. Profile slow patterns in `patterns.py`
4. Consider caching compiled patterns (already done in code)

## Integration

### Example Integration with Leibniz Agent

```python
import httpx

async def classify_user_intent(text: str) -> dict:
    """Classify user intent via Intent service"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://intent-service:8002/api/v1/classify",
            json={"text": text},
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()

# Usage
result = await classify_user_intent("I want to schedule an appointment")
print(f"Intent: {result['intent']}")
print(f"Confidence: {result['confidence']}")
print(f"User Goal: {result['context']['user_goal']}")
```

### Error Handling

```python
try:
    result = await classify_user_intent(text)
    intent = result["intent"]
except httpx.TimeoutException:
    # Fallback to default intent
    intent = "RAG_QUERY"
except httpx.HTTPStatusError as e:
    if e.response.status_code == 503:
        # Service unavailable
        intent = "UNCLEAR"
    else:
        raise
```

## Architecture Decisions

### Why Two-Tier Classification?

- **Performance**: Pattern matching is 20-100x faster than LLM calls
- **Cost**: Reduce Gemini API usage by 80%+
- **Reliability**: Patterns work even if Gemini API is down
- **Transparency**: Pattern matches provide clear reasoning

### Why Redis Caching?

- **Distributed**: Cache shared across multiple service instances
- **Low Latency**: Sub-5ms cache reads
- **Automatic Expiry**: 30-minute TTL prevents stale classifications
- **Graceful Degradation**: Service works without Redis (just slower)

### Why Gemini 2.0 Flash Lite?

- **Speed**: 500-1500ms response time (vs 2-4s for Pro)
- **Cost**: 10x cheaper than Gemini Pro
- **Quality**: Sufficient for 5-category classification
- **Context**: Excellent understanding of university domain

## Contributing

When adding new intents or patterns:

1. **Update `patterns.py`**: Add keywords and regex patterns
2. **Update Tests**: Add test cases to `tests/test_classifier.py`
3. **Update Docs**: Document new intent in README.md
4. **Test Fast Route**: Verify fast pattern hit rate stays >80%
5. **Update Prompt**: Modify `get_system_prompt()` if needed

Example pattern addition:

```python
# In patterns.py
patterns = {
    ...
    "course_registration": {
        "keywords": ["register", "enroll", "add course", "drop course"],
        "regex_patterns": [
            r"\b(register|enroll)\s+(?:for|in)\s+\w+",
            r"\b(add|drop)\s+(?:a\s+)?course\b"
        ],
        "entity_patterns": {
            "course_code": r"\b[A-Z]{2,4}\s*\d{3,4}\b"
        }
    }
}
```

## License

Part of the SINDH Orchestra Complete project. See repository root for license.

## Support

For issues, see:
- Service logs: `docker logs intent-service`
- Health endpoint: `http://localhost:8002/health`
- Metrics endpoint: `http://localhost:8002/metrics`
- Repository issues: GitHub issues tracker
