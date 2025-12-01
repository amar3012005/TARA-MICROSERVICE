# Intent Service - Build and Run Guide

Complete guide for building, testing, and running the Intent Classification microservice.

## Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- Redis (optional, for distributed caching)
- Gemini API Key (hardcoded in config, but can override with env var)

## 1. Local Testing (Test Suite)

### Run the Test Suite

```bash
# Navigate to intent service directory
cd services/intent

# Run the comprehensive test suite
python test_classifier.py
```

This will test:
- ✅ Layer 1 (Regex) classification
- ✅ Layer 2 (SLM/DistilBERT) classification  
- ✅ Layer 3 (LLM/Gemini) classification
- ✅ Layer 2 Accuracy & Performance (with timing)
- ✅ Layer 3 Accuracy & Performance (with timing)
- ✅ Caching functionality
- ✅ Response format consistency

**Expected Output:**
- All 3 layers initialized
- Performance statistics showing layer distribution
- Accuracy metrics with timing for L2 and L3

## 2. Local Development (FastAPI Server)

### Install Dependencies

```bash
cd services/intent

# Install Python dependencies
pip install -r requirements.txt

# Download Spacy model (if not already downloaded)
python -m spacy download en_core_web_sm
```

### Run the Service

```bash
# Option 1: Using uvicorn directly
cd services/intent
uvicorn app:app --host 0.0.0.0 --port 8002 --reload

# Option 2: Using Python module (from project root)
cd leibniz_agent
python -m uvicorn services.intent.app:app --host 0.0.0.0 --port 8002 --reload
```

Service will be available at: `http://localhost:8002`

### Test the API

```bash
# Health check
curl http://localhost:8002/health

# Classify intent
curl -X POST http://localhost:8002/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I want to schedule an appointment with admissions"}'

# Get metrics
curl http://localhost:8002/metrics
```

## 3. Docker Build

### Build Docker Image (CORRECT METHOD)

The Dockerfile has been updated to use relative paths, so you can build from the **project root** (`leibniz_agent` directory).

```bash
# Navigate to project root (leibniz_agent/)
cd C:\Users\AMAR\SINDHv2\SINDH-Orchestra-Complete\leibniz_agent

# Build Docker image
docker build -f services/intent/Dockerfile -t leibniz-intent:latest .
```

**Note**: The build context will be ~1GB+ (includes the entire leibniz_agent directory). This is normal as it needs to copy the DistilBERT model and all service files.

**Build Process:**
1. ✅ Copies `services/intent/requirements.txt` and installs dependencies
2. ✅ Copies `services/intent/` directory to container
3. ✅ Copies `services/shared/` directory to container  
4. ✅ Downloads Spacy model (`en_core_web_sm`)
5. ✅ Verifies DistilBERT model is present
6. ✅ Exposes port 8002

**Expected Build Time**: 2-5 minutes (depending on internet speed for downloading Spacy model and Python packages)

**Build Process:**
1. Multi-stage build (builder + runtime)
2. Installs Python dependencies
3. Copies service code
4. Downloads Spacy model (`en_core_web_sm`)
5. Verifies DistilBERT model is present

**Expected Build Output:**
```
✅ DistilBERT model found - Layer 2 (SLM) will be enabled
✅ Spacy model downloaded
```

## 4. Docker Run

### Basic Run (Standalone)

```bash
docker run -d \
  --name intent-service \
  -p 8002:8002 \
  -e GEMINI_API_KEY=AIzaSyC6cvyEl4FNjIQCV_p5_2wJkOa1cUObFHU \
  leibniz-intent:latest
```

### Run with Redis (Recommended)

```bash
# Start Redis first
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Run intent service with Redis
docker run -d \
  --name intent-service \
  -p 8002:8002 \
  --link redis:redis \
  -e GEMINI_API_KEY=AIzaSyC6cvyEl4FNjIQCV_p5_2wJkOa1cUObFHU \
  -e REDIS_URL=redis://redis:6379 \
  leibniz-intent:latest
```

### Run with Docker Compose

Add to `services/docker-compose.yml`:

```yaml
  intent-service:
    build:
      context: .
      dockerfile: services/intent/Dockerfile
    container_name: intent-service
    ports:
      - "8002:8002"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY:-AIzaSyC6cvyEl4FNjIQCV_p5_2wJkOa1cUObFHU}
      - GEMINI_MODEL=gemini-2.5-flash-lite
      - REDIS_URL=redis://redis:6379
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - leibniz-network
    restart: unless-stopped
```

Then run:
```bash
cd services
docker-compose up -d intent-service
```

## 5. Verify Deployment

### Check Container Status

```bash
# Check if container is running
docker ps | grep intent-service

# Check logs
docker logs intent-service

# Follow logs
docker logs -f intent-service
```

### Test Endpoints

```bash
# Health check
curl http://localhost:8002/health

# Expected response:
# {
#   "status": "healthy",
#   "classifier": "initialized",
#   "layer1_ready": true,
#   "layer2_ready": true,
#   "layer3_ready": true
# }

# Test classification
curl -X POST http://localhost:8002/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I want to schedule an appointment"}'

# Get metrics
curl http://localhost:8002/metrics
```

## 6. Environment Variables

### Required
- `GEMINI_API_KEY` - Gemini API key (hardcoded fallback: `AIzaSyC6cvyEl4FNjIQCV_p5_2wJkOa1cUObFHU`)

### Optional
- `GEMINI_MODEL` - Model name (default: `gemini-2.5-flash-lite`)
- `LEIBNIZ_INTENT_LAYER1_THRESHOLD` - L1 threshold (default: `0.8`)
- `LEIBNIZ_INTENT_LAYER2_THRESHOLD` - L2 threshold (default: `0.7`)
- `LEIBNIZ_INTENT_LAYER2_ENABLED` - Enable L2 (default: `true`)
- `REDIS_URL` - Redis connection URL (default: `redis://localhost:6379`)
- `INTENT_CACHE_TTL` - Cache TTL in seconds (default: `1800`)

## 7. Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs intent-service

# Common issues:
# - Missing DistilBERT model: Check if leibniz_distilbert_intent_v2/ exists
# - Spacy model download failed: Check internet connection
# - Port already in use: Change port mapping
```

### Service Not Responding

```bash
# Check if service is running
docker ps | grep intent-service

# Check health endpoint
curl http://localhost:8002/health

# Check logs for errors
docker logs intent-service | grep ERROR
```

### Layer 2 Not Working

```bash
# Verify DistilBERT model is present
docker exec intent-service ls -la /app/leibniz_agent/services/intent/leibniz_distilbert_intent_v2/

# Check logs for model loading
docker logs intent-service | grep DistilBERT
```

### Layer 3 Not Working

```bash
# Check Gemini API key
docker exec intent-service env | grep GEMINI_API_KEY

# Check logs for Gemini initialization
docker logs intent-service | grep Gemini
```

## 8. Performance Testing

### Run Accuracy Tests

```bash
# From services/intent/
python test_classifier.py
```

This will show:
- Layer 2 hit rate and average time
- Layer 3 hit rate and average time
- Overall performance statistics

### Load Testing

```bash
# Install Apache Bench
apt-get install apache2-utils  # Ubuntu/Debian
brew install httpd            # macOS

# Run load test
ab -n 1000 -c 10 -p test_request.json -T application/json \
   http://localhost:8002/api/v1/classify
```

## 9. Production Deployment

### Recommended Settings

```bash
docker run -d \
  --name intent-service \
  -p 8002:8002 \
  --restart unless-stopped \
  --memory="2g" \
  --cpus="2" \
  -e GEMINI_API_KEY=your_production_key \
  -e GEMINI_MODEL=gemini-2.5-flash-lite \
  -e REDIS_URL=redis://redis:6379 \
  -e LEIBNIZ_INTENT_LAYER1_THRESHOLD=0.8 \
  -e LEIBNIZ_INTENT_LAYER2_THRESHOLD=0.7 \
  leibniz-intent:latest
```

### Monitoring

```bash
# Watch metrics
watch -n 5 'curl -s http://localhost:8002/metrics | jq'

# Check health
watch -n 10 'curl -s http://localhost:8002/health | jq'
```

## Quick Reference

```bash
# Navigate to project root
cd C:\Users\AMAR\SINDHv2\SINDH-Orchestra-Complete\leibniz_agent

# Build
docker build -f services/intent/Dockerfile -t leibniz-intent:latest .

# Run
docker run -d --name intent-service -p 8002:8002 leibniz-intent:latest

# Test
curl -X POST http://localhost:8002/api/v1/classify `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"Hello\"}'

# Logs
docker logs -f intent-service

# Stop
docker stop intent-service
docker rm intent-service
```

**PowerShell Note**: Use backticks (`) for line continuation in PowerShell, and escape JSON quotes with `\"`.

