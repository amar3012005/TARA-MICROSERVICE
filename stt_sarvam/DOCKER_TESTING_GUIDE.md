# ============================================================================
# Docker Testing Guide for STT/VAD Microservice
# ============================================================================

## Prerequisites

1. **Install Docker Desktop** (if not already installed)
   - Download: https://www.docker.com/products/docker-desktop/
   - Ensure Docker is running

2. **Set Environment Variables**
   Create a `.env.leibniz` file in the repository root:
   ```bash
   # Required
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Optional (defaults shown)
   LEIBNIZ_REDIS_HOST_PORT=6379
   LEIBNIZ_STT_SERVICE_PORT=8001
   LEIBNIZ_STT_SERVICE_WORKERS=2
   LEIBNIZ_VAD_MODEL=gemini-2.0-flash-exp
   LEIBNIZ_VAD_LANGUAGE=en-US
   LEIBNIZ_VAD_INITIAL_TIMEOUT=10.0
   ```

## Quick Start - Docker Compose

### 1. Build and Start Services

```powershell
# Navigate to repository root
cd C:\Users\AMAR\SINDHv2\SINDH-Orchestra-Complete

# Build and start services (Redis + STT/VAD)
docker-compose -f docker-compose.leibniz.yml --env-file .env.leibniz up --build -d

# Check logs
docker-compose -f docker-compose.leibniz.yml logs -f
```

### 2. Verify Services are Running

```powershell
# Check container status
docker ps

# Expected output:
# leibniz-stt-vad   (port 8001)
# leibniz-redis     (port 6379)

# Test health endpoint
curl http://localhost:8001/health

# Or with PowerShell
Invoke-WebRequest -Uri http://localhost:8001/health
```

### 3. Run Test Client

```powershell
# Install test dependencies (if not already installed)
pip install websockets httpx numpy

# Run test client
python leibniz_agent/services/stt_vad/test_client.py
```

Expected output:
```
=== Testing Health Check ===
Status: 200
Response: {
  "status": "healthy",
  "service": "stt-vad",
  "vad_manager": true,
  "redis": true,
  ...
}

=== Testing WebSocket with Synthetic Audio ===
✅ WebSocket connected
📤 Sent chunk 1/60
📤 Sent chunk 11/60
...
📥 Received: {"type": "partial", "text": "...", ...}
📥 Received: {"type": "final", "text": "...", ...}
```

### 4. Stop Services

```powershell
# Stop containers
docker-compose -f docker-compose.leibniz.yml down

# Stop and remove volumes (clean slate)
docker-compose -f docker-compose.leibniz.yml down -v
```

## Manual Docker Run (Alternative)

If you prefer to run services manually without docker-compose:

### 1. Start Redis

```powershell
docker run -d `
  --name leibniz-redis `
  -p 6379:6379 `
  redis:7-alpine
```

### 2. Build STT/VAD Image

```powershell
docker build `
  -f leibniz_agent/services/stt_vad/Dockerfile `
  -t leibniz-stt-vad:latest `
  .
```

### 3. Run STT/VAD Container

```powershell
docker run -d `
  --name leibniz-stt-vad `
  -p 8001:8001 `
  -e GEMINI_API_KEY=$env:GEMINI_API_KEY `
  -e LEIBNIZ_REDIS_HOST=host.docker.internal `
  -e LEIBNIZ_REDIS_PORT=6379 `
  leibniz-stt-vad:latest
```

### 4. Check Logs

```powershell
# STT/VAD logs
docker logs -f leibniz-stt-vad

# Redis logs
docker logs -f leibniz-redis
```

### 5. Cleanup

```powershell
docker stop leibniz-stt-vad leibniz-redis
docker rm leibniz-stt-vad leibniz-redis
```

## Debugging

### Check Container Health

```powershell
# Inspect health status
docker inspect leibniz-stt-vad | Select-String "Health"

# Enter container shell
docker exec -it leibniz-stt-vad /bin/sh

# Inside container:
# - Check Python packages: pip list
# - Test health endpoint: curl http://localhost:8001/health
# - View logs: tail -f /var/log/*
```

### Common Issues

**Issue: Container fails to start**
- Check logs: `docker logs leibniz-stt-vad`
- Verify GEMINI_API_KEY is set
- Ensure port 8001 is not in use

**Issue: Redis connection failed**
- Check Redis is running: `docker ps | Select-String redis`
- Verify network: `docker network ls | Select-String leibniz`
- Test Redis: `docker exec leibniz-redis redis-cli ping`

**Issue: Health check failing**
- Check health logs: `docker inspect leibniz-stt-vad --format='{{json .State.Health}}'`
- Verify dependencies installed: `docker exec leibniz-stt-vad pip list | Select-String httpx`

**Issue: WebSocket connection refused**
- Verify port mapping: `docker port leibniz-stt-vad`
- Check firewall: Allow port 8001
- Test locally: `curl http://localhost:8001/health`

## Production Deployment

For production use:

1. **Enable Redis password**
   ```yaml
   command: redis-server --requirepass ${REDIS_PASSWORD}
   ```

2. **Use environment-specific configs**
   ```bash
   docker-compose -f docker-compose.leibniz.yml --env-file .env.production up -d
   ```

3. **Configure resource limits**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 2G
   ```

4. **Enable TLS for WebSocket** (add nginx reverse proxy)

5. **Monitor with Prometheus/Grafana** (add metrics exporter)

## API Endpoints

Once running, the service exposes:

- `GET /health` - Health check
- `GET /metrics` - Performance metrics
- `POST /admin/reset` - Force reset Gemini session
- `WS /api/v1/transcribe/stream` - WebSocket transcription

See `leibniz_agent/services/stt_vad/app.py` for full API documentation.
