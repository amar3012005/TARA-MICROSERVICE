# Leibniz Agent Microservices Architecture

> **Phase 1: Redis Infrastructure Foundation** ✅  
> Comprehensive setup guide for the Leibniz Agent microservices transformation

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Infrastructure Setup (Phase 1)](#infrastructure-setup-phase-1)
- [Shared Utilities](#shared-utilities)
- [Environment Variables](#environment-variables)
- [Development Workflow](#development-workflow)
- [Service Communication](#service-communication)
- [Monitoring and Health Checks](#monitoring-and-health-checks)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## 🎯 Overview

The Leibniz Agent is being transformed from a monolithic architecture (`leibniz_pro.py`) into a distributed microservices system. This transformation enables:

- **Scalability**: Individual services can be scaled independently based on load
- **Resilience**: Service failures don't bring down the entire system
- **Development Velocity**: Teams can work on services independently
- **Resource Optimization**: Deploy only needed services (e.g., RAG-only deployment)
- **Technology Flexibility**: Each service can use optimal tech stack

**Reference**: See [`leibniz_agent/docs/Cloud Transformation.md`](../docs/Cloud%20Transformation.md) for complete architectural vision.

---

## 🏗️ Architecture

### Microservices Breakdown

The Leibniz Agent is decomposed into **7 independent services**:

```
┌─────────────────────────────────────────────────────────────┐
│                     Client (Web/Mobile)                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  Orchestrator  │ (Port 8000)
                    │   Service      │
                    └───┬────┬───┬───┘
                        │    │   │
        ┌───────────────┘    │   └──────────────┐
        │                    │                  │
┌───────▼────────┐  ┌────────▼────────┐  ┌─────▼──────────┐
│   STT/VAD      │  │     Intent      │  │      RAG       │
│   Service      │  │  Classification │  │    Service     │
│  (Port 8001)   │  │   (Port 8002)   │  │  (Port 8003)   │
└────────────────┘  └─────────────────┘  └────────────────┘
        │                    │                  │
        │            ┌───────▼───────┐          │
        │            │  Appointment  │          │
        │            │  FSM Service  │          │
        │            │  (Port 8005)  │          │
        │            └───────────────┘          │
        │                    │                  │
        └────────┬───────────┴──────────────────┘
                 │
         ┌───────▼────────┐
         │  TTS Service   │ (Port 8004)
         └────────────────┘
                 │
         ┌───────▼────────┐
         │     Redis      │ (Port 6379)
         │  Cache/State   │
         └────────────────┘
```

### Service Responsibilities

| Service | Port | Responsibility | Current Status |
|---------|------|----------------|----------------|
| **Redis** | 6379 | Caching, session state, inter-service communication | ✅ **Phase 1 Complete** |
| **STT/VAD** | 8001 | Speech-to-text, voice activity detection | ✅ **Phase 2 Complete** |
| **Intent** | 8002 | Intent classification, entity extraction | ✅ **Phase 3 Complete** |
| **RAG** | 8003 | Knowledge base retrieval, document search | 🔜 Phase 4 |
| **TTS** | 8004 | Text-to-speech synthesis (ElevenLabs/Google/Gemini) | 🔜 Phase 5 |
| **Appointment** | 8005 | Appointment scheduling FSM | ✅ **Phase 6 Complete** |
| **Orchestrator** | 8000 | Main coordinator, session management | 🔜 Phase 7 |

---

## 🚀 Infrastructure Setup (Phase 1)

### Prerequisites

- **Docker** 20.10+ and **Docker Compose** 1.29+
- **Python** 3.8+ with pip
- **Git** (for cloning repository)

### Step 1: Install Dependencies

```powershell
# Install Python dependencies (includes redis[asyncio])
pip install -r requirements.txt

# Verify Redis client installation
python -c "import redis.asyncio; print(f'✅ Redis client version: {redis.__version__}')"
```

### Step 2: Start Redis Infrastructure

```powershell
# Start Redis container
docker-compose -f docker-compose.leibniz.yml up -d redis

# Verify Redis is running
docker-compose -f docker-compose.leibniz.yml ps

# Expected output:
#      Name                   Command                State           Ports
# ---------------------------------------------------------------------------------
# leibniz-redis   docker-entrypoint.sh redis ...   Up      0.0.0.0:6379->6379/tcp
```

### Step 3: Test Redis Connection

```powershell
# Option 1: Using health check utility
python -m leibniz_agent.services.shared.health_check --redis

# Option 2: Using redis_client test
python -m leibniz_agent.services.shared.redis_client

# Option 3: Direct redis-cli (requires redis-cli installed)
docker exec leibniz-redis redis-cli ping
# Expected: PONG
```

### Step 4: Configure Environment Variables

Ensure your `leibniz_agent/.env.leibniz` file contains the Redis configuration (already added in Phase 1):

```bash
# Redis connection settings
LEIBNIZ_REDIS_HOST=localhost
LEIBNIZ_REDIS_PORT=6379
LEIBNIZ_REDIS_DB=0
LEIBNIZ_REDIS_URL=redis://localhost:6379/0
```

**Note**: The Leibniz Agent uses `leibniz_agent/.env.leibniz` for all configuration. For general project environment variables, see `.env.example` at the repository root.

**For Docker services**: Change `LEIBNIZ_REDIS_HOST=redis` (use service name instead of localhost).

### Step 5: Stopping Redis

```powershell
# Stop Redis container (preserves data)
docker-compose -f docker-compose.leibniz.yml stop redis

# Stop and remove containers (preserves data in named volume)
docker-compose -f docker-compose.leibniz.yml down

# DANGER: Remove data volume (complete reset)
docker-compose -f docker-compose.leibniz.yml down -v
```

---

## 🔧 Shared Utilities

The `leibniz_agent/services/shared/` module provides reusable utilities for all microservices.

### Redis Client Usage

```python
from leibniz_agent.services.shared import get_redis_client, close_redis_client

# Get async Redis client (singleton pattern)
redis = await get_redis_client()

# Set key with 1-hour expiration
await redis.set("user:123:session", "active", ex=3600)

# Get key (returns string due to decode_responses=True)
session_state = await redis.get("user:123:session")

# Delete key
await redis.delete("user:123:session")

# Cleanup on shutdown
await close_redis_client()
```

**Important Note on String Encoding**: The Redis client is configured with `decode_responses=True`, which means all values are automatically decoded from bytes to strings. This is suitable for most use cases (caching JSON, session data, intent classifications). If you need to cache binary data (e.g., audio chunks, images), use manual encoding:

```python
import json

# For JSON data (recommended)
await redis.set("user:data", json.dumps({"name": "Alice"}), ex=3600)
data = json.loads(await redis.get("user:data"))

# For binary data (if needed)
binary_data = b"..."
await redis.set("audio:chunk", binary_data.hex(), ex=60)  # Store as hex string
retrieved = bytes.fromhex(await redis.get("audio:chunk"))  # Decode back to bytes
```

### Health Check Usage

```python
from leibniz_agent.services.shared import check_redis_health, check_service_health

# Check Redis
redis_health = await check_redis_health()
if redis_health.is_healthy():
    print(f"✅ Redis is healthy (latency: {redis_health.latency_ms:.2f}ms)")
else:
    print(f"❌ Redis is {redis_health.status}: {redis_health.details}")

# Check HTTP service (requires httpx to be installed)
stt_health = await check_service_health("stt-service", "http://localhost:8001/health")
print(f"STT Service: {stt_health.status}")
```

**Dependencies**: Health checks for HTTP services require `httpx>=0.24.0` (already included in `requirements.txt`).

### Context Manager Pattern

```python
from leibniz_agent.services.shared.redis_client import RedisClientContext

async with RedisClientContext() as redis:
    await redis.set("temp_key", "temp_value", ex=60)
    value = await redis.get("temp_key")
# Client automatically cleaned up on exit
```

---

## 🔐 Environment Variables

### Core Redis Configuration

All variables are prefixed with `LEIBNIZ_` to avoid conflicts. Defined in `leibniz_agent/.env.leibniz` (or `.env.example` at root for reference):

| Variable | Default | Description |
|----------|---------|-------------|
| `LEIBNIZ_REDIS_HOST` | `localhost` | Redis server hostname |
| `LEIBNIZ_REDIS_PORT` | `6379` | Redis server port |
| `LEIBNIZ_REDIS_DB` | `0` | Redis database number (0-15) |
| `LEIBNIZ_REDIS_PASSWORD` | _(empty)_ | Redis password (optional, use in production) |
| `LEIBNIZ_REDIS_URL` | `redis://localhost:6379/0` | Full connection URL (overrides individual settings) |
| `LEIBNIZ_REDIS_HOST_PORT` | `6379` | Host port for Docker mapping (use different value if SINDH Redis is running) |

### Connection Pool Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LEIBNIZ_REDIS_MAX_CONNECTIONS` | `50` | Max connections in pool |
| `LEIBNIZ_REDIS_SOCKET_TIMEOUT` | `5.0` | Socket timeout (seconds) |
| `LEIBNIZ_REDIS_SOCKET_CONNECT_TIMEOUT` | `5.0` | Connection timeout (seconds) |

### Cache TTL Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LEIBNIZ_REDIS_INTENT_CACHE_TTL` | `1800` | Intent classification cache (30 min) |
| `LEIBNIZ_REDIS_RAG_CACHE_TTL` | `3600` | RAG response cache (1 hour) |
| `LEIBNIZ_REDIS_SESSION_TTL` | `1800` | Session state cache (30 min) |

### Docker vs Local Configuration

**Local Development** (`leibniz_agent/.env.leibniz`):
```bash
LEIBNIZ_REDIS_HOST=localhost
LEIBNIZ_REDIS_URL=redis://localhost:6379/0
LEIBNIZ_REDIS_HOST_PORT=6379  # Or 6380 if SINDH Redis is running
```

**Docker Services** (override in docker-compose):
```yaml
environment:
  - LEIBNIZ_REDIS_HOST=redis  # Use Docker service name
  - LEIBNIZ_REDIS_URL=redis://redis:6379/0
```

**Note**: When running containerized Leibniz services, set `LEIBNIZ_REDIS_HOST=redis` to use the Docker service name instead of `localhost`.

---

## 💻 Development Workflow

### Phase-by-Phase Implementation

Each microservice will be implemented in phases:

1. **Phase 1**: ✅ **Redis infrastructure** (current phase)
2. **Phase 2**: STT/VAD service extraction from `leibniz_stt.py` and `leibniz_vad.py`
3. **Phase 3**: Intent classification service from `leibniz_intent_parser.py`
4. **Phase 4**: RAG service from `leibniz_rag.py`
5. **Phase 5**: TTS service from `leibniz_tts.py`
6. **Phase 6**: Appointment FSM service from `leibniz_appointment_fsm.py`
7. **Phase 7**: Main orchestrator service (replaces `leibniz_pro.py`)
8. **Phase 8**: Integration tests and deployment

### Testing Individual Services

Each service will have:

```powershell
# Unit tests (service-specific logic)
python leibniz_agent/services/<service>/test_<service>.py

# Integration tests (with Redis)
python leibniz_agent/services/<service>/test_integration.py

# End-to-end tests (full flow)
python leibniz_agent/services/tests/test_e2e_<scenario>.py
```

### Current Testing (Phase 1)

```powershell
# Test Redis client
python -m leibniz_agent.services.shared.redis_client

# Test health checks
python -m leibniz_agent.services.shared.health_check --redis

# Test all configured services (once deployed)
python -m leibniz_agent.services.shared.health_check --all
```

---

## 🔗 Service Communication

### HTTP REST APIs

Services communicate via HTTP REST endpoints:

```python
# Example: Orchestrator → Intent Service
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8002/classify",
        json={"text": "I need to schedule an appointment"}
    )
    intent = response.json()
```

### Redis for Caching and State

Services use Redis for:

1. **Response Caching**: Avoid redundant processing
   ```python
   cache_key = f"intent:{hash(text)}"
   cached = await redis.get(cache_key)
   if cached:
       return json.loads(cached)
   ```

2. **Session State**: Maintain conversation context
   ```python
   session_key = f"session:{user_id}"
   await redis.hset(session_key, "last_intent", "appointment_scheduling")
   await redis.expire(session_key, 1800)  # 30 min TTL
   ```

3. **Inter-Service Communication**: Pub/Sub for events
   ```python
   # Service A publishes event
   await redis.publish("appointments", json.dumps({"event": "created", "id": 123}))
   
   # Service B subscribes
   pubsub = redis.pubsub()
   await pubsub.subscribe("appointments")
   ```

### WebSocket for Real-Time

The orchestrator service will use WebSockets for:
- Streaming audio (STT input)
- Partial transcripts (real-time feedback)
- TTS audio chunks (low-latency playback)

---

## 📊 Monitoring and Health Checks

### Service Health Endpoints

Each service exposes a `/health` endpoint:

```python
# FastAPI health endpoint pattern
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "stt-vad",
        "version": "1.0.0",
        "uptime_seconds": get_uptime(),
        "dependencies": {
            "redis": (await check_redis_health()).status,
        }
    }
```

### Using Health Check Utility

```powershell
# Check single service
python -m leibniz_agent.services.shared.health_check --service stt http://localhost:8001/health

# Check multiple services
python -m leibniz_agent.services.shared.health_check --redis --service stt http://localhost:8001/health --service intent http://localhost:8002/health

# Check all configured services
python -m leibniz_agent.services.shared.health_check --all
```

**Expected Output**:
```
🔍 Running health checks...

✅ REDIS
   Status: healthy
   Latency: 2.34ms
   Details:
      version: 7.2.4
      uptime_seconds: 3600
      connected_clients: 5
      used_memory: 1.2M

✅ STT-VAD
   Status: healthy
   Latency: 45.67ms
   Details:
      status_code: 200

📊 Summary: 2/2 services healthy
🎉 All services are healthy!
```

### Docker Health Checks

Each service's Dockerfile includes health checks:

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1
```

Redis health check (already configured in `docker-compose.leibniz.yml`):

```yaml
healthcheck:
  test: ["CMD", "redis-cli", "ping"]
  interval: 10s
  timeout: 3s
  retries: 3
```

---

## 🛠️ Troubleshooting

### Issue: Redis Connection Refused

**Symptoms**:
```
❌ Redis health check failed: ConnectionRefusedError: [Errno 111] Connection refused
```

**Solutions**:

1. **Check Redis is running**:
   ```powershell
   docker-compose -f docker-compose.leibniz.yml ps
   ```

2. **Check port binding**:
   ```powershell
   docker-compose -f docker-compose.leibniz.yml logs redis
   ```

3. **Verify environment variables**:
   ```powershell
   $env:LEIBNIZ_REDIS_HOST; $env:LEIBNIZ_REDIS_PORT
   ```

4. **Test with redis-cli**:
   ```powershell
   docker exec leibniz-redis redis-cli ping
   ```

### Issue: Port 6379 Already in Use

**Symptoms**:
```
Error starting userland proxy: listen tcp 0.0.0.0:6379: bind: address already in use
```

**Solutions**:

1. **Find process using port**:
   ```powershell
   netstat -ano | findstr :6379
   ```

2. **Change Redis host port** using environment variable (recommended):
   
   Add to your `.env` or `.env.leibniz` file:
   ```bash
   LEIBNIZ_REDIS_HOST_PORT=6380
   LEIBNIZ_REDIS_PORT=6380
   LEIBNIZ_REDIS_URL=redis://localhost:6380/0
   ```
   
   Then start Redis:
   ```powershell
   docker-compose -f docker-compose.leibniz.yml up -d redis
   ```
   
   This is especially useful when running both SINDH and Leibniz Redis instances simultaneously.

3. **Or manually edit docker-compose.leibniz.yml** (if not using env var):
   ```yaml
   ports:
     - "6380:6379"  # Host:Container
   ```

4. **For Docker-only deployments** (no external access needed):
   
   Remove the port mapping entirely and rely on the `leibniz-network` internal network. Only containerized Leibniz services will access Redis.

### Issue: Docker Network Not Found

**Symptoms**:
```
ERROR: Network leibniz-network declared as external, but could not be found
```

**Solution**:
```powershell
# Create network manually
docker network create leibniz-network

# Or recreate from docker-compose
docker-compose -f docker-compose.leibniz.yml down
docker-compose -f docker-compose.leibniz.yml up -d
```

### Issue: Import Error for redis.asyncio

**Symptoms**:
```python
ImportError: cannot import name 'asyncio' from 'redis'
```

**Solution**:
```powershell
# Install correct version
pip install --upgrade "redis[asyncio]>=5.0.0"

# Verify installation
python -c "import redis.asyncio; print(redis.__version__)"
```

### Issue: Environment Variables Not Loaded

**Symptoms**:
```
Using default Redis host: localhost (expected: redis)
```

**Solution**:

1. **Check .env.leibniz exists** in project root
2. **Load environment manually** (if not using dotenv loader):
   ```python
   from dotenv import load_dotenv
   load_dotenv(".env.leibniz")
   ```
3. **Verify loading**:
   ```powershell
   python -c "import os; from dotenv import load_dotenv; load_dotenv('.env.leibniz'); print(os.getenv('LEIBNIZ_REDIS_HOST'))"
   ```

### Debug Commands Reference

```powershell
# View Redis logs
docker logs leibniz-redis

# Follow Redis logs in real-time
docker logs -f leibniz-redis

# Check Redis container health
docker inspect leibniz-redis | Select-String -Pattern "Health"

# Test Redis connectivity from container
docker exec leibniz-redis redis-cli ping

# View all keys in Redis
docker exec leibniz-redis redis-cli KEYS "*"

# Monitor Redis commands in real-time
docker exec leibniz-redis redis-cli MONITOR

# Check Redis memory usage
docker exec leibniz-redis redis-cli INFO memory

# Restart Redis container
docker-compose -f docker-compose.leibniz.yml restart redis
```

---

## 🚀 Next Steps

### Phase 2: STT/VAD Service Extraction ✅ COMPLETE

**Goal**: Extract speech-to-text and voice activity detection into standalone service.

**Source Files**:
- `leibniz_agent/leibniz_stt.py`
- `leibniz_agent/leibniz_vad.py`

**Completed Structure**:
```
leibniz_agent/services/stt_vad/
├── __init__.py
├── app.py                     # FastAPI app with WebSocket
├── vad_manager.py             # Voice activity detection logic
├── gemini_client.py           # Gemini Live session management
├── config.py                  # VAD configuration
├── utils.py                   # Shared utilities
├── Dockerfile
└── tests/
```

**API Endpoints**:
- `WebSocket /api/v1/transcribe/stream`: Real-time audio streaming with transcription
- `GET /health`: Service health check

**Status**: ✅ Service deployed and tested in Docker (port 8001)

---

### Phase 3: Intent Classification Service ✅ COMPLETE

**Goal**: Extract intent classification into stateless HTTP service with Redis caching.

**Source File**:
- `leibniz_agent/leibniz_intent_parser.py`

**Completed Structure**:
```
leibniz_agent/services/intent/
├── __init__.py
├── app.py                     # FastAPI HTTP REST API
├── intent_classifier.py       # Two-tier classification logic
├── patterns.py                # Pattern definitions and system prompt
├── config.py                  # Service configuration
├── requirements.txt
├── Dockerfile
├── README.md                  # Comprehensive documentation
└── tests/
    ├── __init__.py
    └── test_classifier.py     # 15+ integration tests
```

**API Endpoints**:
- `POST /api/v1/classify`: Classify user intent with caching
- `GET /health`: Service health with component status
- `GET /metrics`: Performance metrics (fast route %, confidence)
- `POST /admin/clear_cache`: Clear Redis cache

**Configuration**:

Add to your `.env.leibniz` file:

```bash
# Gemini API (required for LLM fallback)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-lite

# Intent Classification (optional - defaults shown)
LEIBNIZ_INTENT_PARSER_CONFIDENCE_THRESHOLD=0.8    # Fast route threshold
LEIBNIZ_INTENT_PARSER_GEMINI_TIMEOUT=5.0          # Gemini timeout (seconds)
LEIBNIZ_INTENT_PARSER_LOG_CLASSIFICATIONS=true    # Log all classifications
LEIBNIZ_INTENT_PARSER_FAST_ROUTE_TARGET=0.8       # Target fast route %

# Redis caching (optional - defaults shown)
INTENT_CACHE_TTL=1800  # 30 minutes
```

**Testing**:

```powershell
# Start service (local development)
cd leibniz_agent/services/intent
python app.py
# Service runs on http://localhost:8002

# Start service (Docker)
docker-compose -f docker-compose.leibniz.yml up -d intent

# Run tests
pytest leibniz_agent/services/intent/tests/test_classifier.py -v

# Test classification endpoint
curl -X POST http://localhost:8002/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I want to schedule an appointment"}'

# Check health
curl http://localhost:8002/health

# Get metrics
curl http://localhost:8002/metrics
```

**Expected Performance**:

- **Fast Route**: <50ms per classification (pattern matching)
- **Gemini Fallback**: 500-2000ms per classification (LLM API)
- **Cache Hit**: <5ms per classification
- **Fast Route Target**: 80-90% of requests

**Key Features**:

1. **Two-Tier Classification**:
   - Fast pattern matching (regex + keywords) for 80%+ requests
   - Gemini LLM fallback for complex cases

2. **Rich Context Extraction**:
   - `user_goal`: Clear sentence describing user's objective
   - `key_entities`: Extracted entities (department, datetime, program, etc.)
   - `extracted_meaning`: Paraphrased clean version of user query

3. **Distributed Caching**:
   - Redis-backed 30-minute TTL cache
   - Shared across multiple service instances
   - Graceful degradation if Redis unavailable

4. **Performance Monitoring**:
   - `/metrics` endpoint tracks fast route percentage
   - `/health` endpoint shows component status
   - Structured logging for all classifications

**Status**: ✅ Service deployed and ready for integration (port 8002)

**Documentation**: See `leibniz_agent/services/intent/README.md` for comprehensive guide

---

### Phase 6: Appointment FSM Service ✅ COMPLETE

**Goal**: Extract appointment booking FSM into standalone service with Redis session persistence.

**Source File**:
- `leibniz_agent/leibniz_appointment_fsm.py`

**Completed Structure**:
```
leibniz_agent/services/appointment/
├── __init__.py
├── app.py                     # FastAPI HTTP REST API
├── fsm_manager.py             # 17-state FSM logic with validation
├── models.py                  # Pydantic models and enums
├── config.py                  # Service configuration
├── validation.py              # Input validation utilities
├── requirements.txt
├── Dockerfile
├── tests/
│   ├── __init__.py
│   ├── test_fsm_flow.py       # FSM logic and validation tests
│   └── test_api_integration.py # API endpoint tests
└── README.md                  # Service documentation
```

**API Endpoints**:
- `POST /api/v1/session/create`: Create new appointment session
- `POST /api/v1/session/{session_id}/process`: Process user input
- `GET /api/v1/session/{session_id}/status`: Get session status
- `DELETE /api/v1/session/{session_id}`: Delete session
- `GET /health`: Service health check
- `GET /metrics`: Session statistics
- `POST /admin/clear_sessions`: Admin session cleanup

**Configuration**:

Add to your `.env.leibniz` file:

```bash
# Appointment FSM Service
LEIBNIZ_APPOINTMENT_SERVICE_PORT=8005
APPOINTMENT_SESSION_TTL=1800          # 30 minutes session timeout
APPOINTMENT_MAX_RETRIES=3             # Max retries for invalid inputs
APPOINTMENT_MAX_CONFIRMATION_ATTEMPTS=2  # Max confirmation attempts
```

**Testing**:

```powershell
# Start service (local development)
cd leibniz_agent/services/appointment
python app.py
# Service runs on http://localhost:8005

# Start service (Docker)
docker-compose -f docker-compose.leibniz.yml up -d appointment

# Run tests
pytest leibniz_agent/services/appointment/tests/ -v

# Test complete appointment flow
curl -X POST http://localhost:8005/api/v1/session/create
# Returns: {"session_id": "uuid", "state": "AWAITING_NAME", "response": "..."}

# Process name input
curl -X POST http://localhost:8005/api/v1/session/{session_id}/process \
  -H "Content-Type: application/json" \
  -d '{"user_input": "John Doe"}'

# Continue through full flow: email, phone, department, appointment type, datetime, confirmation

# Check health
curl http://localhost:8005/health

# Get metrics
curl http://localhost:8005/metrics
```

**Key Features**:

1. **17-State FSM**: Complete appointment booking conversation flow
   - Name → Email → Phone → Department → Appointment Type → DateTime → Confirmation

2. **Redis Session Persistence**: 30-minute TTL sessions with automatic cleanup

3. **Comprehensive Validation**: Name, email, phone, datetime parsing with retry logic

4. **Natural Language Processing**: Flexible datetime parsing ("tomorrow at 2pm", "next Monday")

5. **Department/Appointment Types**: Predefined lists with fuzzy matching

6. **Error Handling**: Graceful degradation, retry limits, session recovery

**Expected Performance**:
- **Session Creation**: <10ms
- **Input Processing**: 50-200ms (validation + FSM logic)
- **Session Retrieval**: <5ms (Redis cached)
- **Concurrent Sessions**: 1000+ simultaneous sessions

**Status**: ✅ Service deployed and ready for integration (port 8005)

---

### Phase 4-5,7: Remaining Services

Follow the same pattern for:
- **Phase 3**: Intent classification service
- **Phase 4**: RAG service
- **Phase 5**: TTS service
- **Phase 6**: Appointment FSM service
- **Phase 7**: Main orchestrator service

### Phase 8: Integration Tests

Create comprehensive test suite covering:
- Individual service functionality
- Inter-service communication
- End-to-end conversation flows
- Performance benchmarks
- Failure recovery scenarios

---

## 📚 Additional Resources

- **Cloud Transformation Document**: [`leibniz_agent/docs/Cloud Transformation.md`](../docs/Cloud%20Transformation.md)
- **Main Agent Documentation**: [`leibniz_agent/README.md`](../README.md)
- **Redis Documentation**: [Redis Official Docs](https://redis.io/docs/)
- **FastAPI Documentation**: [FastAPI Official Docs](https://fastapi.tiangolo.com/)
- **Docker Compose Reference**: [Docker Compose Docs](https://docs.docker.com/compose/)

---

## 🤝 Contributing

When adding new services:

1. **Follow Shared Patterns**: Use utilities from `leibniz_agent/services/shared/`
2. **Environment Variables**: Prefix with `LEIBNIZ_`
3. **Health Checks**: Implement `/health` endpoint
4. **Docker**: Include Dockerfile and update `docker-compose.leibniz.yml`
5. **Tests**: Write unit tests, integration tests, and update E2E tests
6. **Documentation**: Update this README with service details

---

**Status**: Phase 6 Complete ✅  
**Last Updated**: October 31, 2025  
**Maintainer**: Leibniz Agent Team
