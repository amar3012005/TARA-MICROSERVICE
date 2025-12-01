# Orchestrator Quick Start Guide

## Prerequisites

- Docker and Docker Compose installed
- Redis, STT-VAD, Intent, and RAG services running (via `docker-compose up`)

## Quick Start

### 1. Build the Orchestrator

```bash
cd services
docker-compose build orchestrator
```

### 2. Start All Services

```bash
docker-compose up -d
```

This will start:
- Redis (port 6379)
- STT-VAD Service (port 8001)
- Intent Service (port 8002)
- RAG Service (port 8003)
- **Orchestrator Service (port 8004)** ← NEW

### 3. Verify Health

```bash
curl http://localhost:8004/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "orchestrator",
  "active_sessions": 0,
  "redis_connected": true,
  "uptime_seconds": 123.45
}
```

### 4. Run Integration Tests

```bash
cd services/orchestrator
python test_integration.py
```

## Architecture Flow

```
User Browser
    ↓ (Audio)
STT-VAD Service (8001)
    ↓ (STT fragments)
Orchestrator (8004) 🧠
    ├─→ Intent Service (8002) ─┐
    └─→ RAG Service (8003) ────┼─→ Parallel Processing
                                ↓
                           Response
                                ↓
                           TTS Service (8005)
                                ↓
                           User Browser
```

## WebSocket API

### Connect

```javascript
const ws = new WebSocket('ws://localhost:8004/orchestrate?session_id=my_session_123');
```

### Send STT Fragment

```javascript
ws.send(JSON.stringify({
    type: "stt_fragment",
    session_id: "my_session_123",
    text: "What are admission requirements?",
    is_final: false,
    timestamp: Date.now() / 1000
}));
```

### Send End of Turn

```javascript
ws.send(JSON.stringify({
    type: "vad_end",
    session_id: "my_session_123",
    confidence: 0.95
}));
```

### Receive Response

```javascript
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "response_ready") {
        console.log("Response:", data.text);
        console.log("Latency:", data.thinking_ms, "ms");
    }
};
```

## State Machine

The orchestrator manages conversation state:

1. **IDLE** 🟢 - Initial state
2. **LISTENING** 🔵 - Buffering STT fragments
3. **THINKING** 🟡 - Processing Intent+RAG in parallel
4. **SPEAKING** 🔴 - Streaming TTS audio
5. **INTERRUPT** ⚡ - Handling barge-in

## Performance Targets

- **STT → Orchestrator**: 50ms
- **Parallel Intent+RAG**: 80ms (vs 130ms sequential)
- **LLM Generation**: 100ms
- **TTS First Chunk**: 75ms
- **Total E2E**: ~445ms

## Troubleshooting

### Orchestrator won't start

```bash
# Check logs
docker-compose logs orchestrator

# Verify Redis is running
docker-compose ps redis

# Verify other services are running
docker-compose ps
```

### WebSocket connection fails

```bash
# Check if orchestrator is listening
curl http://localhost:8004/health

# Check network connectivity
docker-compose exec orchestrator ping intent-service
docker-compose exec orchestrator ping rag-service
```

### High latency

```bash
# Check service response times
time curl -X POST http://localhost:8002/api/v1/classify -d '{"text":"test"}'
time curl -X POST http://localhost:8003/api/v1/query -d '{"query":"test"}'
```

## Next Steps

1. **Integrate with STT-VAD**: Connect STT-VAD service to send fragments to orchestrator
2. **Add TTS Service**: Replace mock TTS with real TTS service integration
3. **Add LLM**: Implement LLM integration for response generation
4. **Load Testing**: Test with multiple concurrent sessions

## References

- `README.md` - Full documentation
- `services/docs/ORCHESTRATOR_GUIDE.md` - Architecture guide
- `services/docs/ORCHESTRATOR_IMPLEMENTATION.md` - Implementation details



