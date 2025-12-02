# StateManager Orchestrator Service

Ultra-low latency voice agent orchestrator with FSM state management, parallel Intent+RAG processing, and barge-in detection.

## Overview

The orchestrator coordinates the conversation flow:
- **STT** → **Intent+RAG (parallel)** → **LLM** → **TTS**

Key features:
- ✅ Parallel Intent+RAG execution (saves ~50ms per turn)
- ✅ FSM state management (IDLE → LISTENING → THINKING → SPEAKING → INTERRUPT)
- ✅ Barge-in detection and TTS cancellation
- ✅ Redis-backed state persistence
- ✅ Sub-500ms E2E latency target

## Architecture

```
Browser (WebRTC) → STT-VAD (8001) → Orchestrator (8004) → Intent (8002) + RAG (8003) → TTS (8005)
                                                              ↓ (parallel)
                                                          Response → Browser
```

## API Endpoints

### WebSocket: `/orchestrate?session_id=<uuid>`

Main conversation endpoint. Accepts messages:
- `stt_fragment`: STT transcription fragments
- `vad_end`: End-of-turn signal
- `user_speaking`: Barge-in signal

### HTTP: `/health`

Health check endpoint.

### HTTP: `/metrics`

Prometheus metrics endpoint.

## Environment Variables

- `REDIS_URL`: Redis connection URL (default: `redis://redis:6379/0`)
- `INTENT_SERVICE_URL`: Intent service URL (default: `http://intent-service:8002`)
- `RAG_SERVICE_URL`: RAG service URL (default: `http://rag-service:8003`)
- `STT_SERVICE_URL`: STT service URL (default: `http://stt-vad-service:8001`)
- `TTS_SERVICE_URL`: TTS service URL (default: `http://tts-service:8005`)
- `SESSION_TTL_SECONDS`: Session TTL in seconds (default: 3600)
- `MAX_CONCURRENT_SESSIONS`: Max concurrent sessions (default: 1000)
- `LOG_LEVEL`: Logging level (default: INFO)

## Building and Running

### Docker Compose

The orchestrator is included in `services/docker-compose.yml`:

```bash
cd services
docker-compose up -d orchestrator
```

### Standalone

```bash
cd services/orchestrator
docker build -t leibniz-orchestrator:latest -f Dockerfile ..
docker run -p 8004:8004 \
  -e REDIS_URL=redis://localhost:6379/0 \
  -e INTENT_SERVICE_URL=http://localhost:8002 \
  -e RAG_SERVICE_URL=http://localhost:8003 \
  leibniz-orchestrator:latest
```

## Testing

### Health Check

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

### WebSocket Test

```python
import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://localhost:8004/orchestrate?session_id=test_001") as ws:
        # Receive connection message
        msg = await ws.recv()
        print("Connected:", json.loads(msg))
        
        # Send STT fragment
        await ws.send(json.dumps({
            "type": "stt_fragment",
            "session_id": "test_001",
            "text": "What are admission requirements?",
            "is_final": True,
            "timestamp": 1234567890.0
        }))
        
        # Send VAD end
        await ws.send(json.dumps({
            "type": "vad_end",
            "session_id": "test_001",
            "confidence": 0.95
        }))
        
        # Receive response
        response = await ws.recv()
        print("Response:", json.loads(response))

asyncio.run(test())
```

## State Machine

The orchestrator uses a 5-state FSM:

1. **IDLE**: Initial state, waiting for connection
2. **LISTENING**: Buffering STT fragments, waiting for end-of-turn
3. **THINKING**: Processing Intent+RAG in parallel, calling LLM
4. **SPEAKING**: Streaming TTS audio to client
5. **INTERRUPT**: Handling barge-in, resetting state

## Performance

Target latencies:
- STT → Orchestrator: 50ms
- Parallel Intent+RAG: 80ms
- LLM Token Generation: 100ms
- TTS First Chunk: 75ms
- **Total E2E: 445ms**

## References

- `services/docs/ORCHESTRATOR_GUIDE.md` - Architecture guide
- `services/docs/ORCHESTRATOR_IMPLEMENTATION.md` - Implementation details
- `services/docs/ORCHESTRATOR_DEPLOYMENT.md` - Deployment guide







