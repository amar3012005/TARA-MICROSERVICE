# 🚀 StateManager Orchestrator - Complete Architecture Guide

> **Enterprise Real-Time Voice Agent Pipeline with State Machines**  
> Ultra-low latency, stateful, interrupt-aware, production-ready

---

## 🎯 Executive Summary

Your current architecture (STT → Intent → RAG → TTS) is **linear and blocking**. This means:
- ❌ User speaks → Waits for STT → Waits for Intent → Waits for RAG → Waits for LLM → Waits for TTS
- ❌ Cannot handle interruptions (barge-in)
- ❌ No conversation context across turns
- ❌ ~2-3 second latency per turn

**The Solution: StateManager Orchestrator (Port 8004)**

This microservice introduces:
- ✅ **State Machine FSM** - IDLE → LISTENING → THINKING → SPEAKING → INTERRUPT
- ✅ **Parallel Processing** - Intent + RAG run concurrently (save 50-100ms)
- ✅ **Barge-in Detection** - User can interrupt TTS immediately
- ✅ **Redis-backed State** - Distributed, scalable, persistent
- ✅ **Sub-500ms Latency** - Human-imperceptible delays

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Browser (WebRTC/FastRTC)                                     │
│ └─ Audio Stream → WebSocket                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ STT-VAD Service (8001)                                       │
│ └─ Transcription Fragments (PARTIAL/FINAL)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ StateManager (8004) 🧠 │  ← NEW ORCHESTRATOR
        │ ┌──────────────────┐   │
        │ │ FSM State        │   │
        │ │ IDLE → LISTENING │   │
        │ │ → THINKING       │   │
        │ │ → SPEAKING       │   │
        │ │ → INTERRUPT      │   │
        │ └──────────────────┘   │
        └────┬─────────────┬─────┘
             │             │
      ┌──────▼──┐    ┌────▼──────┐
      │ Intent  │    │ RAG       │  (Parallel)
      │ (8002)  │    │ (8003)    │
      └──────┬──┘    └────┬──────┘
             │            │
             └──────┬─────┘
                    ▼
          ┌───────────────────┐
          │ LLM (Groq/Gemini) │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │ TTS (8005) 🔊      │
          └─────────┬─────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Browser (Audio Out)   │
        └───────────────────────┘

Redis (6379) - Persistent State, Session Cache, Barge-in Signals
```

---

## 🔄 State Machine Details

### States & Transitions

| State | Triggered By | Actions | Next State(s) | Latency |
|-------|--------------|---------|--------------|---------|
| **IDLE** | WebSocket Connect | Init session, load Redis state | LISTENING | 50ms |
| **LISTENING** | VAD Activity | Buffer STT fragments | THINKING (on end-of-turn) | 90ms |
| **THINKING** | End-of-turn (500ms silence) | Parallel: Intent + RAG + LLM | SPEAKING | 200ms |
| **SPEAKING** | TTS stream ready | Stream audio chunks to browser | IDLE or INTERRUPT | 75ms |
| **INTERRUPT** | User barge-in (VAD activity during SPEAKING) | Cancel TTS, reset buffers | LISTENING | 100ms |

### State Transition Diagram

```
    ┌─────────────────────────────────────────┐
    │           IDLE (🟢)                      │
    │    [Init Session, Load State]            │
    └────────────────┬────────────────────────┘
                     │ (WebSocket Connect)
                     ▼
    ┌─────────────────────────────────────────┐
    │         LISTENING (🔵)                   │
    │    [Buffer STT, Wait for End-of-Turn]    │
    └────────────────┬────────────────────────┘
                     │ (500ms silence)
                     ▼
    ┌─────────────────────────────────────────┐
    │         THINKING (🟡)                    │
    │  [Parallel Intent+RAG, Call LLM]         │
    └────────────────┬────────────────────────┘
                     │ (Response ready)
                     ▼
    ┌─────────────────────────────────────────┐
    │         SPEAKING (🔴)                    │
    │    [Stream TTS Audio, Wait for Turn]     │
    └────────────────┬─────────────┬──────────┘
                     │ (TTS done)  │ (Barge-in)
                  IDLE         INTERRUPT (⚡)
                                 │
                    ┌────────────┘
                    │ (Reset buffers)
                    ▼
                 LISTENING
```

---

## 📁 File Structure (Microservice)

```
orchestrator/
├── app.py                          # FastAPI WebSocket endpoint
├── state_manager.py                # Core FSM engine
├── conversation_state.py           # Redis-backed state persistence
├── parallel_pipeline.py            # Intent+RAG+LLM orchestration
├── interruption_handler.py         # Barge-in detection & handling
├── tts_proxy.py                    # TTS integration (ElevenLabs/Google)
├── config.py                       # Configuration
├── models.py                       # Pydantic schemas
├── Dockerfile                      # Container definition
├── requirements.txt                # Dependencies
└── tests/
    ├── test_state_transitions.py   # Unit tests for FSM
    ├── test_parallel_execution.py  # Integration tests
    └── test_latency.py             # Performance tests
```

---

## 🔌 API & WebSocket Protocol

### Endpoint: `ws://localhost:8004/orchestrate`

#### **Client → Orchestrator Messages**

```json
// Message Type 1: STT Fragment (from STT service)
{
  "type": "stt_fragment",
  "session_id": "session_123",
  "text": "what are admission",
  "is_final": false,
  "timestamp": 1704067200.123
}

// Message Type 2: End of Turn (STT service)
{
  "type": "vad_end",
  "session_id": "session_123",
  "confidence": 0.95
}

// Message Type 3: Barge-in Signal (STT service)
{
  "type": "user_speaking",
  "session_id": "session_123",
  "timestamp": 1704067205.456
}
```

#### **Orchestrator → Client Messages**

```json
// State Change Notification
{
  "type": "state_change",
  "session_id": "session_123",
  "from": "LISTENING",
  "to": "THINKING",
  "timestamp": 1704067200.789,
  "latency_ms": 45
}

// Processing Progress
{
  "type": "processing",
  "session_id": "session_123",
  "stage": "intent_classification",
  "progress": 0.33,
  "details": {
    "intent": "query_admissions",
    "confidence": 0.92
  }
}

// Response Ready (TTS Metadata)
{
  "type": "response_ready",
  "session_id": "session_123",
  "text": "Admission requirements include...",
  "duration_ms": 4200,
  "audio_format": "pcm16"
}

// Audio Chunk (Binary)
// Sent as binary WebSocket frame

// Turn Complete
{
  "type": "turn_complete",
  "session_id": "session_123",
  "turn_number": 3,
  "latency_breakdown": {
    "stt_ms": 150,
    "thinking_ms": 200,
    "tts_ms": 300,
    "total_ms": 650
  }
}
```

---

## ⏱️ Latency Breakdown (Target: 465ms E2E)

| Stage | Latency | Method | Notes |
|-------|---------|--------|-------|
| **STT Fragment → Orchestrator** | 50ms | WebSocket | Direct streaming |
| **Intent Classification (L1 Regex)** | 5ms | In-memory | Cached patterns |
| **RAG Semantic Search** | 80ms | FAISS + Vector DB | Parallel w/ Intent |
| **Intent+RAG Merge** | 15ms | Python dict merge | Negligible |
| **LLM Token Generation** | 100ms | Groq (Llama4-Maverick) | Streaming tokens |
| **TTS Synthesis (First Chunk)** | 75ms | ElevenLabs Flash | Chunked synthesis |
| **Orchestrator Overhead** | 40ms | State transitions + queue ops | Redis atomic ops |
| **Network RTT** | ~0ms (local) | Docker bridge | Negligible |
| **Browser Audio Playback** | ~100ms | Audio buffer | Human perception |
| | | | |
| **TOTAL (Parallel Path)** | **465ms** | Optimized | Sub-500ms! |

---

## 🛠️ Implementation Roadmap (5 Phases)

### Phase 1: Core State Machine (1 day)
- `state_manager.py` - FSM with 5 states
- Redis persistence
- Unit tests

### Phase 2: WebSocket Orchestrator (1 day)
- `app.py` - FastAPI WebSocket endpoint
- Session management
- Message routing

### Phase 3: Parallel Pipeline (1 day)
- `parallel_pipeline.py` - asyncio.gather() for Intent+RAG
- LLM orchestration (Groq/Gemini)
- Response formatting

### Phase 4: Interruption Handling (1 day)
- `interruption_handler.py` - Barge-in detection
- TTS cancellation logic
- Graceful state resets

### Phase 5: TTS Integration (1 day)
- `tts_proxy.py` - ElevenLabs/Google TTS
- Audio chunking & streaming
- Production deployment

---

## 🚀 Production Checklist

- [ ] **Latency**: All stages < 200ms
- [ ] **Reliability**: 99.5% uptime (no message loss)
- [ ] **Scalability**: 1000+ concurrent sessions
- [ ] **Monitoring**: Prometheus metrics + Grafana dashboards
- [ ] **Observability**: Distributed tracing (Jaeger)
- [ ] **Health**: `/health` endpoint with detailed status
- [ ] **Load Testing**: Apache JMeter / Locust
- [ ] **Failover**: Redis Sentinel for HA
- [ ] **Logging**: Structured logs (JSON) to ELK stack
- [ ] **Security**: TLS for WebSocket, API key validation

---

## 📊 Deployment: Docker Compose

```yaml
version: '3.9'

services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  orchestrator:
    build: ./orchestrator
    ports: ["8004:8004"]
    depends_on: [redis]
    environment:
      - REDIS_URL=redis://redis:6379
      - LLM_API_KEY=${GROQ_API_KEY}

  tts-service:
    build: ./tts
    ports: ["8005:8005"]
    environment:
      - ELEVENLABS_API_KEY=${ELEVENLABS_KEY}

  # Existing services...
  stt-vad-service: ...
  intent-service: ...
  rag-service: ...
```

---

## 🎓 Real-World Examples (Inspiration)

1. **LiveKit Agents** - Built-in state management for voice agents
2. **Deepgram Voice Agent API** - Parallel STT+Intent+TTS
3. **ElevenLabs Agents** - Conversation state tracking
4. **Groq LPU** - Sub-100ms LLM inference
5. **VAPI** - Barge-in detection + interruption handling

---

## 📈 Performance Metrics

After implementation, expect:
- **Latency**: <500ms (end-to-end)
- **Throughput**: 1000+ concurrent sessions
- **Availability**: 99.5%
- **Error Rate**: <0.1%
- **Resource Usage**: 2GB RAM, 2 vCPU per 100 sessions

---

## 🔗 Next Steps

1. **Review** this guide (30 min)
2. **Implement** Phase 1 (state_manager.py) - 4 hours
3. **Test** with unit tests - 2 hours
4. **Deploy** to Docker - 1 hour
5. **Load test** with Locust - 2 hours

**Total**: ~9 hours to production-ready orchestrator

---

**Ready to build? Start with Phase 1 → `state_manager.py`**