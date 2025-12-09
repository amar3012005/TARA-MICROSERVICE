# TARA Microservice Developer Notes

## Quick Commands

### Docker Build & Run
```bash
cd /home/prometheus/leibniz_agent/TARA-MICROSERVICE

# Start RAG service
docker --context desktop-linux compose -f docker-compose-tara-task.yml up -d rag

# List running containers
docker ps --format "table {{.Names}}\t{{.Ports}}"

# Stop all containers
docker kill $(docker ps -q)

# Remove all containers
docker rm -f $(docker ps -aq)
```

### ElevenLabs Setup
```bash
# Set your ElevenLabs API key
export ELEVENLABS_API_KEY="sk_b52c643178d624ed09d8b52da3554ee7ff3096d02e55299b"

# Start with ElevenLabs TTS
docker compose --profile elevenlabs up -d tts-labs-service
```

---

## ElevenLabs TTS Production Path

### Overview

The orchestrator supports two TTS streaming modes:

1. **Standard Mode (Sarvam/Generic TTS)**: Uses `config.tts_service_url` to connect to any TTS service with the `/api/v1/stream` WebSocket endpoint.

2. **ElevenLabs Direct Mode**: Uses a dedicated `ElevenLabsTTSClient` that connects directly to the `tts-labs` service with prewarm support for ultra-low latency.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR                                  │
│                                                                         │
│  ┌──────────────┐     ┌───────────────────┐                            │
│  │ VAD Detected │────>│ prewarm_elevenlabs│ (if USE_ELEVENLABS_TTS)    │
│  └──────────────┘     │ _tts()            │                            │
│                       └───────────────────┘                            │
│                              │                                         │
│                              ▼                                         │
│                       ┌───────────────────┐                            │
│                       │ ElevenLabsTTSClient│                           │
│                       │ (prewarmed)        │                           │
│                       └───────────────────┘                            │
│                              │                                         │
│  ┌──────────────┐           │                                         │
│  │ RAG Response │           │                                         │
│  │ (text chunks)│           ▼                                         │
│  └──────────────┘     ┌───────────────────┐                            │
│         │             │stream_tts_from_   │                            │
│         └────────────>│generator()        │                            │
│                       └───────────────────┘                            │
│                              │                                         │
│                              ▼                                         │
│                       ┌───────────────────┐                            │
│                       │ Audio to Client   │                            │
│                       │ (WebSocket/LiveKit)│                           │
│                       └───────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           TTS-LABS SERVICE                             │
│                                                                         │
│  ┌───────────────────┐     ┌───────────────────┐                       │
│  │ /api/v1/stream    │────>│ ElevenLabsStream  │──> ElevenLabs API    │
│  │ WebSocket         │     │ Manager           │                       │
│  └───────────────────┘     └───────────────────┘                       │
│                                                                         │
│  ┌───────────────────┐                                                 │
│  │ /fastrtc          │ (OPTIONAL - Developer preview only)             │
│  │ Gradio UI         │                                                 │
│  └───────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Configuration

Set these environment variables in the orchestrator to enable ElevenLabs direct streaming:

```bash
# Enable ElevenLabs direct streaming
USE_ELEVENLABS_TTS=true

# TTS-Labs service URL
ELEVENLABS_TTS_URL=http://tara-task-tts-labs:8006

# Prewarm connection when VAD detects speech (default: true)
ELEVENLABS_PREWARM_ON_VAD=true

# Streaming mode (continuous for ElevenLabs, buffered for Sarvam)
TTS_STREAMING_MODE=continuous
```

### Production Flow

1. **VAD Detection**: When user starts speaking, the STT service sends partial transcripts via Redis.

2. **Prewarm Trigger**: The orchestrator triggers `prewarm_elevenlabs_tts()` when it receives the first partial STT event. This establishes the WebSocket connection to `tts-labs`, which in turn connects to ElevenLabs.

3. **RAG Processing**: Meanwhile, the orchestrator processes the user's query through RAG.

4. **Text Streaming**: As RAG generates response text, chunks are immediately sent to the prewarmed ElevenLabs connection via `stream_text_to_audio()`.

5. **Audio Delivery**: Audio chunks are yielded as they arrive and forwarded to the client immediately.

### Key Files

| File | Purpose |
|------|---------|
| `orchestrator/eleven_tts_client.py` | ElevenLabs TTS WebSocket client with prewarm support |
| `orchestrator/app.py` | Main orchestrator with `stream_tts_from_generator()` |
| `orchestrator/config.py` | Configuration including `use_elevenlabs_tts` |
| `TTS_LABS/app.py` | TTS-Labs service with ElevenLabs integration |
| `TTS_LABS/elevenlabs_manager.py` | ElevenLabs WebSocket manager |

### FastRTC Note

**FastRTC (`/fastrtc`) is for developer preview only.**

- The production flow does **NOT** depend on FastRTC.
- FastRTC is optionally mounted on the `tts-labs` service for testing/visualization.
- The orchestrator never calls `/fastrtc` directly - it uses `/api/v1/stream`.

### Latency Characteristics

| Scenario | Expected Latency |
|----------|-----------------|
| Pre-warmed first audio | <150ms |
| Cold start first audio | 300-500ms |
| Prewarm connection | ~100-200ms |

### Configuration Knobs

In `TTS_LABS/elevenlabs_manager.py`:

```python
generation_config = {
    "chunk_length_schedule": [50, 120, 250, 290]  # Aggressive first chunk
}
```

- `chunk_length_schedule`: Controls how many characters must buffer before audio generation. Lower first value = faster first chunk.
- `try_trigger_generation`: When `true`, flushes the buffer immediately for lower latency.

### Testing

Run the test script to validate latency:

```bash
# From orchestrator directory
python test_eleven_tts_direct.py --tts-url http://localhost:8006
```

This tests:
1. Single text synthesis
2. Streaming with prewarm
3. Streaming without prewarm (cold start)

### Troubleshooting

**Issue: High first-audio latency**
- Ensure `USE_ELEVENLABS_TTS=true` is set
- Check that prewarm is triggered (look for "ElevenLabs prewarm triggered" in logs)
- Verify `tts-labs` service is running and healthy

**Issue: Audio not streaming**
- Check `ELEVENLABS_API_KEY` is set correctly in `docker-compose-tara-task.yml`
- Verify WebSocket connection to `tts-labs` (check orchestrator logs for connection errors)

**Issue: Prewarm timeout**
- Increase `ELEVENLABS_TTS_URL` network timeout
- Check network connectivity between orchestrator and `tts-labs`