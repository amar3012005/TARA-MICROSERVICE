# TTS_LABS - ElevenLabs Ultra-Low Latency TTS Microservice

Ultra-low latency text-to-speech streaming service using ElevenLabs WebSocket API. Configurable voice and model selection via environment variables.

## Features

- **Ultra-Low Latency**: < 150ms time-to-first-audio using `stream-input` WebSocket API
- **Continuous Streaming**: Bidirectional WebSocket for real-time text-to-audio
- **Sentence Synthesis**: Compatible with orchestrator's sentence-based workflow
- **FastRTC Preview**: Browser-based audio preview via WebRTC
- **Audio Caching**: MD5-based caching for frequently used phrases

## Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| First Chunk Latency | < 150ms | 75-150ms* |
| Model | Configurable | Via ELEVENLABS_MODEL_ID |
| Output Format | Configurable | Via ELEVENLABS_OUTPUT_FORMAT |

*Latency depends on selected model and network conditions

## Configuration

TTS_LABS is fully configurable via environment variables. No hardcoded defaults - all voice and model settings must be provided externally.

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ELEVENLABS_API_KEY` | Your ElevenLabs API key | `sk_...` |
| `ELEVENLABS_VOICE_ID` | Voice ID from ElevenLabs dashboard | `21m00Tcm4TlvDq8ikWAM` |
| `ELEVENLABS_MODEL_ID` | TTS model to use | `eleven_turbo_v2_5` |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ELEVENLABS_LATENCY_OPTIMIZATION` | `4` | Latency vs quality (0-4) |
| `ELEVENLABS_OUTPUT_FORMAT` | `pcm_24000` | Audio format |
| `ELEVENLABS_STABILITY` | `0.5` | Voice stability (0.0-1.0) |
| `ELEVENLABS_SIMILARITY_BOOST` | `0.75` | Voice similarity (0.0-1.0) |

## Quick Start

### 1. Set Environment Variables

```bash
export ELEVENLABS_API_KEY="your-api-key-here"
export ELEVENLABS_VOICE_ID="your_voice_id_here"  # Get from ElevenLabs dashboard
export ELEVENLABS_MODEL_ID="eleven_turbo_v2_5"   # Or other model
```

### 2. Run with Docker Compose

```bash
# Start TTS_LABS with the elevenlabs profile
docker compose --profile elevenlabs up -d tts-labs-service

# Or run the full stack with ElevenLabs TTS
docker compose --profile elevenlabs up -d
```

### 3. Configure Orchestrator to Use TTS_LABS

Set these environment variables in orchestrator:

```bash
TTS_SERVICE_URL=http://tts-labs-service:8006
TTS_STREAMING_MODE=continuous  # Bypass SmartBuffer for ultra-low latency
```

## API Endpoints

### WebSocket: `/api/v1/stream`

Connect with `?session_id=<your-session-id>`

**Sentence-based synthesis (compatible with orchestrator):**
```json
{"type": "synthesize", "text": "Hello, how can I help you today?"}
```

**Continuous streaming (ultra-low latency):**
```json
{"type": "stream_chunk", "text": "Hello"}
{"type": "stream_chunk", "text": ", how can I"}
{"type": "stream_chunk", "text": " help you?"}
{"type": "stream_end"}
```

### HTTP: `POST /api/v1/synthesize`

```json
{
  "text": "Hello, how can I help you today?",
  "voice": "your_configured_voice_id"
}
```

### Health: `GET /health`

Returns service status, model info, and latency optimization level.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ELEVENLABS_API_KEY` | (required) | ElevenLabs API key |
| `ELEVENLABS_VOICE_ID` | *Required* | Voice ID from ElevenLabs dashboard |
| `ELEVENLABS_MODEL_ID` | *Required* | TTS model (eleven_turbo_v2_5, eleven_multilingual_v2, etc.) |
| `ELEVENLABS_LATENCY_OPTIMIZATION` | `4` | Latency level (0-4) |
| `ELEVENLABS_OUTPUT_FORMAT` | `pcm_24000` | Audio format |
| `ELEVENLABS_STABILITY` | `0.5` | Voice stability |
| `ELEVENLABS_SIMILARITY_BOOST` | `0.75` | Voice similarity |
| `TTS_LABS_PORT` | `8006` | Service port |

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Orchestrator  │────▶│    TTS_LABS      │────▶│  ElevenLabs     │
│   (RAG output)  │     │  (WebSocket)     │     │  stream-input   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        │ Text chunks          │ stream_chunk           │ Audio chunks
        ▼                       ▼                        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                    FastRTC / Browser                         │
   └─────────────────────────────────────────────────────────────┘
```

## Switching Between TTS Providers

### Use Sarvam (Telugu, default):
```bash
TTS_SERVICE_URL=http://tts-sarvam-service:8025
TTS_STREAMING_MODE=buffered
```

### Use ElevenLabs (English, ultra-low latency):
```bash
TTS_SERVICE_URL=http://tts-labs-service:8006
TTS_STREAMING_MODE=continuous
```

## Pricing

ElevenLabs pricing varies by model and voice. Check ElevenLabs dashboard for current rates.

## Troubleshooting

### High Latency (> 300ms)
- Check `ELEVENLABS_LATENCY_OPTIMIZATION` is set to 4
- Ensure using `pcm_24000` output format (no MP3 decoding overhead)
- Verify network connectivity to ElevenLabs servers

### Connection Errors
- Verify `ELEVENLABS_API_KEY` is valid
- Check WebSocket firewall rules

### No Audio
- Confirm `ELEVENLABS_VOICE_ID` exists and is accessible
- Check service logs for API errors
