# STT Sarvam Docker Microservice Setup

## Overview

The `stt-sarvam` microservice provides ultra-low latency streaming speech-to-text using Sarvam AI's Saarika model. It supports real-time partial transcripts for RAG pre-LLM incremental retrieval.

## Docker Compose Configuration

The service is configured in `docker-compose-tara-task.yml`:

```yaml
stt-sarvam:
  container_name: tara-task-stt-sarvam
  ports:
    - "2002:8001"   # API + WebSocket
    - "2013:8001"   # FastRTC UI
```

## Ports

- **2002**: Main API and WebSocket endpoint (`/api/v1/transcribe/stream`)
- **2013**: FastRTC UI for browser-based testing (`/fastrtc`)

## Environment Variables

### Required

- `SARVAM_API_SUBSCRIPTION_KEY`: Your Sarvam AI API subscription key

### Optional Configuration

#### Sarvam Model Settings
- `LEIBNIZ_VAD_MODEL`: Model name (default: `saarika:v2.5`)
- `LEIBNIZ_VAD_LANGUAGE`: Language code (default: `unknown` for auto-detect)
- `LEIBNIZ_VAD_ENABLE_LANGUAGE_DETECTION`: Enable auto language detection (default: `true`)
- `LEIBNIZ_VAD_SAMPLE_RATE`: Audio sample rate (default: `16000`)

#### Streaming Parameters
- `SARVAM_HIGH_VAD_SENSITIVITY`: High VAD sensitivity (default: `true`)
- `SARVAM_VAD_SIGNALS`: Enable VAD signals (default: `true`)

#### Timeout Configuration
- `LEIBNIZ_VAD_TIMEOUT_INITIAL`: Initial timeout (default: `20.0` seconds)
- `LEIBNIZ_VAD_TIMEOUT_RETRY`: Retry timeout (default: `10.0` seconds)
- `LEIBNIZ_VAD_TIMEOUT_MAX`: Maximum timeout (default: `30.0` seconds)

## Quick Start

### 1. Set Environment Variable

```bash
export SARVAM_API_SUBSCRIPTION_KEY="your-api-key-here"
```

### 2. Build and Start

```bash
docker-compose -f docker-compose-tara-task.yml up --build stt-sarvam
```

### 3. Verify Health

```bash
curl http://localhost:2002/health
```

### 4. Test FastRTC UI

Navigate to: `http://localhost:2013/fastrtc`

## Switching Between STT Services

The orchestrator can use either `stt-vad` (Gemini) or `stt-sarvam` (Sarvam):

### Use Sarvam STT

```bash
export USE_SARVAM_STT=true
export STT_SERVICE_URL=http://tara-task-stt-sarvam:8001
docker-compose -f docker-compose-tara-task.yml up orchestrator
```

### Use Gemini STT (Default)

```bash
export USE_SARVAM_STT=false
export STT_SERVICE_URL=http://tara-task-stt-vad:8001
docker-compose -f docker-compose-tara-task.yml up orchestrator
```

## Features

### Ultra-Low Latency Streaming
- Real-time partial transcripts (< 500ms latency)
- Final transcripts after speech completion
- Server-side VAD (Voice Activity Detection)

### Redis Integration
- Publishes both partial and final transcripts to `leibniz:events:stt`
- Format: `{"text": "...", "session_id": "...", "is_final": true/false, ...}`
- Enables RAG pre-LLM incremental retrieval

### FastRTC Integration
- Browser-based audio streaming
- Direct microphone input
- Real-time transcript display

## API Endpoints

### WebSocket Transcription
```
ws://localhost:2002/api/v1/transcribe/stream?session_id=<session_id>
```

### Health Check
```
GET http://localhost:2002/health
```

### Metrics
```
GET http://localhost:2002/metrics
```

### Reset Sessions
```
POST http://localhost:2002/admin/reset_session
```

## Monitoring

### Check Logs
```bash
docker logs -f tara-task-stt-sarvam
```

### Monitor Redis Channel
```bash
redis-cli
SUBSCRIBE leibniz:events:stt
```

### View Metrics
```bash
curl http://localhost:2002/metrics | jq
```

## Troubleshooting

### Service Won't Start
1. Check `SARVAM_API_SUBSCRIPTION_KEY` is set
2. Verify Redis is running: `docker ps | grep redis`
3. Check logs: `docker logs tara-task-stt-sarvam`

### No Transcripts Received
1. Verify API key is valid
2. Check network connectivity to Sarvam API
3. Review logs for rate limit errors
4. Test with FastRTC UI: `http://localhost:2013/fastrtc`

### High Latency
1. Ensure `SARVAM_HIGH_VAD_SENSITIVITY=true`
2. Check network latency to Sarvam API
3. Verify audio sample rate is 16kHz
4. Monitor metrics for connection issues

## Architecture

```
Browser → FastRTC → FastRTCSTTHandler → VADManager → SarvamStreamingClient → Sarvam WebSocket API
                                              ↓
                                        Redis (leibniz:events:stt)
                                              ↓
                                    RAG Pre-LLM Incremental Retrieval
```

## Related Services

- **orchestrator**: Master controller (can use stt-sarvam)
- **rag**: Knowledge base service (consumes transcripts from Redis)
- **redis**: Message broker for inter-service communication
