# Leibniz TTS Microservice

Text-to-Speech microservice with multi-provider support, intelligent caching, and automatic fallback.

## Overview

Standalone TTS service extracted from the monolithic Leibniz agent for improved scalability, maintainability, and independent deployment.

**Key Features:**
- 🎙️ **5 TTS Providers**: Google Cloud, ElevenLabs, Gemini Live, XTTS Local, Mock
- 🔄 **Automatic Fallback**: Primary → Fallback provider on failure
- ♻️ **Smart Retry**: Exponential backoff (1s, 2s, 4s delays)
- 💾 **MD5 Caching**: LRU cleanup, 500 entry limit
- 😊 **Emotion Support**: Provider-specific emotion modulation
- 🐳 **Docker Ready**: Multi-stage build, health checks

## Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment (.env file)
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-credentials.json
ELEVENLABS_API_KEY=your_elevenlabs_api_key
LEIBNIZ_TTS_PROVIDER=gemini
LEIBNIZ_TTS_FALLBACK_PROVIDER=google
LEIBNIZ_TTS_CACHE_ENABLED=true

# 3. Run service
python app.py
# Or with uvicorn:
uvicorn app:app --host 0.0.0.0 --port 8004 --reload
```

### Docker

```bash
# Build image
docker build -t leibniz-tts:latest .

# Run container
docker run -d \
  --name leibniz-tts \
  -p 8004:8004 \
  -v $(pwd)/audio_cache:/app/audio_cache \
  -e GEMINI_API_KEY=your_key \
  leibniz-tts:latest
```

### Docker Compose

```yaml
# docker-compose.leibniz.yml
services:
  tts:
    build: ./leibniz_agent/services/tts
    ports:
      - "8004:8004"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
      - LEIBNIZ_TTS_PROVIDER=gemini
      - LEIBNIZ_TTS_FALLBACK_PROVIDER=google
    volumes:
      - ./audio_cache:/app/audio_cache
      - ./credentials.json:/app/credentials.json:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8004/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## API Documentation

### POST /api/v1/synthesize

Synthesize text to speech.

**Request:**
```json
{
  "text": "Hello! Welcome to Leibniz University.",
  "emotion": "helpful",
  "voice": null,
  "language": "en-US"
}
```

**Response:**
```json
{
  "success": true,
  "cache_key": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
  "audio_url": "/api/v1/audio/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
  "duration": 3.5,
  "cached": false,
  "provider": "gemini",
  "elapsed": 1.234
}
```

### GET /api/v1/audio/{cache_key}

Retrieve cached audio file.

**Response:** WAV audio file (audio/wav)

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "providers_available": ["gemini", "google", "elevenlabs"],
  "cache_enabled": true,
  "cache_stats": {
    "size": 247,
    "hits": 1832,
    "misses": 468,
    "hit_rate": 0.7965
  },
  "total_requests": 2300
}
```

## Providers

### Google Cloud TTS
- **Pros**: Stable, 200+ voices, SSML support
- **Cons**: Requires service account credentials
- **Emotion**: Pitch/speaking_rate modulation
- **Setup**: `GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json`

### ElevenLabs
- **Pros**: Premium quality, 50+ voices, streaming
- **Cons**: Paid API, no direct emotion support
- **Emotion**: Via voice selection (not parameters)
- **Setup**: `ELEVENLABS_API_KEY=your_key`

### Gemini Live
- **Pros**: Free tier, emotion-aware, voice characters
- **Cons**: Preview API, occasional 500 errors
- **Emotion**: System prompts (6 emotions)
- **Setup**: `GEMINI_API_KEY=your_key`

### XTTS Local (Optional)
- **Pros**: Voice cloning, no API costs, GPU acceleration
- **Cons**: Large dependency (~2GB torch), slower on CPU
- **Emotion**: Not supported (voice cloning only)
- **Setup**: Requires speaker sample WAV file

### Mock
- **Pros**: No dependencies, instant responses
- **Cons**: Silent audio (testing only)
- **Usage**: `MOCK_TTS=true`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LEIBNIZ_TTS_PROVIDER` | `gemini` | Primary provider (google, elevenlabs, gemini, xtts_local, auto, mock) |
| `LEIBNIZ_TTS_FALLBACK_PROVIDER` | `google` | Fallback provider (google, elevenlabs, gemini) |
| `LEIBNIZ_TTS_ENABLE_FALLBACK` | `true` | Enable automatic fallback on primary failure |
| `LEIBNIZ_TTS_CACHE_ENABLED` | `true` | Enable MD5-based caching |
| `LEIBNIZ_TTS_CACHE_DIR` | `/app/audio_cache` | Cache directory path |
| `LEIBNIZ_TTS_CACHE_MAX_SIZE` | `500` | Maximum cached entries (LRU cleanup) |
| `LEIBNIZ_TTS_TIMEOUT` | `30.0` | Synthesis timeout (seconds) |
| `LEIBNIZ_TTS_RETRY_ATTEMPTS` | `3` | Retry attempts per provider |
| `LEIBNIZ_TTS_SERVICE_PORT` | `8004` | Service port |
| `GEMINI_API_KEY` | - | Gemini API key (required for Gemini provider) |
| `GOOGLE_APPLICATION_CREDENTIALS` | - | Google service account JSON path |
| `ELEVENLABS_API_KEY` | - | ElevenLabs API key |
| `MOCK_TTS` | `false` | Enable mock provider (testing) |

### Emotion Types

| Emotion | Google | ElevenLabs | Gemini | XTTS |
|---------|--------|------------|--------|------|
| `helpful` | ✅ +0.05 pitch | ⚠️ Via voice | ✅ System prompt | ❌ |
| `excited` | ✅ +0.15 pitch, 1.2x rate | ⚠️ Via voice | ✅ System prompt | ❌ |
| `calm` | ✅ 0.0 pitch, 0.95x rate | ⚠️ Via voice | ✅ System prompt | ❌ |
| `professional` | ✅ 0.0 pitch | ⚠️ Via voice | ✅ System prompt | ❌ |
| `neutral` | ✅ Default | ✅ Default | ✅ Default | ✅ Default |

## Performance

### Typical Latency

- **Cache Hit**: 1-5ms (200-600x faster)
- **Cache Miss (Gemini)**: 1-3s
- **Cache Miss (Google)**: 2-4s
- **Cache Miss (ElevenLabs)**: 1-2s
- **Cache Miss (XTTS GPU)**: 500ms-2s

### Expected Cache Hit Rates

- **Repeated Queries**: 80-95%
- **Similar Queries**: 40-60%
- **Unique Queries**: 0-10%

## Troubleshooting

### "No providers available"
- Check API keys are set (`GEMINI_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, etc.)
- Verify provider installation (`pip list | grep google-genai`)
- Enable mock mode for testing: `MOCK_TTS=true`

### "Synthesis failed: TimeoutError"
- Increase timeout: `LEIBNIZ_TTS_TIMEOUT=60.0`
- Check network connectivity
- Try different provider: `LEIBNIZ_TTS_PROVIDER=google`

### "Cache file missing"
- Cache may have been cleared (LRU cleanup at 500 entries)
- Check cache directory permissions
- Verify `LEIBNIZ_TTS_CACHE_DIR` is writable

### Gemini "500 Internal Server Error"
- Use Flash model (more stable): `LEIBNIZ_TTS_GEMINI_MODEL=gemini-2.5-flash-preview-tts`
- Enable fallback: `LEIBNIZ_TTS_ENABLE_FALLBACK=true`
- Retry automatically handled (3 attempts)

## Testing

```bash
# Health check
curl http://localhost:8004/health

# Synthesize text
curl -X POST http://localhost:8004/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "emotion": "helpful"}'

# Retrieve audio
curl http://localhost:8004/api/v1/audio/{cache_key} -o audio.wav

# Play audio (Linux/macOS)
aplay audio.wav  # or: ffplay audio.wav
```

## Integration Examples

### Python Client

```python
import requests

# Synthesize
response = requests.post(
    "http://localhost:8004/api/v1/synthesize",
    json={
        "text": "Welcome to Leibniz University!",
        "emotion": "helpful"
    }
)
result = response.json()

# Download audio
audio_url = result['audio_url']
audio_response = requests.get(f"http://localhost:8004{audio_url}")
with open("output.wav", "wb") as f:
    f.write(audio_response.content)
```

### Leibniz Agent Integration

```python
from leibniz_agent.services.tts import synthesize_speech

# Direct function call (not via HTTP)
audio_file = await synthesize_speech(
    text="Hello!",
    emotion="helpful"
)

# Or use HTTP client
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8004/api/v1/synthesize",
        json={"text": "Hello!", "emotion": "helpful"}
    )
    result = response.json()
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│   POST /api/v1/synthesize | GET /api/v1/audio | GET /health │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    TTSSynthesizer                            │
│  • Provider selection (primary → fallback)                   │
│  • Retry logic (3 attempts, exponential backoff)             │
│  • Cache integration (check → synthesize → cache)            │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┬──────────────┐
         ▼               ▼               ▼              ▼
    ┌────────┐     ┌──────────┐    ┌─────────┐    ┌────────┐
    │ Google │     │ElevenLabs│    │ Gemini  │    │  XTTS  │
    │  TTS   │     │   TTS    │    │  Live   │    │ Local  │
    └────────┘     └──────────┘    └─────────┘    └────────┘
```

## License

Part of SINDH Orchestra Complete project.

## Support

For issues, questions, or contributions, please refer to the main project repository.
