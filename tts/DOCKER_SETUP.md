# TTS Microservice Docker Setup

## Quick Start (Minimal Build)

### 1. Build the minimal Docker image (fast - ~30 seconds)
```bash
docker build -f leibniz_agent/services/tts/Dockerfile -t leibniz-tts-service:latest leibniz_agent/services/tts
```

Or using docker-compose:
```bash
docker-compose -f docker-compose.leibniz.yml build tts
```

### 2. Start the container
```bash
docker-compose -f docker-compose.leibniz.yml up tts -d
```

### 3. Install additional providers INSIDE the running container

#### Option A: Install all optional dependencies
```bash
docker exec -it <container_name> pip install -r requirements.txt
```

#### Option B: Install specific providers only

**Google Cloud TTS:**
```bash
docker exec -it <container_name> pip install google-cloud-texttospeech>=2.14.0 google-auth>=2.23.0
```

**ElevenLabs:**
```bash
docker exec -it <container_name> pip install elevenlabs>=0.2.26 soundfile>=0.12.1 numpy>=1.24.0
```

**Gemini Live:**
```bash
docker exec -it <container_name> pip install google-genai>=1.33.0
```

**XTTS Local (heavy - ~2GB):**
```bash
docker exec -it <container_name> pip install TTS>=0.22.0 torch>=2.1.0 torchaudio>=2.1.0 scipy>=1.11.0
```

### 4. Restart the service (if needed)
```bash
docker-compose -f docker-compose.leibniz.yml restart tts
```

## Testing the Service

### Health Check
```bash
curl http://localhost:8004/health
```

Expected response:
```json
{
  "status": "healthy",
  "providers_available": ["mock"],
  "cache_stats": {...}
}
```

### Synthesize Speech (Mock Provider - always works)
```bash
curl -X POST http://localhost:8004/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "emotion": "neutral"}'
```

Expected response:
```json
{
  "cache_key": "abc123...",
  "audio_url": "/api/v1/audio/abc123...",
  "provider": "mock"
}
```

### Get Audio File
```bash
curl http://localhost:8004/api/v1/audio/<cache_key> --output test.wav
```

## Environment Variables

Set these in `docker-compose.leibniz.yml` or `.env.leibniz`:

```bash
# Provider Selection
LEIBNIZ_TTS_PROVIDER=gemini              # Primary: google, elevenlabs, gemini, xtts_local, mock
LEIBNIZ_TTS_FALLBACK_PROVIDER=google     # Fallback provider

# API Keys
GEMINI_API_KEY=your_key_here
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
ELEVENLABS_API_KEY=your_key_here

# Cache Settings
LEIBNIZ_TTS_CACHE_ENABLED=true
LEIBNIZ_TTS_CACHE_DIR=/app/audio_cache
LEIBNIZ_TTS_CACHE_MAX_SIZE=500
LEIBNIZ_TTS_CACHE_TTL_DAYS=30
```

## Initial State

The minimal Docker image includes:
- ✅ FastAPI + Uvicorn (web server)
- ✅ Pydantic (data validation)
- ✅ python-dotenv (environment variables)
- ✅ FFmpeg (system-level audio processing)
- ✅ Mock TTS provider (always works, no dependencies)
- ❌ Google Cloud TTS (install manually)
- ❌ ElevenLabs (install manually)
- ❌ Gemini Live (install manually)
- ❌ XTTS Local (install manually, ~2GB)

## Why This Approach?

1. **Fast Builds**: Minimal image builds in ~30 seconds vs 20+ minutes
2. **Flexible**: Install only the providers you need
3. **Iteration**: Change provider dependencies without rebuilding image
4. **Development**: Test with mock provider immediately
5. **Production**: Install full dependencies once, persist in volume

## Production Deployment

For production, create a custom image with pre-installed dependencies:

```dockerfile
FROM leibniz-tts-service:latest

# Install all providers
RUN pip install --no-cache-dir -r requirements.txt

# Set production defaults
ENV LEIBNIZ_TTS_PROVIDER=gemini \
    LEIBNIZ_TTS_FALLBACK_PROVIDER=google
```

Build and push:
```bash
docker build -f Dockerfile.production -t leibniz-tts-service:production .
docker push your-registry/leibniz-tts-service:production
```

## Troubleshooting

### Container fails to start
- Check logs: `docker logs <container_name>`
- Verify port 8004 is available
- Check environment variables in docker-compose.yml

### Provider not available
- Install provider dependencies inside container
- Verify API keys are set correctly
- Check `/health` endpoint for `providers_available` list

### Health check failing
- Wait 30 seconds for service startup
- Check container logs for errors
- Verify uvicorn is running: `docker exec <container> ps aux | grep uvicorn`

### Performance issues
- Enable caching: `LEIBNIZ_TTS_CACHE_ENABLED=true`
- Use SSD-backed volume for cache
- Monitor cache hit rate via `/health` endpoint

## Example Workflow

```bash
# 1. Build minimal image (30 seconds)
docker build -f leibniz_agent/services/tts/Dockerfile -t leibniz-tts:latest leibniz_agent/services/tts

# 2. Start container
docker run -d \
  --name tts-service \
  -p 8004:8004 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e LEIBNIZ_TTS_PROVIDER=mock \
  -v tts-cache:/app/audio_cache \
  leibniz-tts:latest

# 3. Test with mock provider (no installation needed)
curl http://localhost:8004/health

# 4. Install Gemini provider (2 minutes)
docker exec tts-service pip install google-genai>=1.33.0

# 5. Update provider setting and restart
docker exec tts-service sh -c 'export LEIBNIZ_TTS_PROVIDER=gemini'
docker restart tts-service

# 6. Test Gemini synthesis
curl -X POST http://localhost:8004/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Gemini!", "emotion": "excited"}'
```
