# Build and Run Individual Leibniz Microservices

This guide provides commands to build and run each Leibniz microservice individually using Docker. This is useful for development, testing, and debugging specific services without running the entire stack.

## Prerequisites

1. **Docker installed** and running
2. **Redis container** for state management:
   ```bash
   docker run -d --name leibniz-redis -p 6379:6379 redis:7-alpine
   ```
3. **Environment variables** set (create `.env` file or export):
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key_here"
   export LEMONFOX_API_KEY="your_lemonfox_api_key_here"
   # Add other required API keys...
   ```

## Build Commands

### Intent Service (Port 8002)
```bash
docker build -f leibniz_agent/services/intent/Dockerfile -t leibniz-intent:latest .
```

### Appointment Service (Port 8005)
```bash
docker build -f leibniz_agent/services/appointment/Dockerfile -t leibniz-appointment:latest .
```

### TTS Service (Port 8004)
```bash
docker build -f leibniz_agent/services/tts/Dockerfile -t leibniz-tts:latest .
```

### STT/VAD Service (Port 8001)
```bash
docker build -f leibniz_agent/services/stt_vad/Dockerfile -t leibniz-stt-vad:latest .
```

### RAG Service (Port 8003)
```bash
docker build -f leibniz_agent/services/rag/Dockerfile -t leibniz-rag:latest .
```

## Run Commands

### Intent Service
```bash
docker run -d --name leibniz-intent \
  --link leibniz-redis:redis \
  -p 8002:8002 \
  -e GEMINI_API_KEY=${GEMINI_API_KEY} \
  -e REDIS_URL=redis://redis:6379 \
  leibniz-intent:latest
```

### Appointment Service
```bash
docker run -d --name leibniz-appointment \
  --link leibniz-redis:redis \
  -p 8005:8005 \
  -e REDIS_URL=redis://redis:6379 \
  leibniz-appointment:latest
```

### TTS Service
```bash
docker run -d --name leibniz-tts \
  -p 8004:8004 \
  -e LEMONFOX_API_KEY=${LEMONFOX_API_KEY} \
  -e GEMINI_API_KEY=${GEMINI_API_KEY} \
  -v $(pwd)/leibniz_agent/audio_cache:/app/audio_cache \
  leibniz-tts:latest
```

### STT/VAD Service
```bash
docker run -d --name leibniz-stt-vad \
  --link leibniz-redis:redis \
  -p 8001:8001 \
  -e GEMINI_API_KEY=${GEMINI_API_KEY} \
  -e LEIBNIZ_REDIS_HOST=redis \
  leibniz-stt-vad:latest
```

### RAG Service
```bash
docker run -d --name leibniz-rag \
  --link leibniz-redis:redis \
  -p 8003:8003 \
  -e GEMINI_API_KEY=${GEMINI_API_KEY} \
  -e LEIBNIZ_REDIS_HOST=redis \
  -v $(pwd)/leibniz_knowledge_base:/app/leibniz_knowledge_base:ro \
  leibniz-rag:latest
```

## Health Check Commands

### Intent Service
```bash
curl http://localhost:8002/health
```

### Appointment Service
```bash
curl http://localhost:8005/health
```

### TTS Service
```bash
curl http://localhost:8004/health
```

### STT/VAD Service
```bash
curl http://localhost:8001/health
```

### RAG Service
```bash
curl http://localhost:8003/health
```

## Sample Test Requests

### Intent Classification
```bash
curl -X POST http://localhost:8002/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I want to schedule an appointment with admissions"}'
```

### TTS Synthesis
```bash
curl -X POST http://localhost:8004/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello! Welcome to Leibniz University. How can I help you today?", "emotion": "helpful"}'
```

### RAG Query
```bash
curl -X POST http://localhost:8003/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the admission requirements?"}'
```

### Appointment Session Creation
```bash
curl -X POST http://localhost:8005/api/v1/session/create
```

### STT/VAD WebSocket Connection
```bash
# Requires WebSocket client like websocat or Python script
python -c "
import asyncio
import websockets
import json

async def test_stt():
    uri = 'ws://localhost:8001/api/v1/transcribe/stream?session_id=test123'
    async with websockets.connect(uri) as websocket:
        # Receive welcome
        welcome = await websocket.recv()
        print('Welcome:', json.loads(welcome))
        
        # Send start command
        await websocket.send(json.dumps({'type': 'start_capture'}))
        
        # Send some dummy audio bytes (this would be real PCM data)
        dummy_audio = b'\x00\x01\x02\x03' * 100  # 400 bytes
        await websocket.send(dummy_audio)
        
        # Receive response
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print('Response:', json.loads(response))
        except asyncio.TimeoutError:
            print('Timeout - no response received')

asyncio.run(test_stt())
"
```

## Troubleshooting

### Build Failures
- **"requirements.txt not found"**: Check Dockerfile COPY paths - they should reference `leibniz_agent/services/<service>/requirements.txt`
- **"Module not found"**: Ensure PYTHONPATH=/app is set and __init__.py files exist
- **"Permission denied"**: Check file ownership and USER directive in Dockerfile

### Runtime Import Errors
- **"ModuleNotFoundError: No module named 'config'"**: Convert relative imports to absolute imports (`from leibniz_agent.services.intent.config import ...`)
- **"ModuleNotFoundError: No module named 'leibniz_agent'"**: Check PYTHONPATH and package structure

### Redis Connection Issues
- **"Connection refused"**: Ensure Redis container is running: `docker ps | grep redis`
- **"Timeout"**: Check network connectivity and firewall
- **"Authentication failed"**: Verify REDIS_PASSWORD if set

### Service-Specific Issues
- **Intent**: "Gemini API key not set" → Set GEMINI_API_KEY environment variable
- **TTS**: "No providers available" → Set LEMONFOX_API_KEY or enable MOCK_TTS=true
- **RAG**: "FAISS index not loaded" → Run index builder or check volume mount
- **STT/VAD**: "Audio device not found" → Not needed in Docker (WebSocket-based)
- **Appointment**: "Session not found" → Check Redis TTL and session_id

### Port Conflicts
- **"Address already in use"**: Check for existing services on ports 8001-8005
- Solution: Stop conflicting services or change port mappings

### Performance Issues
- **Slow responses**: Check Redis cache hit rate
- **High memory usage**: Reduce cache sizes or worker counts
- **Timeout errors**: Increase service-specific timeout values

### Docker-Specific Issues
- **"Cannot connect to Docker daemon"**: Start Docker Desktop
- **"Image not found"**: Build image first with `docker build`
- **"Container exits immediately"**: Check logs with `docker logs <container>`

### Debug Commands
```bash
# View service logs
docker logs -f leibniz-intent

# Exec into container
docker exec -it leibniz-intent /bin/bash

# Check Python imports
docker exec leibniz-intent python -c "import leibniz_agent.services.intent.app"

# Test Redis connectivity
docker exec leibniz-redis redis-cli ping
```

## Cleanup Commands

```bash
# Stop all services
docker stop leibniz-intent leibniz-appointment leibniz-tts leibniz-stt-vad leibniz-rag leibniz-redis

# Remove all containers
docker rm leibniz-intent leibniz-appointment leibniz-tts leibniz-stt-vad leibniz-rag leibniz-redis

# Remove all images
docker rmi leibniz-intent leibniz-appointment leibniz-tts leibniz-stt-vad leibniz-rag

# Clean up unused resources
docker system prune -f
```

## Development Workflow

1. **Make code changes** in the service directory
2. **Rebuild the image**: `docker build -f leibniz_agent/services/<service>/Dockerfile -t leibniz-<service>:latest .`
3. **Stop old container**: `docker stop leibniz-<service>`
4. **Remove old container**: `docker rm leibniz-<service>`
5. **Run new container** with the commands above
6. **Check logs**: `docker logs -f leibniz-<service>`
7. **Test endpoints** with the sample requests above

## Integration Testing

For end-to-end testing across services:

1. Start all services with the run commands above
2. Use the health check commands to verify all services are healthy
3. Test cross-service workflows:
   - Intent → RAG: Classify intent, then query RAG with extracted context
   - Intent → Appointment: Classify appointment intent, then create session
   - RAG → TTS: Query knowledge base, then synthesize response
   - STT → Intent: Send audio, get transcription, classify intent

## Performance Monitoring

Monitor service performance:

```bash
# Check container resource usage
docker stats leibniz-intent leibniz-appointment leibniz-tts leibniz-stt-vad leibniz-rag

# Check Redis memory usage
docker exec leibniz-redis redis-cli info memory

# Monitor service metrics (if implemented)
curl http://localhost:8002/metrics  # Intent service metrics
```

## Environment Variables Reference

### Required for All Services
- `GEMINI_API_KEY`: Google Gemini API key for LLM operations

### Intent Service
- `REDIS_URL`: Redis connection string (default: redis://localhost:6379)
- `INTENT_CACHE_TTL`: Cache TTL in seconds (default: 1800)

### Appointment Service
- `REDIS_URL`: Redis connection string (default: redis://localhost:6379)

### TTS Service
- `LEMONFOX_API_KEY`: LemonFox TTS API key
- `LEIBNIZ_TTS_CACHE_DIR`: Cache directory (default: /app/audio_cache)
- `LEIBNIZ_TTS_PROVIDER`: Primary provider (default: lemonfox)

### STT/VAD Service
- `LEIBNIZ_REDIS_HOST`: Redis host (default: redis)
- `LEIBNIZ_VAD_MODEL`: Gemini model (default: gemini-2.0-flash-exp)

### RAG Service
- `LEIBNIZ_REDIS_HOST`: Redis host (default: redis)
- `LEIBNIZ_RAG_KNOWLEDGE_BASE_PATH`: Knowledge base path
- `LEIBNIZ_RAG_VECTOR_STORE_PATH`: FAISS index path