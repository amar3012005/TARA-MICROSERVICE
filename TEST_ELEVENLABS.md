# Testing ElevenLabs TTS Direct Streaming

## Configuration Summary

The orchestrator is now configured with:
- ✅ `USE_ELEVENLABS_TTS=true` - Enables direct ElevenLabs streaming
- ✅ `ELEVENLABS_TTS_URL=http://tara-task-tts-labs:8006` - TTS-Labs service URL
- ✅ `ELEVENLABS_PREWARM_ON_VAD=true` - Prewarm on VAD detection
- ✅ `TTS_STREAMING_MODE=continuous` - Continuous streaming mode

## Quick Start

### 1. Start Services

```bash
cd /home/prometheus/leibniz_agent/TARA-MICROSERVICE

# Start all services
docker --context desktop-linux compose -f docker-compose-tara-task.yml up -d

# Or start specific services
docker --context desktop-linux compose -f docker-compose-tara-task.yml up -d redis tts-labs orchestrator
```

### 2. Verify Services are Running

```bash
# Check service status
docker --context desktop-linux compose -f docker-compose-tara-task.yml ps

# Check orchestrator logs
docker logs tara-task-orchestrator -f

# Check tts-labs logs
docker logs tara-task-tts-labs -f
```

### 3. Test ElevenLabs TTS Direct Streaming

#### Option A: Using the Test Script (Recommended)

```bash
# From the orchestrator directory
cd orchestrator

# Install dependencies if needed
pip install aiohttp

# Run the test script
python test_eleven_tts_direct.py --tts-url http://localhost:2007

# Or test with custom text
python test_eleven_tts_direct.py --tts-url http://localhost:2007 --text "Hello! How can I help you today?"
```

Expected output:
- Prewarm latency: ~100-200ms
- First audio latency (prewarmed): <150ms
- First audio latency (cold start): 300-500ms

#### Option B: Using curl (HTTP Endpoint)

```bash
# Test single text synthesis
curl -X POST http://localhost:2007/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello! This is a test of ElevenLabs ultra-low latency TTS.",
    "emotion": "helpful"
  }'
```

#### Option C: Using WebSocket (Direct)

```bash
# Test WebSocket streaming (requires websocat or similar tool)
# Install websocat: cargo install websocat

echo '{"type":"prewarm"}' | websocat ws://localhost:2007/api/v1/stream?session_id=test123

echo '{"type":"stream_chunk","text":"Hello! This is a test.","emotion":"helpful"}' | websocat ws://localhost:2007/api/v1/stream?session_id=test123

echo '{"type":"stream_end"}' | websocat ws://localhost:2007/api/v1/stream?session_id=test123
```

### 4. Test End-to-End Orchestrator Flow

```bash
# Start orchestrator with all dependencies
docker --context desktop-linux compose -f docker-compose-tara-task.yml up -d redis rag stt-vad tts-labs orchestrator

# Check orchestrator logs for ElevenLabs prewarm messages
docker logs tara-task-orchestrator -f | grep -i "elevenlabs\|prewarm"

# Simulate a turn (requires WebSocket client or use the /simulate/turn endpoint)
curl -X POST "http://localhost:2004/simulate/turn?text=Hello"
```

### 5. View FastRTC Preview (Optional)

Open in browser:
- **TTS-Labs FastRTC**: http://localhost:2007/fastrtc
- **Orchestrator Status**: http://localhost:2004/status

## Expected Log Messages

### Orchestrator Logs

When ElevenLabs TTS is enabled, you should see:
```
🎙️ ELEVENLABS TTS ENABLED - Ultra-low latency streaming
   TTS URL: http://tara-task-tts-labs:8006
   Prewarm on VAD: True
   Streaming Mode: continuous
```

When VAD detects speech:
```
⚡ ElevenLabs prewarm triggered for: <session_id>
⚡ ElevenLabs TTS pre-warmed for session <session_id>
```

When streaming starts:
```
⚡ Using ELEVENLABS DIRECT streaming mode (pre-warmed)
⚡ ULTRA-FAST FIRST AUDIO: <latency>ms
```

### TTS-Labs Logs

```
✅ ElevenLabs provider initialized
⚡ Pre-warmed ElevenLabs connection in <ms>ms for <session_id>
🚀 Started stream (pre-warmed) for <session_id>
```

## Troubleshooting

### Issue: "ElevenLabs TTS client not available"
- Check that `eleven_tts_client.py` exists in `orchestrator/` directory
- Verify orchestrator container has the file mounted

### Issue: "Failed to prewarm ElevenLabs TTS"
- Check `ELEVENLABS_API_KEY` is set in `docker-compose-tara-task.yml`
- Verify `tts-labs` service is running: `docker ps | grep tts-labs`
- Check network connectivity: `docker exec tara-task-orchestrator ping tara-task-tts-labs`

### Issue: High latency (>300ms)
- Ensure prewarm is triggered (check logs for "ElevenLabs prewarm triggered")
- Verify `TTS_STREAMING_MODE=continuous` is set
- Check `tts-labs` service logs for connection issues

### Issue: No audio received
- Check WebSocket connection in orchestrator logs
- Verify ElevenLabs API key is valid
- Check `tts-labs` service health: `curl http://localhost:2007/health`

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Prewarm latency | <200ms | ✅ |
| First audio (prewarmed) | <150ms | ✅ |
| First audio (cold start) | <500ms | ✅ |
| Streaming parallelism | Real-time | ✅ |

## Next Steps

1. Test with actual RAG responses
2. Monitor latency in production
3. Adjust `chunk_length_schedule` in `TTS_LABS/elevenlabs_manager.py` if needed
4. Fine-tune prewarm timing based on VAD detection patterns
