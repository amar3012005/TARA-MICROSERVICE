# TTS Streaming Microservice

Independent TTS microservice with WebSocket-based parallel queue streaming, sentence-level chunking, and Sarvam AI integration. Optimized for TARA mode with Telugu language support for TASK organization.

## Features

- **WebSocket Streaming**: Real-time bidirectional audio streaming
- **Sentence-level Chunking**: Automatic text splitting with abbreviation protection
- **2-Slot Pipeline**: Parallel synthesis (synthesize N+1 while playing N)
- **Sarvam AI Provider**: Multi-language TTS with Telugu support for TARA mode
- **MD5 Caching**: Audio caching with LRU cleanup and TTL enforcement
- **TARA Mode**: Telugu language support for TASK customer service agent

## Architecture

```
Client (WebSocket) --> TTS Service (Port 8025)
                          |
                          +--> Sentence Splitter
                          |        |
                          +--> TTS Queue (asyncio.Queue)
                          |        |
                          +--> Parallel Synthesizer (2-slot pipeline)
                          |        |
                          +--> Sarvam AI API (te-IN for TARA mode)
                          |        |
                          +--> Audio Cache (MD5 keys)
```

## API Endpoints

### WebSocket: `/api/v1/stream?session_id=<id>`

**Client to Server:**
```json
{"type": "synthesize", "text": "Hello! How are you?", "emotion": "helpful"}
{"type": "cancel"}
{"type": "ping"}
```

**Server to Client:**
```json
{"type": "connected", "session_id": "abc123"}
{"type": "sentence_start", "index": 0, "text": "Hello!"}
{"type": "audio", "data": "<base64>", "index": 0, "sample_rate": 24000}
{"type": "sentence_complete", "index": 0, "duration_ms": 450}
{"type": "complete", "total_sentences": 2, "total_duration_ms": 1200}
```

### HTTP: `POST /api/v1/synthesize`

Synthesize text without streaming (returns base64 audio).

**Request:**
```json
{
  "text": "Hello! How are you?",
  "emotion": "helpful",
  "voice": "sarah",
  "language": "en-us"
}
```

**Response:**
```json
{
  "success": true,
  "audio_data": "<base64-encoded-audio>",
  "sample_rate": 24000,
  "duration_ms": 1200.0,
  "sentences": 2
}
```

### Health: `GET /health`

Returns service health status.

### Metrics: `GET /metrics`

Returns performance metrics (queue stats, cache hit rate).

## Environment Variables

- `LEMONFOX_API_KEY`: LemonFox API key (required)
- `LEIBNIZ_LEMONFOX_VOICE`: Voice name (default: sarah)
- `LEIBNIZ_LEMONFOX_LANGUAGE`: Language code (default: en-us)
- `TTS_STREAMING_PORT`: Service port (default: 8005)
- `LEIBNIZ_TTS_CACHE_DIR`: Cache directory (default: /app/audio_cache)
- `LEIBNIZ_TTS_CACHE_ENABLED`: Enable caching (default: true)
- `LEIBNIZ_TTS_CACHE_MAX_SIZE`: Max cache entries (default: 500)
- `TTS_QUEUE_MAX_SIZE`: Max sentences in queue (default: 10)

## Docker Deployment

```bash
# Build
docker-compose build tts-streaming-service

# Run
docker-compose up -d tts-streaming-service

# Check logs
docker-compose logs -f tts-streaming-service
```

## Usage Example

```python
import asyncio
import websockets
import json
import base64

async def test_tts():
    uri = "ws://localhost:8005/api/v1/stream?session_id=test123"
    async with websockets.connect(uri) as websocket:
        # Receive connection confirmation
        response = await websocket.recv()
        print(f"Connected: {json.loads(response)}")
        
        # Send synthesis request
        await websocket.send(json.dumps({
            "type": "synthesize",
            "text": "Hello! This is a test. How are you?",
            "emotion": "helpful"
        }))
        
        # Receive audio chunks
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            
            if data["type"] == "audio":
                audio_bytes = base64.b64decode(data["data"])
                print(f"Received audio chunk {data['index']}: {len(audio_bytes)} bytes")
            
            elif data["type"] == "complete":
                print(f"Complete: {data['total_sentences']} sentences")
                break
            
            elif data["type"] == "error":
                print(f"Error: {data['message']}")
                break

asyncio.run(test_tts())
```

## Implementation Details

- **Sentence Splitting**: Extracted from `leibniz_pro.py` with abbreviation protection
- **2-Slot Pipeline**: Synthesize next sentence while playing current (from `leibniz_pro.py`)
- **LemonFox Provider**: Hardcoded API endpoint `https://api.lemonfox.ai/v1/audio/speech`
- **Audio Cache**: MD5-based keys with LRU cleanup (from `leibniz_tts.py`)

## FastRTC Integration

The service includes FastRTC support for browser audio playback, similar to STT-VAD.

### Access FastRTC UI

1. **Start the service:**
   ```bash
   docker-compose up -d tts-streaming-service
   ```

2. **Open FastRTC UI:**
   - Local: `http://localhost:8005/fastrtc`
   - **IMPORTANT**: Use the HTTPS URL provided by Gradio (browsers require HTTPS for audio)

3. **Synthesize text via API:**
   ```bash
   curl -X POST http://localhost:8005/api/v1/fastrtc/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello! This is a test.", "emotion": "helpful"}'
   ```

4. **Audio will stream to your browser speakers** via FastRTC!

### How It Works

- FastRTC handler receives synthesized audio from TTS queue
- Audio chunks are streamed via `emit()` method to browser
- Browser plays audio in real-time as it's synthesized
- Similar to STT-VAD but for audio OUTPUT instead of INPUT

## Port

- **8005** - TTS Streaming Service
- **FastRTC UI** - Available at `/fastrtc` endpoint

