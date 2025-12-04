# XTTS-v2 Streaming Microservice

CUDA-accelerated TTS service that streams Coqui XTTS-v2 chunks directly from the model’s native `inference_stream()` API. No sentence queueing or third-party providers—audio starts within ~400 ms on RTX-class GPUs.

## Highlights
- **Native streaming** – token-level chunking (10–30 tokens) delivers 100–200 ms audio buffers immediately.
- **Voice cloning** – reference WAV latents cached in RAM for zero recompute overhead.
- **Cache & Redis reuse** – same MD5 cache + orchestrator events as the legacy `tts_streaming` service.
- **FastRTC support** – every chunk is broadcast to browser clients with <50 ms additional latency.
- **CUDA Docker image** – ships with PyTorch+CUDNN runtime; just mount your XTTS checkpoint folder and enable `--gpus all`.

## API Surface (compatible with `tts_streaming`)
| Endpoint | Description |
| --- | --- |
| `WS /api/v1/stream?session_id=<id>` | Same request/response schema, powered by native XTTS chunks |
| `POST /api/v1/synthesize` | Returns base64 24 kHz mono PCM |
| `POST /api/v1/fastrtc/synthesize` | Streams audio directly into FastRTC sessions |
| `GET /health`, `GET /metrics`, `GET /` | Operational visibility |

### WebSocket example
```jsonc
// Client → server
{"type": "synthesize", "text": "Hello native XTTS!", "voice": "support", "language": "en"}

// Server → client (first few messages)
{"type": "connected", "session_id": "abc"}
{"type": "sentence_start", "index": 0, "text": "Hello native XTTS!"}
{"type": "first_chunk", "latency_ms": 415.6, "cached": false}
{"type": "audio", "data": "<base64>", "index": 0, "sample_rate": 24000}
```

## Key Environment Variables
| Variable | Purpose | Default |
| --- | --- | --- |
| `XTTS_MODEL_DIR` | Folder containing `config.json`, `model.pth`, etc. | `~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2` |
| `XTTS_DEVICE` | `cuda` or `cpu` fallback | `cuda` |
| `LEIBNIZ_XTTS_SPEAKER_WAV` | Default cloning WAV | `/app/leibniz_agent/services/ElevenLabs_enigma.wav` |
| `LEIBNIZ_XTTS_LANGUAGE` | Default language | `en` |
| `LEIBNIZ_XTTS_VOICE_ID` | Cache key voice label | `xtts_voice_default` |
| `LEIBNIZ_XTTS_STREAM_CHUNK_TOKENS` | Tokens per native chunk | `20` |
| `LEIBNIZ_TTS_CACHE_DIR` | Reused audio cache dir | `/app/audio_cache` |
| `LEIBNIZ_TTS_FASTRTC_CHUNK_MS` | FastRTC playback chunk size | `40` |
| `TTS_STREAMING_PORT` | Service port | `8005` |

(Existing cache + Redis env vars from the old service continue to work.)

## Docker Build & Run
```bash
# Build (context = repo/services)
docker build -f tts_xtts_v2/Dockerfile -t leibniz/tts-xtts-v2 .

# Run with GPU and mounted models
docker run --gpus all -p 8005:8005 \
  -e XTTS_MODEL_DIR=/models/xtts_v2 \
  -e LEIBNIZ_XTTS_SPEAKER_WAV=/voices/enigma.wav \
  -v /host/models/xtts_v2:/models/xtts_v2 \
  -v /host/voices:/voices \
  -v tts_cache:/app/audio_cache \
  leibniz/tts-xtts-v2
```
Health check: `curl http://localhost:8005/health`

### Two-Stage Dev Workflow (skip rebuilds)
1. **Build heavy deps once** – the Dockerfile now has a `base-deps` stage. Run `docker compose build --target base-deps tts-xtts-v2-service` (or just `docker compose build tts-xtts-v2-service`) whenever requirements/TTS patches change.
2. **Hot-reload Python edits** – start the bind-mounted dev container:
   ```bash
   cd services
   docker compose --profile dev up -d tts-xtts-v2-dev
   ```
   This service:
   - Reuses the cached `base-deps` layers (no reinstall of Coqui/pip deps)
   - Mounts `./tts_xtts_v2` and `./shared`, so local edits are reflected instantly
   - Shares the same cache/model volumes as production (`tts_xtts_cache`, `XTTS_MODEL_HOST_DIR`)
3. **Production image** – when you’re ready to bake code into an image, `docker compose build tts-xtts-v2-service` (runtime stage) and deploy as usual.

## FastRTC UI
1. Start the container and open `http://localhost:8005/fastrtc` (or the HTTPS URL Gradio prints).
2. Submit text via the WebSocket or `POST /api/v1/fastrtc/synthesize`.
3. Native chunks arrive in the browser with the same latency as the orchestrator feed.

## Local Development
```bash
cd /home/prometheus/leibniz_agent
source services/tts-env/bin/activate
uvicorn leibniz_agent.services.tts_xtts_v2.app:app --host 0.0.0.0 --port 8005
```
Then run `python test_xtts_simple.py` or hit the WebSocket endpoint for smoke testing.

## Migration Plan
1. Deploy `tts_xtts_v2` alongside `tts_streaming`.
2. Use the orchestrator switch to send a small percentage of traffic to XTTS.
3. Monitor `/metrics` (first-chunk latency, chunk counts) and GPU utilization.
4. Flip 100 % of traffic once satisfied; keep the legacy container around for quick rollback.

This service intentionally mirrors the legacy API, so upstream callers can adopt native XTTS streaming without any payload changes.

