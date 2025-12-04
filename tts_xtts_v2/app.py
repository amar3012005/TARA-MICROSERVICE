"""XTTS-v2 Native Streaming Microservice."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import tempfile
import threading
import time
import wave
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .audio_cache import AudioCache
from .config import XTTSStreamingConfig
from .fastrtc_handler import FastRTCTTSHandler
from .xtts_provider import XTTSNativeProvider, XTTSProviderConfig

import gradio as gr
from fastrtc import Stream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from leibniz_agent.services.shared.redis_client import get_redis_client, ping_redis  # type: ignore

    REDIS_AVAILABLE = True
except ImportError:
    try:
        from shared.redis_client import get_redis_client, ping_redis  # type: ignore

        REDIS_AVAILABLE = True
    except ImportError:  # pragma: no cover - optional
        REDIS_AVAILABLE = False

        async def get_redis_client():  # type: ignore
            return None

        async def ping_redis(_client):  # type: ignore
            return False

config: Optional[XTTSStreamingConfig] = None
provider: Optional[XTTSNativeProvider] = None
cache: Optional[AudioCache] = None
app_start_time = time.time()
fastrtc_handler: Optional[FastRTCTTSHandler] = None
fastrtc_stream: Optional[Stream] = None
redis_client = None
active_sessions: Dict[str, Dict[str, Any]] = {}


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: Optional[str] = Field(default=None)
    voice: Optional[str] = Field(default=None)
    speaker: Optional[str] = Field(default=None, description="Absolute path to reference WAV")
    emotion: str = Field(default="neutral")


class SynthesizeResponse(BaseModel):
    success: bool
    audio_data: Optional[str] = None
    sample_rate: Optional[int] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None


async def _connect_redis_background():
    global redis_client

    if not REDIS_AVAILABLE:
        return

    host = os.getenv("LEIBNIZ_REDIS_HOST", os.getenv("REDIS_HOST", "localhost"))
    port = os.getenv("LEIBNIZ_REDIS_PORT", os.getenv("REDIS_PORT", "6379"))
    os.environ.setdefault("LEIBNIZ_REDIS_HOST", host)
    os.environ.setdefault("LEIBNIZ_REDIS_PORT", port)

    for attempt in range(5):
        try:
            redis_client = await asyncio.wait_for(get_redis_client(), timeout=5.0)
            await ping_redis(redis_client)
            logger.info("✅ Redis connected")
            if fastrtc_handler:
                fastrtc_handler.redis_client = redis_client
            return
        except Exception as exc:
            logger.warning("Redis connection failed (%s/5): %s", attempt + 1, exc)
            await asyncio.sleep(2.0)

    logger.warning("⚠️ Redis unavailable - connection events disabled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, provider, cache

    logger.info("=" * 70)
    logger.info("🚀 Starting XTTS Streaming Microservice")
    logger.info("=" * 70)

    config = XTTSStreamingConfig.from_env()
    provider = XTTSNativeProvider(
        XTTSProviderConfig(
            model_dir=config.xtts_model_dir,
            device=config.xtts_device,
            stream_chunk_tokens=config.stream_chunk_tokens,
            max_buffer_chunks=config.max_buffer_chunks,
        )
    )

    cache = AudioCache(str(config.cache_dir), max_size=config.max_cache_size) if config.enable_cache else None

    FastRTCTTSHandler.default_chunk_duration_ms = config.fastrtc_chunk_duration_ms

    asyncio.create_task(_connect_redis_background())

    yield

    if redis_client:
        with contextlib.suppress(Exception):
            await redis_client.close()

    logger.info("✅ XTTS microservice stopped")


app = FastAPI(
    title="Leibniz XTTS Streaming Service",
    description="Ultra-low latency native XTTS streaming",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    fastrtc_handler = FastRTCTTSHandler(tts_queue=None, redis_client=redis_client)
    fastrtc_stream = Stream(
        handler=fastrtc_handler,
        modality="audio",
        mode="send-receive",
        ui_args={
            "title": "XTTS Streaming",
            "description": "Enter text to hear it streamed from XTTS-v2",
        },
    )
    app = gr.mount_gradio_app(app, fastrtc_stream.ui, path="/fastrtc")
    logger.info("✅ FastRTC UI mounted at /fastrtc")
except Exception as exc:  # pragma: no cover - FastRTC optional
    logger.warning("FastRTC init failed: %s", exc)
    fastrtc_handler = None
    fastrtc_stream = None


def _resolve_speaker_path(value: Optional[str]) -> str:
    if value:
        return os.path.abspath(value)
    assert config is not None
    return str(config.default_speaker_wav)


def _resolve_voice_id(value: Optional[str]) -> str:
    assert config is not None
    return value or config.default_voice_id


async def _emit_audio_chunk(
    websocket: WebSocket,
    chunk_bytes: bytes,
    *,
    chunk_index: int,
    sample_rate: int,
    text: str,
    cached: bool,
):
    duration_ms = len(chunk_bytes) / (sample_rate * 2) * 1000.0
    audio_b64 = base64.b64encode(chunk_bytes).decode("ascii")
    await websocket.send_json(
        {
            "type": "audio",
            "data": audio_b64,
            "index": chunk_index,
            "sample_rate": sample_rate,
            "cached": cached,
        }
    )
    await websocket.send_json(
        {
            "type": "sentence_playing",
            "index": 0,
            "duration_ms": duration_ms,
            "expected_complete_at": time.time() + duration_ms / 1000.0,
            "text": text,
        }
    )

    await FastRTCTTSHandler.broadcast_audio(chunk_bytes, sample_rate)


async def _stream_cached_audio(path: str, chunk_size: int = 8192):
    def _reader() -> bytes:
        with open(path, "rb") as fh:
            return fh.read()

    data = await asyncio.to_thread(_reader)
    for idx in range(0, len(data), chunk_size):
        yield data[idx : idx + chunk_size]


async def _write_wav_file(audio_bytes: bytes, sample_rate: int) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()

    def _writer():
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)

    await asyncio.to_thread(_writer)
    return tmp_path


async def _handle_stream_request(
    websocket: WebSocket,
    *,
    text: str,
    language: str,
    voice_id: str,
    speaker_wav: str,
    emotion: str,
    stop_event: threading.Event,
):
    assert provider and config

    start_time = time.time()
    first_chunk_latency_ms: Optional[float] = None
    chunk_index = 0
    total_bytes = 0
    cached = False
    pcm_accumulator = bytearray()

    await websocket.send_json({"type": "sentence_start", "index": 0, "text": text})

    cached_path = None
    if cache:
        cached_path = cache.get_cached_audio(text, voice_id, language, config.cache_provider_name, emotion)

    if cached_path:
        stream_iter = _stream_cached_audio(cached_path)
        cached = True
    else:
        stream_iter = provider.stream_text(
            text,
            language=language,
            speaker_wav=speaker_wav,
            cancel_event=stop_event,
        )

    try:
        async for chunk in stream_iter:
            if stop_event.is_set():
                break

            if first_chunk_latency_ms is None:
                first_chunk_latency_ms = (time.time() - start_time) * 1000.0
                await websocket.send_json(
                    {
                        "type": "first_chunk",
                        "latency_ms": first_chunk_latency_ms,
                        "cached": cached,
                    }
                )

            chunk_bytes = chunk if cached else chunk
            if not cached:
                pcm_accumulator.extend(chunk_bytes)

            await _emit_audio_chunk(
                websocket,
                chunk_bytes,
                chunk_index=chunk_index,
                sample_rate=config.sample_rate,
                text=text,
                cached=cached,
            )

            total_bytes += len(chunk_bytes)
            chunk_index += 1
    finally:
        if not cached and pcm_accumulator and cache and not stop_event.is_set():
            tmp_path = await _write_wav_file(bytes(pcm_accumulator), config.sample_rate)
            try:
                cache.cache_audio(text, voice_id, language, config.cache_provider_name, emotion, tmp_path)
            finally:
                os.unlink(tmp_path)

    duration_ms = total_bytes / (config.sample_rate * 2) * 1000.0
    await websocket.send_json(
        {
            "type": "sentence_complete",
            "index": 0,
            "duration_ms": duration_ms,
            "cached": cached,
        }
    )
    await websocket.send_json(
        {
            "type": "complete",
            "total_duration_ms": duration_ms,
            "chunks": chunk_index,
            "first_chunk_latency_ms": first_chunk_latency_ms,
            "cached": cached,
        }
    )


@app.websocket("/api/v1/stream")
async def stream_tts(websocket: WebSocket, session_id: str = Query(...)):
    await websocket.accept()
    logger.info("🔌 WebSocket connected: %s", session_id)

    if not provider or not config:
        await websocket.send_json({"type": "error", "message": "XTTS provider unavailable"})
        await websocket.close()
        return

    await websocket.send_json({"type": "connected", "session_id": session_id, "timestamp": time.time()})

    session_data = active_sessions.setdefault(session_id, {"task": None, "stop_event": None})

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "timeout", "message": "No activity detected"})
                continue
            except WebSocketDisconnect:
                break

            msg_type = message.get("type")

            if msg_type == "synthesize":
                text = message.get("text", "").strip()
                if not text:
                    await websocket.send_json({"type": "error", "message": "Empty text"})
                    continue

                language = message.get("language") or config.default_language
                voice_id = _resolve_voice_id(message.get("voice"))
                speaker_wav = _resolve_speaker_path(message.get("speaker"))
                emotion = message.get("emotion", "neutral")

                if session_data["task"]:
                    if session_data.get("stop_event"):
                        session_data["stop_event"].set()
                    session_data["task"].cancel()
                    with contextlib.suppress(Exception):
                        await session_data["task"]

                stop_event = threading.Event()
                session_data["stop_event"] = stop_event
                session_data["task"] = asyncio.create_task(
                    _handle_stream_request(
                        websocket,
                        text=text,
                        language=language,
                        voice_id=voice_id,
                        speaker_wav=speaker_wav,
                        emotion=emotion,
                        stop_event=stop_event,
                    )
                )

            elif msg_type == "cancel":
                if session_data["task"]:
                    if session_data.get("stop_event"):
                        session_data["stop_event"].set()
                    session_data["task"].cancel()
                    with contextlib.suppress(Exception):
                        await session_data["task"]
                    session_data["task"] = None
                await websocket.send_json({"type": "cancelled"})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        session = active_sessions.pop(session_id, None)
        if session:
            if session.get("stop_event"):
                session["stop_event"].set()
            task = session.get("task")
            if task:
                task.cancel()
        with contextlib.suppress(Exception):
            await websocket.close()
        logger.info("🔌 WebSocket disconnected: %s", session_id)


@app.post("/api/v1/synthesize", response_model=SynthesizeResponse)
async def synthesize_text(request: SynthesizeRequest):
    if not provider or not config:
        raise HTTPException(status_code=503, detail="XTTS provider unavailable")

    language = request.language or config.default_language
    voice_id = _resolve_voice_id(request.voice)
    speaker_wav = _resolve_speaker_path(request.speaker)

    cached_path = None
    if cache:
        cached_path = cache.get_cached_audio(request.text, voice_id, language, config.cache_provider_name, request.emotion)

    try:
        if cached_path:
            with open(cached_path, "rb") as fh:
                audio_bytes = fh.read()
        else:
            audio_bytes = await provider.synthesize_to_bytes(
                request.text,
                language=language,
                speaker_wav=speaker_wav,
            )
            if cache:
                tmp_path = await _write_wav_file(audio_bytes, config.sample_rate)
                try:
                    cache.cache_audio(request.text, voice_id, language, config.cache_provider_name, request.emotion, tmp_path)
                finally:
                    os.unlink(tmp_path)

        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        duration_ms = len(audio_bytes) / (config.sample_rate * 2) * 1000.0
        return SynthesizeResponse(success=True, audio_data=audio_b64, sample_rate=config.sample_rate, duration_ms=duration_ms)
    except Exception as exc:
        logger.error("HTTP synthesis failed: %s", exc)
        return SynthesizeResponse(success=False, error=str(exc))


@app.post("/api/v1/fastrtc/synthesize")
async def fastrtc_synthesize(request: SynthesizeRequest):
    if not provider or not config:
        raise HTTPException(status_code=503, detail="XTTS provider unavailable")
    if not fastrtc_handler:
        raise HTTPException(status_code=503, detail="FastRTC handler unavailable")

    audio_bytes = await provider.synthesize_to_bytes(
        request.text,
        language=request.language or config.default_language,
        speaker_wav=_resolve_speaker_path(request.speaker),
    )

    chunk_samples = int(config.sample_rate * (config.fastrtc_chunk_duration_ms / 1000.0))
    for idx in range(0, len(audio_bytes), chunk_samples * 2):
        chunk = audio_bytes[idx : idx + chunk_samples * 2]
        await FastRTCTTSHandler.broadcast_audio(chunk, config.sample_rate)
        await asyncio.sleep(config.fastrtc_chunk_duration_ms / 1000.0)

    duration_ms = len(audio_bytes) / (config.sample_rate * 2) * 1000.0
    return {"success": True, "duration_ms": duration_ms}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if provider else "unhealthy",
        "service": "tts-xtts-v2",
        "uptime_seconds": time.time() - app_start_time,
        "provider": "ready" if provider else "not_ready",
        "cache": cache.get_stats() if cache else None,
        "active_sessions": len(active_sessions),
    }


@app.get("/metrics")
async def metrics():
    return {
        "active_sessions": len(active_sessions),
        "uptime_seconds": time.time() - app_start_time,
    }


@app.get("/")
async def root():
    return {
        "service": "Leibniz XTTS Streaming Service",
        "version": "1.0.0",
        "endpoints": {
            "stream": "WebSocket /api/v1/stream?session_id=<id>",
            "synthesize": "POST /api/v1/synthesize",
            "fastrtc_synthesize": "POST /api/v1/fastrtc/synthesize",
            "health": "GET /health",
            "metrics": "GET /metrics",
        },
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    port = int(os.getenv("TTS_STREAMING_PORT", "8005"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
