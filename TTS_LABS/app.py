"""
TTS_LABS Streaming Microservice FastAPI Application

WebSocket-based TTS streaming service using ElevenLabs stream-input API.
Optimized for ultra-low latency (<150ms first audio chunk) with eleven_flash_v2_5.
"""

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import TTSLabsConfig
from .sentence_splitter import split_into_sentences
from .elevenlabs_manager import ElevenLabsProvider, ElevenLabsStreamManager
from .audio_cache import AudioCache

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Suppress noisy libraries
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Optional Redis support for orchestrator coordination
try:
    from leibniz_agent.services.shared.redis_client import get_redis_client, ping_redis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        from shared.redis_client import get_redis_client, ping_redis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        logger.warning("Redis client utilities unavailable - TTS connection events disabled")

        async def get_redis_client():
            return None

        async def ping_redis(_client):
            return False

# Global state
config: Optional[TTSLabsConfig] = None
provider: Optional[ElevenLabsProvider] = None
cache: Optional[AudioCache] = None
active_sessions: Dict[str, Dict[str, Any]] = {}
app_start_time: float = time.time()
redis_client = None

# FastRTC support (optional)
try:
    import gradio as gr
    from fastrtc import Stream, AsyncStreamHandler
    FASTRTC_AVAILABLE = True
except ImportError:
    FASTRTC_AVAILABLE = False
    logger.warning("FastRTC not available - UI preview disabled")


class FastRTCTTSHandler(AsyncStreamHandler if FASTRTC_AVAILABLE else object):
    """FastRTC handler for TTS audio streaming to browser."""
    
    active_instances = set()
    
    def __init__(self, redis_client=None):
        if FASTRTC_AVAILABLE:
            super().__init__()
        self.redis_client = redis_client
        self.session_id = f"fastrtc_tts_{int(time.time())}"
        self._started = False
        self._sample_rate = 24000
        self._audio_queue = asyncio.Queue()
        self._remainder = b""
        self._lock = asyncio.Lock()
        self._chunk_size = 960  # 40ms at 24kHz
        # Instrumentation flags for latency diagnostics
        self._first_audio_logged = False
        self._first_emit_logged = False
        
    async def start_up(self):
        self._started = True
        FastRTCTTSHandler.active_instances.add(self)
        logger.info(f"FastRTC TTS stream started: {self.session_id}")
        
        # Publish connection event
        if self.redis_client:
            try:
                payload = json.dumps({
                    "session_id": self.session_id,
                    "timestamp": time.time(),
                    "event": "tts_connected",
                    "source": "tts_labs_fastrtc"
                })
                await self.redis_client.publish("leibniz:events:tts:connected", payload)
            except Exception as e:
                logger.warning(f"Failed to publish TTS connection event: {e}")
    
    async def receive(self, audio):
        pass  # TTS is output-only
    
    async def emit(self):
        """Emit audio chunks to the browser."""
        try:
            # Wait briefly for audio data to reduce latency vs sending silence
            # But don't block too long to keep stream alive
            sample_rate, audio_data = await asyncio.wait_for(self._audio_queue.get(), timeout=0.02)

            # Log first non-silence emission for this FastRTC session
            if not self._first_emit_logged:
                self._first_emit_logged = True
                logger.info(
                    "FastRTC first non-silence emit for %s at %.3f (queue_size=%d, chunk_samples=%d)",
                    self.session_id,
                    time.time(),
                    self._audio_queue.qsize(),
                    len(audio_data),
                )
            return (sample_rate, audio_data)
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            # Send silence if no audio available
            silence = np.zeros(self._chunk_size, dtype=np.int16)
            return (self._sample_rate, silence)
    
    async def add_audio(self, audio_bytes: bytes, sample_rate: int):
        """Add audio data to the queue, chunking it correctly."""
        async with self._lock:
            # Log first arrival of audio bytes from ElevenLabs for this FastRTC session
            if not self._first_audio_logged and audio_bytes:
                self._first_audio_logged = True
                logger.info(
                    "FastRTC first audio bytes received for %s at %.3f (bytes=%d, sample_rate=%d)",
                    self.session_id,
                    time.time(),
                    len(audio_bytes),
                    sample_rate,
                )

            # Combine with any remainder from previous chunks
            data = self._remainder + audio_bytes
            
            # Calculate bytes per chunk (16-bit = 2 bytes per sample)
            chunk_len_bytes = self._chunk_size * 2
            
            # Process full chunks
            cursor = 0
            while cursor + chunk_len_bytes <= len(data):
                chunk = data[cursor : cursor + chunk_len_bytes]
                # Create numpy array from bytes (copy to ensure safety)
                audio_array = np.frombuffer(chunk, dtype=np.int16).copy()
                self._audio_queue.put_nowait((self._sample_rate, audio_array))
                cursor += chunk_len_bytes
            
            # Save remainder
            self._remainder = data[cursor:]
    
    @classmethod
    async def broadcast_audio(cls, audio_bytes: bytes, sample_rate: int):
        for instance in cls.active_instances:
            try:
                await instance.add_audio(audio_bytes, sample_rate)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
    
    async def shutdown(self):
        FastRTCTTSHandler.active_instances.discard(self)
        self._started = False
        logger.info(f"FastRTC TTS stream closed: {self.session_id}")
    
    def copy(self):
        return FastRTCTTSHandler(redis_client=self.redis_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for application startup/shutdown"""
    global config, provider, cache, redis_client
    
    logger.info("=" * 70)
    logger.info("🚀 Starting TTS_LABS Microservice (ElevenLabs)")
    logger.info("=" * 70)
    
    # Load configuration
    try:
        config = TTSLabsConfig.from_env()
        logger.info(f"📋 Configuration loaded")
        logger.info(f"   Model: {config.elevenlabs_model_id}")
        logger.info(f"   Voice: {config.elevenlabs_voice_id}")
        logger.info(f"   Latency optimization: {config.optimize_streaming_latency}")
        logger.info(f"   Output format: {config.output_format}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise
    
    # Initialize ElevenLabs provider
    try:
        if not config.elevenlabs_api_key:
            logger.warning("⚠️ ELEVENLABS_API_KEY not set - service will not function properly")
            provider = None
        else:
            provider = ElevenLabsProvider(config)
            logger.info("✅ ElevenLabs provider initialized")
            
            # Warmup connection in background
            asyncio.create_task(provider.warmup())
            
    except Exception as e:
        logger.error(f"Failed to initialize ElevenLabs provider: {e}")
        provider = None
    
    # Initialize cache
    try:
        if config.enable_cache:
            cache = AudioCache(
                cache_dir=config.cache_dir,
                max_size=config.max_cache_size
            )
            logger.info(f"✅ Audio cache initialized (max_size={config.max_cache_size})")
        else:
            cache = None
            logger.info("Cache disabled")
    except Exception as e:
        logger.warning(f"Failed to initialize cache: {e}")
        cache = None
    
    # Connect to Redis (background)
    async def connect_redis_background():
        global redis_client
        if not REDIS_AVAILABLE:
            return
        
        logger.info("🔌 Connecting to Redis (background)...")
        
        if not os.getenv("LEIBNIZ_REDIS_HOST"):
            os.environ["LEIBNIZ_REDIS_HOST"] = os.getenv("REDIS_HOST", "localhost")
        if not os.getenv("LEIBNIZ_REDIS_PORT"):
            os.environ["LEIBNIZ_REDIS_PORT"] = os.getenv("REDIS_PORT", "6379")
        
        for attempt in range(5):
            try:
                redis_client = await asyncio.wait_for(get_redis_client(), timeout=5.0)
                await ping_redis(redis_client)
                logger.info(f"✅ Redis connected (attempt {attempt + 1})")
                return
            except Exception as e:
                logger.warning(f"⚠️ Redis connection error (attempt {attempt + 1}/5): {e}")
                if attempt < 4:
                    await asyncio.sleep(2.0)
        
        logger.warning("⚠️ Redis unavailable - TTS connection events will be disabled")
    
    asyncio.create_task(connect_redis_background())
    
    logger.info("=" * 70)
    logger.info("✅ TTS_LABS Microservice Ready")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown
    logger.info("=" * 70)
    logger.info("🛑 Shutting down TTS_LABS microservice...")
    logger.info("=" * 70)
    
    if provider:
        await provider.close()
        logger.info("✅ ElevenLabs provider closed")
    
    if redis_client:
        try:
            await redis_client.close()
            logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.warning(f"⚠️ Failed to close Redis connection: {e}")
    
    logger.info("✅ TTS_LABS microservice stopped")


# Initialize FastAPI app
app = FastAPI(
    title="TTS_LABS Streaming Service",
    description="Ultra-low latency TTS streaming with ElevenLabs WebSocket API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount FastRTC UI if available
fastrtc_handler = None
if FASTRTC_AVAILABLE:
    try:
        fastrtc_handler = FastRTCTTSHandler(redis_client=None)
        fastrtc_stream = Stream(
            handler=fastrtc_handler,
            modality="audio",
            mode="send-receive",
            ui_args={
                "title": "TTS_LABS Streaming Service",
                "description": "ElevenLabs ultra-low latency TTS preview"
            }
        )
        app = gr.mount_gradio_app(app, fastrtc_stream.ui, path="/fastrtc")
        logger.info("✅ FastRTC UI mounted at /fastrtc")
    except Exception as e:
        logger.warning(f"⚠️ FastRTC initialization failed: {e}")


# Request/Response Models
class SynthesizeRequest(BaseModel):
    """Request model for HTTP synthesis endpoint"""
    text: str = Field(..., min_length=1, description="Text to synthesize")
    emotion: str = Field(default="helpful", description="Emotion type (ignored for ElevenLabs)")
    voice: Optional[str] = Field(default=None, description="Voice ID override")
    language: Optional[str] = Field(default=None, description="Language code (optional)")


class SynthesizeResponse(BaseModel):
    """Response model for HTTP synthesis endpoint"""
    success: bool
    audio_data: Optional[str] = Field(default=None, description="Base64-encoded audio")
    sample_rate: Optional[int] = Field(default=None, description="Audio sample rate")
    duration_ms: Optional[float] = Field(default=None, description="Audio duration")
    sentences: int = Field(default=0, description="Number of sentences")
    first_chunk_latency_ms: Optional[float] = Field(default=None, description="Time to first audio chunk")
    error: Optional[str] = Field(default=None, description="Error message if success=False")


# WebSocket endpoint - Compatible with Orchestrator
@app.websocket("/api/v1/stream")
async def stream_tts(websocket: WebSocket, session_id: str = Query(...)):
    """
    WebSocket endpoint for streaming TTS synthesis.
    
    Compatible with the orchestrator's TTS service interface.
    Supports both sentence-based synthesis and continuous streaming.
    
    Message protocol:
    Client -> Server:
        {"type": "synthesize", "text": "...", "emotion": "helpful"}
        {"type": "stream_chunk", "text": "...", "emotion": "helpful"}  # Continuous mode
        {"type": "stream_end"}  # End continuous stream
        {"type": "cancel"}
        {"type": "ping"}
    
    Server -> Client:
        {"type": "connected", "session_id": "..."}
        {"type": "sentence_start", "index": 0, "text": "..."}
        {"type": "audio", "data": "<base64>", "index": 0, "sample_rate": 24000}
        {"type": "sentence_complete", "index": 0, "duration_ms": 450}
        {"type": "complete", "total_sentences": 2, "total_duration_ms": 1200}
        {"type": "error", "message": "..."}
    """
    await websocket.accept()
    
    logger.info(f"🔌 WebSocket session established: {session_id}")
    
    # Send connection confirmation
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "provider": "elevenlabs",
        "model": config.elevenlabs_model_id if config else "unknown",
        "timestamp": time.time()
    })
    
    # Check if provider is available
    if not provider:
        await websocket.send_json({
            "type": "error",
            "message": "ElevenLabs provider not initialized. Check ELEVENLABS_API_KEY."
        })
        await websocket.close()
        return
    
    # Continuous stream state
    stream_manager: Optional[ElevenLabsStreamManager] = None
    text_queue: Optional[asyncio.Queue] = None
    receiver_task: Optional[asyncio.Task] = None
    
    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "timeout",
                    "message": "No activity for 30 seconds"
                })
                break
            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket disconnected: {session_id}")
                break
            
            msg_type = message.get("type")
            
            # =================================================================
            # SYNTHESIZE: Sentence-based synthesis (compatible with orchestrator)
            # =================================================================
            if msg_type == "synthesize":
                text = message.get("text", "")
                voice = message.get("voice")
                
                if not text.strip():
                    await websocket.send_json({
                        "type": "error",
                        "message": "Empty text provided"
                    })
                    continue
                
                # Split into sentences
                sentences = split_into_sentences(text)
                if not sentences:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No valid sentences found"
                    })
                    continue
                
                logger.info(f"📝 Processing {len(sentences)} sentences for session {session_id}")
                
                # Process each sentence
                total_duration_ms = 0
                request_start = time.time()
                first_audio_time = None
                
                for i, sentence in enumerate(sentences):
                    await websocket.send_json({
                        "type": "sentence_start",
                        "index": i,
                        "text": sentence
                    })
                    
                    try:
                        # Create stream manager for this sentence
                        stream_mgr = ElevenLabsStreamManager(config)
                        audio_chunks = []
                        
                        async def collect_and_send(audio_bytes: bytes, sample_rate: int, metadata: Dict[str, Any]):
                            nonlocal first_audio_time
                            audio_chunks.append(audio_bytes)
                            
                            # Track first audio latency
                            if first_audio_time is None:
                                first_audio_time = time.time()
                                latency_ms = (first_audio_time - request_start) * 1000
                                logger.info(f"⚡ Time to First Audio: {latency_ms:.0f}ms")
                            
                            # Send to WebSocket
                            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                            await websocket.send_json({
                                "type": "audio",
                                "data": audio_b64,
                                "index": i,
                                "sample_rate": sample_rate,
                                "sentence": sentence,
                                "cached": False
                            })
                            
                            # Broadcast to FastRTC
                            await FastRTCTTSHandler.broadcast_audio(audio_bytes, sample_rate)
                        
                        await stream_mgr.synthesize_text(sentence, collect_and_send)
                        await stream_mgr.disconnect()
                        
                        # Calculate duration
                        combined_audio = b''.join(audio_chunks)
                        duration_ms = len(combined_audio) / (config.sample_rate * 2) * 1000
                        total_duration_ms += duration_ms
                        
                        await websocket.send_json({
                            "type": "sentence_complete",
                            "index": i,
                            "duration_ms": duration_ms
                        })
                        
                        # Send playback state for orchestrator tracking
                        await websocket.send_json({
                            "type": "sentence_playing",
                            "index": i,
                            "duration_ms": duration_ms,
                            "expected_complete_at": time.time() + (duration_ms / 1000.0)
                        })
                        
                    except Exception as e:
                        logger.error(f"Error synthesizing sentence {i}: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Synthesis error: {str(e)}"
                        })
                
                # Send completion
                first_chunk_latency = (first_audio_time - request_start) * 1000 if first_audio_time else 0
                await websocket.send_json({
                    "type": "complete",
                    "total_sentences": len(sentences),
                    "total_duration_ms": total_duration_ms,
                    "first_chunk_latency_ms": first_chunk_latency,
                    "cache_hits": 0,
                    "cache_misses": len(sentences)
                })
            
            # =================================================================
            # PREWARM: Pre-establish ElevenLabs connection for ultra-low latency
            # Triggered when VAD detects user speech (before LLM response)
            # Only connects - does NOT send BOS until first real text arrives
            # =================================================================
            elif msg_type == "prewarm":
                # If already prewarmed/streaming, acknowledge and continue
                if stream_manager is not None and stream_manager.is_connected:
                    await websocket.send_json({
                        "type": "prewarmed",
                        "status": "already_connected",
                        "timestamp": time.time()
                    })
                    continue
                
                # Create and connect stream manager (but don't send BOS yet!)
                stream_manager = ElevenLabsStreamManager(config)
                
                prewarm_start = time.time()
                if await stream_manager.connect():
                    # DO NOT send BOS here - ElevenLabs will close if we don't send text quickly
                    # BOS will be sent with first real text chunk for true ultra-low latency
                    
                    prewarm_duration_ms = (time.time() - prewarm_start) * 1000
                    logger.info(f"⚡ Pre-warmed ElevenLabs connection in {prewarm_duration_ms:.0f}ms for {session_id}")
                    
                    await websocket.send_json({
                        "type": "prewarmed",
                        "status": "ready",
                        "prewarm_duration_ms": prewarm_duration_ms,
                        "timestamp": time.time()
                    })
                else:
                    logger.error(f"❌ Prewarm failed for {session_id}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to prewarm ElevenLabs connection"
                    })
                    stream_manager = None
            
            # =================================================================
            # STREAM_CHUNK: Continuous streaming mode (ultra-low latency)
            # If prewarmed, uses already-open connection for faster first audio
            # =================================================================
            elif msg_type == "stream_chunk":
                text = message.get("text", "")
                
                if not text:
                    continue
                
                # Define the receiver function (used in both paths)
                async def stream_receiver():
                    """Receive audio and send to WebSocket."""
                    nonlocal stream_manager
                    async for audio_bytes, metadata in stream_manager.receive_audio():
                        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                        await websocket.send_json({
                            "type": "audio",
                            "data": audio_b64,
                            "index": metadata.get("chunk_index", 0),
                            "sample_rate": config.sample_rate,
                            "is_final": metadata.get("is_final", False)
                        })
                        await FastRTCTTSHandler.broadcast_audio(audio_bytes, config.sample_rate)
                
                # If prewarmed (connected but BOS not sent yet), send BOS with first text
                if stream_manager is not None and stream_manager.is_connected and not stream_manager.is_streaming:
                    # Ultra-fast path: connection already open, send BOS with text now
                    await stream_manager.send_bos(text)
                    receiver_task = asyncio.create_task(stream_receiver())
                    logger.info(f"🚀 Started stream (pre-warmed) for {session_id}")
                    
                elif stream_manager is not None and stream_manager.is_streaming:
                    # Already streaming, just send text chunk
                    await stream_manager.send_text_chunk(text, flush=True)
                    
                else:
                    # Cold start: Initialize stream manager from scratch
                    stream_manager = ElevenLabsStreamManager(config)
                    
                    # Connect and start receiver (cold start path)
                    if await stream_manager.connect():
                        await stream_manager.send_bos(text)
                        receiver_task = asyncio.create_task(stream_receiver())
                        logger.info(f"🎤 Started continuous stream (cold) for {session_id}")
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to connect to ElevenLabs"
                        })
                        stream_manager = None
            
            # =================================================================
            # STREAM_END: End continuous stream
            # =================================================================
            elif msg_type == "stream_end":
                if stream_manager:
                    await stream_manager.send_eos()
                    
                    # Wait for receiver to complete
                    if receiver_task:
                        try:
                            await asyncio.wait_for(receiver_task, timeout=10.0)
                        except asyncio.TimeoutError:
                            receiver_task.cancel()
                    
                    await stream_manager.disconnect()
                    stream_manager = None
                    receiver_task = None
                    
                    await websocket.send_json({
                        "type": "stream_complete",
                        "message": "Continuous stream ended"
                    })
            
            # =================================================================
            # CANCEL: Cancel current synthesis
            # =================================================================
            elif msg_type == "cancel":
                if stream_manager:
                    await stream_manager.disconnect()
                    stream_manager = None
                    if receiver_task:
                        receiver_task.cancel()
                        receiver_task = None
                
                await websocket.send_json({
                    "type": "cancelled",
                    "message": "Synthesis cancelled"
                })
            
            # =================================================================
            # PING: Keep-alive
            # =================================================================
            elif msg_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": time.time()
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error for session {session_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except Exception:
            pass
    finally:
        # Cleanup
        if stream_manager:
            await stream_manager.disconnect()
        if receiver_task and not receiver_task.done():
            receiver_task.cancel()
        
        if session_id in active_sessions:
            del active_sessions[session_id]


# HTTP Endpoints
@app.post("/api/v1/synthesize", response_model=SynthesizeResponse)
async def synthesize_text(request: SynthesizeRequest):
    """
    HTTP endpoint for single text synthesis (non-streaming).
    
    Returns base64-encoded audio data.
    """
    if not provider:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs provider not initialized. Check ELEVENLABS_API_KEY."
        )
    
    try:
        start_time = time.time()
        first_chunk_time = None
        
        # Split into sentences
        sentences = split_into_sentences(request.text)
        
        if not sentences:
            return SynthesizeResponse(
                success=False,
                error="No valid sentences found"
            )
        
        # Synthesize all sentences
        all_audio_chunks = []
        
        stream_mgr = ElevenLabsStreamManager(config)
        
        async def collect_audio(audio_bytes: bytes, sample_rate: int, metadata: Dict[str, Any]):
            nonlocal first_chunk_time
            all_audio_chunks.append(audio_bytes)
            if first_chunk_time is None:
                first_chunk_time = time.time()
            await FastRTCTTSHandler.broadcast_audio(audio_bytes, sample_rate)
        
        for sentence in sentences:
            await stream_mgr.synthesize_text(sentence, collect_audio)
        
        await stream_mgr.disconnect()
        
        # Concatenate audio
        combined_audio = b''.join(all_audio_chunks)
        duration_ms = len(combined_audio) / (config.sample_rate * 2) * 1000
        
        # Calculate latency
        first_chunk_latency_ms = None
        if first_chunk_time:
            first_chunk_latency_ms = (first_chunk_time - start_time) * 1000
        
        # Encode as base64
        audio_b64 = base64.b64encode(combined_audio).decode('utf-8')
        
        return SynthesizeResponse(
            success=True,
            audio_data=audio_b64,
            sample_rate=config.sample_rate,
            duration_ms=duration_ms,
            sentences=len(sentences),
            first_chunk_latency_ms=first_chunk_latency_ms
        )
    
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return SynthesizeResponse(
            success=False,
            error=str(e)
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    uptime_seconds = time.time() - app_start_time
    
    provider_status = "available" if provider else "unavailable"
    cache_status = "enabled" if cache else "disabled"
    
    status = "healthy" if provider else "unhealthy"
    
    cache_stats = {}
    if cache:
        cache_stats = cache.get_stats()
    
    return {
        "status": status,
        "service": "tts-labs",
        "provider": "elevenlabs",
        "model": config.elevenlabs_model_id if config else "unknown",
        "voice": config.elevenlabs_voice_id if config else "unknown",
        "uptime_seconds": uptime_seconds,
        "provider_status": provider_status,
        "cache": cache_status,
        "cache_stats": cache_stats,
        "active_sessions": len(active_sessions),
        "latency_optimization": config.optimize_streaming_latency if config else 0
    }


@app.get("/metrics")
async def get_metrics():
    """Get service performance metrics"""
    metrics = {
        "active_sessions": len(active_sessions),
        "uptime_seconds": time.time() - app_start_time,
        "provider": "elevenlabs",
        "model": config.elevenlabs_model_id if config else "unknown"
    }
    
    if cache:
        metrics["cache_stats"] = cache.get_stats()
    
    return metrics


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "TTS_LABS Streaming Service",
        "provider": "ElevenLabs",
        "model": config.elevenlabs_model_id if config else "unknown",
        "version": "1.0.0",
        "endpoints": {
            "stream": "WebSocket /api/v1/stream?session_id=<id>",
            "synthesize": "POST /api/v1/synthesize",
            "fastrtc_ui": "GET /fastrtc",
            "health": "GET /health",
            "metrics": "GET /metrics"
        },
        "features": {
            "ultra_low_latency": True,
            "continuous_streaming": True,
            "sentence_synthesis": True,
            "fastrtc_preview": FASTRTC_AVAILABLE
        }
    }


if __name__ == "__main__":
    try:
        import uvicorn
        
        port = int(os.getenv("TTS_LABS_PORT", "8006"))
        
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except ImportError:
        logger.error("uvicorn not installed")
