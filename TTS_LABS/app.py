"""
TTS_LABS Streaming Microservice FastAPI Application

WebSocket-based TTS streaming service using ElevenLabs stream-input API.
Optimized for ultra-low latency (<150ms first audio chunk) with eleven_turbo_v2_5.
"""

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, Tuple, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import TTSLabsConfig
from .sentence_splitter import split_into_sentences
from .elevenlabs_manager import ElevenLabsProvider, ElevenLabsStreamManager, stream_text_to_audio
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
    from leibniz_agent.services.shared.events import VoiceEvent, EventTypes
    from leibniz_agent.services.shared.event_broker import EventBroker
    REDIS_AVAILABLE = True
except ImportError:
    try:
        from shared.redis_client import get_redis_client, ping_redis
        from shared.events import VoiceEvent, EventTypes
        from shared.event_broker import EventBroker
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        logger.warning("Redis client utilities unavailable - TTS connection events disabled")

        async def get_redis_client():
            return None

        async def ping_redis(_client):
            return False
        
        # Stub classes when Redis not available
        VoiceEvent = None
        EventTypes = None
        EventBroker = None

# Global state
config: Optional[TTSLabsConfig] = None
provider: Optional[ElevenLabsProvider] = None
cache: Optional[AudioCache] = None
active_sessions: Dict[str, Dict[str, Any]] = {}
app_start_time: float = time.time()
redis_client = None
event_broker = None  # Event-driven architecture broker

# FastRTC support (optional)
try:
    import gradio as gr
    from fastrtc import Stream, AsyncStreamHandler
    FASTRTC_AVAILABLE = True
except ImportError:
    FASTRTC_AVAILABLE = False
    logger.warning("FastRTC not available - UI preview disabled")


class StreamState:
    """
    FIX #4: Track stream lifecycle to prevent orphaned handlers.
    
    Centralizes stream management to ensure:
    - Proper cleanup on disconnect
    - No orphaned tasks
    - Graceful error recovery
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.stream_manager: Optional[ElevenLabsStreamManager] = None
        self.receiver_task: Optional[asyncio.Task] = None
        self.is_streaming = False
        self.lock = asyncio.Lock()
        self.audio_chunks_received = 0
    
    async def cleanup(self):
        """Gracefully cleanup all resources."""
        async with self.lock:
            # Cancel receiver task if still running
            if self.receiver_task and not self.receiver_task.done():
                logger.debug(f"Cancelling receiver for {self.session_id}")
                self.receiver_task.cancel()
                try:
                    await asyncio.wait_for(self.receiver_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Disconnect stream manager
            if self.stream_manager and self.stream_manager.is_connected:
                logger.debug(f"Disconnecting stream manager for {self.session_id}")
                await self.stream_manager.disconnect()
            
            self.is_streaming = False
            logger.info(f"✅ Cleaned up {self.session_id} ({self.audio_chunks_received} chunks)")


# Global tracking of active streams
active_streams: Dict[str, StreamState] = {}


class FastRTCTTSHandler(AsyncStreamHandler if FASTRTC_AVAILABLE else object):
    """
    FastRTC AsyncStreamHandler that streams TTS audio to browser.

    DESIGN (matching Sarvam TTS):
    - Queue stores full audio chunks (sentences / large segments), not tiny frames
    - emit() slices those chunks into real-time frames (40ms by default)
    - No per-frame queue overflow; streaming feels continuous as soon as first chunk arrives
    """

    # Registry of active handler instances
    active_instances: Set["FastRTCTTSHandler"] = set()
    default_chunk_duration_ms: int = 40
    default_min_buffer_chunks: int = 1  # Start playing as soon as audio arrives

    def __init__(
        self,
        tts_queue=None,
        redis_client=None,
        chunk_duration_ms: Optional[int] = None,
        min_buffer_chunks: Optional[int] = None,
    ):
        if FASTRTC_AVAILABLE:
            super().__init__()

        self.tts_queue = tts_queue
        self.redis_client = redis_client
        self.session_id = f"fastrtc_tts_{int(time.time())}"
        self._started = False
        self._sample_rate = 24000

        # Streaming parameters and buffers
        self._chunk_duration_ms: Optional[int] = None
        self._chunk_size_samples: Optional[int] = None
        self._min_buffer_chunks: Optional[int] = None
        self._buffer_warmed: bool = False

        self._configure_stream_parameters(
            chunk_duration_ms=chunk_duration_ms,
            min_buffer_chunks=min_buffer_chunks,
        )

        # Queue of full audio chunks (np.int16 arrays) to be sliced in emit()
        self._audio_output_queue: asyncio.Queue = asyncio.Queue()
        self._current_audio_chunk: Optional[Tuple[np.ndarray, int]] = None
        self._chunk_position: int = 0
        self._playback_end_time: float = 0.0

        logger.info("🔊 FastRTC TTS Handler initialized")
        logger.info(f"   Handler instance: {id(self)}")
        logger.info(f"   Session ID: {self.session_id}")

    def _configure_stream_parameters(
        self,
        chunk_duration_ms: Optional[int] = None,
        min_buffer_chunks: Optional[int] = None,
    ) -> None:
        """Configure streaming parameters such as chunk duration and buffering."""
        resolved_chunk_ms = (
            chunk_duration_ms
            if chunk_duration_ms is not None
            else FastRTCTTSHandler.default_chunk_duration_ms
        )
        resolved_buffer_chunks = (
            min_buffer_chunks
            if min_buffer_chunks is not None
            else FastRTCTTSHandler.default_min_buffer_chunks
        )

        self._chunk_duration_ms = max(5, int(resolved_chunk_ms))
        self._min_buffer_chunks = max(1, int(resolved_buffer_chunks))
        self._chunk_size_samples = max(
            1, int(self._sample_rate * (self._chunk_duration_ms / 1000.0))
        )

    async def start_up(self):
        """Called when WebRTC stream starts."""
        self._started = True
        self._audio_output_queue = asyncio.Queue()
        self._current_audio_chunk = None
        self._chunk_position = 0
        self._buffer_warmed = False

        # Register this instance
        FastRTCTTSHandler.active_instances.add(self)

        logger.info("=" * 70)
        logger.info(f"🔌 FastRTC Stream STARTED | Session: {self.session_id}")
        logger.info(f"   Active instances: {len(FastRTCTTSHandler.active_instances)}")
        logger.info("=" * 70)
        logger.info("🚀 FastRTC TTS stream started")
        logger.info(f"   Handler instance: {id(self)} | Session: {self.session_id}")
        logger.info(f"   Active instances: {len(FastRTCTTSHandler.active_instances)}")
        logger.info(
            f"   Chunk duration: {self._chunk_duration_ms}ms | "
            f"Min buffer chunks: {self._min_buffer_chunks}"
        )
        logger.info("=" * 70)

        if not self.tts_queue:
            logger.info(
                "ℹ️ TTS queue not directly injected - waiting for broadcast audio"
            )
        else:
            logger.info("✅ TTS queue ready | Ready for audio streaming")
            logger.info("📊 Flow: Text → TTS Queue → FastRTC → Browser Speakers")

        await self._publish_connection_event()

    async def receive(self, audio) -> None:
        """
        Required by AsyncStreamHandler interface; not used for TTS output.
        """
        return  # TTS is output-only

    async def emit(self) -> Tuple[int, np.ndarray]:
        """
        Emit audio chunks to browser for playback.

        Behaviour:
        - Uses a small internal buffer (min_buffer_chunks) before starting playback
        - Then slices current audio chunk into frames of size _chunk_size_samples
        """
        # If FastRTC is not available, just return silence
        if not FASTRTC_AVAILABLE or not self._started:
            silence = np.zeros(self._chunk_size_samples or 1, dtype=np.int16)
            await asyncio.sleep((self._chunk_duration_ms or 40) / 1000.0)
            return self._sample_rate, silence

        silence_chunk = np.zeros(self._chunk_size_samples, dtype=np.int16)
        sleep_interval = self._chunk_duration_ms / 1000.0

        if self._audio_output_queue.empty() and self._current_audio_chunk is None:
            await asyncio.sleep(sleep_interval)
            return self._sample_rate, silence_chunk

        # Buffer warming: wait until we have enough audio to start playback
        if not self._buffer_warmed:
            buffered_chunks = self._audio_output_queue.qsize()
            if self._current_audio_chunk is not None:
                buffered_chunks += 1

            if buffered_chunks < self._min_buffer_chunks:
                await asyncio.sleep(sleep_interval)
                return self._sample_rate, silence_chunk

            self._buffer_warmed = True

        # Process current chunk or get next one
        if self._current_audio_chunk is None:
            try:
                self._current_audio_chunk = self._audio_output_queue.get_nowait()
                self._chunk_position = 0
            except asyncio.QueueEmpty:
                await asyncio.sleep(sleep_interval)
                return self._sample_rate, silence_chunk

        if self._current_audio_chunk is not None:
            audio_data, sample_rate = self._current_audio_chunk

            # Ensure audio_data is numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.int16)

            # Get next frame from current audio
            remaining = len(audio_data) - self._chunk_position
            if remaining > 0:
                take_samples = min(self._chunk_size_samples, remaining)
                chunk = audio_data[
                    self._chunk_position : self._chunk_position + take_samples
                ]
                self._chunk_position += take_samples

                # Ensure 1D array (FastRTC handles 1D as mono)
                if chunk.ndim > 1:
                    chunk = chunk.flatten()

                # Ensure int16
                if chunk.dtype != np.int16:
                    chunk = chunk.astype(np.int16)

                return sample_rate, chunk
            else:
                # Current chunk finished, get next
                self._current_audio_chunk = None
                self._chunk_position = 0

        await asyncio.sleep(sleep_interval)
        return self._sample_rate, silence_chunk

    async def add_audio(self, audio_bytes: bytes, sample_rate: int) -> None:
        """
        Add a full audio chunk to the output queue for streaming.

        Unlike the previous implementation, we do NOT split into 960-sample
        frames here. We store the full chunk and let emit() slice it, which
        avoids queue explosions and matches Sarvam's behaviour.
        """
        if not self._started or not audio_bytes:
            return

        try:
            # Convert bytes to numpy array (int16)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

            # Track when this audio chunk will finish playing
            duration_s = len(audio_bytes) / (sample_rate * 2)  # int16 = 2 bytes/sample
            self._playback_end_time = time.time() + duration_s

            # Add to queue as int16 (unbounded queue, async put)
            await self._audio_output_queue.put((audio_int16, sample_rate))
        except Exception as e:
            logger.error(f"Error adding audio chunk: {e}")

    @classmethod
    async def broadcast_audio(cls, audio_bytes: bytes, sample_rate: int) -> None:
        """
        Broadcast audio to all active FastRTC handler instances.
        """
        if not cls.active_instances:
            logger.debug("No active FastRTC instances to broadcast to")
            return

        for instance in cls.active_instances:
            try:
                await instance.add_audio(audio_bytes, sample_rate)
            except Exception as e:
                logger.error(f"Failed to broadcast to instance {id(instance)}: {e}")

    async def shutdown(self) -> None:
        """Cleanup resources when stream closes."""
        if self in FastRTCTTSHandler.active_instances:
            FastRTCTTSHandler.active_instances.remove(self)

        logger.info("=" * 70)
        logger.info("🛑 FastRTC TTS stream shutting down...")
        logger.info(f"   Handler instance: {id(self)} | Started: {self._started}")
        logger.info(
            f"   Remaining instances: {len(FastRTCTTSHandler.active_instances)}"
        )
        logger.info("=" * 70)

        self._started = False
        self._current_audio_chunk = None
        self._chunk_position = 0
        self._buffer_warmed = False

        # Clear queue
        while not self._audio_output_queue.empty():
            try:
                self._audio_output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("✅ FastRTC TTS stream closed")

    def copy(self) -> "FastRTCTTSHandler":
        """Create a copy of this handler for FastRTC."""
        return FastRTCTTSHandler(
            tts_queue=self.tts_queue,
            redis_client=self.redis_client,
            chunk_duration_ms=self._chunk_duration_ms,
            min_buffer_chunks=self._min_buffer_chunks,
        )

    async def _publish_connection_event(self) -> None:
        """Publish FastRTC connection event to Redis for orchestrator coordination."""
        if not self.redis_client:
            logger.warning(
                "⚠️ Redis client unavailable - cannot publish TTS connection event"
            )
            return

        payload = json.dumps(
            {
                "session_id": self.session_id,
                "timestamp": time.time(),
                "event": "tts_connected",
                "source": "tts_labs_fastrtc",
            }
        )
        channel = "leibniz:events:tts:connected"

        try:
            await self.redis_client.publish(channel, payload)
            logger.info(f"📡 Published TTS connection event → {channel}")
        except Exception as exc:
            logger.warning(f"⚠️ Failed to publish TTS connection event: {exc}")


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
        global redis_client, event_broker
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
                
                # Initialize event broker for event-driven architecture
                if EventBroker is not None:
                    event_broker = EventBroker(redis_client)
                    logger.info("✅ Event broker initialized for TTS_LABS")
                    
                    # Start event consumer in background
                    asyncio.create_task(_tts_event_consumer_loop())
                    logger.info("✅ TTS event consumer started")
                
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


# =============================================================================
# EVENT-DRIVEN TTS CONSUMER
# =============================================================================

async def _handle_tts_request_event(event: "VoiceEvent"):
    """
    Handle a TTS request event from Redis Stream.
    
    Synthesizes text using ElevenLabs and emits audio chunks back to 
    session-specific stream.
    
    Event payload:
        - text: str - Text to synthesize
        - voice: str (optional) - Voice ID override
        
    Emits:
        - EventTypes.TTS_CHUNK_READY - For each audio chunk
        - EventTypes.TTS_COMPLETE - When synthesis is done
    """
    global event_broker, provider, config
    
    if not event_broker or not provider or not config:
        logger.warning("TTS event handler called but broker/provider/config not ready")
        return
    
    session_id = event.session_id
    correlation_id = event.correlation_id
    text = event.payload.get("text", "")
    voice = event.payload.get("voice")
    
    if not text.strip():
        logger.warning(f"Empty text in TTS request for session {session_id}")
        return
    
    logger.info(f"🎤 [EVENT] TTS request for session {session_id}: {text[:50]}...")
    
    # Target stream for this session's TTS output
    output_stream = f"voice:tts:session:{session_id}"
    
    synthesis_start = time.time()
    first_chunk_time = None
    chunk_index = 0
    total_audio_bytes = 0
    
    try:
        # Create stream manager for synthesis
        stream_mgr = ElevenLabsStreamManager(config)
        
        async def send_audio_chunk(audio_bytes: bytes, sample_rate: int, metadata: Dict[str, Any]):
            nonlocal first_chunk_time, chunk_index, total_audio_bytes
            
            if first_chunk_time is None:
                first_chunk_time = time.time()
                latency_ms = (first_chunk_time - synthesis_start) * 1000
                logger.info(f"⚡ [EVENT] First TTS chunk latency: {latency_ms:.0f}ms")
            
            total_audio_bytes += len(audio_bytes)
            
            # Emit chunk ready event
            chunk_event = VoiceEvent(
                event_type=EventTypes.TTS_CHUNK_READY,
                session_id=session_id,
                source="tts_labs",
                correlation_id=correlation_id,
                payload={
                    "audio_base64": base64.b64encode(audio_bytes).decode('utf-8'),
                    "sample_rate": sample_rate,
                    "chunk_index": chunk_index,
                    "is_final": metadata.get("is_final", False),
                    "provider": "elevenlabs"
                },
                metadata={
                    "text_snippet": text[:30] if chunk_index == 0 else None
                }
            )
            
            await event_broker.publish(output_stream, chunk_event)
            chunk_index += 1
            
            # Also broadcast to FastRTC if available
            await FastRTCTTSHandler.broadcast_audio(audio_bytes, sample_rate)
        
        # Synthesize the text
        await stream_mgr.synthesize_text(text, send_audio_chunk)
        await stream_mgr.disconnect()
        
        # Emit completion event
        total_duration_ms = (time.time() - synthesis_start) * 1000
        audio_duration_ms = total_audio_bytes / (config.sample_rate * 2) * 1000 if total_audio_bytes > 0 else 0
        
        complete_event = VoiceEvent(
            event_type=EventTypes.TTS_COMPLETE,
            session_id=session_id,
            source="tts_labs",
            correlation_id=correlation_id,
            payload={
                "total_chunks": chunk_index,
                "total_audio_bytes": total_audio_bytes,
                "audio_duration_ms": audio_duration_ms,
                "synthesis_duration_ms": total_duration_ms,
                "provider": "elevenlabs"
            }
        )
        await event_broker.publish(output_stream, complete_event)
        
        logger.info(f"✅ [EVENT] TTS complete for {session_id}: {chunk_index} chunks, {audio_duration_ms:.0f}ms audio")
        
    except Exception as e:
        logger.error(f"❌ [EVENT] TTS synthesis error for {session_id}: {e}")
        
        # Emit error event
        error_event = VoiceEvent(
            event_type=EventTypes.ORCHESTRATOR_ERROR,
            session_id=session_id,
            source="tts_labs",
            correlation_id=correlation_id,
            payload={
                "error": str(e),
                "error_type": "tts_synthesis_error"
            }
        )
        await event_broker.publish(output_stream, error_event)


async def _tts_event_consumer_loop():
    """
    Background task that consumes TTS request events from Redis Streams.
    
    Listens to: voice:tts:requests (global request stream)
    """
    global event_broker, redis_client
    
    if not event_broker or not redis_client:
        logger.warning("Event consumer cannot start - broker/redis not available")
        return
    
    stream_key = "voice:tts:requests"
    last_id = "0"  # Start from beginning on startup
    
    logger.info(f"🎧 TTS event consumer listening on: {stream_key}")
    
    while True:
        try:
            # Read from stream with blocking
            result = await redis_client.xread(
                streams={stream_key: last_id},
                count=10,
                block=1000  # 1 second blocking
            )
            
            if not result:
                continue
            
            for stream_name, messages in result:
                for message_id, message_data in messages:
                    last_id = message_id
                    
                    try:
                        # Parse event from message
                        event = VoiceEvent.from_redis_dict(message_data)
                        
                        # Handle TTS request events
                        if event.event_type == EventTypes.TTS_REQUEST:
                            # Process in background to not block consumer
                            asyncio.create_task(_handle_tts_request_event(event))
                        else:
                            logger.debug(f"Ignoring event type: {event.event_type}")
                            
                    except Exception as e:
                        logger.error(f"Error processing TTS event {message_id}: {e}")
                        
        except asyncio.CancelledError:
            logger.info("TTS event consumer cancelled")
            break
        except Exception as e:
            logger.error(f"TTS event consumer error: {e}")
            await asyncio.sleep(1.0)  # Back off on error
    
    logger.info("TTS event consumer stopped")


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
    
    FIX #4: Uses StreamState for proper lifecycle tracking.
    
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
    
    # ✅ FIX 4.4: Use StreamState for lifecycle tracking
    stream_state = StreamState(session_id)
    active_streams[session_id] = stream_state
    
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
    
    # Continuous stream state (use stream_state for tracking, local vars for convenience)
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
                        # Track audio chunks received
                        stream_state.audio_chunks_received += 1
                        
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
                    stream_state.is_streaming = True
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
                        stream_state.is_streaming = True
                        logger.info(f"🎤 Started continuous stream (cold) for {session_id}")
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to connect to ElevenLabs"
                        })
                        stream_manager = None
            
            # =================================================================
            # STREAM_END: End continuous stream
            # FIX #3: Proper Async Task Completion & Cleanup
            # =================================================================
            elif msg_type == "stream_end":
                logger.debug(f"Stream end requested for {session_id}")
                
                # ✅ FIX 3.1: Proper EOS sending with grace period
                if stream_manager and stream_manager.is_connected:
                    logger.debug(f"Sending EOS to finalize stream")
                    await stream_manager.send_eos()
                    
                    # Give ElevenLabs time to finish and send isFinal
                    await asyncio.sleep(0.5)  # 500ms grace period
                
                # ✅ FIX 3.2: Proper receiver task completion with longer timeout
                if receiver_task:
                    try:
                        logger.debug(f"Waiting for receiver (max 15s)")
                        # Wait longer for graceful completion
                        await asyncio.wait_for(receiver_task, timeout=15.0)  # Was: 10.0
                        logger.info(f"✅ Receiver completed gracefully")
                    except asyncio.TimeoutError:
                        logger.warning(f"⚠️ Receiver timeout, forcing shutdown")
                        receiver_task.cancel()
                        try:
                            # Give it 2 seconds to handle cancellation
                            await asyncio.wait_for(receiver_task, timeout=2.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            logger.debug(f"Receiver stopped")
                    except Exception as e:
                        logger.error(f"❌ Receiver error: {e}")
                
                # ✅ FIX 3.3: Ensure proper disconnect
                if stream_manager and stream_manager.is_connected:
                    logger.debug(f"Disconnecting stream manager")
                    await stream_manager.disconnect()
                
                stream_manager = None
                receiver_task = None
                
                await websocket.send_json({
                    "type": "stream_complete",
                    "message": "Continuous stream ended",
                    "timestamp": time.time()
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
        # ✅ FIX 4.5: Cleanup via StreamState for proper resource management
        logger.debug(f"Cleaning up session {session_id}")
        
        # Update stream_state with current references for cleanup
        stream_state.stream_manager = stream_manager
        stream_state.receiver_task = receiver_task
        
        # Graceful cleanup
        await stream_state.cleanup()
        
        # Remove from active streams
        if session_id in active_streams:
            del active_streams[session_id]
        if session_id in active_sessions:
            del active_sessions[session_id]
        
        logger.info(f"✅ Session cleaned: {session_id}")


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
        
        # Collect audio chunks
        all_audio_chunks: list[bytes] = []
        
        async def collect_audio(audio_bytes: bytes, sample_rate: int, metadata: Dict[str, Any]):
            nonlocal first_chunk_time
            all_audio_chunks.append(audio_bytes)
            if first_chunk_time is None:
                first_chunk_time = time.time()
            # Broadcast to FastRTC preview UI
            try:
                await FastRTCTTSHandler.broadcast_audio(audio_bytes, sample_rate)
            except Exception as e:
                logger.debug(f"FastRTC broadcast failed (non-fatal): {e}")
        
        # Use stateless stream_text_to_audio for clean connection lifecycle
        # Join sentences with space to send as one continuous text
        full_text = " ".join(sentences)
        
        stats = await stream_text_to_audio(config, full_text, collect_audio)
        
        if stats.get("error"):
            logger.error(f"Streaming failed: {stats['error']}")
            return SynthesizeResponse(
                success=False,
                error=f"Streaming failed: {stats['error']}"
            )
        
        # Concatenate audio
        combined_audio = b''.join(all_audio_chunks)
        duration_ms = len(combined_audio) / (config.sample_rate * 2) * 1000
        
        # Calculate latency
        first_chunk_latency_ms = stats.get("first_chunk_latency_ms")
        
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
