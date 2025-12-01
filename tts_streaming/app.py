"""
TTS Streaming Microservice FastAPI Application

WebSocket-based TTS streaming service with parallel queue processing.
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

from .config import TTSStreamingConfig
from .sentence_splitter import split_into_sentences
from .lemonfox_provider import LemonFoxProvider
from .audio_cache import AudioCache
from .tts_queue import TTSStreamingQueue
from .fastrtc_handler import FastRTCTTSHandler

import gradio as gr
from fastrtc import Stream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
config: Optional[TTSStreamingConfig] = None
provider: Optional[LemonFoxProvider] = None
cache: Optional[AudioCache] = None
active_sessions: Dict[str, Dict[str, Any]] = {}
app_start_time: float = time.time()
fastrtc_handler: Optional[FastRTCTTSHandler] = None
fastrtc_stream: Optional[Stream] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for application startup/shutdown"""
    global config, provider, cache
    
    logger.info("=" * 70)
    logger.info("🚀 Starting TTS Streaming Microservice")
    logger.info("=" * 70)
    
    # Load configuration
    try:
        config = TTSStreamingConfig.from_env()
        logger.info(f"📋 Configuration loaded | Voice: {config.lemonfox_voice} | Language: {config.lemonfox_language}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise
    
    # Initialize LemonFox provider
    try:
        if not config.lemonfox_api_key:
            logger.warning("⚠️ LEMONFOX_API_KEY not set - service will not function properly")
            provider = None
        else:
            provider = LemonFoxProvider(
                api_key=config.lemonfox_api_key,
                voice=config.lemonfox_voice,
                language=config.lemonfox_language
            )
            logger.info("✅ LemonFox provider initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LemonFox provider: {e}")
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
    
    # Initialize FastRTC handler (will be set up after app creation)
    global fastrtc_handler, fastrtc_stream
    fastrtc_handler = None
    fastrtc_stream = None
    
    logger.info("=" * 70)
    logger.info("✅ TTS Streaming Microservice Ready")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown
    logger.info("=" * 70)
    logger.info("🛑 Shutting down TTS Streaming microservice...")
    logger.info("=" * 70)
    
    # Close provider session
    if provider:
        await provider.close()
        logger.info("✅ LemonFox provider closed")
    
    logger.info("✅ TTS Streaming microservice stopped")


# Initialize FastAPI app
app = FastAPI(
    title="Leibniz TTS Streaming Service",
    description="WebSocket-based TTS streaming with parallel queue processing",
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

# Initialize FastRTC handler and mount UI
try:
    fastrtc_handler = FastRTCTTSHandler(tts_queue=None)  # Will be injected per session
    fastrtc_stream = Stream(
        handler=fastrtc_handler,
        modality="audio",
        mode="send-receive",  # send-receive for bidirectional
        ui_args={
            "title": "Leibniz TTS Streaming Service",
            "description": "Enter text and hear it synthesized with real-time streaming. "
                         "Audio streams directly from the TTS service to your browser speakers. "
                         "Use the HTTP API endpoint /api/v1/fastrtc/synthesize to send text."
        }
    )
    # Mount FastRTC UI
    app = gr.mount_gradio_app(app, fastrtc_stream.ui, path="/fastrtc")
    logger.info("✅ FastRTC UI mounted at /fastrtc")
except Exception as e:
    logger.warning(f"⚠️ FastRTC initialization failed: {e}")
    logger.warning("   FastRTC UI will not be available")
    fastrtc_handler = None
    fastrtc_stream = None


# Request/Response Models
class SynthesizeRequest(BaseModel):
    """Request model for HTTP synthesis endpoint"""
    text: str = Field(..., min_length=1, description="Text to synthesize")
    emotion: str = Field(default="helpful", description="Emotion type")
    voice: Optional[str] = Field(default=None, description="Voice name")
    language: Optional[str] = Field(default=None, description="Language code")


class SynthesizeResponse(BaseModel):
    """Response model for HTTP synthesis endpoint"""
    success: bool
    audio_data: Optional[str] = Field(default=None, description="Base64-encoded audio")
    sample_rate: Optional[int] = Field(default=None, description="Audio sample rate")
    duration_ms: Optional[float] = Field(default=None, description="Audio duration")
    sentences: int = Field(default=0, description="Number of sentences")
    error: Optional[str] = Field(default=None, description="Error message if success=False")


# WebSocket endpoint
@app.websocket("/api/v1/stream")
async def stream_tts(websocket: WebSocket, session_id: str = Query(...)):
    """
    WebSocket endpoint for streaming TTS synthesis.
    
    Message protocol:
    Client -> Server:
        {"type": "synthesize", "text": "...", "emotion": "helpful"}
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
        "timestamp": time.time()
    })
    
    # Check if provider is available
    if not provider:
        await websocket.send_json({
            "type": "error",
            "message": "LemonFox provider not initialized. Check LEMONFOX_API_KEY."
        })
        await websocket.close()
        return
    
    # Create TTS queue for this session
    queue = None
    consumer_task = None
    
    try:
        # Audio callback for WebSocket streaming and FastRTC
        async def audio_callback(audio_bytes: bytes, sample_rate: int, metadata: Dict[str, Any]):
            """Callback to send audio chunks via WebSocket and FastRTC"""
            try:
                # Send to WebSocket
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                await websocket.send_json({
                    "type": "audio",
                    "data": audio_b64,
                    "index": metadata.get('sentence_index', 0),
                    "sample_rate": sample_rate,
                    "sentence": metadata.get('sentence', ''),
                    "cached": metadata.get('cached', False)
                })
                
                # Also send to FastRTC handler if available
                if fastrtc_handler:
                    await fastrtc_handler.add_audio_chunk(audio_bytes, sample_rate)
            except Exception as e:
                logger.error(f"Error sending audio chunk: {e}")
        
        # Initialize queue
        queue = TTSStreamingQueue(
            provider=provider,
            cache=cache,
            config=config,
            audio_callback=audio_callback
        )
        
        active_sessions[session_id] = {
            "queue": queue,
            "websocket": websocket,
            "consumer_task": None
        }
        
        # Message loop
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                # Send timeout message
                await websocket.send_json({
                    "type": "timeout",
                    "message": "No activity for 30 seconds"
                })
                break
            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket disconnected: {session_id}")
                break
            
            msg_type = message.get("type")
            
            if msg_type == "synthesize":
                # Extract parameters
                text = message.get("text", "")
                emotion = message.get("emotion", "helpful")
                voice = message.get("voice")
                language = message.get("language")
                
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
                
                # Cancel any existing consumer
                if consumer_task and not consumer_task.done():
                    queue.cancel()
                    try:
                        await consumer_task
                    except Exception:
                        pass
                
                # Reset queue
                queue.reset()
                
                # Enqueue sentences
                await queue.enqueue_sentences(sentences, emotion, voice, language)
                
                # Send sentence start notifications
                for i, sentence in enumerate(sentences):
                    await websocket.send_json({
                        "type": "sentence_start",
                        "index": i,
                        "text": sentence
                    })
                
                # Start consumer task
                consumer_task = asyncio.create_task(queue.consume_queue())
                active_sessions[session_id]["consumer_task"] = consumer_task
                
                # Wait for consumer to complete
                try:
                    stats = await consumer_task
                    
                    # Send completion message
                    await websocket.send_json({
                        "type": "complete",
                        "total_sentences": len(sentences),
                        "total_duration_ms": stats.get('total_duration_ms', 0),
                        "sentences_played": stats.get('sentences_played', 0),
                        "sentences_failed": stats.get('sentences_failed', 0),
                        "cache_hits": stats.get('cache_hits', 0),
                        "cache_misses": stats.get('cache_misses', 0)
                    })
                except Exception as e:
                    logger.error(f"Consumer error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Processing error: {str(e)}"
                    })
            
            elif msg_type == "cancel":
                # Cancel current synthesis
                if queue:
                    queue.cancel()
                if consumer_task and not consumer_task.done():
                    consumer_task.cancel()
                    try:
                        await consumer_task
                    except Exception:
                        pass
                
                await websocket.send_json({
                    "type": "cancelled",
                    "message": "Synthesis cancelled"
                })
            
            elif msg_type == "ping":
                # Respond to ping
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
        if session_id in active_sessions:
            session_data = active_sessions[session_id]
            consumer_task = session_data.get("consumer_task")
            
            if consumer_task and not consumer_task.done():
                if queue:
                    queue.cancel()
                consumer_task.cancel()
                try:
                    await consumer_task
                except Exception:
                    pass
            
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
            detail="LemonFox provider not initialized. Check LEMONFOX_API_KEY."
        )
    
    try:
        # Split into sentences
        sentences = split_into_sentences(request.text)
        
        if not sentences:
            return SynthesizeResponse(
                success=False,
                error="No valid sentences found"
            )
        
        # Synthesize all sentences
        all_audio_chunks = []
        total_duration_ms = 0.0
        
        for sentence in sentences:
            try:
                # Check cache
                cached_path = None
                if cache:
                    cached_path = cache.get_cached_audio(
                        sentence,
                        request.voice or config.lemonfox_voice,
                        request.language or config.lemonfox_language,
                        "lemonfox",
                        request.emotion
                    )
                
                if cached_path:
                    import soundfile as sf
                    audio_data, sample_rate = await asyncio.to_thread(sf.read, cached_path)
                    audio_bytes = audio_data.tobytes()
                else:
                    # Synthesize
                    audio_bytes = await provider.synthesize(
                        text=sentence,
                        voice=request.voice,
                        language=request.language,
                        emotion=request.emotion
                    )
                    sample_rate = config.sample_rate
                
                all_audio_chunks.append(audio_bytes)
                duration_ms = len(audio_bytes) / (sample_rate * 2) * 1000
                total_duration_ms += duration_ms
                
            except Exception as e:
                logger.error(f"Error synthesizing sentence: {e}")
                return SynthesizeResponse(
                    success=False,
                    error=f"Synthesis failed: {str(e)}"
                )
        
        # Concatenate all audio chunks
        combined_audio = b''.join(all_audio_chunks)
        
        # Encode as base64
        audio_b64 = base64.b64encode(combined_audio).decode('utf-8')
        
        return SynthesizeResponse(
            success=True,
            audio_data=audio_b64,
            sample_rate=sample_rate,
            duration_ms=total_duration_ms,
            sentences=len(sentences)
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
    
    # Determine overall status
    if not provider:
        status = "unhealthy"
    else:
        status = "healthy"
    
    cache_stats = {}
    if cache:
        cache_stats = cache.get_stats()
    
    return {
        "status": status,
        "service": "tts-streaming",
        "uptime_seconds": uptime_seconds,
        "provider": provider_status,
        "cache": cache_status,
        "cache_stats": cache_stats,
        "active_sessions": len(active_sessions)
    }


@app.get("/metrics")
async def get_metrics():
    """Get service performance metrics"""
    metrics = {
        "active_sessions": len(active_sessions),
        "uptime_seconds": time.time() - app_start_time
    }
    
    # Aggregate queue stats from active sessions
    total_queued = 0
    total_synthesized = 0
    total_failed = 0
    
    for session_data in active_sessions.values():
        queue = session_data.get("queue")
        if queue:
            stats = queue.get_stats()
            total_queued += stats.get('sentences_queued', 0)
            total_synthesized += stats.get('sentences_synthesized', 0)
            total_failed += stats.get('sentences_failed', 0)
    
    metrics.update({
        "total_sentences_queued": total_queued,
        "total_sentences_synthesized": total_synthesized,
        "total_sentences_failed": total_failed
    })
    
    if cache:
        metrics["cache_stats"] = cache.get_stats()
    
    return metrics


@app.post("/api/v1/fastrtc/synthesize")
async def fastrtc_synthesize(request: SynthesizeRequest):
    """
    HTTP endpoint for FastRTC text synthesis.
    Synthesizes text and streams to FastRTC handler for browser playback.
    """
    if not provider:
        raise HTTPException(
            status_code=503,
            detail="LemonFox provider not initialized. Check LEMONFOX_API_KEY."
        )
    
    if not fastrtc_handler:
        raise HTTPException(
            status_code=503,
            detail="FastRTC handler not initialized"
        )
    
    try:
        # Split into sentences
        sentences = split_into_sentences(request.text)
        
        if not sentences:
            raise HTTPException(status_code=400, detail="No valid sentences found")
        
        # Create TTS queue with FastRTC callback
        async def fastrtc_audio_callback(audio_bytes: bytes, sample_rate: int, metadata: Dict[str, Any]):
            """Callback to send audio to FastRTC"""
            await fastrtc_handler.add_audio_chunk(audio_bytes, sample_rate)
        
        queue = TTSStreamingQueue(
            provider=provider,
            cache=cache,
            config=config,
            audio_callback=fastrtc_audio_callback
        )
        
        # Enqueue sentences
        await queue.enqueue_sentences(sentences, request.emotion, request.voice, request.language)
        
        # Consume queue (this will stream audio to FastRTC)
        stats = await queue.consume_queue()
        
        return {
            "success": True,
            "sentences": len(sentences),
            "sentences_played": stats.get('sentences_played', 0),
            "total_duration_ms": stats.get('total_duration_ms', 0)
        }
    
    except Exception as e:
        logger.error(f"FastRTC synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Leibniz TTS Streaming Service",
        "version": "1.0.0",
        "endpoints": {
            "stream": "WebSocket /api/v1/stream?session_id=<id>",
            "synthesize": "POST /api/v1/synthesize",
            "fastrtc_synthesize": "POST /api/v1/fastrtc/synthesize",
            "fastrtc_ui": "GET /fastrtc",
            "health": "GET /health",
            "metrics": "GET /metrics"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("TTS_STREAMING_PORT", "8005"))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

