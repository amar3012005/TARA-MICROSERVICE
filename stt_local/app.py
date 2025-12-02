"""
STT Local Microservice FastAPI Application

WebSocket-based speech transcription service using Silero VAD and Faster Whisper.

Endpoints:
    WebSocket /api/v1/transcribe/stream - Real-time speech transcription
    GET /health - Health check
    GET /metrics - Service metrics
    POST /admin/reset_session - Force reset session

Reference:
    services/stt_vad/app.py - FastAPI app structure
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from config import STTLocalConfig
from stt_manager import STTManager
from utils import validate_audio_chunk
from shared.redis_client import get_redis_client, close_redis_client, ping_redis
from fastrtc import Stream
from fastrtc_handler import FastRTCSTTLocalHandler
import gradio as gr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
stt_manager: STTManager = None
redis_client = None
active_sessions: Dict[str, Any] = {}
app_start_time: float = time.time()
fastrtc_handler: FastRTCSTTLocalHandler = None
fastrtc_stream: Stream = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for application startup/shutdown"""
    global stt_manager, redis_client
    
    logger.info("=" * 70)
    logger.info("🚀 Starting STT Local Microservice")
    logger.info("=" * 70)
    
    # Load configuration
    config = STTLocalConfig.from_env()
    logger.info(f"📋 Configuration loaded | Model: {config.whisper_model_size} | Device: {config.whisper_device}")
    
    # Initialize STT manager FIRST (before Redis) so service can start quickly
    logger.info("🎙️ Initializing STT Manager...")
    stt_manager = STTManager(config, None)  # Start without Redis
    logger.info("✅ STT Manager initialized")
    
    # Update FastRTC handler with STTManager
    global fastrtc_handler
    if fastrtc_handler:
        fastrtc_handler.stt_manager = stt_manager
        logger.info("✅ FastRTC handler updated with STTManager")
    
    logger.info("=" * 70)
    logger.info("✅ STT Local Microservice Ready (Redis connecting in background)")
    logger.info("=" * 70)
    
    # Start Redis connection in background task (non-blocking)
    async def connect_redis_background():
        """Connect to Redis in background without blocking service startup"""
        global redis_client
        logger.info("🔌 Connecting to Redis (background)...")
        
        # Set Redis environment variables if not already set
        if not os.getenv("LEIBNIZ_REDIS_HOST"):
            redis_host = os.getenv("REDIS_HOST", "localhost")
            os.environ["LEIBNIZ_REDIS_HOST"] = redis_host
        if not os.getenv("LEIBNIZ_REDIS_PORT"):
            redis_port = os.getenv("REDIS_PORT", "6379")
            os.environ["LEIBNIZ_REDIS_PORT"] = redis_port
        
        for retry_attempt in range(5):  # 5 retry attempts
            try:
                redis_client = await asyncio.wait_for(
                    get_redis_client(),
                    timeout=5.0  # Shorter timeout per attempt
                )
                # Test connection
                await ping_redis(redis_client)
                logger.info(f"✅ Redis connected (attempt {retry_attempt + 1})")
                # Update STT manager with Redis client
                stt_manager.redis_client = redis_client
                break
            except asyncio.TimeoutError:
                logger.warning(f"⏳ Redis connection timeout (attempt {retry_attempt + 1}/5)")
                if retry_attempt < 4:
                    await asyncio.sleep(2.0)  # Wait before retry
            except Exception as e:
                logger.warning(f"⚠️ Redis error: {e} (attempt {retry_attempt + 1}/5)")
                if retry_attempt < 4:
                    await asyncio.sleep(2.0)
        
        if redis_client is None:
            logger.warning("⚠️ Redis unavailable - service running in degraded mode")
    
    # Start Redis connection in background
    asyncio.create_task(connect_redis_background())
    
    yield
    
    # Shutdown
    logger.info("=" * 70)
    logger.info("🛑 Shutting down STT Local microservice...")
    logger.info("=" * 70)
    
    # Close Redis
    logger.info("🔌 Closing Redis connection...")
    await close_redis_client(redis_client)
    logger.info("✅ Redis connection closed")
    
    logger.info("✅ STT Local microservice stopped")


# Initialize FastAPI app
app = FastAPI(
    title="Leibniz STT Local Service",
    description="Speech transcription service using Silero VAD and Faster Whisper",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create FastRTC handler (stt_manager will be injected in lifespan)
fastrtc_handler = FastRTCSTTLocalHandler(stt_manager=None)

# Create FastRTC stream
fastrtc_stream = Stream(
    handler=fastrtc_handler,
    modality="audio",
    mode="send-receive",
    ui_args={
        "title": "Leibniz STT Local Transcription Service",
        "description": "Speak into your browser microphone. Audio streams directly to the STT Local service for real-time transcription using Silero VAD and Faster Whisper. "
                     "Check Docker console logs for pipeline progress, speech detection, and transcript fragments."
    }
)

# Mount FastRTC stream UI to FastAPI app
try:
    app = gr.mount_gradio_app(app, fastrtc_stream.ui, path="/fastrtc")
    logger.info("✅ FastRTC stream UI mounted at /fastrtc")
except Exception as e:
    logger.error(f"❌ Failed to mount FastRTC UI: {e}")
    logger.info("⚠️ FastRTC UI will not be available")

@app.get("/client")
async def serve_client():
    # Ensure static directory exists
    if not os.path.exists("static/client.html"):
        # Create simple fallback client if missing
        os.makedirs("static", exist_ok=True)
        with open("static/client.html", "w") as f:
            f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leibniz STT Local Client</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
        h1 { color: #2c3e50; }
        #status { padding: 1rem; border-radius: 4px; margin-bottom: 1rem; background: #eee; }
        #transcripts { height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 1rem; border-radius: 4px; background: #f9f9f9; }
        .fragment { color: #666; }
        .final { color: #000; font-weight: bold; }
        button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; background: #3498db; color: white; border: none; border-radius: 4px; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        button.stop { background: #e74c3c; }
    </style>
</head>
<body>
    <h1>🎙️ Leibniz STT Local Client</h1>
    <div id="status">Disconnected</div>
    <div>
        <button id="startBtn">Start Microphone</button>
        <button id="stopBtn" class="stop" disabled>Stop</button>
    </div>
    <h3>Transcripts</h3>
    <div id="transcripts"></div>

    <script>
        let socket;
        let mediaStream;
        let audioContext;
        let processor;
        const status = document.getElementById('status');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const transcripts = document.getElementById('transcripts');

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/v1/transcribe/stream?session_id=client_${Date.now()}`;
            
            status.textContent = 'Connecting...';
            socket = new WebSocket(wsUrl);

            socket.onopen = () => {
                status.textContent = 'Connected - Ready to record';
                status.style.background = '#d4edda';
                startBtn.disabled = false;
            };

            socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'fragment') {
                    const p = document.createElement('div');
                    p.textContent = `[${new Date().toLocaleTimeString()}] ${data.text}`;
                    p.className = data.is_final ? 'final' : 'fragment';
                    transcripts.appendChild(p);
                    transcripts.scrollTop = transcripts.scrollHeight;
                }
            };

            socket.onclose = () => {
                status.textContent = 'Disconnected';
                status.style.background = '#f8d7da';
                startBtn.disabled = false;
                stopBtn.disabled = true;
            };
        }

        startBtn.onclick = async () => {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);

                source.connect(processor);
                processor.connect(audioContext.destination);

                processor.onaudioprocess = (e) => {
                    if (socket && socket.readyState === WebSocket.OPEN) {
                        const inputData = e.inputBuffer.getChannelData(0);
                        // Convert float32 to int16
                        const pcmData = new Int16Array(inputData.length);
                        for (let i = 0; i < inputData.length; i++) {
                            pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                        }
                        socket.send(pcmData.buffer);
                    }
                };

                startBtn.disabled = true;
                stopBtn.disabled = false;
                status.textContent = 'Recording...';
                
                if (socket.readyState !== WebSocket.OPEN) connect();
                
            } catch (err) {
                console.error(err);
                status.textContent = 'Error: ' + err.message;
            }
        };

        stopBtn.onclick = () => {
            if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
            if (audioContext) audioContext.close();
            if (socket) socket.close();
            startBtn.disabled = false;
            stopBtn.disabled = true;
            status.textContent = 'Stopped';
        };

        // Initial connection
        connect();
    </script>
</body>
</html>
            """)
    return FileResponse("static/client.html")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.websocket("/api/v1/transcribe/stream")
async def transcribe_stream(websocket: WebSocket, session_id: str = Query(...)):
    """
    WebSocket endpoint for real-time speech transcription.
    
    Supports continuous streaming mode for ultra-low latency.
    
    Args:
        websocket: WebSocket connection
        session_id: Session identifier
    """
    await websocket.accept()

    # Send welcome message
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "timestamp": time.time(),
        "message": "Ready for audio streaming"
    })

    # Create audio queue for this session
    audio_queue = asyncio.Queue(maxsize=20)
    active_sessions[session_id] = {
        "audio_queue": audio_queue,
        "websocket": websocket,
        "last_activity": time.time(),
        "continuous_mode": False,
        "capture_task": None
    }
    
    logger.info("=" * 70)
    logger.info(f"🔌 WebSocket session established")
    logger.info(f"   Session ID: {session_id}")
    logger.info(f"   Remote: {websocket.client}")
    logger.info("=" * 70)
    
    logger.info(f"📡 Ready to receive audio chunks")

    try:
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=30.0  # 30s idle timeout
                )
            except (WebSocketDisconnect, RuntimeError):
                logger.info(f"🔌 WebSocket disconnected/closed for session {session_id}")
                break
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({
                        "type": "timeout",
                        "text": "",
                        "session_id": session_id,
                        "timestamp": time.time()
                    })
                except:
                    pass
                break

            # Handle binary audio data
            if "bytes" in message:
                audio_data = message["bytes"]
                
                logger.debug(f"📥 Audio chunk received | Size: {len(audio_data)} bytes")
                
                # Leaky Bucket Strategy: Add to queue, dropping oldest if full
                try:
                    audio_queue.put_nowait(audio_data)
                except asyncio.QueueFull:
                    try:
                        dropped_chunk = audio_queue.get_nowait()
                        audio_queue.put_nowait(audio_data)
                        logger.debug(f"⚠️ Queue full - dropped oldest chunk ({len(dropped_chunk)} bytes)")
                    except asyncio.QueueEmpty:
                        logger.warning(f"⚠️ Queue state inconsistent")
                        try:
                            audio_queue.put_nowait(audio_data)
                        except asyncio.QueueFull:
                            logger.error(f"❌ Failed to add chunk even after dropping oldest")
                            continue
                
                active_sessions[session_id]["last_activity"] = time.time()
                
                # Auto-start continuous capture
                if not active_sessions[session_id]["continuous_mode"]:
                    logger.info("=" * 70)
                    logger.info("🚀 Auto-starting continuous capture")
                    logger.info("   Processing audio in real-time...")
                    logger.info("=" * 70)
                    
                    active_sessions[session_id]["continuous_mode"] = True
                    capture_task = asyncio.create_task(
                        continuous_capture_loop(session_id, audio_queue, websocket)
                    )
                    active_sessions[session_id]["capture_task"] = capture_task

            elif "text" in message:
                # Parse JSON command
                import json
                try:
                    command = json.loads(message["text"])

                    if command.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "session_id": session_id,
                            "timestamp": time.time()
                        })

                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "text": "Invalid JSON command",
                        "session_id": session_id,
                        "timestamp": time.time()
                    })

    except (WebSocketDisconnect, RuntimeError):
        logger.info(f"🔌 WebSocket disconnected/closed for session {session_id}")
    
    except Exception as e:
        logger.error(f"❌ Unexpected WebSocket error for session {session_id}: {e}")

    finally:
        # Cleanup
        if session_id in active_sessions:
            session_data = active_sessions[session_id]

            # Cancel any running capture task
            capture_task = session_data.get("capture_task")
            if capture_task and not capture_task.done():
                capture_task.cancel()
                try:
                    await capture_task
                except asyncio.CancelledError:
                    pass

            del active_sessions[session_id]


async def continuous_capture_loop(session_id: str, audio_queue: asyncio.Queue, websocket: WebSocket):
    """
    Continuous capture loop for real-time transcription.
    
    Processes audio chunks as they arrive and streams transcripts back.
    """
    logger.info("=" * 70)
    logger.info(f"🔄 Starting continuous capture loop | Session: {session_id}")
    logger.info("📊 Processing audio chunks in real-time")
    logger.info("=" * 70)

    # Streaming callback for real-time transcripts
    def streaming_callback(text: str, is_final: bool):
        """Send transcript fragments"""
        async def send_fragment():
            try:
                fragment = {
                    "type": "fragment",
                    "text": text,
                    "is_final": is_final,
                    "session_id": session_id,
                    "timestamp": time.time()
                }
                await websocket.send_json(fragment)
                
                # Log to Docker console
                status = "✅ FINAL" if is_final else "🔄 PARTIAL"
                logger.info(f"📝 [{status}] STT Fragment")
                logger.info(f"   Text: '{text[:150]}'")
                if is_final:
                    logger.info(f"   Session: {session_id}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to send fragment: {e}")
    
        asyncio.create_task(send_fragment())

    try:
        while True:
            # Get audio chunk with short timeout
            try:
                audio_chunk = await asyncio.wait_for(
                    audio_queue.get(),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                continue

            # Process audio chunk immediately
            try:
                if stt_manager:
                    chunk_size = len(audio_chunk)
                    logger.debug(f"📥 Audio chunk received | Size: {chunk_size} bytes | Session: {session_id}")
                    
                    await stt_manager.process_audio_chunk_streaming(
                        session_id,
                        audio_chunk,
                        streaming_callback
                    )

            except Exception as e:
                logger.error(f"❌ Error processing audio chunk for session {session_id}: {e}")

    except asyncio.CancelledError:
        logger.info(f"🛑 Continuous capture loop cancelled | Session: {session_id}")
        raise
    except Exception as e:
        logger.error(f"❌ Continuous capture loop error | Session: {session_id} | Error: {e}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Service health status
    """
    uptime_seconds = time.time() - app_start_time
    
    # Check Redis
    redis_connected = False
    if redis_client:
        try:
            await ping_redis(redis_client)
            redis_connected = True
        except Exception:
            redis_connected = False
    
    # Get STT metrics
    stt_metrics = {}
    if stt_manager:
        try:
            stt_metrics = stt_manager.get_performance_metrics()
        except AttributeError:
            stt_metrics = {
                "total_captures": getattr(stt_manager, 'capture_count', 0),
                "avg_capture_time_ms": 0
            }
    
    # Determine status
    status = "healthy"
    if not redis_connected:
        status = "degraded"
    
    return {
        "status": status,
        "service": "stt-local",
        "uptime_seconds": uptime_seconds,
        "active_sessions": len(active_sessions),
        "redis_connected": redis_connected,
        "total_captures": stt_metrics.get("total_captures", 0),
        "avg_capture_time_ms": stt_metrics.get("avg_capture_time_ms", 0)
    }


@app.get("/metrics")
async def get_metrics():
    """
    Get service performance metrics.
    
    Returns:
        dict: Performance metrics
    """
    if not stt_manager:
        return {"error": "STT manager not initialized"}
    
    metrics = stt_manager.get_performance_metrics()
    metrics["active_sessions"] = len(active_sessions)
    
    return metrics


@app.post("/admin/reset_session")
async def reset_session():
    """
    Force reset session (admin endpoint).
    
    Returns:
        dict: Reset confirmation
    """
    if stt_manager:
        stt_manager.consecutive_timeouts = 0
    
    return {"status": "success", "message": "Session reset"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8006,
        log_level="info",
        reload=False
    )

