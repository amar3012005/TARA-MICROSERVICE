"""
STT/VAD Microservice FastAPI Application

WebSocket-based speech transcription service using Sarvam AI's Saarika model.

Endpoints:
    WebSocket /api/v1/transcribe/stream - Real-time speech transcription
    GET /health - Health check
    GET /metrics - Service metrics
    POST /admin/reset_session - Reset Sarvam streaming buffers

Reference:
    https://docs.sarvam.ai/api-reference-docs/getting-started/models/saarika
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

from config import VADConfig
from vad_manager import VADManager
from utils import validate_audio_chunk, format_transcript_fragment
from sarvam_client import SarvamSTTClient
from shared.redis_client import get_redis_client, close_redis_client, ping_redis, get_event_broker
from shared.events import VoiceEvent, EventTypes
from fastrtc import Stream
from fastrtc_handler import FastRTCSTTHandler
import gradio as gr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
vad_manager: VADManager = None
redis_client = None
event_broker = None
sarvam_client: SarvamSTTClient = None
active_sessions: Dict[str, Any] = {}
app_start_time: float = time.time()
fastrtc_handler: FastRTCSTTHandler = None
fastrtc_stream: Stream = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for application startup/shutdown"""
    global vad_manager, redis_client, event_broker, sarvam_client
    
    logger.info("=" * 70)
    logger.info("🚀 Starting STT/VAD Microservice (Sarvam Saarika)")
    logger.info("=" * 70)
    
    # Load configuration
    config = VADConfig.from_env()
    logger.info(f"📋 Configuration loaded | Model: {config.model_name} | Language hint: {config.language_code}")
    
    # Initialize Sarvam client (Legacy/Fallback)
    # Note: Streaming uses SarvamStreamingClient inside VADManager
    logger.info("🌐 Initializing Sarvam STT client...")
    sarvam_client = SarvamSTTClient(
        api_key=config.sarvam_api_key,
        model=config.model_name,
        sample_rate=config.sample_rate,
        channels=config.channels,
        endpoint=config.sarvam_endpoint,
    )
    logger.info("✅ Sarvam client ready")
    
    # Initialize VAD manager FIRST (before Redis) so service can start quickly
    logger.info("🎙️ Initializing VAD Manager...")
    vad_manager = VADManager(config, None, sarvam_client=sarvam_client)
    logger.info("✅ VAD Manager initialized")
    
    # Update FastRTC handler with VADManager
    global fastrtc_handler
    if fastrtc_handler:
        fastrtc_handler.vad_manager = vad_manager
        logger.info("✅ FastRTC handler updated with VADManager")
    
    logger.info("=" * 70)
    logger.info("✅ STT/VAD Microservice Ready (Redis connecting in background)")
    logger.info("=" * 70)
    
    # Start Redis + EventBroker connection in background task (non-blocking)
    async def connect_redis_background():
        """Connect to Redis in background without blocking service startup"""
        global redis_client, event_broker
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

                # Initialize Event Broker
                try:
                    event_broker = await get_event_broker()
                    logger.info("✅ EventBroker initialized for STT events")
                except Exception as e:
                    event_broker = None
                    logger.warning(f"⚠️ Failed to initialize EventBroker: {e}")

                # Update VAD manager with Redis client
                vad_manager.redis_client = redis_client
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
    logger.info("🛑 Shutting down STT/VAD microservice...")
    logger.info("=" * 70)
    
    # Close Sarvam client
    logger.info("🔌 Closing Sarvam client...")
    if sarvam_client:
        await sarvam_client.aclose()
        sarvam_client = None
    logger.info("✅ Sarvam client closed")
    
    # Close Redis
    logger.info("🔌 Closing Redis connection...")
    await close_redis_client(redis_client)
    logger.info("✅ Redis connection closed")
    
    logger.info("✅ STT/VAD microservice stopped")


# Initialize FastAPI app
app = FastAPI(
    title="Leibniz STT/VAD Service",
    description="Speech transcription service powered by Sarvam AI Saarika",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create FastRTC handler (vad_manager will be injected in lifespan)
fastrtc_handler = FastRTCSTTHandler(vad_manager=None)

# Create FastRTC stream
fastrtc_stream = Stream(
    handler=fastrtc_handler,
    modality="audio",
    mode="send-receive",
    ui_args={
        "title": "Leibniz STT/VAD Transcription Service",
        "description": "Speak into your browser microphone. Audio streams directly to the STT/VAD service for real-time transcription. "
                     "Check Docker console logs for pipeline progress, speech detection, and transcript fragments."
    }
)

# Mount FastRTC stream UI to FastAPI app
try:
    # Use Gradio's mount_gradio_app which correctly handles the mounting
    # and path resolution for Gradio Blocks apps
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
    <title>Leibniz STT/VAD Client</title>
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
    <h1>🎙️ Leibniz STT/VAD Client</h1>
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

    Supports both command-based and continuous streaming modes:
    - Command mode: Send JSON commands like {"type": "start_capture"}
    - Continuous mode: Send audio bytes directly for ultra-low latency

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
    # Reduced to 20 (~1 second buffer) for ultra-low latency
    # Leaky bucket strategy: drops oldest chunks when full to prioritize real-time audio
    audio_queue = asyncio.Queue(maxsize=20)
    
    # Detect session type based on session_id pattern
    session_type = "orchestrator" if session_id.startswith("ws_session_") else "fastrtc"
    
    active_sessions[session_id] = {
        "audio_queue": audio_queue,
        "websocket": websocket,
        "last_activity": time.time(),
        "continuous_mode": False,
        "capture_task": None,
        "audio_chunks_received": 0
    }
    
    logger.info("=" * 70)
    logger.info(f"[{session_id}] 🔌 WebSocket session established")
    logger.info(f"[{session_id}]    Session ID: {session_id}")
    logger.info(f"[{session_id}]    Session type: {session_type}")
    logger.info(f"[{session_id}]    Remote: {websocket.client}")
    logger.info("=" * 70)
    
    logger.info(f"[{session_id}] 📡 Ready to receive audio chunks")

    try:
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=30.0  # 30s idle timeout
                )
            except (WebSocketDisconnect, RuntimeError):
                # Normal disconnection or "Cannot call receive..." error
                logger.info(f"[{session_id}] 🔌 WebSocket disconnected/closed")
                break
            except asyncio.TimeoutError:
                # Send timeout message and close
                try:
                    await websocket.send_json({
                        "type": "timeout",
                        "text": "",
                        "session_id": session_id,
                        "timestamp": time.time()
                    })
                except:
                    pass  # Ignore send errors on timeout
                break

            # Handle binary audio data
            if "bytes" in message:
                audio_data = message["bytes"]
                
                # Increment chunk counter
                active_sessions[session_id]["audio_chunks_received"] += 1
                chunk_count = active_sessions[session_id]["audio_chunks_received"]
                
                # Log first chunk and every 50th chunk at INFO level
                if chunk_count == 1:
                    logger.info(f"[{session_id}] 📥 First audio chunk received | Size: {len(audio_data)} bytes")
                elif chunk_count % 50 == 0:
                    logger.info(f"[{session_id}] 📥 Audio chunk #{chunk_count} received | Size: {len(audio_data)} bytes")
                else:
                    logger.debug(f"[{session_id}] 📥 Audio chunk #{chunk_count} received | Size: {len(audio_data)} bytes")
                
                # Leaky Bucket Strategy: Add to queue, dropping oldest if full
                # This ensures we always process the most recent audio for ultra-low latency
                try:
                    # Try to put without blocking
                    audio_queue.put_nowait(audio_data)
                except asyncio.QueueFull:
                    # Queue is full - drop oldest chunk and add new one (leaky bucket)
                    try:
                        dropped_chunk = audio_queue.get_nowait()
                        audio_queue.put_nowait(audio_data)
                        logger.debug(f"[{session_id}] ⚠️ Queue full - dropped oldest chunk ({len(dropped_chunk)} bytes) to make room for new audio")
                    except asyncio.QueueEmpty:
                        # Shouldn't happen, but handle gracefully
                        logger.warning(f"[{session_id}] ⚠️ Queue state inconsistent - attempting to add chunk anyway")
                        try:
                            audio_queue.put_nowait(audio_data)
                        except asyncio.QueueFull:
                            logger.error(f"[{session_id}] ❌ Failed to add chunk even after dropping oldest")
                            continue
                
                active_sessions[session_id]["last_activity"] = time.time()
                
                # Auto-start continuous capture
                if not active_sessions[session_id]["continuous_mode"]:
                    logger.info("=" * 70)
                    logger.info(f"[{session_id}] 🚀 Auto-starting continuous capture")
                    logger.info(f"[{session_id}]    Processing audio in real-time...")
                    logger.info(f"[{session_id}]    Session type: {session_type}")
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

                    if command.get("type") == "start_capture":
                        # Start capture mode
                        if not active_sessions[session_id]["continuous_mode"]:
                            logger.info(f"[{session_id}] 🚀 Starting capture (manual command)")
                            active_sessions[session_id]["continuous_mode"] = True

                            # Start continuous capture task
                            capture_task = asyncio.create_task(
                                continuous_capture_loop(session_id, audio_queue, websocket)
                            )
                            active_sessions[session_id]["capture_task"] = capture_task

                            await websocket.send_json({
                                "type": "capture_started",
                                "session_id": session_id,
                                "timestamp": time.time()
                            })

                    elif command.get("type") == "stop_capture":
                        # Stop capture mode
                        logger.info(f"[{session_id}] 🛑 Stopping capture (manual command)")
                        active_sessions[session_id]["continuous_mode"] = False

                        # Cancel capture task
                        capture_task = active_sessions[session_id]["capture_task"]
                        if capture_task and not capture_task.done():
                            capture_task.cancel()
                            try:
                                await capture_task
                            except asyncio.CancelledError:
                                pass

                        active_sessions[session_id]["capture_task"] = None

                        await websocket.send_json({
                            "type": "capture_stopped",
                            "session_id": session_id,
                            "timestamp": time.time()
                        })

                    elif command.get("type") == "ping":
                        # Respond to ping
                        await websocket.send_json({
                            "type": "pong",
                            "session_id": session_id,
                            "timestamp": time.time()
                        })

                except json.JSONDecodeError:
                    logger.warning(f"[{session_id}] ⚠️ Invalid JSON command received")
                    await websocket.send_json({
                        "type": "error",
                        "text": "Invalid JSON command",
                        "session_id": session_id,
                        "timestamp": time.time()
                    })

    except (WebSocketDisconnect, RuntimeError):
        logger.info(f"[{session_id}] 🔌 WebSocket disconnected/closed")
    
    except Exception as e:
        logger.error(f"[{session_id}] ❌ Unexpected WebSocket error: {e}", exc_info=True)

    finally:
        # Cleanup
        logger.info(f"[{session_id}] 🧹 Cleaning up session")
        if session_id in active_sessions:
            session_data = active_sessions[session_id]

            # Cancel any running capture task
            capture_task = session_data.get("capture_task")
            if capture_task and not capture_task.done():
                logger.info(f"[{session_id}]    Cancelling capture task")
                capture_task.cancel()
                try:
                    await capture_task
                except asyncio.CancelledError:
                    pass

            chunks_received = session_data.get("audio_chunks_received", 0)
            logger.info(f"[{session_id}]    Session processed {chunks_received} audio chunks")
            del active_sessions[session_id]
        
        # Unregister from Sarvam Streaming (Cleanup)
        if vad_manager:
            await vad_manager.unregister_session(session_id)
            logger.info(f"[{session_id}]    Unregistered from VAD manager")


async def continuous_capture_loop(session_id: str, audio_queue: asyncio.Queue, websocket: WebSocket):
    """
    Continuous capture loop for real-time transcription.

    Processes audio chunks as they arrive and streams transcripts back.
    Optimized for ultra-low latency with minimal buffering.

    Args:
        session_id: Session identifier
        audio_queue: Queue containing audio chunks
        websocket: WebSocket for sending transcripts
    """
    logger.info("=" * 70)
    logger.info(f"[{session_id}] 🔄 Starting continuous capture loop")
    logger.info(f"[{session_id}] 📊 Processing audio chunks in real-time (no buffering)")
    logger.info("=" * 70)

    # Streaming callback for real-time transcripts
    def streaming_callback(text: str, is_final: bool):
        """Send transcript fragments to WebSocket and emit events to Redis Streams."""

        async def send_fragment():
            try:
                fragment = {
                    "type": "fragment",
                    "text": text,
                    "is_final": is_final,
                    "session_id": session_id,
                    "timestamp": time.time(),
                }
                
                # Log BEFORE sending for debugging
                status = "✅ FINAL" if is_final else "🔄 PARTIAL"
                logger.info(f"[{session_id}] 📝 [{status}] STT Fragment")
                logger.info(f"[{session_id}]    Text: '{text[:150]}'")
                
                # 1) Send to WebSocket client (for UI feedback)
                # Check WebSocket state before sending (if available)
                try:
                    # Check if WebSocket is still connected (FastAPI WebSocket state)
                    if hasattr(websocket, 'client_state'):
                        if websocket.client_state.name != "CONNECTED":
                            logger.warning(f"[{session_id}] ⚠️ WebSocket not connected (state: {websocket.client_state.name}), cannot send fragment")
                            return
                except (AttributeError, Exception) as state_check_error:
                    # Fallback - try to send anyway, will catch error if disconnected
                    logger.debug(f"[{session_id}] WebSocket state check skipped: {state_check_error}")
                
                try:
                    await websocket.send_json(fragment)
                    logger.info(f"[{session_id}]    📤 Sent fragment to WebSocket client")
                except Exception as send_error:
                    logger.error(f"[{session_id}] ❌ Failed to send fragment to WebSocket: {send_error}")
                    # Don't raise - fragment sending failure shouldn't break the loop
                    return

                # 2) Emit event to Redis Streams for orchestrator FSM
                if event_broker:
                    try:
                        evt_type = EventTypes.STT_FINAL if is_final else EventTypes.STT_PARTIAL
                        event = VoiceEvent(
                            event_type=evt_type,
                            session_id=session_id,
                            source="stt-sarvam",
                            payload={
                                "text": text,
                                "confidence": 1.0,  # TODO: wire actual confidence from VAD/Sarvam if available
                                "is_final": is_final,
                            },
                        )
                        await event_broker.publish(f"voice:stt:session:{session_id}", event)
                        logger.info(f"[{session_id}]    📢 Published to Redis Streams")
                    except Exception as e:
                        logger.warning(f"[{session_id}] ⚠️ Failed to emit STT event to Redis Streams: {e}")

            except Exception as e:
                logger.warning(f"[{session_id}] ⚠️ Failed to handle streaming callback: {e}", exc_info=True)

        asyncio.create_task(send_fragment())

    try:
        while True:
            # Get audio chunk with short timeout for responsiveness
            try:
                audio_chunk = await asyncio.wait_for(
                    audio_queue.get(),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                # No audio available, continue loop
                continue

            # Process audio chunk immediately
            try:
                # Use VAD manager to process chunk
                if vad_manager:
                    chunk_size = len(audio_chunk)
                    logger.debug(f"[{session_id}] 🔄 Processing audio chunk | Size: {chunk_size} bytes")
                    
                    # For continuous mode, we process each chunk individually
                    # This provides real-time streaming without waiting for turn completion
                    await vad_manager.process_audio_chunk_streaming(
                        session_id,
                        audio_chunk,
                        streaming_callback
                    )
                else:
                    logger.warning(f"[{session_id}] ⚠️ VAD manager not available, cannot process audio chunk")

            except Exception as e:
                logger.error(f"[{session_id}] ❌ Error processing audio chunk: {e}", exc_info=True)
                # Continue processing other chunks even if one fails

    except asyncio.CancelledError:
        logger.info(f"[{session_id}] 🛑 Continuous capture loop cancelled")
        raise
    except Exception as e:
        logger.error(f"[{session_id}] ❌ Continuous capture loop error: {e}", exc_info=True)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Service health status
    """
    uptime_seconds = time.time() - app_start_time
    
    # Get Sarvam stats (VADManager handles active sessions)
    sarvam_stats = {}
    
    # Check Redis
    redis_connected = False
    if redis_client:
        try:
            await ping_redis(redis_client)
            redis_connected = True
        except Exception:
            redis_connected = False
    
    # Get VAD metrics
    vad_metrics = {}
    if vad_manager:
        vad_metrics = vad_manager.get_performance_metrics()
    
    # Determine status
    status = "healthy"
    if not redis_connected:
        status = "degraded"
    
    return {
        "status": status,
        "service": "stt-sarvam",
        "uptime_seconds": uptime_seconds,
        "active_sessions": len(active_sessions),
        "sarvam_metrics": vad_metrics,
        "redis_connected": redis_connected
    }


@app.get("/metrics")
async def get_metrics():
    """
    Get service performance metrics.
    
    Returns:
        dict: Performance metrics
    """
    if not vad_manager:
        return {"error": "VAD manager not initialized"}
    
    metrics = vad_manager.get_performance_metrics()
    metrics["active_ws_sessions"] = len(active_sessions)
    
    return metrics


@app.post("/admin/reset_session")
async def reset_session():
    """
    Force reset Sarvam streaming buffers (admin endpoint).
    
    Returns:
        dict: Reset confirmation
    """
    if vad_manager:
        vad_manager.reset_streaming_state()
    
    return {"status": "success", "message": "Sarvam streaming sessions cleared"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False
    )
