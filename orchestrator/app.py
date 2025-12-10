"""
StateManager Orchestrator FastAPI Application

WebSocket-based orchestrator that coordinates STT → Intent+RAG (parallel) → LLM → TTS flow
with FSM state management and barge-in detection.

Reference:
    services/docs/ORCHESTRATOR_IMPLEMENTATION.md - Implementation details
    services/docs/ORCHESTRATOR_GUIDE.md - Architecture guide
"""

import asyncio
import json
import logging
import os
import time
import base64
import audioop
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, AsyncGenerator

import aiohttp
from livekit.api import AccessToken, VideoGrants

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Optional FastRTC/Gradio UI for unified audio streaming
try:
    import gradio as gr
    from fastrtc import Stream

    FASTRTC_UI_AVAILABLE = True
except ImportError:  # pragma: no cover - degraded mode without FastRTC UI
    gr = None
    Stream = None
    FASTRTC_UI_AVAILABLE = False

from leibniz_agent.services.orchestrator.config import OrchestratorConfig, TARA_INTRO_GREETING, DEFAULT_INTRO_GREETING
from leibniz_agent.services.orchestrator.state_manager import StateManager, State
from leibniz_agent.services.orchestrator.parallel_pipeline import (
    process_intent_rag_llm, 
    process_rag_direct,
    process_rag_incremental,
    buffer_rag_incremental
)
from leibniz_agent.services.orchestrator.interruption_handler import InterruptionHandler
from leibniz_agent.services.orchestrator.service_manager import ServiceManager
from leibniz_agent.services.orchestrator.dialogue_manager import DialogueManager, DialogueType
from leibniz_agent.services.shared.redis_client import get_redis_client, close_redis_client, ping_redis

# ElevenLabs TTS client for ultra-low latency streaming
try:
    from leibniz_agent.services.orchestrator.eleven_tts_client import ElevenLabsTTSClient
    ELEVENLABS_CLIENT_AVAILABLE = True
except ImportError:
    try:
        from .eleven_tts_client import ElevenLabsTTSClient
        ELEVENLABS_CLIENT_AVAILABLE = True
    except ImportError:
        ELEVENLABS_CLIENT_AVAILABLE = False
        ElevenLabsTTSClient = None
        # Warning logged after logger is configured

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
redis_client = None
redis_subscriber = None
redis_listener_task = None
timeout_monitor_task = None
active_sessions: Dict[str, Any] = {}
config: Optional[OrchestratorConfig] = None
dialogue_manager: Optional[DialogueManager] = None
app_start_time: float = time.time()
STT_GRADIO_URL = os.getenv("STT_GRADIO_URL", "http://localhost:8001/fastrtc")
TTS_GRADIO_URL = os.getenv("TTS_GRADIO_URL", "http://localhost:8005/fastrtc")
service_connections = {
    "stt": {"connected": False, "session_id": None, "timestamp": None},
    "tts": {"connected": False, "session_id": None, "timestamp": None}
}

# ElevenLabs TTS client instances per session (for prewarm and streaming)
eleven_tts_clients: Dict[str, ElevenLabsTTSClient] = {} if ELEVENLABS_CLIENT_AVAILABLE else {}

# Workflow control - wait for manual start trigger
workflow_ready = False  # True when both STT and TTS are connected
workflow_triggered = False  # True when /start is called

# Intro greeting text - will be set from config in lifespan
# Uses TARA Telugu greeting if TARA_MODE=true, otherwise default English
INTRO_GREETING = None  # Set in lifespan from config


async def check_service_health(name: str, url: str) -> bool:
    """Check health of a dependent service"""
    if not url:
        return True
        
    try:
        async with aiohttp.ClientSession() as session:
            # Try /health endpoint first
            health_url = f"{url}/health"
            try:
                async with session.get(health_url, timeout=5.0) as response:
                    if response.status == 200:
                        logger.info(f"✅ {name} is healthy ({url})")
                        return True
            except:
                pass
                
            # Fallback to root if /health fails or doesn't exist
            async with session.get(url, timeout=5.0) as response:
                if response.status < 500:
                    logger.info(f"✅ {name} is reachable ({url})")
                    return True
    except Exception as e:
        logger.debug(f"⏳ {name} not ready yet ({url}): {e}")
        return False
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for application startup/shutdown"""
    global redis_client, config, INTRO_GREETING, dialogue_manager
    
    logger.info("=" * 70)
    logger.info("🚀 Starting StateManager Orchestrator")
    logger.info("=" * 70)
    
    # Load configuration
    config = OrchestratorConfig.from_env()
    logger.info(f"📋 Configuration loaded")
    
    # Initialize Dialogue Manager
    dialogue_manager = DialogueManager(tara_mode=config.tara_mode)
    logger.info(f"✅ Dialogue Manager initialized")
    
    # Set intro greeting from config (supports TARA Telugu mode)
    INTRO_GREETING = config.intro_greeting
    logger.info(f"🎤 Intro greeting: {INTRO_GREETING[:50]}...")
    
    # ------------------------------------------------------------------
    # 1. Auto-start Services (Master Controller Mode)
    # ------------------------------------------------------------------
    auto_start_services = os.getenv("AUTO_START_SERVICES", "true").lower() == "true"
    docker_compose_file = os.getenv("DOCKER_COMPOSE_FILE", None)
    docker_context = os.getenv("DOCKER_CONTEXT", "desktop-linux")
    
    if auto_start_services:
        logger.info("=" * 70)
        logger.info("🎛️ MASTER CONTROLLER MODE: Auto-starting services...")
        logger.info(f"   Docker Context: {docker_context}")
        logger.info("=" * 70)
        
        # Detect project name from compose file or use default
        project_name = os.getenv("DOCKER_PROJECT_NAME", "tara-task")
        
        service_manager = ServiceManager(
            docker_compose_file=docker_compose_file,
            docker_context=docker_context
        )
        
        # Ensure network exists (auto-detected from project name)
        await service_manager.ensure_network()
        
        # Start all services (skip intent and appointment)
        skip_services = []
        if config.skip_intent_service:
            skip_services.append("intent")
        if config.skip_appointment_service:
            skip_services.append("appointment")
        
        logger.info("🚀 Starting services in order: Redis → STT → RAG → TTS")
        start_results = await service_manager.start_all_services(skip_services=skip_services)
        
        # Check if all services started successfully
        failed_services = [name for name, success in start_results.items() if not success]
        if failed_services:
            logger.error(f"❌ Failed to start services: {', '.join(failed_services)}")
            logger.error("⚠️ Continuing anyway - services may need manual startup")
        else:
            logger.info("✅ All services started successfully!")
        
        # Get service URLs for display
        service_urls = service_manager.get_service_urls()
        if "stt" in service_urls:
            global STT_GRADIO_URL
            STT_GRADIO_URL = service_urls["stt"]
        if "tts" in service_urls:
            global TTS_GRADIO_URL
            TTS_GRADIO_URL = service_urls["tts"]
    
    # ------------------------------------------------------------------
    # 2. Wait for Dependent Services to be Healthy
    # ------------------------------------------------------------------
    services_to_check = [
        ("STT Service", config.stt_service_url),
        ("TTS Service", config.tts_service_url),
        ("RAG Service", config.rag_service_url),
    ]
    
    # Skip Appointment Service in TARA mode or if explicitly skipped
    if not config.tara_mode and not config.skip_appointment_service and config.appointment_service_url:
        services_to_check.append(("Appointment Service", config.appointment_service_url))
    
    # Filter out None values
    services_to_check = [(n, u) for n, u in services_to_check if u]
    
    logger.info("=" * 70)
    logger.info("🏥 Checking health of all services...")
    logger.info("=" * 70)
    
    # Loop until all services are healthy
    max_wait_time = 120  # Maximum 2 minutes
    start_wait = time.time()
    while True:
        pending = []
        for name, url in services_to_check:
            is_ready = await check_service_health(name, url)
            if not is_ready:
                pending.append(name)
        
        if not pending:
            logger.info("✅ All dependent services are READY")
            break
        
        # Check timeout
        if time.time() - start_wait > max_wait_time:
            logger.error(f"❌ Timeout waiting for services: {', '.join(pending)}")
            logger.error("⚠️ Some services may not be ready - check manually")
            break
            
        logger.info(f"⏳ Waiting for: {', '.join(pending)}...")
        await asyncio.sleep(5.0)
    
    # Connect to Redis with timeout and retries
    redis_client = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting Redis connection (attempt {attempt + 1}/{max_retries})...")
            redis_client = await asyncio.wait_for(get_redis_client(), timeout=15.0)
            await asyncio.wait_for(ping_redis(redis_client), timeout=5.0)
            logger.info("✅ Redis connected")
            break
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ Redis connection timeout (attempt {attempt + 1}/{max_retries}), retrying...")
                await asyncio.sleep(2.0)
            else:
                logger.warning(f"⚠️ Redis connection timeout after {max_retries} attempts - service will run in degraded mode")
                redis_client = None
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ Redis connection failed (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                await asyncio.sleep(2.0)
            else:
                logger.warning(f"⚠️ Redis connection failed after {max_retries} attempts: {e}")
                redis_client = None
    
    # Start Redis subscriber background task
    global redis_listener_task, timeout_monitor_task
    if redis_client:
        redis_listener_task = asyncio.create_task(listen_to_redis_events())
        logger.info("✅ Redis event listener started")
    
    # Start timeout monitor background task
    timeout_monitor_task = asyncio.create_task(monitor_timeouts())
    logger.info("✅ Timeout monitor started")
    
    logger.info("=" * 70)
    logger.info("✅ StateManager Orchestrator Ready")
    logger.info("=" * 70)
    
    # Determine the unified FastRTC URL
    orchestrator_port = os.getenv('ORCHESTRATOR_PORT', os.getenv('PORT', '5204'))
    unified_fastrtc_url = f"http://localhost:{orchestrator_port}/fastrtc"
    orchestrator_api_url = f"http://localhost:{orchestrator_port}"
    
    logger.info("📋 SERVICE LINKS:")
    logger.info(f"   🎯 UNIFIED FastRTC UI (RECOMMENDED): {unified_fastrtc_url}")
    logger.info(f"   🔗 Orchestrator API: {orchestrator_api_url}")
    logger.info(f"   🔗 STT FastRTC UI (standalone): {STT_GRADIO_URL}")
    logger.info(f"   🔗 TTS FastRTC UI (standalone): {TTS_GRADIO_URL}")
    logger.info("=" * 70)
    logger.info("🚀 QUICK START:")
    logger.info(f"   1. Open Unified FastRTC UI in browser: {unified_fastrtc_url}")
    logger.info(f"   2. Click 'Record' to connect (handles both STT + TTS)")
    logger.info(f"   3. Send POST /start to trigger: curl -X POST {orchestrator_api_url}/start")
    logger.info("=" * 70)
    logger.info("⏳ ALTERNATIVE (Separate UIs):")
    logger.info(f"   1. Open STT FastRTC UI: {STT_GRADIO_URL}")
    logger.info(f"   2. Open TTS FastRTC UI: {TTS_GRADIO_URL}")
    logger.info("   3. Connections will be detected automatically via Redis events")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown background tasks
    if redis_listener_task:
        redis_listener_task.cancel()
        try:
            await redis_listener_task
        except asyncio.CancelledError:
            pass
    
    if timeout_monitor_task:
        timeout_monitor_task.cancel()
        try:
            await timeout_monitor_task
        except asyncio.CancelledError:
            pass
    
    logger.info("=" * 70)
    logger.info("🛑 Shutting down StateManager Orchestrator")
    logger.info("=" * 70)
    
    if redis_client:
        await close_redis_client()


app = FastAPI(
    title="Leibniz StateManager Orchestrator",
    description="Ultra-low latency voice agent orchestrator",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    import os
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")


# ============================================================================
# Unified FastRTC Gradio UI (single connection for STT + TTS)
# ============================================================================

unified_fastrtc_stream = None
_unified_handler = None  # Global reference for state broadcasting

# Import UnifiedFastRTCHandler (may fail if dependencies missing)
try:
    from .unified_fastrtc import UnifiedFastRTCHandler, create_unified_handler
    UNIFIED_FASTRTC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ UnifiedFastRTCHandler not available: {e}")
    UnifiedFastRTCHandler = None
    create_unified_handler = None
    UNIFIED_FASTRTC_AVAILABLE = False


async def broadcast_orchestrator_state(state_value: str) -> None:
    """
    Broadcast orchestrator state to all UnifiedFastRTCHandler instances.
    
    This enables state-aware input gating (mute mic during THINKING/SPEAKING).
    
    Args:
        state_value: State string value (e.g., "listening", "speaking")
    """
    if not UNIFIED_FASTRTC_AVAILABLE or UnifiedFastRTCHandler is None:
        return
    
    try:
        await UnifiedFastRTCHandler.broadcast_state_change(state_value)
    except Exception as e:
        logger.debug(f"State broadcast error: {e}")


# Forward declarations for callbacks (defined below, after routing helpers)
async def _unified_on_stt_transcript(
    fastrtc_session_id: str,
    text: str,
    is_final: bool,
) -> None:
    """Callback from UnifiedFastRTCHandler when STT text is available."""
    # Implementation is below - this is just for reference
    pass  # Actual implementation replaces this


async def _unified_on_connection_change(
    fastrtc_session_id: str,
    connected: bool,
) -> None:
    """Callback when a Unified FastRTC browser client connects or disconnects."""
    # Implementation is below - this is just for reference
    pass  # Actual implementation replaces this


def _setup_unified_fastrtc():
    """
    Lazy setup of UnifiedFastRTC UI after callbacks are defined.
    Called at the end of module initialization.
    """
    global unified_fastrtc_stream, _unified_handler, app
    
    if not UNIFIED_FASTRTC_AVAILABLE or not FASTRTC_UI_AVAILABLE:
        logger.warning("FastRTC / Gradio not available - unified UI disabled")
        return
    
    try:
        # Create handler
        _unified_handler = create_unified_handler()
        
        # Note: Callbacks are registered after they're defined (see bottom of file)
        
        unified_fastrtc_stream = Stream(
            handler=_unified_handler,
            modality="audio",
            mode="send-receive",
            ui_args={
                "title": "Leibniz Unified Voice Interface",
                "description": (
                    "Single FastRTC connection for ultra-low latency STT + TTS. "
                    "Connect your microphone and speakers here. "
                    "Use the /start API on the orchestrator to trigger the intro."
                ),
            },
        )
        
        # Mount Gradio UI on the main FastAPI app
        app = gr.mount_gradio_app(app, unified_fastrtc_stream.ui, path="/fastrtc")
        logger.info("✅ Unified FastRTC UI mounted at /fastrtc")
        
    except Exception as e:
        unified_fastrtc_stream = None
        _unified_handler = None
        logger.warning(f"⚠️ Failed to initialize unified FastRTC UI: {e}")


@app.get("/")
async def root():
    """Serve the unified client HTML"""
    static_file = os.path.join(os.path.dirname(__file__), "static", "client.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "Leibniz Orchestrator API", "client": "/static/client.html"}


@app.get("/livekit")
async def livekit_client():
    """Serve the LiveKit client HTML"""
    static_file = os.path.join(os.path.dirname(__file__), "static", "livekit_client.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "LiveKit client not found", "client": "/static/livekit_client.html"}


@app.get("/token")
async def get_token(room_name: str = Query("room-1"), participant_name: str = Query("user")):
    """
    Generate a LiveKit access token for the frontend client.
    """
    api_key = os.getenv("LIVEKIT_API_KEY", "devkey")
    api_secret = os.getenv("LIVEKIT_API_SECRET", "secret")

    grant = VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True
    )

    token = AccessToken(api_key, api_secret) \
        .with_grants(grant) \
        .with_identity(participant_name) \
        .with_name(participant_name)
    
    return {"token": token.to_jwt(), "room_name": room_name, "participant_name": participant_name}


@app.websocket("/orchestrate")
async def orchestrate(websocket: WebSocket, session_id: str = Query(...)):
    """
    Main WebSocket orchestrator endpoint.
    
    Coordinates flow: STT → Intent+RAG (parallel) → LLM → TTS
    Handles interruptions and state transitions.
    """
    await websocket.accept()
    
    logger.info("=" * 70)
    logger.info(f"🔌 Session connected: {session_id}")
    logger.info("=" * 70)
    
    # Initialize state manager
    state_mgr = StateManager(session_id, redis_client)
    await state_mgr.initialize()
    
    # Initialize interruption handler
    interrupt_handler = InterruptionHandler(session_id)
    
    active_sessions[session_id] = {
        "state_mgr": state_mgr,
        "interrupt_handler": interrupt_handler,
        "websocket": websocket,
        "tts_task": None,
        "intro_task": None,
        "workflow_started": False
    }
    
    # Send initial connection message
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "state": state_mgr.state.value
    })
    
    await send_service_status(websocket, session_id, "Waiting for STT/TTS FastRTC clients to connect...")
    await check_and_start_workflow("session_connected")
    
    try:
        while True:
            try:
                # Use a longer timeout (5 minutes) since STT events come via Redis, not WebSocket
                # The WebSocket only receives client messages (manual controls, etc.)
                message = await asyncio.wait_for(websocket.receive_json(), timeout=300.0)
            except asyncio.TimeoutError:
                # Send a keep-alive ping to keep the connection open
                try:
                    await websocket.send_json({"type": "ping", "session_id": session_id})
                    logger.debug(f"🔄 Keep-alive ping for session {session_id}")
                    continue  # Don't break, just continue waiting
                except:
                    logger.warning(f"⏱️ Session {session_id} disconnected")
                    break
            
            msg_type = message.get("type")
            
            # ═══════════════════════════════════════════════════════════════
            # LISTENING STATE - Buffer STT fragments
            # ═══════════════════════════════════════════════════════════════
            if msg_type == "stt_fragment":
                # Only process STT when in LISTENING state
                if state_mgr.state != State.LISTENING:
                    logger.debug(f"⏸️ Ignoring STT fragment - current state: {state_mgr.state.value}")
                    continue
                
                text = message.get("text", "")
                is_final = message.get("is_final", False)
                
                if text:
                    logger.info(f"📝 [{state_mgr.state.value}] STT: {text[:50]}...")
                    
                    await state_mgr.transition(State.LISTENING, "stt_fragment", {"text": text})
                    
                    await websocket.send_json({
                        "type": "state_update",
                        "session_id": session_id,
                        "state": State.LISTENING.value,
                        "text_buffer": state_mgr.context.text_buffer
                    })
            
            # ═══════════════════════════════════════════════════════════════
            # THINKING STATE - Parallel Intent+RAG+LLM
            # ═══════════════════════════════════════════════════════════════
            elif msg_type == "vad_end":
                # Only process VAD end when in LISTENING state
                if state_mgr.state != State.LISTENING:
                    logger.debug(f"⏸️ Ignoring VAD end - current state: {state_mgr.state.value}")
                    continue
                
                logger.info("=" * 70)
                logger.info(f"🤐 End of turn detected")
                logger.info(f"📝 Text: {' '.join(state_mgr.context.text_buffer)}")
                logger.info("=" * 70)
                
                await state_mgr.transition(State.THINKING, "vad_end", {})
                
                # CRITICAL: Parallel execution
                user_text = " ".join(state_mgr.context.text_buffer)
                
                if not user_text.strip():
                    logger.warning("⚠️ Empty user text, skipping processing")
                    await state_mgr.transition(State.IDLE, "empty_text", {})
                    continue
                
                start_time = time.time()
                
                # Process with parallel pipeline (RAG optional)
                if config.rag_service_url:
                    logger.info("⚡ Starting parallel Intent+RAG processing...")
                else:
                    logger.info("⚡ Starting Intent processing (RAG not configured)...")
                
                result = await process_intent_rag_llm(
                    user_text, 
                    session_id,
                    config.intent_service_url,
                    config.rag_service_url  # Can be None if RAG not configured
                )
                
                thinking_time = (time.time() - start_time) * 1000
                response_text = result.get("response", "")
                
                logger.info(f"✅ Response ready in {thinking_time:.0f}ms: {response_text[:100]}...")
                
                await state_mgr.transition(State.SPEAKING, "response_ready", {
                    "response": response_text,
                    "intent": result.get("intent"),
                    "rag_results": result.get("rag")
                })
                
                # Send response ready message
                await websocket.send_json({
                    "type": "response_ready",
                    "session_id": session_id,
                    "text": response_text,
                    "thinking_ms": thinking_time,
                    "intent": result.get("intent", {}).get("intent", "unknown"),
                    "latency_breakdown": {
                        "thinking_ms": thinking_time,
                        "intent_ms": result.get("intent", {}).get("response_time", 0) * 1000,
                        "rag_ms": result.get("rag", {}).get("timing_breakdown", {}).get("total_ms", 0) if result.get("rag") else 0
                    }
                })
                
                # Simulate TTS streaming (replace with real TTS service call)
                tts_task = asyncio.create_task(
                    stream_tts_audio(session_id, response_text, websocket, state_mgr)
                )
                active_sessions[session_id]["tts_task"] = tts_task
            
            # ═══════════════════════════════════════════════════════════════
            # INTERRUPT STATE - Barge-in
            # ═══════════════════════════════════════════════════════════════
            elif msg_type == "user_speaking":
                if state_mgr.state == State.SPEAKING:
                    logger.warning(f"⚡ INTERRUPT: User started speaking during TTS")
                    
                    # Cancel TTS
                    tts_task = active_sessions[session_id].get("tts_task")
                    if tts_task and not tts_task.done():
                        tts_task.cancel()
                    
                    # Reset to listening
                    state_mgr.context.text_buffer = []  # Clear buffer
                    await state_mgr.transition(State.INTERRUPT, "barge_in", {})
                    
                    # Immediately go back to listening
                    await state_mgr.transition(State.LISTENING, "resume_listening", {})
                    
                    await websocket.send_json({
                        "type": "interrupted",
                        "session_id": session_id,
                        "message": "Listening to your response..."
                    })
    
    except WebSocketDisconnect:
        logger.info(f"🔌 Session disconnected: {session_id}")
    
    except Exception as e:
        logger.error(f"❌ Session error: {e}", exc_info=True)
    
    finally:
        if session_id in active_sessions:
            tts_task = active_sessions[session_id].get("tts_task")
            intro_task = active_sessions[session_id].get("intro_task")
            if tts_task and not tts_task.done():
                tts_task.cancel()
            if intro_task and not intro_task.done():
                intro_task.cancel()
            del active_sessions[session_id]
        
        # Clean up ElevenLabs TTS client if exists
        if session_id in eleven_tts_clients:
            await cleanup_elevenlabs_client(session_id)
        
        logger.info(f"✅ Session cleanup complete: {session_id}")


async def stream_audio_file(
    session_id: str,
    audio_file_path: str,
    websocket: Optional[WebSocket],
    state_mgr: StateManager,
    skip_state_transition: bool = False
):
    """Stream pre-synthesized audio file via TTS service WebSocket
    
    Reads audio file and sends it to TTS service for playback.
    This simulates TTS streaming but uses pre-recorded audio.
    """
    try:
        import wave
        import struct
        
        # Read audio file
        with wave.open(audio_file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            logger.info(f"📁 Audio file: {audio_file_path}")
            logger.info(f"   Sample rate: {sample_rate}Hz, Channels: {num_channels}, Width: {sample_width} bytes")
            
            # Connect to TTS service WebSocket (for routing to FastRTC UI)
            tts_ws_url = config.tts_service_url.replace("http://", "ws://").replace("https://", "wss://")
            tts_ws_url = f"{tts_ws_url}/api/v1/stream?session_id={session_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(tts_ws_url) as tts_ws:
                    # Send audio file metadata
                    await tts_ws.send_json({
                        "type": "audio_file",
                        "sample_rate": sample_rate,
                        "channels": num_channels,
                        "sample_width": sample_width
                    })
                    
                    # Read and send audio chunks
                    chunk_size = 4096  # 4KB chunks
                    total_bytes = 0
                    
                    # Resampling state for audioop
                    resample_state = None
                    target_rate = 24000
                    
                    while True:
                        audio_data = wav_file.readframes(chunk_size // sample_width)
                        if not audio_data:
                            break
                        
                        # Encode audio data as base64
                        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                        
                        # Send audio chunk
                        await tts_ws.send_json({
                            "type": "audio_chunk",
                            "data": audio_b64
                        })
                        
                        total_bytes += len(audio_data)
                        
                        # Forward to client WebSocket if exists
                        if websocket:
                            try:
                                await websocket.send_bytes(audio_data)
                            except Exception as ws_err:
                                logger.debug(f"Could not forward audio to client: {ws_err}")
                        
                        # Broadcast to Unified FastRTC (CRITICAL FIX for auto-sessions)
                        try:
                            if UNIFIED_FASTRTC_AVAILABLE and UnifiedFastRTCHandler:
                                # Resample if necessary (e.g. 22050Hz -> 24000Hz)
                                broadcast_data = audio_data
                                broadcast_rate = sample_rate
                                
                                if sample_rate != target_rate and sample_width == 2:
                                    try:
                                        broadcast_data, resample_state = audioop.ratecv(
                                            audio_data, 
                                            sample_width, 
                                            1, 
                                            sample_rate, 
                                            target_rate, 
                                            resample_state
                                        )
                                        broadcast_rate = target_rate
                                    except Exception as e:
                                        logger.warning(f"Resampling failed: {e}")
                                
                                await UnifiedFastRTCHandler.broadcast_audio(broadcast_data, broadcast_rate)
                        except Exception as e:
                            logger.debug(f"Failed to broadcast audio file chunk: {e}")
                        
                        # Small delay to simulate streaming
                        await asyncio.sleep(0.01)
                    
                    # Send completion
                    await tts_ws.send_json({
                        "type": "audio_complete",
                        "total_bytes": total_bytes
                    })
                    
                    logger.info(f"✅ Audio file streaming complete: {total_bytes} bytes")
                    
                    # Handle state transition (unless skipping for fillers)
                    if not skip_state_transition:
                        await state_mgr.transition(State.IDLE, "audio_complete", {})
                        state_mgr.context.turn_number += 1
                        state_mgr.context.text_buffer = []
                        await state_mgr.save_state()
                        logger.info(f"✅ Audio playback complete, ready for next turn")
                    else:
                        logger.info(f"✅ Audio playback complete (filler, state unchanged)")
                    
    except Exception as e:
        logger.error(f"❌ Error streaming audio file: {e}", exc_info=True)
        raise


async def play_intro_greeting(session_id: str, websocket: Optional[WebSocket], state_mgr: StateManager):
    """Play intro greeting via Dialogue Manager (supports pre-synthesized audio)"""
    try:
        if dialogue_manager:
            intro_asset = dialogue_manager.get_asset(DialogueType.INTRO)
            logger.info(f"🎤 Playing intro greeting: {intro_asset.text[:50]}...")
            await stream_tts_audio(
                session_id, 
                intro_asset.text, 
                websocket, 
                state_mgr, 
                emotion=intro_asset.emotion,
                audio_file_path=intro_asset.audio_path if intro_asset.has_audio() else None
            )
        else:
            # Fallback to config intro greeting
            logger.info(f"🎤 Playing intro greeting (fallback): {INTRO_GREETING[:50]}...")
            await stream_tts_audio(session_id, INTRO_GREETING, websocket, state_mgr, emotion="helpful")
    except Exception as e:
        logger.error(f"❌ Intro greeting error: {e}")


async def stream_tts_audio(
    session_id: str, 
    text: str, 
    websocket: Optional[WebSocket], 
    state_mgr: StateManager,
    emotion: str = "helpful",
    audio_file_path: Optional[str] = None,
    skip_state_transition: bool = False
):
    """Stream TTS audio from tts-streaming-service via WebSocket or play pre-synthesized audio
    
    If audio_file_path is provided and file exists, streams the audio file directly.
    Otherwise, synthesizes text via TTS service.
    
    If websocket is None (auto-created session), audio still streams to TTS FastRTC UI.
    The TTS service handles audio playback directly.
    """
    # Check if we have a pre-synthesized audio file
    if audio_file_path and os.path.exists(audio_file_path):
        try:
            logger.info(f"🔊 Playing pre-synthesized audio: {audio_file_path}")
            await stream_audio_file(session_id, audio_file_path, websocket, state_mgr, skip_state_transition)
            return
        except Exception as e:
            logger.warning(f"⚠️ Failed to play audio file, falling back to TTS: {e}")
            # Fall through to TTS synthesis
    
    # Synthesize via TTS service
    tts_ws_url = config.tts_service_url.replace("http://", "ws://").replace("https://", "wss://")
    tts_ws_url = f"{tts_ws_url}/api/v1/stream?session_id={session_id}"
    
    try:
        logger.info(f"🔊 Connecting to TTS service: {tts_ws_url}")
        logger.info(f"📝 Text to synthesize: {text[:50]}...")
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(tts_ws_url) as tts_ws:
                # Send synthesis request
                await tts_ws.send_json({
                    "type": "synthesize",
                    "text": text,
                    "emotion": emotion
                })
                
                # Receive audio chunks and forward to client
                sentence_count = 0
                total_duration_ms = 0.0
                
                async for msg in tts_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        
                        if msg_type == "connected":
                            logger.info(f"✅ TTS service connected")
                        
                        elif msg_type == "sentence_start":
                            sentence_count += 1
                            logger.info(f"📢 Sentence {sentence_count} starting: {data.get('text', '')[:50]}...")
                        
                        elif msg_type == "audio":
                            # Forward audio chunk to WebSocket client (if exists)
                            audio_b64 = data.get("data", "")
                            sample_rate = data.get("sample_rate", config.tts_sample_rate if hasattr(config, "tts_sample_rate") else 24000)  # type: ignore[assignment]

                            if audio_b64:
                                audio_bytes = base64.b64decode(audio_b64)

                                # 1) Send to any connected orchestrator WebSocket client
                                if websocket:
                                    try:
                                        await websocket.send_bytes(audio_bytes)
                                    except Exception as ws_err:
                                        logger.debug(f"Could not forward audio to client: {ws_err}")

                                # 2) Broadcast to all Unified FastRTC sessions (primary audio path)
                                try:
                                    from .unified_fastrtc import UnifiedFastRTCHandler  # Local import to avoid circulars

                                    await UnifiedFastRTCHandler.broadcast_audio(
                                        audio_bytes, sample_rate
                                    )
                                except Exception as broadcast_err:
                                    logger.debug(f"Could not broadcast audio to Unified FastRTC: {broadcast_err}")
                        
                        elif msg_type == "sentence_playing":
                            # Track when sentence is playing in browser
                            duration_ms = data.get("duration_ms", 0)
                            expected_complete_at = data.get("expected_complete_at", 0)
                            logger.info(f"🔊 Sentence {data.get('index', 0)} playing ({duration_ms:.0f}ms)")
                            # Wait for browser playback to complete before processing next
                            await asyncio.sleep(duration_ms / 1000.0)
                            logger.debug(f"✅ Sentence {data.get('index', 0)} playback complete")
                        
                        elif msg_type == "sentence_complete":
                            duration_ms = data.get("duration_ms", 0)
                            total_duration_ms += duration_ms
                            logger.debug(f"✅ Sentence {data.get('index', 0)} synthesis complete ({duration_ms:.0f}ms)")
                        
                        elif msg_type == "complete":
                            logger.info(f"✅ TTS complete: {data.get('total_sentences', 0)} sentences, {data.get('total_duration_ms', 0):.0f}ms")
                            break
                        
                        elif msg_type == "error":
                            logger.error(f"❌ TTS error: {data.get('message', 'Unknown error')}")
                            if websocket:
                                try:
                                    await websocket.send_json({
                                        "type": "tts_error",
                                        "message": data.get("message", "TTS synthesis failed")
                                    })
                                except:
                                    pass
                            return
                    
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"❌ TTS WebSocket error: {tts_ws.exception()}")
                        break
                
                # TTS complete - transition to IDLE (unless skipping for fillers)
                if not skip_state_transition:
                    await state_mgr.transition(State.IDLE, "tts_complete", {})
                    await broadcast_orchestrator_state(State.IDLE.value)  # Open mic
                    state_mgr.context.turn_number += 1
                    state_mgr.context.text_buffer = []  # Clear buffer for next turn
                    await state_mgr.save_state()
                    logger.info(f"✅ TTS streaming complete, ready for next turn")
                else:
                    logger.info(f"✅ TTS streaming complete (filler, state unchanged)")
                
                # Send turn complete message (if WebSocket exists)
                if websocket:
                    try:
                        await websocket.send_json({
                            "type": "turn_complete",
                            "session_id": session_id,
                            "turn_number": state_mgr.context.turn_number,
                            "state": state_mgr.state.value
                        })
                    except:
                        pass
        
    except asyncio.CancelledError:
        logger.warning(f"⚡ TTS cancelled (barge-in)")
        raise
    except Exception as e:
        logger.error(f"❌ TTS streaming error: {e}", exc_info=True)
        if websocket:
            try:
                await websocket.send_json({
                    "type": "tts_error",
                    "message": f"TTS streaming failed: {str(e)}"
                })
            except:
                pass


def get_missing_services():
    missing = []
    if not service_connections["stt"]["connected"]:
        missing.append("STT FastRTC")
    if not service_connections["tts"]["connected"]:
        missing.append("TTS FastRTC")
    return missing


async def send_service_status(websocket: WebSocket, session_id: str, note: str = ""):
    """Send current service connectivity status to a client session."""
    if not websocket:
        return
    message = note
    if not message:
        missing = get_missing_services()
        if missing:
            message = f"Waiting for {' & '.join(missing)} to connect..."
        else:
            message = "All FastRTC clients connected."
    try:
        await websocket.send_json({
            "type": "service_status",
            "session_id": session_id,
            "stt_ready": service_connections["stt"]["connected"],
            "tts_ready": service_connections["tts"]["connected"],
            "message": message
        })
    except Exception as exc:
        logger.warning(f"⚠️ Failed to send service status to {session_id}: {exc}")


async def broadcast_service_status(note: str = ""):
    """Broadcast service readiness to all connected orchestrator sessions."""
    if not active_sessions:
        return
    for session_id, session_data in active_sessions.items():
        websocket = session_data.get("websocket")
        await send_service_status(websocket, session_id, note)


async def handle_service_connection(service: str, payload: Dict[str, Any]):
    """Handle STT/TTS FastRTC connection signals from Redis."""
    service = service.lower()
    if service not in service_connections:
        logger.warning(f"⚠️ Unknown service type '{service}' in connection handler")
        return
    
    info = service_connections[service]
    info["connected"] = True
    info["session_id"] = payload.get("session_id")
    info["timestamp"] = payload.get("timestamp", time.time())
    
    logger.info("=" * 70)
    logger.info(f"📡 {service.upper()} FastRTC connected | session: {info['session_id'] or 'unknown'}")
    logger.info(f"   Source: {payload.get('source', 'unknown')}")
    missing = get_missing_services()
    if missing:
        logger.info(f"   Waiting for: {', '.join(missing)}")
    else:
        logger.info("   All FastRTC clients connected. Preparing workflow...")
    logger.info("=" * 70)
    
    await broadcast_service_status(f"{service.upper()} connected.")
    await check_and_start_workflow(f"{service}_connected")


async def check_and_start_workflow(trigger: str):
    """
    Check if workflow can start. Does NOT auto-start - waits for /start trigger.
    
    Sets workflow_ready=True when both STT and TTS are connected.
    Actual workflow start only happens when /start is called.
    """
    global workflow_ready
    
    missing = get_missing_services()
    if missing:
        workflow_ready = False
        logger.info(f"⏳ [{trigger}] Waiting for {', '.join(missing)} before workflow can start")
        return
    
    # Both services connected - mark as ready
    workflow_ready = True
    logger.info("=" * 70)
    logger.info(f"✅ [{trigger}] All FastRTC clients connected - workflow READY")
    logger.info("=" * 70)
    
    # Auto-create a session if none exists (for /start endpoint)
    if not active_sessions:
        auto_session_id = f"auto_session_{int(time.time())}"
        logger.info(f"🤖 Auto-creating session: {auto_session_id}")
        
        # Create state manager
        state_mgr = StateManager(auto_session_id, redis_client)
        await state_mgr.initialize()
        
        interrupt_handler = InterruptionHandler(auto_session_id)
        
        # Store session WITHOUT WebSocket (auto-created session)
        # TTS will stream directly to TTS FastRTC UI
        active_sessions[auto_session_id] = {
            "state_mgr": state_mgr,
            "interrupt_handler": interrupt_handler,
            "websocket": None,  # No client WebSocket - TTS streams to FastRTC UI directly
            "tts_task": None,
            "intro_task": None,
            "workflow_started": False,
            "is_auto_created": True
        }
        
        logger.info(f"✅ Auto-created session: {auto_session_id}")
        logger.info(f"📊 Active sessions: {len(active_sessions)}")
        logger.info(f"🎯 Send POST /start to trigger the intro greeting")
        logger.info(f"   Example: curl -X POST http://localhost:5204/start")
        logger.info("=" * 70)
    
    # Notify all connected sessions that we're ready
    await broadcast_service_status("All services connected. Send POST /start to begin.")


async def start_intro_sequence(session_id: str):
    """
    Play intro greeting and transition to listening once complete.
    
    State flow:
    1. IDLE -> SPEAKING (intro greeting via TTS)
    2. SPEAKING -> LISTENING (waiting for user speech via STT)
    
    Works with both WebSocket sessions and auto-created sessions (websocket=None).
    For auto-created sessions, TTS streams directly to TTS FastRTC UI.
    """
    session_data = active_sessions.get(session_id)
    if not session_data:
        logger.warning(f"⚠️ Session {session_id} not found for intro sequence")
        return
    
    websocket = session_data.get("websocket")  # Can be None for auto-created sessions
    state_mgr = session_data.get("state_mgr")
    
    if not state_mgr:
        logger.warning(f"⚠️ Session {session_id} missing state_mgr")
        return
    
    try:
        # Transition to SPEAKING state for intro
        await state_mgr.transition(State.SPEAKING, "intro_start", {"text": INTRO_GREETING})
        await broadcast_orchestrator_state(State.SPEAKING.value)  # Gate mic
        
        if websocket:
            await send_service_status(
                websocket,
                session_id,
                "Playing intro greeting..."
            )
        else:
            logger.info(f"🎤 [AUTO SESSION] Playing intro greeting: {INTRO_GREETING[:50]}...")
        
        logger.info("=" * 70)
        logger.info(f"🎤 [SPEAKING] Playing intro greeting for session: {session_id}")
        logger.info(f"📝 Text: {INTRO_GREETING}")
        logger.info("=" * 70)
        
        # Play intro via TTS
        await play_intro_greeting(session_id, websocket, state_mgr)
        
        # Reset timeout timer explicitly after intro finishes
        state_mgr.context.last_activity_time = time.time()
        
        # Transition to LISTENING state - now waiting for user speech
        await state_mgr.transition(State.LISTENING, "intro_complete", {})
        await broadcast_orchestrator_state(State.LISTENING.value)  # Open mic
        
        if websocket:
            try:
                await websocket.send_json({
                    "type": "intro_complete",
                    "session_id": session_id,
                    "state": State.LISTENING.value,
                    "message": "Intro complete. Listening for your response..."
                })
            except:
                pass
        else:
            logger.info("=" * 70)
            logger.info(f"🎧 [LISTENING] Auto session ready - waiting for user speech via STT")
            logger.info("=" * 70)
        
        logger.info("=" * 70)
        logger.info(f"🎧 [LISTENING] Waiting for user speech via STT")
        logger.info(f"   STT service should now be active")
        logger.info("=" * 70)
    
    except Exception as exc:
        logger.error(f"❌ Intro workflow error for session {session_id}: {exc}", exc_info=True)
        # Reset to IDLE on error
        await state_mgr.transition(State.IDLE, "intro_error", {"error": str(exc)})
    finally:
        session_entry = active_sessions.get(session_id)
        if session_entry:
            session_entry["intro_task"] = None


async def monitor_timeouts():
    """Background task to monitor timeout for sessions in LISTENING state"""
    global config, dialogue_manager
    
    while True:
        try:
            await asyncio.sleep(1.0)  # Check every second
            
            if not config or not dialogue_manager:
                continue
            
            timeout_seconds = config.timeout_seconds
            current_time = time.time()
            
            # Check all active sessions
            for session_id, session_data in list(active_sessions.items()):
                state_mgr = session_data.get("state_mgr")
                websocket = session_data.get("websocket")
                workflow_started = session_data.get("workflow_started", False)
                
                if not state_mgr or not workflow_started:
                    continue
                
                # Only check timeout for sessions in LISTENING state
                if state_mgr.state != State.LISTENING:
                    continue
                
                # Check if timeout has occurred
                last_activity = state_mgr.context.last_activity_time
                if last_activity and (current_time - last_activity) >= timeout_seconds:
                    logger.info("=" * 70)
                    logger.info(f"⏱️ TIMEOUT DETECTED for session {session_id}")
                    logger.info(f"   Last activity: {last_activity:.2f}, Current: {current_time:.2f}")
                    logger.info(f"   Timeout duration: {timeout_seconds}s")
                    logger.info("=" * 70)
                    
                    # Get timeout dialogue from Dialogue Manager (JSON-driven)
                    timeout_asset = dialogue_manager.get_timeout_prompt()
                    logger.info(f"🎤 Playing timeout prompt: {timeout_asset.text[:50]}...")
                    
                    # Transition to SPEAKING for timeout message
                    try:
                        await state_mgr.transition(State.SPEAKING, "timeout_detected", {})
                        await broadcast_orchestrator_state(State.SPEAKING.value)  # Gate mic
                        
                        # Play timeout dialogue
                        await stream_tts_audio(
                            session_id,
                            timeout_asset.text,
                            websocket,
                            state_mgr,
                            emotion=timeout_asset.emotion,
                            audio_file_path=timeout_asset.audio_path if timeout_asset.has_audio() else None
                        )
                        
                        # Transition back to LISTENING and reset activity time
                        await state_mgr.transition(State.LISTENING, "timeout_complete", {})
                        await broadcast_orchestrator_state(State.LISTENING.value)  # Open mic
                        state_mgr.context.last_activity_time = time.time()  # Reset timeout timer
                        
                        if websocket:
                            try:
                                await websocket.send_json({
                                    "type": "timeout",
                                    "session_id": session_id,
                                    "message": "No response detected, prompting user..."
                                })
                            except:
                                pass
                    except Exception as e:
                        logger.error(f"❌ Error handling timeout for session {session_id}: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"❌ Error in timeout monitor: {e}", exc_info=True)
            await asyncio.sleep(5.0)  # Wait longer on error


async def listen_to_redis_events():
    """Background task to listen for STT transcripts and service connection events via Redis"""
    global redis_client, redis_subscriber
    
    if not redis_client:
        logger.warning("⚠️ Redis client not available, skipping event listener")
        return
    
    try:
        logger.info("=" * 70)
        logger.info("👂 Starting Redis event listener")
        logger.info("=" * 70)
        
        channels = [
            "leibniz:events:stt",
            "leibniz:events:stt:connected",
            "leibniz:events:tts:connected"
        ]
        
        # Create pubsub subscriber
        redis_subscriber = redis_client.pubsub()
        await redis_subscriber.subscribe(*channels)
        
        logger.info(f"✅ Subscribed to Redis channels: {', '.join(channels)}")
        
        # Listen for messages
        while True:
            try:
                message = await redis_subscriber.get_message(ignore_subscribe_messages=True, timeout=1.0)
                
                if message and message.get("type") == "message":
                    channel = message.get("channel")
                    raw_data = message.get("data", "{}")
                    try:
                        event_data = json.loads(raw_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse Redis event from {channel}: {e}")
                        continue
                    
                    if channel == "leibniz:events:stt":
                        text = event_data.get("text", "")
                        event_session_id = event_data.get("session_id", "")
                        is_final = event_data.get("is_final", False)
                        
                        if text:
                            logger.info("=" * 70)
                            logger.info(f"📨 Received STT event from Redis")
                            logger.info(f"   STT Session: {event_session_id}")
                            logger.info(f"   Text: {text[:100]}...")
                            logger.info(f"   Is Final: {is_final}")
                            logger.info("=" * 70)
                            
                            # Route STT events using shared routing logic
                            routed = await route_stt_text_to_active_sessions(
                                text=text,
                                is_final=is_final,
                                event_session_id=event_session_id,
                            )
                            if not routed:
                                logger.warning("⚠️ No active LISTENING sessions to route STT event")
                    elif channel in ("leibniz:events:stt:connected", "leibniz:events:tts:connected"):
                        service = "stt" if channel.endswith("stt:connected") else "tts"
                        await handle_service_connection(service, event_data)
            
            except asyncio.TimeoutError:
                # Timeout is normal, continue listening
                continue
        
    except asyncio.CancelledError:
        logger.info("🛑 Redis event listener cancelled")
        if redis_subscriber:
            await redis_subscriber.unsubscribe("leibniz:events:stt", "leibniz:events:stt:connected", "leibniz:events:tts:connected")
            await redis_subscriber.close()
    except Exception as e:
        logger.error(f"❌ Redis event listener error: {e}", exc_info=True)


async def route_stt_text_to_active_sessions(
    text: str,
    is_final: bool,
    event_session_id: str = "",
) -> bool:
    """
    Route STT text (from Redis or Unified FastRTC) to active orchestrator sessions.

    This centralizes the logic so both Redis-based STT and the unified FastRTC
    handler behave identically.
    """
    routed = False

    # Use list() to create a copy of items to avoid "dictionary changed size during iteration"
    for ws_session_id, session_data in list(active_sessions.items()):
        state_mgr = session_data.get("state_mgr")
        websocket = session_data.get("websocket")
        workflow_started = session_data.get("workflow_started", False)

        if not workflow_started:
            logger.debug(f"Skipping session {ws_session_id}: workflow not started")
            continue

        # Ignore STT while agent is speaking (TARA mode)
        if state_mgr and state_mgr.state == State.SPEAKING:
            if config and config.ignore_stt_while_speaking:
                logger.info(
                    f"🔇 IGNORING STT - Agent is SPEAKING (session: {ws_session_id})"
                )
                logger.debug(f"   Text ignored: {text[:50]}...")
                continue

        if state_mgr and state_mgr.state == State.LISTENING:
            logger.info(f"🎯 Routing STT to session: {ws_session_id}")
            # Process final transcripts to trigger the pipeline
            # Works for both WebSocket and auto-created sessions
            if is_final:
                await handle_stt_event(ws_session_id, text, websocket, state_mgr)
            else:
                # Incremental buffering for RAG
                if config and config.tara_mode and config.rag_service_url:
                    asyncio.create_task(
                        buffer_rag_incremental(
                            text=text,
                            session_id=ws_session_id,
                            rag_url=config.rag_service_url,
                            language=config.response_language,
                            organization=config.organization_name,
                        )
                    )
                    logger.debug(
                        f"📦 Incremental buffer triggered for: {text[:30]}..."
                    )

                # ElevenLabs prewarm for ultra-low latency
                if (
                    config
                    and config.use_elevenlabs_tts
                    and config.elevenlabs_prewarm_on_vad
                ):
                    if ws_session_id not in eleven_tts_clients:
                        asyncio.create_task(prewarm_elevenlabs_tts(ws_session_id))
                        logger.debug(
                            f"⚡ ElevenLabs prewarm triggered for: {ws_session_id}"
                        )

                # Notify client if WebSocket exists
                if websocket:
                    try:
                        await websocket.send_json(
                            {
                                "type": "stt_partial",
                                "text": text,
                                "session_id": ws_session_id,
                            }
                        )
                    except Exception:
                        pass
                else:
                    logger.debug(f"📝 Partial STT: {text[:50]}...")

            routed = True
        else:
            if state_mgr:
                logger.debug(
                    f"Skipping session {ws_session_id}: state={state_mgr.state.value}"
                )

    return routed


# ============================================================================
# Unified FastRTC callbacks (STT + connection events)
# These are the ACTUAL implementations - registered at module load time
# ============================================================================

async def _unified_on_stt_transcript_impl(
    fastrtc_session_id: str,
    text: str,
    is_final: bool,
) -> None:
    """
    Callback from UnifiedFastRTCHandler when STT text is available.

    We simply reuse the same routing logic as Redis-based STT so that
    orchestrator behaviour is identical irrespective of STT source.
    """
    if not text:
        return

    logger.info("=" * 70)
    logger.info("📨 Received STT event from Unified FastRTC")
    logger.info(f"   FastRTC Session: {fastrtc_session_id}")
    logger.info(f"   Text: {text[:100]}...")
    logger.info(f"   Is Final: {is_final}")
    logger.info("=" * 70)

    await route_stt_text_to_active_sessions(
        text=text,
        is_final=is_final,
        event_session_id=fastrtc_session_id,
    )


async def _unified_on_connection_change_impl(
    fastrtc_session_id: str,
    connected: bool,
) -> None:
    """
    Callback when a Unified FastRTC browser client connects or disconnects.

    We treat this as both STT and TTS being available for the orchestrator,
    so the workflow can become READY without separate FastRTC UIs.
    """
    global service_connections

    if connected:
        timestamp = time.time()
        service_connections["stt"] = {
            "connected": True,
            "session_id": fastrtc_session_id,
            "timestamp": timestamp,
        }
        service_connections["tts"] = {
            "connected": True,
            "session_id": fastrtc_session_id,
            "timestamp": timestamp,
        }
        logger.info("=" * 70)
        logger.info(f"🔌 Unified FastRTC connected | session: {fastrtc_session_id}")
        logger.info("   Marking STT and TTS as connected")
        logger.info("=" * 70)
    else:
        service_connections["stt"]["connected"] = False
        service_connections["tts"]["connected"] = False
        logger.info("=" * 70)
        logger.info(f"🔌 Unified FastRTC disconnected | session: {fastrtc_session_id}")
        logger.info("   Marking STT and TTS as disconnected")
        logger.info("=" * 70)

    # Notify all orchestrator sessions and recompute workflow readiness
    await broadcast_service_status(
        "Unified FastRTC connected" if connected else "Unified FastRTC disconnected"
    )
    await check_and_start_workflow("unified_fastrtc_connected" if connected else "unified_fastrtc_disconnected")


# Register callbacks and setup UI now that implementations are defined
if UNIFIED_FASTRTC_AVAILABLE and UnifiedFastRTCHandler is not None:
    UnifiedFastRTCHandler.on_stt_transcript = _unified_on_stt_transcript_impl
    UnifiedFastRTCHandler.on_connection_change = _unified_on_connection_change_impl
    logger.info("✅ Unified FastRTC callbacks registered")

# Setup the FastRTC UI (must be after callbacks are registered)
_setup_unified_fastrtc()

async def prewarm_elevenlabs_tts(session_id: str) -> bool:
    """
    Pre-warm ElevenLabs TTS connection for ultra-low latency.
    
    Called when VAD detects user speech (partial STT) to establish
    the TTS connection before the RAG response is ready.
    
    Args:
        session_id: Session identifier for the TTS connection
        
    Returns:
        True if prewarm successful, False otherwise
    """
    global eleven_tts_clients, config
    
    # Check if ElevenLabs TTS is enabled
    if not config or not config.use_elevenlabs_tts:
        return False
    
    if not ELEVENLABS_CLIENT_AVAILABLE:
        logger.warning("ElevenLabs TTS client not available")
        return False
    
    # Check if already prewarmed for this session
    if session_id in eleven_tts_clients:
        client = eleven_tts_clients[session_id]
        if client.is_prewarmed:
            logger.debug(f"ElevenLabs already prewarmed for {session_id}")
            return True
    
    try:
        # Create new client
        client = ElevenLabsTTSClient(
            tts_service_url=config.elevenlabs_tts_url,
            session_id=f"eleven_{session_id}_{int(time.time())}"
        )
        
        # Prewarm connection
        if await client.prewarm():
            eleven_tts_clients[session_id] = client
            logger.info(f"⚡ ElevenLabs TTS pre-warmed for session {session_id}")
            return True
        else:
            await client.close()
            logger.warning(f"Failed to prewarm ElevenLabs TTS for {session_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error prewarming ElevenLabs TTS: {e}")
        return False


async def cleanup_elevenlabs_client(session_id: str):
    """Clean up ElevenLabs TTS client for a session."""
    global eleven_tts_clients
    
    if session_id in eleven_tts_clients:
        client = eleven_tts_clients.pop(session_id)
        try:
            await client.close()
        except Exception as e:
            logger.debug(f"Error closing ElevenLabs client: {e}")


class SmartBuffer:
    """
    Buffers tokens and releases complete sentences for TTS.
    Ensures natural prosody by avoiding sending partial sentences.
    """
    def __init__(self, min_length: int = 20):
        self.buffer = ""
        self.min_length = min_length
        self.sentence_endings = {'.', '!', '?', '।', '\n'}
        self.abbreviations = {'Mr.', 'Mrs.', 'Dr.', 'St.', 'Prof.'}

    def add_token(self, token: str) -> Optional[str]:
        self.buffer += token
        
        # Check if we have a complete sentence
        if len(self.buffer) < self.min_length:
            return None
            
        # Find the last sentence ending
        last_end = -1
        for i, char in enumerate(self.buffer):
            if char in self.sentence_endings:
                # Check for abbreviations
                if i > 0 and self.buffer[i-1:i+1] in self.abbreviations:
                    continue
                last_end = i
        
        if last_end != -1:
            sentence = self.buffer[:last_end+1].strip()
            self.buffer = self.buffer[last_end+1:]
            if sentence:
                return sentence
        
        return None

    def flush(self) -> Optional[str]:
        """Release remaining buffer content"""
        if self.buffer.strip():
            content = self.buffer.strip()
            self.buffer = ""
            return content
        return None


async def _handle_tts_responses(tts_ws, websocket):
    """Helper to handle incoming TTS messages"""
    try:
        async for msg in tts_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get("type")
                
                if msg_type == "audio":
                    if websocket:
                        audio_b64 = data.get("data", "")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            try:
                                await websocket.send_bytes(audio_bytes)
                            except:
                                pass
                elif msg_type == "sentence_start":
                    logger.info(f"📢 TTS started speaking: {data.get('text', '')[:30]}...")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"❌ Error receiving from TTS: {e}")


async def stream_tts_from_generator(
    session_id: str,
    generator: AsyncGenerator[str, None],
    websocket: Optional[WebSocket],
    state_mgr: StateManager,
    emotion: str = "helpful"
):
    """
    Consumes text generator and streams to TTS service.
    
    Supports three modes:
    - "elevenlabs": Direct ElevenLabs streaming with prewarmed connection (ultra-low latency)
    - "continuous": Stream chunks immediately via standard TTS WebSocket (for ElevenLabs tts-labs)
    - "buffered": Wait for complete sentences (SmartBuffer) for better prosody (Sarvam)
    """
    global eleven_tts_clients
    
    # Check if we should use direct ElevenLabs streaming
    use_elevenlabs_direct = (
        config.use_elevenlabs_tts and 
        ELEVENLABS_CLIENT_AVAILABLE and 
        session_id in eleven_tts_clients
    )
    
    if use_elevenlabs_direct:
        # ═══════════════════════════════════════════════════════════════════
        # ELEVENLABS DIRECT MODE: Use pre-warmed ElevenLabs client
        # Matches sarvam's behavior EXACTLY: forward audio immediately as it arrives
        # ═══════════════════════════════════════════════════════════════════
        logger.info("⚡ Using ELEVENLABS DIRECT streaming mode (pre-warmed)")
        client = eleven_tts_clients[session_id]
        
        try:
            first_audio_sent = False
            chunks_sent = 0
            total_bytes_sent = 0
            
            async def on_first_audio(latency_ms: float):
                nonlocal first_audio_sent
                if not first_audio_sent:
                    first_audio_sent = True
                    if websocket:
                        try:
                            await websocket.send_json({
                                "type": "first_audio",
                                "latency_ms": latency_ms,
                                "mode": "elevenlabs_direct"
                            })
                        except Exception:
                            pass
            
            # Stream text to audio and forward IMMEDIATELY (exactly like sarvam's audio_callback)
            # No buffering, no queuing - forward as soon as we receive each chunk
            async for audio_bytes, metadata in client.stream_text_to_audio(generator, on_first_audio):
                # Forward audio to WebSocket client IMMEDIATELY (matches sarvam's pattern exactly)
                # This is the same pattern as sarvam's audio_callback function
                chunk_received_time = time.time()
                
                if websocket:
                    try:
                        # Send audio bytes directly (same as sarvam after base64 decode)
                        await websocket.send_bytes(audio_bytes)
                        chunks_sent += 1
                        total_bytes_sent += len(audio_bytes)
                        forward_time = time.time()
                        logger.info(f"📤 Forwarded audio chunk {chunks_sent}: {len(audio_bytes)} bytes (forwarded in {(forward_time - chunk_received_time)*1000:.1f}ms)")
                    except Exception as e:
                        logger.error(f"❌ Could not forward audio to client: {e}")
                        # Don't break, continue forwarding remaining chunks (like sarvam)
                else:
                    logger.warning("⚠️ No WebSocket connection to forward audio to")
                
                # Log progress
                if metadata.get("is_final"):
                    logger.info(f"✅ ElevenLabs stream complete (forwarded {chunks_sent} chunks, {total_bytes_sent} bytes)")
            
            # Keep connection open briefly to ensure all chunks are forwarded
            # Matches sarvam's pattern: allow time for final chunks to arrive
            await asyncio.sleep(0.5)
            
            # Get metrics
            metrics = client.get_metrics()
            logger.info(f"📊 ElevenLabs metrics: received {metrics.get('chunks_received')} chunks, "
                       f"{metrics.get('total_audio_bytes')} bytes | "
                       f"forwarded {chunks_sent} chunks, {total_bytes_sent} bytes | "
                       f"first audio: {metrics.get('first_audio_latency_ms', 'N/A')}ms")
            
        except Exception as e:
            logger.error(f"❌ Error in ElevenLabs direct streaming: {e}")
        finally:
            # Clean up the client after streaming
            await cleanup_elevenlabs_client(session_id)
        
        return
    
    # ═══════════════════════════════════════════════════════════════════
    # STANDARD MODE: Use generic TTS WebSocket (tts-sarvam or tts-labs)
    # ═══════════════════════════════════════════════════════════════════
    tts_ws_url = config.tts_service_url.replace("http://", "ws://").replace("https://", "wss://")
    tts_ws_url = f"{tts_ws_url}/api/v1/stream?session_id={session_id}"
    
    # Determine streaming mode
    streaming_mode = getattr(config, 'tts_streaming_mode', 'buffered')
    use_continuous_mode = streaming_mode == "continuous"
    
    buffer = SmartBuffer() if not use_continuous_mode else None
    full_text = ""
    
    try:
        logger.info(f"🔊 Connecting to TTS service for streaming: {tts_ws_url}")
        logger.info(f"   Streaming mode: {streaming_mode}")
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(tts_ws_url) as tts_ws:
                # Send initial connection message
                await tts_ws.send_json({
                    "type": "connect",
                    "session_id": session_id
                })
                
                # Start a background task to receive TTS events (audio, etc.)
                receive_task = asyncio.create_task(
                    _handle_tts_responses(tts_ws, websocket)
                )
                
                try:
                    if use_continuous_mode:
                        # =======================================================
                        # CONTINUOUS MODE: Stream chunks directly (ultra-low latency)
                        # Bypasses SmartBuffer for ElevenLabs stream-input API
                        # =======================================================
                        logger.info("⚡ Using CONTINUOUS streaming mode (bypassing SmartBuffer)")
                        chunk_count = 0
                        
                        async for token in generator:
                            if not token:
                                continue
                            full_text += token
                            chunk_count += 1
                            
                            # Send each token/chunk immediately
                            logger.debug(f"📤 Sending chunk {chunk_count}: {len(token)} chars")
                            await tts_ws.send_json({
                                "type": "stream_chunk",
                                "text": token,
                                "emotion": emotion
                            })
                        
                        logger.info(f"📤 Sent {chunk_count} chunks in continuous mode")
                    else:
                        # =======================================================
                        # BUFFERED MODE: Wait for complete sentences (better prosody)
                        # Uses SmartBuffer for natural speech patterns
                        # =======================================================
                        logger.info("📝 Using BUFFERED streaming mode (SmartBuffer)")
                        
                        async for token in generator:
                            full_text += token
                            sentence = buffer.add_token(token)
                            if sentence:
                                logger.info(f"📤 Sending sentence to TTS: {sentence[:50]}...")
                                await tts_ws.send_json({
                                    "type": "stream_chunk",
                                    "text": sentence,
                                    "emotion": emotion
                                })
                        
                        # Flush remaining buffer
                        remaining = buffer.flush()
                        if remaining:
                            logger.info(f"📤 Sending final chunk to TTS: {remaining[:50]}...")
                            await tts_ws.send_json({
                                "type": "stream_chunk",
                                "text": remaining,
                                "emotion": emotion
                            })
                        
                    # Send end of stream
                    await tts_ws.send_json({
                        "type": "stream_end"
                    })
                    
                    # Keep connection open for a bit to receive remaining audio
                    await asyncio.sleep(2.0)
                    
                except Exception as e:
                    logger.error(f"❌ Error in generator consumption: {e}")
                finally:
                    receive_task.cancel()
                    try:
                        await receive_task
                    except asyncio.CancelledError:
                        pass

    except Exception as e:
        logger.error(f"❌ Error in stream_tts_from_generator: {e}")


async def handle_stt_event(session_id: str, text: str, websocket: Optional[WebSocket], state_mgr: StateManager):
    """Handle STT event from Redis - trigger Intent+RAG+LLM pipeline or direct RAG (TARA mode)"""
    try:
        logger.info("=" * 70)
        logger.info(f"🤐 Processing STT event")
        logger.info(f"📝 Text: {text}")
        if config.tara_mode:
            logger.info(f"🇮🇳 TARA MODE: Direct RAG (skip Intent)")
        logger.info("=" * 70)
        
        session_data = active_sessions.get(session_id, {})
        
        # Check for exit keywords
        text_lower = text.lower().strip()
        # Prefer keywords from DialogueManager (JSON), fall back to config
        exit_keywords: list[str] = []
        if dialogue_manager and dialogue_manager.exit_keywords:
            exit_keywords.extend(dialogue_manager.exit_keywords)
        if hasattr(config, "exit_keywords"):
            exit_keywords.extend(
                kw.strip().lower() for kw in config.exit_keywords if kw.strip()
            )
        # Deduplicate
        exit_keywords = sorted(set(exit_keywords))
        if any(keyword in text_lower for keyword in exit_keywords):
            logger.info("=" * 70)
            logger.info(f"🚪 EXIT DETECTED: User said '{text}'")
            logger.info("=" * 70)
            
            # Get exit dialogue from Dialogue Manager
            if dialogue_manager:
                exit_asset = dialogue_manager.get_random_exit()
                logger.info(f"🎤 Playing exit dialogue: {exit_asset.text[:50]}...")
                
                # Transition to SPEAKING for exit message
                await state_mgr.transition(State.SPEAKING, "exit_detected", {})
                
                # Play exit dialogue
                await stream_tts_audio(
                    session_id,
                    exit_asset.text,
                    websocket,
                    state_mgr,
                    emotion=exit_asset.emotion,
                    audio_file_path=exit_asset.audio_path if exit_asset.has_audio() else None
                )
                
                # Transition to IDLE and mark session for cleanup
                await state_mgr.transition(State.IDLE, "exit_complete", {})
                
                if websocket:
                    try:
                        await websocket.send_json({
                            "type": "exit",
                            "session_id": session_id,
                            "message": "Session ending..."
                        })
                    except:
                        pass
                
                # Remove session after a short delay
                await asyncio.sleep(1.0)
                if session_id in active_sessions:
                    del active_sessions[session_id]
                    logger.info(f"✅ Session {session_id} ended and cleaned up")
                
                return
        
        # Update state to THINKING
        await state_mgr.transition(State.THINKING, "stt_received", {"text": text})
        await broadcast_orchestrator_state(State.THINKING.value)  # Gate mic during processing
        
        # Handle unclear / empty text with a dedicated prompt
        if not text.strip():
            logger.warning("⚠️ Empty/unclear user text, playing UNCLEAR prompt if available")
            if dialogue_manager:
                unclear_asset = dialogue_manager.get_unclear_prompt()
                await state_mgr.transition(State.SPEAKING, "unclear_prompt", {})
                await stream_tts_audio(
                    session_id,
                    unclear_asset.text,
                    websocket,
                    state_mgr,
                    emotion=unclear_asset.emotion,
                    audio_file_path=unclear_asset.audio_path if unclear_asset.has_audio() else None,
                )
                await state_mgr.transition(State.LISTENING, "unclear_complete", {})
            else:
                await state_mgr.transition(State.IDLE, "empty_text", {})
            return
        
        # ──────────────────────────────────────────────────────────────
        # 1) IMMEDIATE FILLER – right after user stops (hold the floor)
        # ──────────────────────────────────────────────────────────────
        latency_filler_task = None
        if dialogue_manager:
            immediate_asset = dialogue_manager.get_immediate_filler()
            logger.info(f"💭 Playing immediate filler: {immediate_asset.text[:50]}...")
            asyncio.create_task(
                stream_tts_audio(
                    session_id,
                    immediate_asset.text,
                    websocket,
                    state_mgr,
                    emotion=immediate_asset.emotion,
                    audio_file_path=immediate_asset.audio_path if immediate_asset.has_audio() else None,
                    skip_state_transition=True,  # Don't change FSM state for fillers
                )
            )
            
            # ──────────────────────────────────────────────────────────
            # 2) DELAYED FILLER – only if RAG/LLM+TTS takes > ~1.5s
            # ──────────────────────────────────────────────────────────
            async def delayed_latency_filler():
                try:
                    # Wait for latency threshold
                    await asyncio.sleep(1.5)
                    latency_asset = dialogue_manager.get_latency_filler()
                    logger.info(f"⏳ Playing latency filler: {latency_asset.text[:50]}...")
                    await stream_tts_audio(
                        session_id,
                        latency_asset.text,
                        websocket,
                        state_mgr,
                        emotion=latency_asset.emotion,
                        audio_file_path=latency_asset.audio_path if latency_asset.has_audio() else None,
                        skip_state_transition=True,  # Don't change FSM state
                    )
                except asyncio.CancelledError:
                    logger.info(f"Latency filler task cancelled for session {session_id}")
                    raise
                except Exception as exc:
                    logger.error(f"Error in latency filler for session {session_id}: {exc}", exc_info=True)

            latency_filler_task = asyncio.create_task(delayed_latency_filler())
            if isinstance(session_data, dict):
                session_data["latency_filler_task"] = latency_filler_task
        
        start_time = time.time()
        
        # ═══════════════════════════════════════════════════════════════
        # TARA MODE: Direct RAG call (skip Intent service)
        # For Telugu TASK customer service agent
        # Now uses INCREMENTAL RAG for lower latency
        # ═══════════════════════════════════════════════════════════════
        generator = None
        
        if config.skip_intent_service or config.tara_mode:
            if config.rag_service_url:
                logger.info("🇮🇳 TARA: Incremental RAG processing (using buffered context)...")
                
                # Use incremental RAG which leverages pre-buffered documents
                generator = process_rag_incremental(
                    user_text=text,
                    session_id=session_id,
                    rag_url=config.rag_service_url,
                    is_final=True,  # This is the final text, trigger generation
                    language=config.response_language,
                    organization=config.organization_name
                )
            else:
                logger.error("❌ TARA mode requires RAG service URL")
                await state_mgr.transition(State.IDLE, "error", {"error": "RAG service not configured"})
                return
        else:
            # Standard mode: Parallel Intent + RAG processing
            if config.rag_service_url:
                logger.info("⚡ Starting parallel Intent+RAG processing...")
            else:
                logger.info("⚡ Starting Intent processing (RAG not configured)...")
            
            generator = process_intent_rag_llm(
                text, 
                session_id,
                config.intent_service_url,
                config.rag_service_url  # Can be None if RAG not configured
            )
        
        # Cancel latency filler once we're ready to start speaking
        if isinstance(session_data, dict):
            task = session_data.get("latency_filler_task")
            if task and not task.done():
                task.cancel()
            session_data["latency_filler_task"] = None
        
        # Transition to SPEAKING immediately as we start streaming
        await state_mgr.transition(State.SPEAKING, "streaming_started", {})
        await broadcast_orchestrator_state(State.SPEAKING.value)  # Gate mic during TTS
        
        # Stream TTS from the generator
        if generator:
            await stream_tts_from_generator(session_id, generator, websocket, state_mgr)
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Interaction completed in {total_time:.0f}ms")
        
        # Transition back to LISTENING (or IDLE)
        await state_mgr.transition(State.LISTENING, "interaction_complete", {})
        await broadcast_orchestrator_state(State.LISTENING.value)  # Open mic for next turn
        
    except Exception as e:
        logger.error(f"❌ Error in handle_stt_event: {e}", exc_info=True)
        await state_mgr.transition(State.IDLE, "error", {"error": str(e)})


@app.post("/start")
async def start_workflow():
    """
    Trigger the workflow to start.
    
    Plays intro greeting via TTS, then transitions to LISTENING state
    to wait for user speech via STT.
    
    Prerequisites:
    - Both STT and TTS FastRTC clients must be connected
    - At least one WebSocket session must be active
    
    Example:
        curl -X POST http://localhost:8004/start
    """
    global workflow_triggered
    
    # Check if services are ready
    missing = get_missing_services()
    if missing:
        return {
            "success": False,
            "error": f"Services not ready. Missing: {', '.join(missing)}",
            "stt_connected": service_connections["stt"]["connected"],
            "tts_connected": service_connections["tts"]["connected"]
        }
    
    # Check if we have active sessions
    if not active_sessions:
        # Get the actual port from environment or use default
        orchestrator_port = os.getenv("ORCHESTRATOR_PORT", "5204")
        return {
            "success": False,
            "error": "No active WebSocket sessions. Connect to /orchestrate first.",
            "hint": f"Open a WebSocket connection to ws://localhost:{orchestrator_port}/orchestrate?session_id=my-session",
            "client_url": f"http://localhost:{orchestrator_port}"
        }
    
    # Check if workflow already triggered
    if workflow_triggered:
        return {
            "success": False,
            "error": "Workflow already triggered",
            "active_sessions": list(active_sessions.keys())
        }
    
    workflow_triggered = True
    
    logger.info("=" * 70)
    logger.info("🚀 WORKFLOW START TRIGGERED via /start endpoint")
    logger.info("=" * 70)
    
    # Start intro sequence for all active sessions
    started_sessions = []
    for session_id, session_data in active_sessions.items():
        if session_data.get("workflow_started"):
            logger.info(f"⏭️ Session {session_id} already has workflow running")
            continue
        
        session_data["workflow_started"] = True
        intro_task = asyncio.create_task(start_intro_sequence(session_id))
        session_data["intro_task"] = intro_task
        started_sessions.append(session_id)
        logger.info(f"🎬 Started intro sequence for session: {session_id}")
    
    return {
        "success": True,
        "message": "Workflow started - playing intro greeting",
        "sessions_started": started_sessions,
        "state_flow": [
            "IDLE -> SPEAKING (intro via TTS)",
            "SPEAKING -> LISTENING (waiting for user via STT)",
            "LISTENING -> THINKING (processing user input)",
            "THINKING -> SPEAKING (response via TTS)",
            "SPEAKING -> LISTENING (next turn)"
        ]
    }


@app.post("/simulate/turn")
async def simulate_turn(text: str = Query(..., description="User text to simulate"), session_id: Optional[str] = Query(None)):
    """
    Simulate a user turn by injecting text as if it came from STT.
    Useful for testing the RAG -> TTS flow without speaking.
    """
    target_session_id = session_id
    
    # If no session ID provided, pick the first active one
    if not target_session_id and active_sessions:
        target_session_id = next(iter(active_sessions))
    
    if not target_session_id or target_session_id not in active_sessions:
        # Try to find an auto-created session
        for sid, data in active_sessions.items():
            if data.get("is_auto_created"):
                target_session_id = sid
                break
        
        if not target_session_id:
             return {
                "success": False,
                "error": "No active session found. Connect via WebSocket or use /start to auto-create one."
            }

    session_data = active_sessions[target_session_id]
    state_mgr = session_data.get("state_mgr")
    websocket = session_data.get("websocket")
    
    logger.info(f"🤖 Simulating turn for session {target_session_id}: {text}")
    
    # Manually trigger the STT event handler
    # This will run the full pipeline: Intent -> RAG -> TTS
    asyncio.create_task(handle_stt_event(target_session_id, text, websocket, state_mgr))
    
    return {
        "success": True,
        "message": f"Turn simulation started for session {target_session_id}",
        "text": text,
        "pipeline": "Intent -> RAG -> TTS"
    }


@app.get("/status")
async def get_status():
    """
    Get current orchestrator status including service connections and session states.
    """
    session_states = {}
    for session_id, session_data in active_sessions.items():
        state_mgr = session_data.get("state_mgr")
        session_states[session_id] = {
            "state": state_mgr.state.value if state_mgr else "unknown",
            "workflow_started": session_data.get("workflow_started", False),
            "turn_number": state_mgr.context.turn_number if state_mgr else 0
        }
    
    # Get UnifiedFastRTC handler states
    unified_fastrtc_states = {}
    if UNIFIED_FASTRTC_AVAILABLE and UnifiedFastRTCHandler is not None:
        try:
            unified_fastrtc_states = UnifiedFastRTCHandler.get_all_states()
        except Exception as e:
            unified_fastrtc_states = {"error": str(e)}
    
    # Get the unified URL
    unified_url = f"http://localhost:{os.getenv('ORCHESTRATOR_PORT', '5204')}/fastrtc"
    
    return {
        "workflow_ready": workflow_ready,
        "workflow_triggered": workflow_triggered,
        "tara_mode": {
            "enabled": config.tara_mode if config else False,
            "skip_intent": config.skip_intent_service if config else False,
            "language": config.response_language if config else "en",
            "organization": config.organization_name if config else "Unknown",
            "ignore_stt_while_speaking": config.ignore_stt_while_speaking if config else True
        },
        "services": {
            "stt": {
                "connected": service_connections["stt"]["connected"],
                "session_id": service_connections["stt"]["session_id"]
            },
            "tts": {
                "connected": service_connections["tts"]["connected"],
                "session_id": service_connections["tts"]["session_id"]
            }
        },
        "active_sessions": session_states,
        "unified_fastrtc": {
            "available": UNIFIED_FASTRTC_AVAILABLE,
            "url": unified_url,
            "active_handlers": unified_fastrtc_states
        },
        "gradio_urls": {
            "unified": unified_url,
            "stt": STT_GRADIO_URL,
            "tts": TTS_GRADIO_URL
        }
    }


@app.post("/reset")
async def reset_workflow():
    """
    Reset the workflow state. Use this to restart after completion or error.
    """
    global workflow_triggered, workflow_ready
    
    workflow_triggered = False
    
    # Reset workflow_started for all sessions
    for session_id, session_data in active_sessions.items():
        session_data["workflow_started"] = False
        # Cancel any running intro tasks
        intro_task = session_data.get("intro_task")
        if intro_task and not intro_task.done():
            intro_task.cancel()
        session_data["intro_task"] = None
        
        # Reset state to IDLE
        state_mgr = session_data.get("state_mgr")
        if state_mgr:
            await state_mgr.transition(State.IDLE, "reset", {})
    
    logger.info("🔄 Workflow reset - ready for new /start trigger")
    
    return {
        "success": True,
        "message": "Workflow reset. Send POST /start to begin again.",
        "workflow_ready": workflow_ready
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    redis_connected = False
    if redis_client:
        try:
            redis_connected = await ping_redis(redis_client)
        except:
            pass
    
    return {
        "status": "healthy" if redis_connected else "degraded",
        "service": "orchestrator",
        "mode": "tara_telugu" if (config and config.tara_mode) else "standard",
        "active_sessions": len(active_sessions),
        "redis_connected": redis_connected,
        "workflow_ready": workflow_ready,
        "workflow_triggered": workflow_triggered,
        "uptime_seconds": time.time() - app_start_time
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return {
        "active_sessions": len(active_sessions),
        "uptime_seconds": time.time() - app_start_time,
        "total_turns": sum(
            s["state_mgr"].context.turn_number 
            for s in active_sessions.values()
        )
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")

