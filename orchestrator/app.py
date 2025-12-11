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
from typing import Dict, Any, Optional, AsyncGenerator, Coroutine

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
from leibniz_agent.services.orchestrator.orchestrator_fsm import OrchestratorFSM
from leibniz_agent.services.orchestrator.stt_event_handler import STTEventHandler
from leibniz_agent.services.orchestrator.parallel_pipeline import (
    process_intent_rag_llm, 
    process_rag_direct,
    process_rag_incremental,
    buffer_rag_incremental
)
from leibniz_agent.services.orchestrator.interruption_handler import InterruptionHandler
from leibniz_agent.services.orchestrator.service_manager import ServiceManager
from leibniz_agent.services.orchestrator.dialogue_manager import DialogueManager, DialogueType
from leibniz_agent.services.orchestrator.orchestrator_ws_handler import OrchestratorWSHandler
from leibniz_agent.services.shared.redis_client import (
    get_redis_client,
    close_redis_client,
    ping_redis,
    get_event_broker,
)

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
event_broker = None
redis_subscriber = None
redis_listener_task = None
timeout_monitor_task = None
orchestrator_ws_handler: Optional[OrchestratorWSHandler] = None
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

# Timeout prompt limit - after MAX_TIMEOUT_PROMPTS timeouts, end session with goodbye
MAX_TIMEOUT_PROMPTS = 3

# =====================================================================
# Session task management helpers (Phase 1: ensure single tracked task)
# =====================================================================
async def replace_session_task(
    session_id: str,
    coro: Optional[Coroutine],
    reason: str = "unspecified",
):
    """
    Cancel any existing per-session task and replace it with a new one.
    
    Designed for per-session audio tasks (TTS, fillers, etc.) to prevent
    overlapping playback. Returns the newly created task so callers can
    await it when sequential behaviour is required.
    """
    session_data = active_sessions.get(session_id)
    if not session_data:
        logger.warning(f"[{session_id}] Cannot set task - session not found")
        return None
    
    old_task = session_data.get("current_task")
    if old_task and not old_task.done():
        logger.info(f"[{session_id}] Cancelling current task (reason={reason})")
        old_task.cancel()
        try:
            await old_task
        except asyncio.CancelledError:
            logger.debug(f"[{session_id}] Previous task cancelled")
    
    if coro is None:
        session_data["current_task"] = None
        return None
    
    new_task = asyncio.create_task(coro)
    session_data["current_task"] = new_task
    logger.info(f"[{session_id}] ▶️ Task started (reason={reason})")
    return new_task


async def cancel_session_task(session_id: str, reason: str = "unspecified") -> None:
    """Cancel the currently tracked per-session task, if any."""
    await replace_session_task(session_id, None, reason=reason)


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
    global redis_client, event_broker, config, INTRO_GREETING, dialogue_manager
    
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
    
    # Initialize Event Broker (used by StateManager for audit logging)
    # NOTE: Redis Event Consumer removed in Phase 2 - all real-time events now flow via /ws WebSocket
    global redis_listener_task, timeout_monitor_task, event_broker
    if redis_client:
        try:
            event_broker = await get_event_broker()
            logger.info("✅ Event broker initialized (Redis Streams - audit logging only)")
            
            # Set event broker on UnifiedFastRTCHandler for WebRTC event emission
            if UNIFIED_FASTRTC_AVAILABLE and UnifiedFastRTCHandler is not None:
                UnifiedFastRTCHandler.set_event_broker(event_broker)
                logger.info("✅ Event broker set on UnifiedFastRTCHandler")
                
        except Exception as e:
            event_broker = None
            logger.warning(f"⚠️ Failed to initialize EventBroker: {e}")
    
    # Start timeout monitor background task
    timeout_monitor_task = asyncio.create_task(monitor_timeouts())
    logger.info("✅ Timeout monitor started")
    
    # Initialize unified WebSocket handler
    global orchestrator_ws_handler
    orchestrator_ws_handler = OrchestratorWSHandler(
        dialogue_manager=dialogue_manager,
        config=config,
        redis_client=redis_client,
        event_broker=event_broker
    )
    logger.info("✅ OrchestratorWSHandler initialized")
    
    # Determine URLs
    orchestrator_port = os.getenv('ORCHESTRATOR_PORT', os.getenv('PORT', '5204'))
    unified_fastrtc_url = f"http://localhost:{orchestrator_port}/fastrtc"
    orchestrator_api_url = f"http://localhost:{orchestrator_port}"
    ws_url = f"ws://localhost:{orchestrator_port}/ws"
    
    logger.info("=" * 70)
    logger.info("✅ StateManager Orchestrator Ready (Phase 2)")
    logger.info("=" * 70)
    logger.info("")
    logger.info("🎯 PRIMARY ENDPOINT (RECOMMENDED):")
    logger.info(f"   WebSocket: {ws_url}")
    logger.info("   ✓ Single bidirectional connection for audio + events")
    logger.info("   ✓ Real-time state sync with browser (playback_done confirmation)")
    logger.info("   ✓ No Redis event routing overhead")
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 REDIS ARCHITECTURE (Phase 2):")
    logger.info("   Redis is used for STATE PERSISTENCE only:")
    logger.info("   ✓ Session state survives restarts")
    logger.info("   ✓ Audit logging via EventBroker (fire-and-forget)")
    logger.info("   ✗ NO Redis event routing for /ws sessions")
    logger.info("=" * 70)
    logger.info("")
    logger.info("⚠️ LEGACY ENDPOINTS (deprecated):")
    logger.info(f"   /orchestrate - Legacy WebSocket (use /ws instead)")
    logger.info(f"   /fastrtc     - Legacy Gradio FastRTC UI: {unified_fastrtc_url}")
    logger.info(f"   External STT: {STT_GRADIO_URL}")
    logger.info(f"   External TTS: {TTS_GRADIO_URL}")
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"🔗 API Base: {orchestrator_api_url}")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown background tasks
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
    Broadcast orchestrator state to all UnifiedFastRTCHandler instances and WebSocket clients.
    
    This enables state-aware input gating (mute mic during THINKING/SPEAKING) and
    keeps browser clients synchronized with server state.
    
    Args:
        state_value: State string value (e.g., "listening", "speaking")
    """
    # 1. Broadcast to UnifiedFastRTCHandler instances
    if UNIFIED_FASTRTC_AVAILABLE and UnifiedFastRTCHandler:
        try:
            await UnifiedFastRTCHandler.broadcast_state_change(state_value)
        except Exception as e:
            logger.debug(f"State broadcast to FastRTC error: {e}")
    
    # 2. Notify all WebSocket clients
    for session_id, session_data in list(active_sessions.items()):
        websocket = session_data.get("websocket")
        if websocket:
            try:
                await websocket.send_json({
                    "type": "state_update",
                    "session_id": session_id,
                    "state": state_value
                })
            except Exception as e:
                logger.debug(f"Failed to send state update to WebSocket {session_id}: {e}")


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


@app.websocket("/ws")
async def websocket_unified(websocket: WebSocket, session_id: Optional[str] = Query(None)):
    """
    PHASE 1: Single unified WebSocket endpoint.
    
    Replaces fragmented FastRTC STT/TTS connections with a single bidirectional WebSocket.
    Browser sends: audio_chunk, playback_done, interrupt, start_session, end_session
    Server sends: session_ready, state_update, audio_chunk, agent_response, playback_control
    
    This is the recommended endpoint for new clients.
    """
    if orchestrator_ws_handler is None:
        logger.error("OrchestratorWSHandler not initialized")
        await websocket.close(code=1011, reason="Handler not initialized")
        return
    
    await orchestrator_ws_handler.handle_connection(websocket, session_id)


@app.websocket("/orchestrate")
async def orchestrate(websocket: WebSocket, session_id: str = Query(...)):
    """
    Legacy WebSocket orchestrator endpoint (requires session_id).
    
    Coordinates flow: STT → Intent+RAG (parallel) → LLM → TTS
    Handles interruptions and state transitions.
    
    NOTE: New clients should use /ws instead for unified single-connection experience.
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

    # Optionally start event-driven FSM (behind config flag)
    fsm: Optional[OrchestratorFSM] = None
    if config and config.use_event_driven and event_broker:
        fsm = OrchestratorFSM(session_id, redis_client, event_broker)

        # Output handler for TTS chunks and other outbound events
        async def fsm_output_handler(payload: Dict[str, Any]):
            # Expect base64 audio for TTS chunks
            audio_b64 = payload.get("audio_base64")
            if audio_b64:
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    await websocket.send_bytes(audio_bytes)
                except Exception as e:
                    logger.debug(f"FSM output handler failed to send audio: {e}")

        fsm.set_output_handler(fsm_output_handler)
        await fsm.start()
        logger.info(f"[{session_id}] Event-driven FSM enabled for session")
    elif config and not config.use_event_driven:
        logger.info(f"[{session_id}] Event-driven FSM disabled by config (USE_EVENT_DRIVEN=false)")
    else:
        logger.warning(f"[{session_id}] Event-driven FSM not started (missing broker or config)")
    
    active_sessions[session_id] = {
        # Core session resources
        "state_manager": state_mgr,
        "interrupt_handler": interrupt_handler,
        "websocket": websocket,
        # Task tracking (single tracked task for audio/fillers)
        "current_task": None,
        # Unified FastRTC handler reference for session routing
        "unified_handler": None,
        # Event-driven FSM
        "fsm": fsm,
        "fsm_task": None,
        # Session control flags
        "workflow_started": False,
        # Timeout tracking
        "timeout_count": 0,
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
            # PLAYBACK EVENTS - Browser confirms audio playback status
            # ═══════════════════════════════════════════════════════════════
            # Note: playback_started and playback_done are handled below in legacy handlers
            # but also route through handle_playback_event for consistency
            if msg_type in ("playback_event", "chunk_complete", "playback_stopped", "playback_error"):
                await handle_playback_event(session_id, message, state_mgr)
                continue
            
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
                
                # Stream TTS audio (tracked to prevent overlap)
                await replace_session_task(
                    session_id,
                    stream_tts_audio(session_id, response_text, websocket, state_mgr),
                    reason="response_tts",
                )
            
            # ═══════════════════════════════════════════════════════════════
            # INTERRUPT STATE - Barge-in (legacy from user_speaking)
            # ═══════════════════════════════════════════════════════════════
            elif msg_type == "user_speaking":
                if state_mgr.state == State.SPEAKING:
                    logger.warning(f"⚡ INTERRUPT: User started speaking during TTS")
                    
                    # Cancel active TTS/filler task
                    await cancel_session_task(session_id, reason="user_speaking")
                    
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
            
            # ═══════════════════════════════════════════════════════════════
            # WEBRTC EVENTS - Playback tracking from client
            # ═══════════════════════════════════════════════════════════════
            elif msg_type == "playback_started":
                logger.info(f"🔊 Client reported playback started: {session_id}")
                
                # Handle playback event directly
                await handle_playback_event(session_id, message, state_mgr)
                
                # Also emit to Redis Stream for event-driven FSM (if enabled)
                if event_broker and fsm:
                    try:
                        from leibniz_agent.services.shared.events import VoiceEvent, EventTypes
                        playback_event = VoiceEvent(
                            event_type=EventTypes.PLAYBACK_STARTED,
                            session_id=session_id,
                            source="client_websocket",
                            payload={"timestamp": message.get("timestamp", time.time())}
                        )
                        await event_broker.publish(f"voice:webrtc:session:{session_id}", playback_event)
                    except Exception as e:
                        logger.debug(f"Could not emit playback_started event: {e}")
            
            elif msg_type == "playback_done":
                duration_ms = message.get("duration_ms", 0)
                logger.info(f"🔊 Client reported playback done: {session_id} ({duration_ms}ms)")
                
                # Handle playback event directly (this will transition state)
                await handle_playback_event(session_id, message, state_mgr)
                
                # Also emit to Redis Stream for event-driven FSM (if enabled)
                if event_broker and fsm:
                    try:
                        from leibniz_agent.services.shared.events import VoiceEvent, EventTypes
                        playback_event = VoiceEvent(
                            event_type=EventTypes.PLAYBACK_DONE,
                            session_id=session_id,
                            source="client_websocket",
                            payload={
                                "duration_ms": duration_ms,
                                "timestamp": message.get("timestamp", time.time())
                            }
                        )
                        await event_broker.publish(f"voice:webrtc:session:{session_id}", playback_event)
                    except Exception as e:
                        logger.debug(f"Could not emit playback_done event: {e}")
            
            elif msg_type == "barge_in":
                reason = message.get("reason", "user_speaking")
                logger.warning(f"⚡ BARGE-IN from client: {reason} (session: {session_id})")
                
                if state_mgr.state == State.SPEAKING:
                    # Cancel active TTS/filler task
                    await cancel_session_task(session_id, reason="barge_in")
                    
                    # Reset to listening
                    state_mgr.context.text_buffer = []  # Clear buffer
                    await state_mgr.transition(State.INTERRUPT, "barge_in", {"reason": reason})
                    
                    # Emit barge-in event to Redis Stream
                    if event_broker and fsm:
                        try:
                            from leibniz_agent.services.shared.events import VoiceEvent, EventTypes
                            barge_event = VoiceEvent(
                                event_type=EventTypes.BARGE_IN,
                                session_id=session_id,
                                source="client_websocket",
                                payload={
                                    "reason": reason,
                                    "timestamp": message.get("timestamp", time.time())
                                }
                            )
                            await event_broker.publish(f"voice:webrtc:session:{session_id}", barge_event)
                        except Exception as e:
                            logger.debug(f"Could not emit barge_in event: {e}")
                    
                    # Broadcast to FastRTC handlers
                    if UNIFIED_FASTRTC_AVAILABLE and UnifiedFastRTCHandler is not None:
                        await UnifiedFastRTCHandler.broadcast_barge_in(reason)
                    
                    # Immediately go back to listening
                    await state_mgr.transition(State.LISTENING, "resume_listening", {})
                    await broadcast_orchestrator_state(State.LISTENING.value)
                    
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
            session_data = active_sessions[session_id]
            fsm = session_data.get("fsm")
            fsm_task = session_data.get("fsm_task")

            # Cancel any tracked per-session task (TTS/fillers)
            await cancel_session_task(session_id, reason="session_cleanup")

            # Cancel intro sequence task if present
            intro_task = session_data.get("intro_task")
            if intro_task and not intro_task.done():
                intro_task.cancel()
            # Stop FSM/task if running
            if fsm_task and not fsm_task.done():
                fsm_task.cancel()
                try:
                    await fsm_task
                except asyncio.CancelledError:
                    pass
            if fsm:
                try:
                    await fsm.stop()
                except Exception as e:
                    logger.debug(f"Error stopping FSM for session {session_id}: {e}")
            del active_sessions[session_id]
        
        # Clean up ElevenLabs TTS client if exists
        if session_id in eleven_tts_clients:
            await cleanup_elevenlabs_client(session_id)
        
        logger.info(f"✅ Session cleanup complete: {session_id}")


async def handle_playback_event(session_id: str, event_data: dict, state_mgr: StateManager):
    """
    Handle playback events from browser - source of truth for audio completion.
    
    The browser tracks actual Web Audio API playback and sends events when:
    - playback_started: First audio chunk begins playing
    - playback_done: ALL audio chunks have finished playing
    
    Only transition state when browser confirms playback completion.
    """
    event_type = event_data.get('event_type') or event_data.get('type')
    
    if event_type == 'playback_started':
        logger.info(f"[{session_id}] 🔊 Browser confirmed: Playback STARTED")
        # Update activity time
        state_mgr.context.last_activity_time = time.time()
        
    elif event_type == 'playback_done':
        duration_ms = event_data.get('duration_ms', 0)
        logger.info(f"[{session_id}] ✅ Browser confirmed: Playback DONE ({duration_ms:.0f}ms)")
        
        # Check if this is a session-ending playback (goodbye message)
        session_data = active_sessions.get(session_id)
        is_session_ending = session_data.get("is_ending", False) if session_data else False
        
        # Only transition if currently SPEAKING
        if state_mgr.state == State.SPEAKING:
            if is_session_ending:
                # For session end, transition to IDLE for cleanup
                await state_mgr.transition(
                    State.IDLE,
                    "session_ended",
                    {"source": "browser", "duration_ms": duration_ms}
                )
                await broadcast_orchestrator_state(State.IDLE.value)
                logger.info(f"[{session_id}] ✅ Goodbye playback complete, session ended")
                
                # Cleanup will happen in finally block or separate cleanup call
            else:
                # Normal playback completion - transition to LISTENING
                await state_mgr.transition(
                    State.LISTENING,
                    "playback_done",
                    {"source": "browser", "duration_ms": duration_ms}
                )
                await broadcast_orchestrator_state(State.LISTENING.value)
                
                # Update turn tracking
                state_mgr.context.turn_number += 1
                state_mgr.context.text_buffer = []
                await state_mgr.save_state()
                
                # Reset timeout timer after playback completes
                state_mgr.context.last_activity_time = time.time()
                
                logger.info(f"[{session_id}] ✅ Audio playback complete, ready for next turn")
        else:
            logger.warning(
                f"[{session_id}] playback_done received in unexpected state: {state_mgr.state.value} "
                f"(expected SPEAKING)"
            )
        
        # Update activity time
        state_mgr.context.last_activity_time = time.time()
        
    elif event_type == 'chunk_complete':
        chunk_id = event_data.get('chunk_id')
        logger.debug(f"[{session_id}] ✅ Browser chunk complete: {chunk_id}")
        
    elif event_type == 'playback_stopped':
        logger.info(f"[{session_id}] ⛔ Browser playback stopped (barge-in?)")
        # Update activity time
        state_mgr.context.last_activity_time = time.time()
        
    elif event_type == 'playback_error':
        error = event_data.get('error', 'Unknown error')
        chunk_id = event_data.get('chunk_id', 'unknown')
        logger.error(f"[{session_id}] ❌ Browser playback error on chunk {chunk_id}: {error}")


# Legacy listen_to_fastrtc_playback_events() removed - replaced by RedisEventConsumer._consume_webrtc_streams()
# This function handled WebRTC playback events which are now handled by RedisEventConsumer

async def _emit_playback_event(session_id: str, event_type: str, payload: dict):
    """
    Emit playback event to Redis Streams and UnifiedFastRTCHandler.
    
    Args:
        session_id: Session identifier
        event_type: Event type ("playback_started" or "playback_done")
        payload: Event payload dictionary
    """
    # 1. Emit to Redis Streams for event-driven FSM
    global event_broker
    if event_broker:
        try:
            from leibniz_agent.services.shared.events import VoiceEvent, EventTypes
            event_type_enum = EventTypes.PLAYBACK_STARTED if event_type == "playback_started" else EventTypes.PLAYBACK_DONE
            event = VoiceEvent(
                event_type=event_type_enum,
                session_id=session_id,
                source="orchestrator",
                payload=payload
            )
            await event_broker.publish(f"voice:webrtc:session:{session_id}", event)
            logger.debug(f"[{session_id}] Emitted {event_type} to Redis Stream")
        except Exception as e:
            logger.debug(f"Failed to emit {event_type} to Redis Stream: {e}")
    
    # 2. Notify UnifiedFastRTCHandler
    if UNIFIED_FASTRTC_AVAILABLE and UnifiedFastRTCHandler:
        try:
            for handler in UnifiedFastRTCHandler.active_instances.values():
                if handler.session_id == session_id or session_id.startswith("auto_session"):
                    if event_type == "playback_started":
                        await handler.emit_playback_started(chunk_count=payload.get("chunk_count", 1))
                    elif event_type == "playback_done":
                        await handler.emit_playback_done(
                            total_chunks=payload.get("total_chunks", 0),
                            duration_ms=payload.get("duration_ms", 0)
                        )
        except Exception as e:
            logger.debug(f"Failed to emit {event_type} to UnifiedFastRTCHandler: {e}")


async def stream_audio_file(
    session_id: str,
    audio_file_path: str,
    websocket: Optional[WebSocket],
    state_mgr: StateManager,
    skip_state_transition: bool = False,
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
            num_frames = wav_file.getnframes()
            
            # Estimate duration in milliseconds
            duration_ms = (num_frames / sample_rate) * 1000 if sample_rate > 0 else 0
            
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
                    first_chunk_sent = False  # Track when first chunk is sent
                    
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
                        broadcast_success = False
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
                                broadcast_success = True
                        except Exception as e:
                            logger.debug(f"Failed to broadcast audio file chunk: {e}")
                        
                        # CRITICAL: Emit playback_started AFTER first chunk is actually sent (to WebSocket or FastRTC)
                        # This ensures accurate timing - playback starts when audio actually reaches the client
                        if not first_chunk_sent:
                            first_chunk_sent = True
                            # Emit playback_started event (accurate timing)
                            await _emit_playback_event(session_id, "playback_started", {
                                "source": "pre_synthesized",
                                "file": audio_file_path,
                                "chunk_count": 1,
                                "timestamp": time.time()
                            })
                            
                            # For auto-created sessions, also trigger handle_playback_event for accurate start tracking
                            if websocket is None:
                                event_data = {
                                    "type": "playback_started",
                                    "event_type": "playback_started",
                                    "source": "fastrtc_fallback",
                                    "timestamp": time.time()
                                }
                                await handle_playback_event(session_id, event_data, state_mgr)
                        
                        # Small delay to simulate streaming
                        await asyncio.sleep(0.01)
                    
                    # Send completion
                    await tts_ws.send_json({
                        "type": "audio_complete",
                        "total_bytes": total_bytes
                    })
                    
                    logger.info(f"✅ Audio file streaming complete: {total_bytes} bytes")
                    
                    # UNIFIED SYNCHRONIZATION: Always wait for playback completion for auto-sessions
                    # This ensures ALL phrases (intro, timeout, goodbye, fillers) are synchronized
                    # Duration is already calculated accurately from WAV metadata (line 1190)
                    
                    if websocket is None:
                        # Auto-created session: Use calculated duration from WAV file (accurate)
                        expected_duration_s = duration_ms / 1000.0 if duration_ms > 0 else 0
                        if expected_duration_s > 0:
                            logger.info(f"[{session_id}] Auto-session: waiting {expected_duration_s:.2f}s for FastRTC playback completion")
                            await asyncio.sleep(expected_duration_s + 0.3)  # Add 300ms buffer
                            logger.info(f"[{session_id}] Auto-session: playback duration elapsed")
                            
                            # Manually trigger playback_done event (unified for all phrases)
                            event_data = {
                                "type": "playback_done",
                                "event_type": "playback_done",
                                "duration_ms": duration_ms,
                                "source": "fastrtc_fallback",
                                "timestamp": time.time()
                            }
                            logger.info(f"[{session_id}] 🎧 Manually triggering playback_done for audio file (auto-session)")
                            await handle_playback_event(session_id, event_data, state_mgr)
                        else:
                            # Fallback: estimate based on file size
                            estimated_duration_s = (total_bytes / sample_rate / sample_width) if sample_rate > 0 else 2.0
                            logger.info(f"[{session_id}] Auto-session: estimated wait {estimated_duration_s:.2f}s")
                            await asyncio.sleep(estimated_duration_s + 0.3)
                            event_data = {
                                "type": "playback_done",
                                "event_type": "playback_done",
                                "duration_ms": estimated_duration_s * 1000,
                                "source": "fastrtc_fallback",
                                "timestamp": time.time()
                            }
                            logger.info(f"[{session_id}] 🎧 Manually triggering playback_done after estimated wait (audio file)")
                            await handle_playback_event(session_id, event_data, state_mgr)
                    else:
                        # WebSocket session: Wait for browser playback_done event
                        logger.info(f"[{session_id}] Audio streaming complete, waiting for browser playback confirmation")
                        max_wait_time = 10.0
                        wait_start = time.time()
                        while state_mgr.state == State.SPEAKING and (time.time() - wait_start) < max_wait_time:
                            await asyncio.sleep(0.1)
                        
                        if state_mgr.state == State.SPEAKING:
                            logger.warning(f"[{session_id}] Timeout waiting for audio playback completion, forcing transition")
                            if not skip_state_transition:
                                await state_mgr.transition(State.LISTENING, "audio_complete_timeout", {})
                                await broadcast_orchestrator_state(State.LISTENING.value)
                    
                    # State transition handling (after synchronization)
                    if skip_state_transition:
                        logger.info(f"[{session_id}] Audio playback synchronized (state managed by caller)")
                    else:
                        # State transition already handled by handle_playback_event above
                        logger.info(f"[{session_id}] Audio playback synchronized and state transitioned")
                    
    except Exception as e:
        logger.error(f"❌ Error streaming audio file: {e}", exc_info=True)
        raise


async def play_intro_greeting(session_id: str, websocket: Optional[WebSocket], state_mgr: StateManager):
    """Play intro greeting via Dialogue Manager (supports pre-synthesized audio)"""
    try:
        if dialogue_manager:
            intro_asset = dialogue_manager.get_asset(DialogueType.INTRO)
            logger.info(f"🎤 Playing intro greeting: {intro_asset.text[:50]}...")
            intro_task = await replace_session_task(
                session_id,
                stream_tts_audio(
                    session_id,
                    intro_asset.text,
                    websocket,
                    state_mgr,
                    emotion=intro_asset.emotion,
                    audio_file_path=intro_asset.audio_path if intro_asset.has_audio() else None,
                    # State transitions around intro are managed by start_intro_sequence
                    skip_state_transition=True,
                ),
                reason="intro_greeting",
            )
            if intro_task:
                await intro_task
        else:
            # Fallback to config intro greeting
            logger.info(f"🎤 Playing intro greeting (fallback): {INTRO_GREETING[:50]}...")
            fallback_intro = await replace_session_task(
                session_id,
                stream_tts_audio(
                    session_id,
                    INTRO_GREETING,
                    websocket,
                    state_mgr,
                    emotion="helpful",
                    # State transitions around intro are managed by start_intro_sequence
                    skip_state_transition=True,
                ),
                reason="intro_greeting_fallback",
            )
            if fallback_intro:
                await fallback_intro
    except Exception as e:
        logger.error(f"❌ Intro greeting error: {e}")


async def stream_tts_audio(
    session_id: str, 
    text: str, 
    websocket: Optional[WebSocket], 
    state_mgr: StateManager,
    emotion: str = "helpful",
    audio_file_path: Optional[str] = None,
    skip_state_transition: bool = False,
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
    
    # Verify we're using tts_sarvam, not ElevenLabs
    if config.use_elevenlabs_tts:
        logger.warning("⚠️ ElevenLabs TTS enabled but should use tts_sarvam only - falling back to tts_sarvam")
    
    # Verify TTS URL points to tts_sarvam
    if "tts-sarvam" not in config.tts_service_url and "tts_sarvam" not in config.tts_service_url:
        logger.warning(f"⚠️ TTS URL may not be tts_sarvam: {config.tts_service_url}")
    else:
        logger.debug(f"✅ TTS URL verified: using tts_sarvam service")
    
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
                total_chunks = 0
                total_audio_bytes = 0  # Track total audio bytes for duration calculation
                actual_sample_rate = None  # Will be set from first audio chunk
                first_chunk_sent = False  # Track when first chunk is sent to FastRTC
                
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
                            
                            # Track actual sample rate from first chunk
                            if actual_sample_rate is None:
                                actual_sample_rate = sample_rate

                            if audio_b64:
                                audio_bytes = base64.b64decode(audio_b64)
                                total_chunks += 1
                                total_audio_bytes += len(audio_bytes)  # Accumulate audio bytes for duration calculation

                                # 1) Send to any connected orchestrator WebSocket client
                                if websocket:
                                    try:
                                        await websocket.send_bytes(audio_bytes)
                                    except Exception as ws_err:
                                        logger.debug(f"Could not forward audio to client: {ws_err}")

                                # 2) Broadcast to all Unified FastRTC sessions (primary audio path)
                                broadcast_success = False
                                try:
                                    from .unified_fastrtc import UnifiedFastRTCHandler  # Local import to avoid circulars

                                    await UnifiedFastRTCHandler.broadcast_audio(
                                        audio_bytes, sample_rate
                                    )
                                    broadcast_success = True
                                except Exception as broadcast_err:
                                    logger.debug(f"Could not broadcast audio to Unified FastRTC: {broadcast_err}")
                                
                                # CRITICAL: Emit playback_started AFTER first chunk is actually sent (to WebSocket or FastRTC)
                                # This ensures accurate timing - playback starts when audio actually reaches the client
                                if not first_chunk_sent:
                                    first_chunk_sent = True
                                    # Emit playback_started event (accurate timing)
                                    await _emit_playback_event(session_id, "playback_started", {
                                        "source": "tts_generated",
                                        "text": text[:100],
                                        "emotion": emotion,
                                        "timestamp": time.time()
                                    })
                                    
                                    # For auto-created sessions, also trigger handle_playback_event for accurate start tracking
                                    if websocket is None:
                                        event_data = {
                                            "type": "playback_started",
                                            "event_type": "playback_started",
                                            "source": "fastrtc_fallback",
                                            "timestamp": time.time()
                                        }
                                        await handle_playback_event(session_id, event_data, state_mgr)
                        
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
                            # Try to get total_duration_ms from TTS service, but calculate from audio if not available
                            reported_duration_ms = data.get('total_duration_ms', 0)
                            if reported_duration_ms > 0:
                                total_duration_ms = reported_duration_ms
                            else:
                                # Calculate duration from actual audio bytes received (16-bit PCM = 2 bytes per sample)
                                if total_audio_bytes > 0 and actual_sample_rate and actual_sample_rate > 0:
                                    total_samples = total_audio_bytes // 2  # 16-bit = 2 bytes per sample
                                    total_duration_ms = (total_samples / actual_sample_rate) * 1000.0
                                    logger.info(f"📊 Calculated TTS duration from audio bytes: {total_duration_ms:.0f}ms ({total_audio_bytes} bytes, {actual_sample_rate}Hz)")
                                elif total_audio_bytes > 0:
                                    # Fallback to config sample rate if not received from chunks
                                    fallback_rate = config.tts_sample_rate if hasattr(config, "tts_sample_rate") else 24000  # type: ignore[assignment]
                                    total_samples = total_audio_bytes // 2
                                    total_duration_ms = (total_samples / fallback_rate) * 1000.0
                                    logger.info(f"📊 Calculated TTS duration from audio bytes (fallback rate): {total_duration_ms:.0f}ms ({total_audio_bytes} bytes, {fallback_rate}Hz)")
                                else:
                                    # No audio bytes received, use text-length estimation as final fallback
                                    logger.warning(f"[{session_id}] No audio bytes received for duration calculation, will use text-length estimation")
                            
                            logger.info(f"✅ TTS streaming complete: {data.get('total_sentences', 0)} sentences, {total_duration_ms:.0f}ms")
                            
                            # DO NOT emit playback_done here - this is server-side guessing
                            # Browser will confirm when audio actually finishes playing
                            logger.info(f"[{session_id}] TTS streaming complete, waiting for browser playback confirmation")
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
                
                # UNIFIED SYNCHRONIZATION: Always wait for playback completion for auto-sessions
                # This ensures ALL phrases (intro, timeout, goodbye, fillers, regular responses) are synchronized
                # Duration is calculated from audio bytes received (accurate) or text length (fallback)
                
                if websocket is None:
                    # Auto-created session: Use calculated duration from audio bytes (accurate)
                    if total_duration_ms > 0:
                        duration_s = total_duration_ms / 1000.0
                        logger.info(f"[{session_id}] Auto-session: waiting {duration_s:.2f}s for TTS playback completion")
                        await asyncio.sleep(duration_s + 0.3)  # Add 300ms buffer
                        logger.info(f"[{session_id}] Auto-session: TTS playback duration elapsed")
                        
                        # Manually trigger playback_done event (unified for all phrases)
                        event_data = {
                            "type": "playback_done",
                            "event_type": "playback_done",
                            "duration_ms": total_duration_ms,
                            "source": "fastrtc_fallback",
                            "timestamp": time.time()
                        }
                        logger.info(f"[{session_id}] 🎧 Manually triggering playback_done for TTS (auto-session)")
                        await handle_playback_event(session_id, event_data, state_mgr)
                    else:
                        # Fallback: estimate based on text length
                        estimated_duration_s = len(text) * 0.05  # ~50ms per character
                        logger.info(f"[{session_id}] Auto-session: estimated wait {estimated_duration_s:.2f}s for TTS")
                        await asyncio.sleep(estimated_duration_s + 0.3)
                        event_data = {
                            "type": "playback_done",
                            "event_type": "playback_done",
                            "duration_ms": estimated_duration_s * 1000,
                            "source": "fastrtc_fallback",
                            "timestamp": time.time()
                        }
                        logger.info(f"[{session_id}] 🎧 Manually triggering playback_done after estimated wait (TTS)")
                        await handle_playback_event(session_id, event_data, state_mgr)
                else:
                    # WebSocket session: Wait for browser playback_done event
                    logger.info(f"[{session_id}] TTS streaming complete, waiting for browser playback confirmation")
                    max_wait_time = 10.0
                    wait_start = time.time()
                    while state_mgr.state == State.SPEAKING and (time.time() - wait_start) < max_wait_time:
                        await asyncio.sleep(0.1)
                    
                    if state_mgr.state == State.SPEAKING:
                        logger.warning(f"[{session_id}] Timeout waiting for TTS playback completion, forcing transition")
                        if not skip_state_transition:
                            await state_mgr.transition(State.LISTENING, "tts_complete_timeout", {})
                            await broadcast_orchestrator_state(State.LISTENING.value)
                            state_mgr.context.turn_number += 1
                            state_mgr.context.text_buffer = []
                            await state_mgr.save_state()
                
                # State transition handling (after synchronization)
                if skip_state_transition:
                    logger.info(f"[{session_id}] TTS playback synchronized (state managed by caller)")
                else:
                    # State transition already handled by handle_playback_event above
                    logger.info(f"[{session_id}] TTS playback synchronized and state transitioned")
                
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
    """
    Handle STT/TTS FastRTC connection signals from Redis.
    
    DEPRECATED (Phase 2): This function is for legacy /orchestrate and /fastrtc endpoints only.
    New /ws sessions handle service connections directly via the WebSocket handshake.
    """
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
            "state_manager": state_mgr,
            "interrupt_handler": interrupt_handler,
            "websocket": None,  # No client WebSocket - TTS streams to FastRTC UI directly
            "current_task": None,
            "unified_handler": None,
            "fsm": None,
            "fsm_task": None,
            "workflow_started": False,
            "is_auto_created": True,
            # Timeout tracking
            "timeout_count": 0,
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
    state_mgr = session_data.get("state_manager")
    
    if not state_mgr:
        logger.warning(f"⚠️ Session {session_id} missing state_manager")
        return
    
    try:
        # Transition to SPEAKING state for intro
        # Use "response" key to satisfy SPEAKING precondition in StateManager
        await state_mgr.transition(
            State.SPEAKING,
            "intro_start",
            {"response": INTRO_GREETING},
        )
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
        # Synchronization is now handled inside stream_audio_file (unified for all phrases)
        await play_intro_greeting(session_id, websocket, state_mgr)
        
        # State transition is handled by stream_audio_file -> handle_playback_event
        # No need to wait here - synchronization is unified in the core function
        
        # Now log LISTENING state (state has actually transitioned)
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


async def end_session_with_goodbye(session_id: str, state_mgr: StateManager, websocket: Optional[WebSocket]):
    """
    End a session gracefully by playing a goodbye message and cleaning up resources.
    
    Called when MAX_TIMEOUT_PROMPTS is reached or session should end.
    
    Args:
        session_id: Session identifier
        state_mgr: StateManager instance for the session
        websocket: Optional WebSocket connection to send goodbye message
    """
    global dialogue_manager
    
    logger.info("=" * 70)
    logger.info(f"👋 Ending session {session_id} with goodbye message")
    logger.info("=" * 70)
    
    try:
        # Get exit dialogue from Dialogue Manager
        exit_asset = dialogue_manager.get_random_exit()
        logger.info(f"🎤 Playing goodbye: {exit_asset.text[:50]}...")
        
        # Transition to SPEAKING for goodbye message
        await state_mgr.transition(
            State.SPEAKING,
            "session_ending",
            {"response": exit_asset.text},
        )
        await broadcast_orchestrator_state(State.SPEAKING.value)  # Gate mic
        
        # Mark session as ending so handle_playback_event knows to transition to IDLE
        session_data = active_sessions.get(session_id)
        if session_data:
            session_data["is_ending"] = True
        
        # Play goodbye dialogue
        goodbye_task = await replace_session_task(
            session_id,
            stream_tts_audio(
                session_id,
                exit_asset.text,
                websocket,
                state_mgr,
                emotion=exit_asset.emotion,
                audio_file_path=exit_asset.audio_path if exit_asset.has_audio() else None,
                skip_state_transition=True,  # State transitions managed explicitly
            ),
            reason="session_goodbye",
        )
        if goodbye_task:
            await goodbye_task
        
        # Synchronization is now handled inside stream_tts_audio/stream_audio_file (unified for all phrases)
        # State transition is handled by handle_playback_event
        # No need to wait here - synchronization is unified in the core function
        
        # Send goodbye message to client if websocket exists
        if websocket:
            try:
                await websocket.send_json({
                    "type": "session_ended",
                    "session_id": session_id,
                    "message": "Session ended due to inactivity",
                    "goodbye": exit_asset.text
                })
            except Exception as e:
                logger.debug(f"Failed to send goodbye message to WebSocket: {e}")
        
        # Clean up session resources
        session_data = active_sessions.get(session_id)
        if session_data:
            # Cancel any tracked per-session task
            await cancel_session_task(session_id, reason="session_ending")
            
            # Stop FSM if running
            fsm = session_data.get("fsm")
            fsm_task = session_data.get("fsm_task")
            if fsm_task and not fsm_task.done():
                fsm_task.cancel()
                try:
                    await fsm_task
                except asyncio.CancelledError:
                    pass
            if fsm:
                try:
                    await fsm.stop()
                except Exception as e:
                    logger.debug(f"Error stopping FSM for session {session_id}: {e}")
            
            # Clean up ElevenLabs client if exists
            await cleanup_elevenlabs_client(session_id)
            
            # Remove session from active sessions
            del active_sessions[session_id]
            logger.info(f"✅ Session {session_id} cleaned up and removed")
        
    except Exception as e:
        logger.error(f"❌ Error in end_session_with_goodbye for {session_id}: {e}", exc_info=True)
        # Ensure cleanup happens even on error
        if session_id in active_sessions:
            try:
                await cancel_session_task(session_id, reason="error_cleanup")
                await cleanup_elevenlabs_client(session_id)
                del active_sessions[session_id]
            except:
                pass


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
                state_mgr = session_data.get("state_manager")
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
                    # Get timeout count for this session
                    timeout_count = session_data.get("timeout_count", 0)
                    timeout_count += 1
                    session_data["timeout_count"] = timeout_count
                    
                    logger.info("=" * 70)
                    logger.info(f"⏱️ TIMEOUT DETECTED for session {session_id} (count: {timeout_count}/{MAX_TIMEOUT_PROMPTS})")
                    logger.info(f"   Last activity: {last_activity:.2f}, Current: {current_time:.2f}")
                    logger.info(f"   Timeout duration: {timeout_seconds}s")
                    logger.info("=" * 70)
                    
                    # Check if we've exceeded the maximum timeout prompts
                    if timeout_count >= MAX_TIMEOUT_PROMPTS:
                        logger.info(f"🛑 Maximum timeout prompts ({MAX_TIMEOUT_PROMPTS}) reached. Ending session with goodbye.")
                        try:
                            await end_session_with_goodbye(session_id, state_mgr, websocket)
                        except Exception as e:
                            logger.error(f"❌ Error ending session with goodbye for {session_id}: {e}", exc_info=True)
                    else:
                        # Get timeout dialogue from Dialogue Manager (JSON-driven)
                        timeout_asset = dialogue_manager.get_timeout_prompt()
                        logger.info(f"🎤 Playing timeout prompt: {timeout_asset.text[:50]}...")
                        
                        # Transition to SPEAKING for timeout message
                        try:
                            await state_mgr.transition(
                                State.SPEAKING,
                                "timeout_detected",
                                # Use "response" to satisfy SPEAKING precondition
                                {"response": timeout_asset.text},
                            )
                            await broadcast_orchestrator_state(State.SPEAKING.value)  # Gate mic
                            
                            # Play timeout dialogue
                            timeout_task = await replace_session_task(
                                session_id,
                                stream_tts_audio(
                                    session_id,
                                    timeout_asset.text,
                                    websocket,
                                    state_mgr,
                                    emotion=timeout_asset.emotion,
                                    audio_file_path=timeout_asset.audio_path if timeout_asset.has_audio() else None,
                                    # State transitions are managed explicitly before/after playback
                                    skip_state_transition=True,
                                ),
                                reason="timeout_prompt",
                            )
                            if timeout_task:
                                await timeout_task
                            
                            # Synchronization is now handled inside stream_tts_audio/stream_audio_file (unified for all phrases)
                            # State transition is handled by handle_playback_event
                            # No need to wait here - synchronization is unified in the core function
                            
                            if websocket:
                                try:
                                    await websocket.send_json({
                                        "type": "timeout",
                                        "session_id": session_id,
                                        "message": "No response detected, prompting user...",
                                        "timeout_count": timeout_count,
                                        "max_timeouts": MAX_TIMEOUT_PROMPTS
                                    })
                                except:
                                    pass
                        except Exception as e:
                            logger.error(f"❌ Error handling timeout for session {session_id}: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"❌ Error in timeout monitor: {e}", exc_info=True)
            await asyncio.sleep(5.0)  # Wait longer on error


# Legacy listen_to_redis_events() removed - replaced by RedisEventConsumer
# This function handled Pub/Sub channels which are now handled by RedisEventConsumer._consume_pubsub_events()

async def resolve_session_for_stt(event_session_id: str):
    """
    Map an incoming STT session ID to the correct orchestrator session.
    
    Rules:
    1) Direct match in active_sessions
    2) Match via UnifiedFastRTCHandler.active_instances -> active_sessions[unified_handler]
    3) Otherwise drop with clear logging
    """
    if not event_session_id:
        logger.error("STT event missing session_id; dropping event")
        return None, None
    
    # Rule 1: Direct match
    if event_session_id in active_sessions:
        return event_session_id, active_sessions[event_session_id]
    
    # Rule 2: Match via UnifiedFastRTC handler registry
    if UNIFIED_FASTRTC_AVAILABLE and UnifiedFastRTCHandler is not None:
        handler = UnifiedFastRTCHandler.active_instances.get(event_session_id)
        if handler:
            for sid, sdata in active_sessions.items():
                if sdata.get("unified_handler") == handler:
                    return sid, sdata
    
    # Rule 3: No match
    logger.error(
        f"Cannot map STT session {event_session_id} to any orchestrator session"
    )
    logger.error(f"Active orchestrator sessions: {list(active_sessions.keys())}")
    if UNIFIED_FASTRTC_AVAILABLE and UnifiedFastRTCHandler is not None:
        logger.error(
            f"Active UnifiedFastRTC handlers: {list(UnifiedFastRTCHandler.active_instances.keys())}"
        )
    return None, None


async def route_stt_text_to_active_sessions(
    text: str,
    is_final: bool,
    event_session_id: str = "",
    source: str = "unknown",
) -> bool:
    """
    Route STT text (from Redis or Unified FastRTC) to active orchestrator sessions.

    This centralizes the logic so both Redis-based STT and the unified FastRTC
    handler behave identically.
    
    DEPRECATED (Phase 2): This function is for legacy /orchestrate and /fastrtc endpoints only.
    New /ws sessions receive STT directly via OrchestratorWSHandler._stt_receive_loop().
    """
    # Resolve orchestrator session using deterministic rules
    ws_session_id, session_data = await resolve_session_for_stt(event_session_id)
    if not ws_session_id or not session_data:
        return False

    state_mgr = session_data.get("state_manager")
    websocket = session_data.get("websocket")
    workflow_started = session_data.get("workflow_started", False)

    if not workflow_started:
        logger.debug(f"Skipping session {ws_session_id}: workflow not started")
        return False

    # Ignore STT while agent is speaking (TARA mode)
    if state_mgr and state_mgr.state == State.SPEAKING:
        if config and config.ignore_stt_while_speaking:
            logger.info(
                f"🔇 IGNORING STT - Agent is SPEAKING (session: {ws_session_id})"
            )
            logger.debug(f"   Text ignored: {text[:50]}...")
            return False

    if state_mgr and state_mgr.state == State.LISTENING:
        logger.info(
            f"[{event_session_id}] STT route -> orchestrator session [{ws_session_id}]"
        )
        # Process final transcripts to trigger the pipeline
        if is_final:
            await handle_stt_event(ws_session_id, text, websocket, state_mgr, source=source)
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

        return True

    if state_mgr:
        logger.debug(
            f"Skipping session {ws_session_id}: state={state_mgr.state.value}"
        )
    return False


# ============================================================================
# Unified FastRTC callbacks (STT + connection events)
# DEPRECATED (Phase 2): These callbacks are for legacy /fastrtc endpoint only.
# New /ws sessions use OrchestratorWSHandler for all event handling.
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
    
    DEPRECATED (Phase 2): This callback is for legacy /fastrtc endpoint only.
    New /ws sessions receive STT directly via OrchestratorWSHandler._stt_receive_loop().
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
        source="unified_handler",
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

        # Link FastRTC handler to orchestrator session for routing
        handler = None
        if UNIFIED_FASTRTC_AVAILABLE and UnifiedFastRTCHandler is not None:
            handler = UnifiedFastRTCHandler.active_instances.get(fastrtc_session_id)
        
        if handler:
            target_session_id = None
            if fastrtc_session_id in active_sessions:
                target_session_id = fastrtc_session_id
            elif len(active_sessions) == 1:
                target_session_id = next(iter(active_sessions))
            else:
                # Prefer auto-created sessions for linking
                for sid, sdata in active_sessions.items():
                    if sdata.get("is_auto_created"):
                        target_session_id = sid
                        break
            
            if target_session_id:
                active_sessions[target_session_id]["unified_handler"] = handler
                logger.info(
                    f"[{target_session_id}] Linked UnifiedFastRTC handler {fastrtc_session_id}"
                )
            else:
                logger.warning(
                    f"Could not link UnifiedFastRTC handler {fastrtc_session_id} to a session"
                )
        else:
            logger.debug(f"No UnifiedFastRTC handler found for {fastrtc_session_id}")
        logger.info("=" * 70)
        logger.info(f"🔌 Unified FastRTC connected | session: {fastrtc_session_id}")
        logger.info("   Marking STT and TTS as connected")
        logger.info("=" * 70)
    else:
        service_connections["stt"]["connected"] = False
        service_connections["tts"]["connected"] = False
        if UNIFIED_FASTRTC_AVAILABLE and UnifiedFastRTCHandler is not None:
            handler = UnifiedFastRTCHandler.active_instances.get(fastrtc_session_id)
            if handler:
                for sid, sdata in active_sessions.items():
                    if sdata.get("unified_handler") == handler:
                        sdata["unified_handler"] = None
                        logger.info(f"[{sid}] Unlinked UnifiedFastRTC handler {fastrtc_session_id}")
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


async def handle_stt_event(
    session_id: str, 
    text: str, 
    websocket: Optional[WebSocket], 
    state_mgr: StateManager,
    source: str = "unknown"
):
    """
    Handle STT event via STTEventHandler - trigger Intent+RAG+LLM pipeline
    
    Args:
        session_id: Session identifier
        text: STT transcript text
        websocket: Optional WebSocket connection
        state_mgr: StateManager instance
        source: Event source ("redis_stream", "unified_handler", "websocket", etc.)
    """
    try:
        # Reset timeout count on user activity (any STT input resets the counter)
        session_data = active_sessions.get(session_id)
        if session_data and session_data.get("timeout_count", 0) > 0:
            old_count = session_data["timeout_count"]
            session_data["timeout_count"] = 0
            logger.info(f"[{session_id}] ✅ User activity detected - reset timeout count ({old_count} → 0)")
        
        handler = STTEventHandler(session_id, state_mgr, config, dialogue_manager)
        
        # Callback to start fillers when THINKING state is entered
        async def on_thinking():
            session_data = active_sessions.get(session_id)
            if not session_data:
                return
                
            # 1) Immediate Filler
            if dialogue_manager:
                immediate_asset = dialogue_manager.get_immediate_filler()
                logger.info(f"💭 Playing immediate filler: {immediate_asset.text[:50]}...")
                await replace_session_task(
                    session_id,
                    stream_tts_audio(
                        session_id,
                        immediate_asset.text,
                        websocket,
                        state_mgr,
                        emotion=immediate_asset.emotion,
                        audio_file_path=immediate_asset.audio_path if immediate_asset.has_audio() else None,
                        skip_state_transition=True,
                    ),
                    reason="immediate_filler",
                )
                
                # 2) Delayed Latency Filler
                async def delayed_latency_filler():
                    try:
                        await asyncio.sleep(1.5)
                        latency_asset = dialogue_manager.get_latency_filler()
                        logger.info(f"⏳ Playing latency filler: {latency_asset.text[:50]}...")
                        await replace_session_task(
                            session_id,
                            stream_tts_audio(
                                session_id,
                                latency_asset.text,
                                websocket,
                                state_mgr,
                                emotion=latency_asset.emotion,
                                audio_file_path=latency_asset.audio_path if latency_asset.has_audio() else None,
                                skip_state_transition=True,
                            ),
                            reason="latency_filler",
                        )
                    except asyncio.CancelledError:
                        logger.debug(f"[{session_id}] Latency filler cancelled")
                    except Exception as e:
                        logger.error(f"Error in latency filler: {e}")

                # Track delayed filler task
                latency_task = asyncio.create_task(delayed_latency_filler())
                session_data["latency_filler_task"] = latency_task

        # Execute handler
        result = await handler.handle_stt_final(
            text, 
            is_final=True, 
            source=source,
            on_thinking=on_thinking
        )
        
        # Cancel latency filler if pending
        session_data = active_sessions.get(session_id, {})
        latency_task = session_data.get("latency_filler_task")
        if latency_task and not latency_task.done():
            latency_task.cancel()
            session_data["latency_filler_task"] = None

        # 1. Handle Exit (result is True)
        if result is True:
            if dialogue_manager:
                exit_asset = dialogue_manager.get_random_exit()
                exit_task = await replace_session_task(
                    session_id,
                    stream_tts_audio(
                        session_id,
                        exit_asset.text,
                        websocket,
                        state_mgr,
                        emotion=exit_asset.emotion,
                        audio_file_path=exit_asset.audio_path if exit_asset.has_audio() else None
                    ),
                    reason="exit_dialogue",
                )
                if exit_task:
                    await exit_task
            
            await state_mgr.transition(State.IDLE, "exit_complete", {})
            
            # Remove session
            await asyncio.sleep(1.0)
            if session_id in active_sessions:
                del active_sessions[session_id]
                logger.info(f"✅ Session {session_id} ended and cleaned up")
            return

        # 2. Handle Generator (Response)
        if result:
            # Stream TTS (tracked)
            stream_task = await replace_session_task(
                session_id,
                stream_tts_from_generator(session_id, result, websocket, state_mgr),
                reason="response_stream",
            )
            if stream_task:
                await stream_task
            
            await state_mgr.transition(State.LISTENING, "interaction_complete", {})
            await broadcast_orchestrator_state(State.LISTENING.value)

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
    state_mgr = session_data.get("state_manager")
    websocket = session_data.get("websocket")
    
    logger.info(f"🤖 Simulating turn for session {target_session_id}: {text}")
    
    # Manually trigger the STT event handler
    # This will run the full pipeline: Intent -> RAG -> TTS
    asyncio.create_task(handle_stt_event(target_session_id, text, websocket, state_mgr, source="api_endpoint"))
    
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
        state_mgr = session_data.get("state_manager")
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
        # Cancel any tracked per-session task
        await cancel_session_task(session_id, reason="workflow_reset")
        # Cancel any running intro tasks
        intro_task = session_data.get("intro_task")
        if intro_task and not intro_task.done():
            intro_task.cancel()
        session_data["intro_task"] = None
        
        # Reset state to IDLE
        state_mgr = session_data.get("state_manager")
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
            s["state_manager"].context.turn_number 
            for s in active_sessions.values()
            if s.get("state_manager")
        )
    }


@app.get("/latency/{session_id}")
async def get_latency(session_id: str):
    """
    Get per-session latency breakdown and current state.

    This exposes the in-memory latency samples tracked by StateManager
    for quick debugging and observability during experiments.
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = active_sessions[session_id]
    state_mgr = session_data.get("state_manager")
    if not state_mgr:
        raise HTTPException(status_code=500, detail="StateManager not initialized for session")

    # Basic latency statistics (average per transition key)
    latency_breakdown = await state_mgr.get_latency_breakdown()

    return {
        "session_id": session_id,
        "state": state_mgr.state.value,
        "latencies_ms": latency_breakdown,
        "timestamps": state_mgr.context.timestamps,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")

