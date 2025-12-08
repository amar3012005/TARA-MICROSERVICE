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
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, AsyncGenerator

import aiohttp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from leibniz_agent.services.orchestrator.config import OrchestratorConfig, TARA_INTRO_GREETING, DEFAULT_INTRO_GREETING
from leibniz_agent.services.orchestrator.state_manager import StateManager, State
from leibniz_agent.services.orchestrator.parallel_pipeline import (
    process_intent_rag_llm,
    process_rag_direct,
    process_rag_incremental,
    buffer_rag_incremental,
    reset_chunk_sequence
)
from leibniz_agent.services.orchestrator.interruption_handler import InterruptionHandler
from leibniz_agent.services.orchestrator.service_manager import ServiceManager
from leibniz_agent.services.shared.redis_client import get_redis_client, close_redis_client, ping_redis

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
active_sessions: Dict[str, Any] = {}
config: Optional[OrchestratorConfig] = None
app_start_time: float = time.time()
STT_GRADIO_URL = os.getenv("STT_GRADIO_URL", "http://localhost:8001/fastrtc")
TTS_GRADIO_URL = os.getenv("TTS_GRADIO_URL", "http://localhost:8005/fastrtc")
service_connections = {
    "stt": {"connected": False, "session_id": None, "timestamp": None},
    "tts": {"connected": False, "session_id": None, "timestamp": None}
}

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
    global redis_client, config, INTRO_GREETING
    
    logger.info("=" * 70)
    logger.info("🚀 Starting StateManager Orchestrator")
    logger.info("=" * 70)
    
    # Load configuration
    config = OrchestratorConfig.from_env()
    logger.info(f"📋 Configuration loaded")
    
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
    global redis_listener_task
    if redis_client:
        redis_listener_task = asyncio.create_task(listen_to_redis_events())
        logger.info("✅ Redis event listener started")
    
    logger.info("=" * 70)
    logger.info("✅ StateManager Orchestrator Ready")
    logger.info("=" * 70)
    logger.info("📋 SERVICE LINKS:")
    logger.info(f"   🔗 STT FastRTC UI: {STT_GRADIO_URL}")
    logger.info(f"   🔗 TTS FastRTC UI: {TTS_GRADIO_URL}")
    logger.info(f"   🔗 Orchestrator API: http://localhost:{os.getenv('PORT', '8004')}")
    logger.info("=" * 70)
    logger.info("⏳ WAITING FOR CONNECTIONS:")
    logger.info(f"   1. Open STT FastRTC UI in browser: {STT_GRADIO_URL}")
    logger.info(f"   2. Open TTS FastRTC UI in browser: {TTS_GRADIO_URL}")
    logger.info("   3. Connections will be detected automatically via Redis events")
    logger.info("   4. Once both STT and TTS connect, workflow will be ready")
    logger.info(f"   5. Send POST /start to trigger: curl -X POST http://localhost:{os.getenv('PORT', '8004')}/start")
    logger.info("=" * 70)
    
    # Wait for STT and TTS connections (non-blocking - connections happen via Redis events)
    logger.info("=" * 70)
    logger.info("⏳ WAITING FOR CONNECTIONS:")
    logger.info("   Connections will be detected automatically via Redis events")
    logger.info("   Once both STT and TTS connect, workflow will be ready")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown Redis listener
    if redis_listener_task:
        redis_listener_task.cancel()
        try:
            await redis_listener_task
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


@app.get("/")
async def root():
    """Serve the unified client HTML"""
    static_file = os.path.join(os.path.dirname(__file__), "static", "client.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "Leibniz Orchestrator API", "client": "/static/client.html"}


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
        
        logger.info(f"✅ Session cleanup complete: {session_id}")


async def play_intro_greeting(session_id: str, websocket: Optional[WebSocket], state_mgr: StateManager):
    """Play intro greeting via TTS streaming service"""
    try:
        logger.info(f"🎤 Playing intro greeting: {INTRO_GREETING[:50]}...")
        await stream_tts_audio(session_id, INTRO_GREETING, websocket, state_mgr, emotion="helpful")
    except Exception as e:
        logger.error(f"❌ Intro greeting error: {e}")


async def stream_tts_audio(
    session_id: str, 
    text: str, 
    websocket: Optional[WebSocket], 
    state_mgr: StateManager,
    emotion: str = "helpful"
):
    """Stream TTS audio from tts-streaming-service via WebSocket
    
    If websocket is None (auto-created session), audio still streams to TTS FastRTC UI.
    The TTS service handles audio playback directly.
    """
    tts_ws_url = config.tts_service_url.replace("http://", "ws://").replace("https://", "wss://")
    tts_ws_url = f"{tts_ws_url}/api/v1/stream?session_id={session_id}"
    
    try:
        logger.info(f"🔊 Connecting to TTS service: {tts_ws_url}")
        logger.info(f"📝 Text to synthesize: {text[:50]}...")
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(tts_ws_url) as tts_ws:
                # Send synthesis request with streaming enabled for ultra-low latency
                await tts_ws.send_json({
                    "type": "synthesize",
                    "text": text,
                    "emotion": emotion,
                    "streaming": True  # Enable Sarvam streaming API for fastest audio chunks
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
                        
                        elif msg_type == "streaming_started":
                            # Ultra-fast streaming mode started
                            is_ultra_fast = data.get("ultra_fast", False)
                            if is_ultra_fast:
                                logger.info(f"⚡ Ultra-fast TTS streaming started (target: <500ms first chunk)")
                            else:
                                logger.info(f"🌊 Streaming synthesis started")
                        
                        elif msg_type == "sentence_start":
                            sentence_count += 1
                            logger.info(f"📢 Sentence {sentence_count} starting: {data.get('text', '')[:50]}...")
                        
                        elif msg_type == "audio":
                            # Forward audio chunk to client (if WebSocket exists)
                            # For auto-created sessions, audio goes directly to TTS FastRTC UI
                            if websocket:
                                audio_b64 = data.get("data", "")
                                if audio_b64:
                                    audio_bytes = base64.b64decode(audio_b64)
                                    try:
                                        await websocket.send_bytes(audio_bytes)
                                    except Exception as ws_err:
                                        logger.debug(f"Could not forward audio to client: {ws_err}")
                            else:
                                # Auto-created session - audio streams directly to TTS FastRTC UI
                                logger.debug(f"📡 Audio streaming to TTS FastRTC UI (no client WebSocket)")
                        
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
                            is_ultra_fast = data.get("ultra_fast", False)
                            if is_ultra_fast:
                                first_chunk_latency = data.get("first_chunk_latency_ms", 0)
                                chunks_delivered = data.get("chunks_delivered", 0)
                                logger.info(f"⚡ Ultra-fast TTS complete: {chunks_delivered} chunks, first chunk: {first_chunk_latency:.0f}ms, total: {data.get('total_duration_ms', 0):.0f}ms")
                            else:
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
                
                # TTS complete - transition to IDLE
                await state_mgr.transition(State.IDLE, "tts_complete", {})
                state_mgr.context.turn_number += 1
                state_mgr.context.text_buffer = []  # Clear buffer for next turn
                await state_mgr.save_state()
                
                logger.info(f"✅ TTS streaming complete, ready for next turn")
                
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
        
        # Transition to LISTENING state - now waiting for user speech
        await state_mgr.transition(State.LISTENING, "intro_complete", {})
        
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
                            
                            # Route STT events to ALL active sessions in LISTENING state
                            # (STT session ID is different from WebSocket session ID)
                            routed = False
                            # Use list() to create a copy of items to avoid "dictionary changed size during iteration"
                            for ws_session_id, session_data in list(active_sessions.items()):
                                state_mgr = session_data.get("state_mgr")
                                websocket = session_data.get("websocket")
                                workflow_started = session_data.get("workflow_started", False)
                                
                                if not workflow_started:
                                    logger.debug(f"Skipping session {ws_session_id}: workflow not started")
                                    continue
                                
                                # ═══════════════════════════════════════════════════════════════
                                # CRITICAL: Ignore STT while agent is SPEAKING (TARA mode)
                                # This prevents transcription interference during TTS playback
                                # ═══════════════════════════════════════════════════════════════
                                if state_mgr and state_mgr.state == State.SPEAKING:
                                    if config.ignore_stt_while_speaking:
                                        logger.info(f"🔇 IGNORING STT - Agent is SPEAKING (session: {ws_session_id})")
                                        logger.debug(f"   Text ignored: {text[:50]}...")
                                        continue  # Skip this session - agent is speaking
                                
                                if state_mgr and state_mgr.state == State.LISTENING:
                                     logger.info(f"🎯 Routing STT to session: {ws_session_id}")
                                     # Process final transcripts to trigger the pipeline
                                     # Works for both WebSocket and auto-created sessions
                                     if is_final:
                                          await handle_stt_event(ws_session_id, text, websocket, state_mgr)
                                     else:
                                          # ═══════════════════════════════════════════════════════════════
                                          # PRE-LLM ACCUMULATION: Trigger RAG pre-processing on partial STT
                                          # RAG now performs full pre-LLM work during speech:
                                          # - Pattern detection
                                          # - Document retrieval
                                          # - Information extraction
                                          # - Prompt construction
                                          # This reduces final generation latency from ~400ms to ~355ms
                                          # ═══════════════════════════════════════════════════════════════
                                          if config.tara_mode and config.rag_service_url:
                                              # Track chunk sequence in state manager
                                              state_mgr.context.increment_chunk(text)
                                              chunk_seq = state_mgr.context.chunk_sequence
                                              
                                              # Fire-and-forget pre-LLM accumulation (don't await)
                                              asyncio.create_task(buffer_rag_incremental(
                                                  text=text,
                                                  session_id=ws_session_id,
                                                  rag_url=config.rag_service_url,
                                                  language=config.response_language,
                                                  organization=config.organization_name,
                                                  chunk_sequence=chunk_seq
                                              ))
                                              logger.info(
                                                  f"[SESSION:{ws_session_id}] Partial chunk {chunk_seq}: "
                                                  f"triggering RAG pre-LLM accumulation"
                                              )
                                          
                                          # Notify client if WebSocket exists
                                          if websocket:
                                              try:
                                                  await websocket.send_json({
                                                      "type": "stt_partial",
                                                      "text": text,
                                                      "session_id": ws_session_id,
                                                      "chunk_sequence": state_mgr.context.chunk_sequence if config.tara_mode else 0
                                                  })
                                              except:
                                                  pass
                                          else:
                                              logger.debug(f"📝 Partial STT (chunk {state_mgr.context.chunk_sequence}): {text[:50]}...")
                                          routed = True
                                else:
                                    if state_mgr:
                                        logger.debug(f"Skipping session {ws_session_id}: state={state_mgr.state.value}")
                            
                            if not routed:
                                logger.warning(f"⚠️ No active LISTENING sessions to route STT event")
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


class SmartBuffer:
    """
    Buffers tokens and releases complete sentences or phrases for TTS.
    Ensures natural prosody by avoiding sending partial words, but allows phrases.
    """
    def __init__(self, min_length: int = 10):
        self.buffer = ""
        self.min_length = min_length
        self.sentence_endings = {'.', '!', '?', '।', '\n', ',', ';', ':'}
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
    Consumes text generator and streams sentences to TTS service.
    """
    tts_ws_url = config.tts_service_url.replace("http://", "ws://").replace("https://", "wss://")
    tts_ws_url = f"{tts_ws_url}/api/v1/stream?session_id={session_id}"
    
    buffer = SmartBuffer()
    full_text = ""
    
    try:
        logger.info(f"🔊 Connecting to TTS service for streaming: {tts_ws_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(tts_ws_url) as tts_ws:
                # Start a background task to receive TTS events (audio, etc.)
                receive_task = asyncio.create_task(
                    _handle_tts_responses(tts_ws, websocket)
                )
                
                try:
                    async for token in generator:
                        full_text += token
                        sentence = buffer.add_token(token)
                        if sentence:
                            logger.info(f"📤 Sending phrase to TTS: {sentence[:50]}...")
                            await tts_ws.send_json({
                                "type": "synthesize",
                                "text": sentence,
                                "streaming": True,
                                "emotion": emotion
                            })
                    
                    # Flush remaining buffer
                    remaining = buffer.flush()
                    if remaining:
                        logger.info(f"📤 Sending final phrase to TTS: {remaining[:50]}...")
                        await tts_ws.send_json({
                            "type": "synthesize",
                            "text": remaining,
                            "streaming": True,
                            "emotion": emotion
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
    """
    Handle final STT event from Redis - trigger Intent+RAG+LLM pipeline or direct RAG (TARA mode).
    
    Pre-LLM Accumulation Optimization:
    - When this is called (is_final=true), RAG already has pre-built prompt from partial chunks
    - Generation uses the pre-accumulated context for faster response (~355ms vs ~400ms)
    """
    try:
        # Log accumulation stats
        chunks_processed = state_mgr.context.chunk_sequence
        accumulation_active = state_mgr.context.rag_accumulation_active
        
        logger.info("=" * 70)
        logger.info(f"[SESSION:{session_id}] 🤐 Processing final STT event")
        logger.info(f"   📝 Text: {text}")
        logger.info(f"   📊 Pre-LLM Stats: {chunks_processed} chunks accumulated, active={accumulation_active}")
        if config.tara_mode:
            logger.info(f"   🇮🇳 TARA MODE: Using optimized RAG generation (pre-built prompt)")
        logger.info("=" * 70)
        
        # Update state to THINKING
        await state_mgr.transition(State.THINKING, "stt_received", {"text": text})
        
        if not text.strip():
            logger.warning(f"[SESSION:{session_id}] ⚠️ Empty user text, skipping processing")
            # Reset accumulation for next utterance
            state_mgr.context.reset_accumulation()
            reset_chunk_sequence(session_id)
            await state_mgr.transition(State.IDLE, "empty_text", {})
            return
        
        start_time = time.time()
        
        # ═══════════════════════════════════════════════════════════════════════
        # TARA MODE: Optimized RAG call with Pre-LLM Accumulation
        # For Telugu TASK customer service agent
        #
        # Optimization Flow:
        # 1. Partial chunks → RAG pre-processing (pattern, retrieval, extraction, prompt)
        # 2. Final text → RAG generation using pre-built prompt (fast ~355ms)
        # ═══════════════════════════════════════════════════════════════════════
        generator = None
        
        if config.skip_intent_service or config.tara_mode:
            if config.rag_service_url:
                logger.info(
                    f"[SESSION:{session_id}] 🇮🇳 TARA: Requesting optimized generation "
                    f"(pre-accumulated from {chunks_processed} chunks)..."
                )
                
                # Use incremental RAG which leverages pre-built prompt
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
        
        # Transition to SPEAKING immediately as we start streaming
        await state_mgr.transition(State.SPEAKING, "streaming_started", {})
        
        # Stream TTS from the generator
        if generator:
            await stream_tts_from_generator(session_id, generator, websocket, state_mgr)
        
        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"[SESSION:{session_id}] ✅ Interaction completed in {total_time:.0f}ms "
            f"(with {chunks_processed} pre-accumulated chunks)"
        )
        
        # Reset accumulation state for next utterance
        state_mgr.context.reset_accumulation()
        reset_chunk_sequence(session_id)
        
        # Transition back to LISTENING (or IDLE)
        await state_mgr.transition(State.LISTENING, "interaction_complete", {})
        
    except Exception as e:
        logger.error(f"[SESSION:{session_id}] ❌ Error in handle_stt_event: {e}", exc_info=True)
        # Reset accumulation on error
        state_mgr.context.reset_accumulation()
        reset_chunk_sequence(session_id)
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
        "gradio_urls": {
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

