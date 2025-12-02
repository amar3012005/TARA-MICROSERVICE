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
from typing import Dict, Any, Optional

import aiohttp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from leibniz_agent.services.orchestrator.config import OrchestratorConfig
from leibniz_agent.services.orchestrator.state_manager import StateManager, State
from leibniz_agent.services.orchestrator.parallel_pipeline import process_intent_rag_llm
from leibniz_agent.services.orchestrator.interruption_handler import InterruptionHandler
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

# Intro greeting text (configurable via environment)
INTRO_GREETING = os.getenv("INTRO_GREETING", "Hello! I'm your assistant at Leibniz University. How can I help you today?")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for application startup/shutdown"""
    global redis_client, config
    
    logger.info("=" * 70)
    logger.info("🚀 Starting StateManager Orchestrator")
    logger.info("=" * 70)
    
    # Load configuration
    config = OrchestratorConfig.from_env()
    logger.info(f"📋 Configuration loaded")
    
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
    logger.info(f"🔗 STT FastRTC UI: {STT_GRADIO_URL}")
    logger.info(f"🔗 TTS FastRTC UI: {TTS_GRADIO_URL}")
    
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


async def play_intro_greeting(session_id: str, websocket: WebSocket, state_mgr: StateManager):
    """Play intro greeting via TTS streaming service"""
    try:
        logger.info(f"🎤 Playing intro greeting: {INTRO_GREETING[:50]}...")
        await stream_tts_audio(session_id, INTRO_GREETING, websocket, state_mgr, emotion="helpful")
    except Exception as e:
        logger.error(f"❌ Intro greeting error: {e}")


async def stream_tts_audio(
    session_id: str, 
    text: str, 
    websocket: WebSocket, 
    state_mgr: StateManager,
    emotion: str = "helpful"
):
    """Stream TTS audio from tts-streaming-service via WebSocket"""
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
                            # Forward audio chunk to client
                            audio_b64 = data.get("data", "")
                            if audio_b64:
                                audio_bytes = base64.b64decode(audio_b64)
                                await websocket.send_bytes(audio_bytes)
                        
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
                            await websocket.send_json({
                                "type": "tts_error",
                                "message": data.get("message", "TTS synthesis failed")
                            })
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
                
                # Send turn complete message
                await websocket.send_json({
                    "type": "turn_complete",
                    "session_id": session_id,
                    "turn_number": state_mgr.context.turn_number,
                    "state": state_mgr.state.value
                })
        
    except asyncio.CancelledError:
        logger.warning(f"⚡ TTS cancelled (barge-in)")
        raise
    except Exception as e:
        logger.error(f"❌ TTS streaming error: {e}", exc_info=True)
        await websocket.send_json({
            "type": "tts_error",
            "message": f"TTS streaming failed: {str(e)}"
        })


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
    
    # Both services connected - mark as ready but DON'T auto-start
    workflow_ready = True
    logger.info("=" * 70)
    logger.info(f"✅ [{trigger}] All FastRTC clients connected - workflow READY")
    logger.info(f"🎯 Send POST /start to trigger the intro greeting")
    logger.info(f"   Example: curl -X POST http://localhost:8004/start")
    logger.info("=" * 70)
    
    # Notify all connected sessions that we're ready
    await broadcast_service_status("All services connected. Send POST /start to begin.")


async def start_intro_sequence(session_id: str):
    """
    Play intro greeting and transition to listening once complete.
    
    State flow:
    1. IDLE -> SPEAKING (intro greeting via TTS)
    2. SPEAKING -> LISTENING (waiting for user speech via STT)
    """
    session_data = active_sessions.get(session_id)
    if not session_data:
        logger.warning(f"⚠️ Session {session_id} not found for intro sequence")
        return
    
    websocket = session_data.get("websocket")
    state_mgr = session_data.get("state_mgr")
    
    if not websocket or not state_mgr:
        logger.warning(f"⚠️ Session {session_id} missing websocket or state_mgr")
        return
    
    try:
        # Transition to SPEAKING state for intro
        await state_mgr.transition(State.SPEAKING, "intro_start", {"text": INTRO_GREETING})
        
        await send_service_status(
            websocket,
            session_id,
            "Playing intro greeting..."
        )
        
        logger.info("=" * 70)
        logger.info(f"🎤 [SPEAKING] Playing intro greeting for session: {session_id}")
        logger.info(f"📝 Text: {INTRO_GREETING}")
        logger.info("=" * 70)
        
        # Play intro via TTS
        await play_intro_greeting(session_id, websocket, state_mgr)
        
        # Transition to LISTENING state - now waiting for user speech
        await state_mgr.transition(State.LISTENING, "intro_complete", {})
        
        await websocket.send_json({
            "type": "intro_complete",
            "session_id": session_id,
            "state": State.LISTENING.value,
            "message": "Intro complete. Listening for your response..."
        })
        
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
                            for ws_session_id, session_data in active_sessions.items():
                                state_mgr = session_data.get("state_mgr")
                                websocket = session_data.get("websocket")
                                workflow_started = session_data.get("workflow_started", False)
                                
                                if not workflow_started:
                                    logger.debug(f"Skipping session {ws_session_id}: workflow not started")
                                    continue
                                    
                                if state_mgr and state_mgr.state == State.LISTENING:
                                    if websocket:
                                        logger.info(f"🎯 Routing STT to session: {ws_session_id}")
                                        # Only process final transcripts to trigger the pipeline
                                        if is_final:
                                            await handle_stt_event(ws_session_id, text, websocket, state_mgr)
                                        else:
                                            # For partial transcripts, just notify the client
                                            try:
                                                await websocket.send_json({
                                                    "type": "stt_partial",
                                                    "text": text,
                                                    "session_id": ws_session_id
                                                })
                                            except:
                                                pass
                                        routed = True
                                    else:
                                        logger.warning(f"⚠️ Session {ws_session_id} missing websocket")
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


async def handle_stt_event(session_id: str, text: str, websocket: WebSocket, state_mgr: StateManager):
    """Handle STT event from Redis - trigger Intent+RAG+LLM pipeline"""
    try:
        logger.info("=" * 70)
        logger.info(f"🤐 Processing STT event")
        logger.info(f"📝 Text: {text}")
        logger.info("=" * 70)
        
        # Update state to THINKING
        await state_mgr.transition(State.THINKING, "stt_received", {"text": text})
        
        if not text.strip():
            logger.warning("⚠️ Empty user text, skipping processing")
            await state_mgr.transition(State.IDLE, "empty_text", {})
            return
        
        start_time = time.time()
        
        # Process with parallel pipeline (RAG optional)
        if config.rag_service_url:
            logger.info("⚡ Starting parallel Intent+RAG processing...")
        else:
            logger.info("⚡ Starting Intent processing (RAG not configured)...")
        
        result = await process_intent_rag_llm(
            text, 
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
        
        # Stream TTS audio
        tts_task = asyncio.create_task(
            stream_tts_audio(session_id, response_text, websocket, state_mgr)
        )
        active_sessions[session_id]["tts_task"] = tts_task
        
    except Exception as e:
        logger.error(f"❌ Error handling STT event: {e}", exc_info=True)
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
        return {
            "success": False,
            "error": "No active WebSocket sessions. Connect to /orchestrate first.",
            "hint": "Open a WebSocket connection to ws://localhost:8004/orchestrate?session_id=my-session"
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

