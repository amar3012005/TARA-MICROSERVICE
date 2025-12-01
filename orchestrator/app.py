"""
StateManager Orchestrator FastAPI Application

WebSocket-based orchestrator that coordinates STT → Intent+RAG (parallel) → LLM → TTS flow
with FSM state management and barge-in detection.

Reference:
    services/docs/ORCHESTRATOR_IMPLEMENTATION.md - Implementation details
    services/docs/ORCHESTRATOR_GUIDE.md - Architecture guide
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

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
active_sessions: Dict[str, Any] = {}
config: Optional[OrchestratorConfig] = None
app_start_time: float = time.time()


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
    
    # Connect to Redis with timeout
    try:
        redis_client = await asyncio.wait_for(get_redis_client(), timeout=15.0)
        await asyncio.wait_for(ping_redis(redis_client), timeout=5.0)
        logger.info("✅ Redis connected")
    except asyncio.TimeoutError:
        logger.warning(f"⚠️ Redis connection timeout - service will run in degraded mode")
        redis_client = None
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}")
        redis_client = None
    
    logger.info("=" * 70)
    logger.info("✅ StateManager Orchestrator Ready")
    logger.info("=" * 70)
    
    yield
    
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
        "tts_task": None
    }
    
    # Send initial connection message
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "state": state_mgr.state.value
    })
    
    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Session {session_id} timeout")
                break
            
            msg_type = message.get("type")
            
            # ═══════════════════════════════════════════════════════════════
            # LISTENING STATE - Buffer STT fragments
            # ═══════════════════════════════════════════════════════════════
            if msg_type == "stt_fragment":
                if state_mgr.state != State.LISTENING:
                    await state_mgr.transition(State.LISTENING, "stt_start", {})
                
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
            if tts_task and not tts_task.done():
                tts_task.cancel()
            del active_sessions[session_id]
        
        logger.info(f"✅ Session cleanup complete: {session_id}")


async def stream_tts_audio(session_id: str, text: str, websocket: WebSocket, state_mgr: StateManager):
    """Simulate TTS audio streaming (replace with real TTS service call)"""
    try:
        # TODO: Replace with real TTS service call
        logger.info(f"🔊 Streaming TTS for: {text[:50]}...")
        
        # Simulate chunked audio streaming
        chunk_count = max(1, len(text) // 50)  # Rough estimate
        for i in range(min(chunk_count, 10)):  # Max 10 chunks
            await asyncio.sleep(0.1)  # Simulate TTS latency
            
            # Send mock audio chunk
            await websocket.send_bytes(b"audio_chunk_" + str(i).encode())
        
        # TTS complete
        await state_mgr.transition(State.IDLE, "tts_complete", {})
        state_mgr.context.turn_number += 1
        state_mgr.context.text_buffer = []  # Clear buffer for next turn
        await state_mgr.save_state()
        
        logger.info(f"✅ TTS complete, ready for next turn")
        
        # Send turn complete message
        await websocket.send_json({
            "type": "turn_complete",
            "session_id": session_id,
            "turn_number": state_mgr.context.turn_number,
            "state": state_mgr.state.value
        })
        
    except asyncio.CancelledError:
        logger.warning(f"⚡ TTS cancelled (barge-in)")
    except Exception as e:
        logger.error(f"❌ TTS streaming error: {e}")


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

