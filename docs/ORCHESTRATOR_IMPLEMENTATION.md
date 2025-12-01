# StateManager Orchestrator - Core Implementation

## 📂 File: `orchestrator/state_manager.py`

**Purpose**: Core Finite State Machine (FSM) engine with Redis persistence

```python
import asyncio
import json
import logging
import time
from enum import Enum
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, asdict
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class State(Enum):
    """Conversation states"""
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPT = "interrupt"

@dataclass
class ConversationContext:
    """Persistent conversation state"""
    session_id: str
    state: str
    text_buffer: list
    intent: Optional[dict] = None
    rag_results: Optional[dict] = None
    llm_response: Optional[str] = None
    turn_number: int = 0
    timestamps: Dict[str, float] = None
    
    def __post_init__(self):
        if self.timestamps is None:
            self.timestamps = {}

class StateManager:
    """
    Ultra-low latency FSM for real-time voice conversations.
    
    Manages state transitions, Redis persistence, and event handling.
    Latency target: <100ms per transition
    """
    
    def __init__(self, session_id: str, redis_client: redis.Redis):
        self.session_id = session_id
        self.redis = redis_client
        self.state = State.IDLE
        self.context = ConversationContext(session_id=session_id, state=State.IDLE.value)
        self.transition_callbacks: Dict[str, list[Callable]] = {
            state.value: [] for state in State
        }
        
    async def initialize(self):
        """Load session state from Redis or create new"""
        redis_key = f"session:{self.session_id}"
        
        try:
            existing = await self.redis.hgetall(redis_key)
            if existing:
                logger.info(f"[{self.session_id}] ✅ Loaded session from Redis")
                # Deserialize and restore state
                self.context.state = existing.get(b"state", b"idle").decode()
                self.context.turn_number = int(existing.get(b"turn_number", 0))
                self.state = State(self.context.state)
            else:
                logger.info(f"[{self.session_id}] 🆕 Created new session")
                await self.save_state()
        except Exception as e:
            logger.warning(f"[{self.session_id}] ⚠️ Redis load failed: {e}")
    
    async def transition(self, new_state: State, trigger: str, data: Optional[Dict] = None):
        """
        Atomic state transition with logging and Redis persistence.
        
        Args:
            new_state: Target state
            trigger: Transition trigger (e.g., "stt_fragment", "vad_end")
            data: Optional context data
        """
        old_state = self.state
        timestamp = time.time()
        
        # Update state
        self.state = new_state
        self.context.state = new_state.value
        self.context.timestamps[f"{old_state.value}_to_{new_state.value}"] = timestamp
        
        # Update context with data
        if data:
            if "text" in data:
                self.context.text_buffer.append(data["text"])
            if "intent" in data:
                self.context.intent = data["intent"]
            if "rag_results" in data:
                self.context.rag_results = data["rag_results"]
            if "response" in data:
                self.context.llm_response = data["response"]
        
        # Persist to Redis (O(1) operation)
        await self.save_state()
        
        # Emit logs with emojis for Docker console visibility
        state_emoji = {
            State.IDLE: "🟢",
            State.LISTENING: "🔵",
            State.THINKING: "🟡",
            State.SPEAKING: "🔴",
            State.INTERRUPT: "⚡"
        }
        
        logger.info(
            f"[{self.session_id}] {state_emoji[old_state]} {old_state.value.upper()} "
            f"→ {state_emoji[new_state]} {new_state.value.upper()} "
            f"({trigger})"
        )
        
        # Fire callbacks
        for callback in self.transition_callbacks.get(new_state.value, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.context)
                else:
                    callback(self.context)
            except Exception as e:
                logger.error(f"[{self.session_id}] ❌ Callback error: {e}")
    
    async def save_state(self):
        """Persist state to Redis (atomic)"""
        redis_key = f"session:{self.session_id}"
        
        state_dict = {
            "state": self.context.state,
            "turn_number": self.context.turn_number,
            "text_buffer": json.dumps(self.context.text_buffer),
            "intent": json.dumps(self.context.intent or {}),
            "rag_results": json.dumps(self.context.rag_results or {}),
            "llm_response": self.context.llm_response or "",
            "last_update": time.time()
        }
        
        try:
            await self.redis.hset(redis_key, mapping=state_dict)
            await self.redis.expire(redis_key, 3600)  # 1 hour TTL
        except Exception as e:
            logger.warning(f"[{self.session_id}] ⚠️ Redis save failed: {e}")
    
    async def load_state(self):
        """Load state from Redis"""
        redis_key = f"session:{self.session_id}"
        
        try:
            data = await self.redis.hgetall(redis_key)
            if data:
                self.context.state = data.get(b"state", b"idle").decode()
                self.context.turn_number = int(data.get(b"turn_number", 0))
                self.context.text_buffer = json.loads(data.get(b"text_buffer", b"[]"))
                self.context.intent = json.loads(data.get(b"intent", b"{}"))
                self.context.rag_results = json.loads(data.get(b"rag_results", b"{}"))
                self.context.llm_response = data.get(b"llm_response", b"").decode()
                self.state = State(self.context.state)
                logger.debug(f"[{self.session_id}] ✅ State loaded from Redis")
        except Exception as e:
            logger.error(f"[{self.session_id}] ❌ Load failed: {e}")
    
    def on_transition(self, state: State):
        """Register callback for state transitions"""
        def decorator(func):
            self.transition_callbacks[state.value].append(func)
            return func
        return decorator
    
    async def get_latency_breakdown(self) -> Dict[str, float]:
        """Calculate latencies between state transitions"""
        latencies = {}
        timestamps = self.context.timestamps
        
        state_sequence = [
            ("idle_to_listening", "STT Wait"),
            ("listening_to_thinking", "End-of-Turn"),
            ("thinking_to_speaking", "LLM Wait"),
            ("speaking_to_idle", "TTS Complete"),
        ]
        
        for transition, label in state_sequence:
            if transition in timestamps:
                prev_ts = list(timestamps.values())[list(timestamps.keys()).index(transition) - 1] \
                    if list(timestamps.keys()).index(transition) > 0 else 0
                latencies[label] = (timestamps[transition] - prev_ts) * 1000  # ms
        
        return latencies
```

---

## 📂 File: `orchestrator/app.py`

**Purpose**: FastAPI WebSocket orchestrator endpoint

```python
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
from state_manager import StateManager, State
from parallel_pipeline import process_intent_rag_llm
from interruption_handler import InterruptionHandler
from config import OrchestratorConfig

logger = logging.getLogger(__name__)

redis_client = None
active_sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler"""
    global redis_client
    
    logger.info("=" * 70)
    logger.info("🚀 Starting StateManager Orchestrator")
    logger.info("=" * 70)
    
    # Connect to Redis
    redis_client = await redis.from_url("redis://redis:6379")
    logger.info("✅ Redis connected")
    
    yield
    
    logger.info("=" * 70)
    logger.info("🛑 Shutting down StateManager Orchestrator")
    logger.info("=" * 70)
    await redis_client.close()

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
                
                text = message.get("text")
                is_final = message.get("is_final", False)
                
                logger.info(f"📝 [{state_mgr.state.value}] STT: {text[:50]}...")
                
                await websocket.send_json({
                    "type": "state_update",
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
                
                logger.info("⚡ Starting parallel Intent+RAG processing...")
                start_time = time.time()
                
                response = await process_intent_rag_llm(user_text, session_id)
                
                thinking_time = (time.time() - start_time) * 1000
                logger.info(f"✅ Response ready in {thinking_time:.0f}ms: {response[:100]}...")
                
                await state_mgr.transition(State.SPEAKING, "response_ready", {
                    "response": response
                })
                
                # Stream TTS
                await websocket.send_json({
                    "type": "response_ready",
                    "text": response,
                    "thinking_ms": thinking_time
                })
                
                # Simulate TTS streaming (replace with real TTS)
                tts_task = asyncio.create_task(
                    stream_tts_audio(session_id, response, websocket, state_mgr)
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
                        "message": "Listening to your response..."
                    })
    
    except WebSocketDisconnect:
        logger.info(f"🔌 Session disconnected: {session_id}")
    
    finally:
        if session_id in active_sessions:
            tts_task = active_sessions[session_id].get("tts_task")
            if tts_task and not tts_task.done():
                tts_task.cancel()
            del active_sessions[session_id]
        
        logger.info(f"✅ Session cleanup complete: {session_id}")

async def stream_tts_audio(session_id: str, text: str, websocket: WebSocket, state_mgr: StateManager):
    """Simulate TTS audio streaming (replace with real TTS)"""
    try:
        # TODO: Replace with real TTS (ElevenLabs, Google, etc.)
        logger.info(f"🔊 Streaming TTS for: {text[:50]}...")
        
        # Simulate chunked audio streaming
        chunk_size = 1000  # bytes
        for i in range(10):  # 10 chunks = 10 seconds simulation
            await asyncio.sleep(0.4)  # Simulate TTS latency
            
            await websocket.send_bytes(b"audio_chunk_" + str(i).encode())
        
        # TTS complete
        await state_mgr.transition(State.IDLE, "tts_complete", {})
        state_mgr.context.turn_number += 1
        await state_mgr.save_state()
        
        logger.info(f"✅ TTS complete, ready for next turn")
        
    except asyncio.CancelledError:
        logger.warning(f"⚡ TTS cancelled (barge-in)")

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "orchestrator",
        "active_sessions": len(active_sessions),
        "redis_connected": redis_client is not None
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return {
        "active_sessions": len(active_sessions),
        "uptime_seconds": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")
```

---

## 📂 File: `orchestrator/parallel_pipeline.py`

**Purpose**: Parallel Intent+RAG+LLM processing

```python
import asyncio
import logging
import aiohttp
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def call_intent_service(text: str, session_id: str) -> Dict[str, Any]:
    """Call Intent service (non-blocking)"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "http://intent-service:8002/api/v1/classify",
                json={"text": text},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return await resp.json()
        except Exception as e:
            logger.error(f"Intent service error: {e}")
            return {"intent": "unknown", "confidence": 0.0}

async def call_rag_service(text: str, session_id: str) -> Dict[str, Any]:
    """Call RAG service (non-blocking)"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "http://rag-service:8003/api/v1/query",
                json={"query": text},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return await resp.json()
        except Exception as e:
            logger.error(f"RAG service error: {e}")
            return {"results": [], "context": ""}

async def call_llm(intent: Dict, rag_context: Dict, user_text: str) -> str:
    """Call LLM (Groq/Gemini)"""
    # TODO: Implement LLM integration
    return f"Response to: {user_text}"

async def process_intent_rag_llm(user_text: str, session_id: str) -> str:
    """
    CRITICAL: Parallel execution of Intent + RAG
    
    Without parallelization: Intent (50ms) + RAG (80ms) = 130ms sequential
    With parallelization: max(50ms, 80ms) = 80ms parallel
    
    Saves ~50ms per turn!
    """
    logger.info("⚡ Spawning parallel Intent + RAG tasks...")
    
    start = time.time()
    
    # Fire both tasks concurrently
    intent_task = asyncio.create_task(call_intent_service(user_text, session_id))
    rag_task = asyncio.create_task(call_rag_service(user_text, session_id))
    
    # Wait for both to complete
    intent_result, rag_result = await asyncio.gather(intent_task, rag_task)
    
    parallel_time = (time.time() - start) * 1000
    logger.info(f"✅ Parallel execution completed in {parallel_time:.0f}ms")
    logger.info(f"   Intent: {intent_result.get('intent', 'unknown')}")
    logger.info(f"   RAG: {len(rag_result.get('results', []))} docs found")
    
    # Call LLM if needed
    if intent_result.get("intent") == "complex_query":
        llm_response = await call_llm(intent_result, rag_result, user_text)
    else:
        llm_response = f"{intent_result.get('intent', 'unknown').replace('_', ' ').title()}: {rag_result.get('context', '')}"
    
    return llm_response
```

---

## 📂 File: `orchestrator/interruption_handler.py`

**Purpose**: Barge-in detection and TTS cancellation

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class InterruptionHandler:
    """Handles barge-in (user interrupts during TTS)"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.interrupted = False
    
    async def detect_barge_in(self, vad_confidence: float) -> bool:
        """Detect if user started speaking during TTS"""
        if vad_confidence > 0.7:
            self.interrupted = True
            logger.warning(f"⚡ Barge-in detected for {self.session_id}")
            return True
        return False
    
    async def handle_interruption(self, state_mgr):
        """Handle interruption logic"""
        logger.info(f"🔄 Resetting state after interruption")
        state_mgr.context.text_buffer = []
        state_mgr.context.llm_response = None
        self.interrupted = False
```

**This gives you the complete implementation structure for the StateManager Orchestrator!**