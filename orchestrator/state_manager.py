"""
Core Finite State Machine (FSM) engine with Redis persistence

Manages conversation state transitions: IDLE → LISTENING → THINKING → SPEAKING → INTERRUPT
Includes RAG Pre-LLM Accumulation tracking for optimized response times.
"""

import asyncio
import json
import logging
import time
from enum import Enum
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, asdict

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
    # RAG Pre-LLM Accumulation tracking
    chunk_sequence: int = 0
    rag_accumulation_active: bool = False
    last_partial_text: str = ""
    accumulation_start_time: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamps is None:
            self.timestamps = {}
        if self.text_buffer is None:
            self.text_buffer = []
    
    def reset_accumulation(self):
        """Reset RAG accumulation state for new utterance"""
        self.chunk_sequence = 0
        self.rag_accumulation_active = False
        self.last_partial_text = ""
        self.accumulation_start_time = None
    
    def start_accumulation(self):
        """Mark start of RAG accumulation"""
        self.rag_accumulation_active = True
        self.accumulation_start_time = time.time()
    
    def increment_chunk(self, text: str):
        """Increment chunk sequence and track partial text"""
        self.chunk_sequence += 1
        self.last_partial_text = text
        if not self.rag_accumulation_active:
            self.start_accumulation()


class StateManager:
    """
    Ultra-low latency FSM for real-time voice conversations.
    
    Manages state transitions, Redis persistence, and event handling.
    Latency target: <100ms per transition
    """
    
    def __init__(self, session_id: str, redis_client):
        self.session_id = session_id
        self.redis = redis_client
        self.state = State.IDLE
        self.context = ConversationContext(
            session_id=session_id, 
            state=State.IDLE.value,
            text_buffer=[]
        )
        self.transition_callbacks: Dict[str, list[Callable]] = {
            state.value: [] for state in State
        }
        
    async def initialize(self):
        """Load session state from Redis or create new"""
        redis_key = f"orchestrator:session:{self.session_id}"
        
        try:
            if self.redis:
                existing = await self.redis.hgetall(redis_key)
                if existing:
                    logger.info(f"[{self.session_id}] ✅ Loaded session from Redis")
                    # Deserialize and restore state
                    self.context.state = existing.get("state", "idle")
                    self.context.turn_number = int(existing.get("turn_number", 0))
                    if existing.get("text_buffer"):
                        self.context.text_buffer = json.loads(existing.get("text_buffer", "[]"))
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
        redis_key = f"orchestrator:session:{self.session_id}"
        
        state_dict = {
            "state": self.context.state,
            "turn_number": str(self.context.turn_number),
            "text_buffer": json.dumps(self.context.text_buffer),
            "intent": json.dumps(self.context.intent or {}),
            "rag_results": json.dumps(self.context.rag_results or {}),
            "llm_response": self.context.llm_response or "",
            "last_update": str(time.time())
        }
        
        try:
            if self.redis:
                await self.redis.hset(redis_key, mapping=state_dict)
                await self.redis.expire(redis_key, 3600)  # 1 hour TTL
        except Exception as e:
            logger.warning(f"[{self.session_id}] ⚠️ Redis save failed: {e}")
    
    async def load_state(self):
        """Load state from Redis"""
        redis_key = f"orchestrator:session:{self.session_id}"
        
        try:
            if self.redis:
                data = await self.redis.hgetall(redis_key)
                if data:
                    self.context.state = data.get("state", "idle")
                    self.context.turn_number = int(data.get("turn_number", 0))
                    if data.get("text_buffer"):
                        self.context.text_buffer = json.loads(data.get("text_buffer", "[]"))
                    if data.get("intent"):
                        self.context.intent = json.loads(data.get("intent", "{}"))
                    if data.get("rag_results"):
                        self.context.rag_results = json.loads(data.get("rag_results", "{}"))
                    self.context.llm_response = data.get("llm_response", "")
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
        
        prev_ts = 0
        for transition, label in state_sequence:
            if transition in timestamps:
                latencies[label] = (timestamps[transition] - prev_ts) * 1000  # ms
                prev_ts = timestamps[transition]
        
        return latencies







