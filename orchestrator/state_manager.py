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
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, asdict

from redis.asyncio import Redis
from leibniz_agent.services.shared.events import VoiceEvent, EventTypes
from leibniz_agent.services.shared.event_broker import EventBroker
from .structured_logger import StructuredLogger

logger = logging.getLogger(__name__)


class State(Enum):
    """Conversation states"""
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPT = "interrupt"


class StateContract:
    """Define what MUST happen when entering each state."""
    
    # Map state value (str) to contract details
    CONTRACTS = {
        State.IDLE.value: {
            "microphone": "CLOSED",
            "audio_playback": "STOPPED",
            "side_effects": [],
        },
        State.LISTENING.value: {
            "microphone": "OPEN",
            "audio_playback": "STOPPED",
            "side_effects": [],
        },
        State.THINKING.value: {
            "microphone": "GATED",
            "audio_playback": "STOPPED",
            "side_effects": ["play_immediate_filler"],
        },
        State.SPEAKING.value: {
            "microphone": "GATED",
            "audio_playback": "STREAMING",
            "side_effects": ["cancel_filler"],
        },
        State.INTERRUPT.value: {
            "microphone": "GATED",
            "audio_playback": "STOPPED",
            "side_effects": ["cancel_tts", "cancel_filler"],
        }
    }
    
    @classmethod
    def get(cls, state: State) -> dict:
        return cls.CONTRACTS.get(state.value, {})


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
    # Activity tracking for timeout detection
    last_activity_time: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamps is None:
            self.timestamps = {}
        if self.text_buffer is None:
            self.text_buffer = []
        if self.last_activity_time is None:
            self.last_activity_time = time.time()
    
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
    
    def __init__(self, session_id: str, redis_client: Redis, broker: Optional[EventBroker] = None):
        self.session_id = session_id
        self.redis = redis_client
        self.broker = broker
        self.state = State.IDLE
        self.structured_logger = StructuredLogger(logger)
        self.context = ConversationContext(
            session_id=session_id, 
            state=State.IDLE.value,
            text_buffer=[]
        )
        self.transition_callbacks: Dict[str, list[Callable]] = {
            state.value: [] for state in State
        }
        
        # Valid state transitions
        self.valid_transitions = {
            State.IDLE: [State.LISTENING, State.SPEAKING],  # SPEAKING for intro/timeout prompts
            State.LISTENING: [State.THINKING, State.INTERRUPT, State.IDLE, State.SPEAKING],  # SPEAKING for fillers/timeout
            State.THINKING: [State.SPEAKING, State.IDLE, State.LISTENING],  # LISTENING for quick returns
            State.SPEAKING: [State.LISTENING, State.INTERRUPT, State.IDLE],
            State.INTERRUPT: [State.LISTENING, State.IDLE],
        }
        
        # Side-effect handlers per state
        self.handlers = {
            State.IDLE: self.handle_idle_state,
            State.LISTENING: self.handle_listening_state,
            State.THINKING: self.handle_thinking_state,
            State.SPEAKING: self.handle_speaking_state,
            State.INTERRUPT: self.handle_interrupt_state,
        }
        # Simple in-memory latency tracking per transition key
        self.latencies: Dict[str, List[float]] = {}
        self._last_transition_ts: float = time.time()
        
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
                    if existing.get("last_activity_time"):
                        self.context.last_activity_time = float(existing.get("last_activity_time", time.time()))
                    try:
                        self.state = State(self.context.state)
                    except ValueError:
                        logger.warning(f"Invalid state in Redis: {self.context.state}, resetting to IDLE")
                        self.state = State.IDLE
                        self.context.state = State.IDLE.value
                else:
                    logger.info(f"[{self.session_id}] 🆕 Created new session")
                    await self.save_state()
        except Exception as e:
            logger.warning(f"[{self.session_id}] ⚠️ Redis load failed: {e}")
    
    async def transition(self, new_state: State, trigger: str, data: Optional[Dict] = None):
        """
        Atomic state transition with logging, Redis persistence, and event emission.
        
        Args:
            new_state: Target state
            trigger: Transition trigger (e.g., "stt_fragment", "vad_end")
            data: Optional context data
        """
        old_state = self.state
        
        # Validate transition
        if new_state not in self.valid_transitions.get(old_state, []):
            # Allow resetting to IDLE from anywhere on error
            if new_state == State.IDLE and trigger == "error":
                pass
            else:
                logger.error(
                    f"[{self.session_id}] ❌ INVALID TRANSITION: "
                    f"{old_state.value.upper()} → {new_state.value.upper()} "
                    f"(trigger: {trigger})"
                )
                logger.error(
                    f"[{self.session_id}] Valid next states from {old_state.value}: "
                    f"{[s.value for s in self.valid_transitions.get(old_state, [])]}"
                )
                return

        # Preconditions for specific states
        if new_state == State.SPEAKING:
            # Accept either "response" or "text" key for SPEAKING transitions
            # This allows flexibility for different call sites (intro/timeout use "response", 
            # while some legacy code might use "text")
            if not data or (not data.get("response") and not data.get("text")):
                logger.error(
                    f"[{self.session_id}] Cannot transition to SPEAKING without response text "
                    f"(data keys: {list(data.keys()) if data else 'None'})"
                )
                return
        
        timestamp = time.time()
        transition_key = f"{old_state.value}→{new_state.value}"

        # Latency since last transition (best-effort)
        delta_ms = (timestamp - self._last_transition_ts) * 1000.0
        self._last_transition_ts = timestamp

        # Track latency history per transition
        self.latencies.setdefault(transition_key, []).append(delta_ms)

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
        
        # Update activity time for user interaction triggers
        if trigger in ("stt_fragment", "vad_end", "stt_received"):
            self.context.last_activity_time = timestamp
        
        # Persist to Redis (O(1) operation)
        await self.save_state()
        
        # Emit state change event (non-blocking, audit logging only)
        # Phase 2: Event emission is fire-and-forget, does not block transitions
        if self.broker:
            event = VoiceEvent(
                event_type=EventTypes.ORCHESTRATOR_STATE,
                session_id=self.session_id,
                source="orchestrator",
                payload={
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "trigger": trigger,
                    "data": data or {}
                }
            )
            asyncio.create_task(
                self._emit_event_async(event),
                name=f"audit_log_{self.session_id}_{new_state.value}"
            )
        
        # Emit logs with emojis for Docker console visibility
        state_emoji = {
            State.IDLE: "🟢",
            State.LISTENING: "🔵",
            State.THINKING: "🟡",
            State.SPEAKING: "🔴",
            State.INTERRUPT: "⚡"
        }
        
        logger.info(
            f"[{self.session_id}] {state_emoji.get(old_state, '')} {old_state.value.upper()} "
            f"→ {state_emoji.get(new_state, '')} {new_state.value.upper()} "
            f"({trigger})"
        )
        # Structured state transition + latency record
        self.structured_logger.state_transition(
            self.session_id,
            old_state.value,
            new_state.value,
            trigger,
            data=data or {},
        )
        self.structured_logger.latency_recorded(
            self.session_id,
            operation=f"{old_state.value}_to_{new_state.value}",
            duration_ms=delta_ms,
        )
        
        # Execute contract
        await self.execute_contract(new_state, data)
        
        # Execute legacy side-effect handlers (if any remain)
        handler = self.handlers.get(new_state)
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(trigger, data)
                else:
                    handler(trigger, data)
            except Exception as e:
                logger.error(f"[{self.session_id}] ❌ State handler error: {e}", exc_info=True)
        
        # Fire registered callbacks
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
            "last_update": str(time.time()),
            "last_activity_time": str(self.context.last_activity_time or time.time())
        }
        
        try:
            if self.redis:
                await self.redis.hset(redis_key, mapping=state_dict)
                await self.redis.expire(redis_key, 3600)  # 1 hour TTL
        except Exception as e:
            logger.warning(f"[{self.session_id}] ⚠️ Redis save failed: {e}")
    
    async def _emit_event_async(self, event: VoiceEvent):
        """
        Non-blocking event emission for audit logging.
        
        Errors are logged but do not affect the main flow.
        This is fire-and-forget - used for state change audit trail only.
        """
        try:
            await self.broker.publish(f"voice:orchestrator:session:{self.session_id}", event)
        except Exception as e:
            # Log but don't fail - audit logging is optional
            logger.debug(f"[{self.session_id}] Audit event emission failed (non-critical): {e}")
    
    async def load_state(self):
        """Load state from Redis"""
        # Alias for initialize to keep compatibility
        await self.initialize()
    
    def on_transition(self, state: State):
        """Register callback for state transitions"""
        def decorator(func):
            self.transition_callbacks[state.value].append(func)
            return func
        return decorator
    
    async def get_latency_breakdown(self) -> Dict[str, float]:
        """
        Calculate simple latency statistics between state transitions.

        Returns a mapping of transition key -> average latency in milliseconds.
        """
        breakdown: Dict[str, float] = {}
        for key, samples in self.latencies.items():
            if samples:
                breakdown[key] = sum(samples) / len(samples)
        return breakdown

    async def execute_contract(self, state: State, data: Optional[Dict]):
        """Execute state contract side effects."""
        contract = StateContract.get(state)
        
        # 1. Microphone control
        mic_state = contract.get("microphone")
        if mic_state == "OPEN":
            await self._control_mic(gate_off=True)
            logger.info(f"[{self.session_id}] 🎤 Microphone OPEN")
        elif mic_state in ("GATED", "CLOSED"):
            await self._control_mic(gate_off=False)
            logger.info(f"[{self.session_id}] 🎤 Microphone GATED")
            
        # 2. Side effects (logging for now, as app.py handles most)
        for effect in contract.get("side_effects", []):
            logger.debug(f"[{self.session_id}] Contract requires side effect: {effect}")

    # =========================================================================
    # State Handlers (Legacy / Specific Logic)
    # =========================================================================
    
    async def handle_idle_state(self, trigger: str, data: dict):
        """Side effects for IDLE state"""
        # Contract handles mic
        # Maybe clean up buffers?
        self.context.text_buffer = []

    async def handle_listening_state(self, trigger: str, data: dict):
        """Side effects for LISTENING state"""
        # Contract handles mic
        pass

    async def handle_thinking_state(self, trigger: str, data: dict):
        """Side effects for THINKING state"""
        # Contract handles mic
        
        if data and data.get("text"):
            text = data["text"]
            logger.info(f"[{self.session_id}] Processing: '{text[:50]}...'")

    async def handle_speaking_state(self, trigger: str, data: dict):
        """Side effects for SPEAKING state"""
        # Contract handles mic
        pass

    async def handle_interrupt_state(self, trigger: str, data: dict):
        """Side effects for INTERRUPT state (barge-in)"""
        logger.info(f"[{self.session_id}] User interrupted agent speech")
        # Contract handles mic
        # TTS cancellation handled by app.py task management


    async def _control_mic(self, gate_off: bool):
        """Emit mic control event to Unified FastRTC via Redis"""
        if self.broker:
            # Action: gate_off=True -> mic open (gate disabled)
            # Action: gate_off=False -> mic closed/gated (gate enabled)
            action = "gate_off" if gate_off else "gate_on"
            
            # This relies on unified_fastrtc listening to this stream
            # or orchestrator forwarding it.
            # Ideally orchestrator consumes this state event and calls broadcast_orchestrator_state
            # But we can also emit a direct control event if needed.
            
            # For now, the orchestrator app.py (FSM consumer) will see the state change
            # and call broadcast_orchestrator_state().
            pass
