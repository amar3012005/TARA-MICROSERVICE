# 🚨 ORCHESTRATOR DEBUG PRIORITY GUIDE
## Enterprise-Ready Debugging Strategy for Event-Driven Architecture

**Status:** Your orchestrator has integrated event-driven patterns, but logs are chaotic due to:
1. Multiple overlapping execution paths (legacy + new event-driven)
2. Race conditions between state transitions and event handlers
3. Session ID routing mismatches (unified_XXXX vs auto_session_XXX)
4. Uncontrolled async task spawning
5. No central orchestration control flow

**Fix Duration:** ~6-8 hours, phased approach

---

## PHASE 1: CRITICAL STABILIZATION (1-2 hours)
### Goal: Stop the bleeding, get clean logs

### Priority 1.1: KILL ASYNC TASK CHAOS ⚠️ 
**Problem:** Every state transition spawns new async tasks (fillers, TTSs, etc.) without tracking
**Evidence:** Logs show overlapping "Playing filler", "Playing timeout", "Processing STT" 
**Fix Location:** `app.py` lines ~500-800

```python
# BEFORE (chaotic):
asyncio.create_task(stream_tts_audio(...))  # Fire and forget
asyncio.create_task(play_timeout_prompt(...))  # Another fire and forget

# AFTER (controlled):
# Store in session_data for lifecycle management
if session_id in active_sessions:
    old_task = active_sessions[session_id].get("current_task")
    if old_task and not old_task.done():
        old_task.cancel()  # Cancel previous
        try:
            await old_task
        except asyncio.CancelledError:
            pass
    
    new_task = asyncio.create_task(...)
    active_sessions[session_id]["current_task"] = new_task  # Track it
```

**Checklist:**
- [ ] Add `current_task` field to `active_sessions[session_id]` dict
- [ ] Always cancel old task before spawning new
- [ ] Log task cancellations with reason
- [ ] Test: Only ONE async task active per session at a time

**Expected Outcome:** Logs go from chaotic to linear (one action per line)

---

### Priority 1.2: FIX SESSION ID ROUTING ⚠️
**Problem:** STT events from `fastrtc_XXXX` routed to `auto_session_YYY`, causing state confusion
**Evidence:** 
```
STT Session: fastrtc_1765407692
Text: లవ్ నౌ దాన్....
Routing STT to session: auto_session_1765407579  ❌ WRONG SESSION!
```

**Root Cause:** 
- Unified FastRTC creates one handler per WebRTC connection (`fastrtc_XXXX`)
- Orchestrator creates another session ID (`auto_session_XXX`) for WebSocket
- Redis events map `fastrtc_XXXX` → `auto_session_XXX` inconsistently

**Fix Location:** `app.py` lines ~200-300 (`handle_stt_event` function)

```python
# BEFORE (messy):
# In listen_to_redis_events()
for message in subscriber.listen():
    if message["type"] == "message":
        channel = message["channel"].decode()
        # Try to guess which session this is for
        if "fastrtc" in channel:
            stt_session = extract_session_id(channel)
            # Now find matching orchestrator session
            for orch_sid, sess_data in active_sessions.items():
                if somehow_matches(stt_session):  # Fragile!
                    ...

# AFTER (clear):
async def handle_stt_event_from_redis(event):
    """
    Map STT event to correct orchestrator session.
    
    Rules:
    1. If event.session_id is in active_sessions, use it directly
    2. Otherwise, check if UnifiedFastRTC handler knows about it
    3. Last resort: reject with clear log
    """
    stt_session_id = event.session_id
    orch_session_id = None
    
    # Rule 1: Direct match
    if stt_session_id in active_sessions:
        orch_session_id = stt_session_id
    
    # Rule 2: Check UnifiedFastRTC registry
    elif stt_session_id in UnifiedFastRTCHandler.active_instances:
        handler = UnifiedFastRTCHandler.active_instances[stt_session_id]
        # Find matching orchestrator session by correlating handler
        for sid, sdata in active_sessions.items():
            if sdata.get("unified_handler") == handler:
                orch_session_id = sid
                break
    
    # Rule 3: Reject
    if not orch_session_id:
        logger.error(f"Cannot map STT session {stt_session_id} to any orchestrator session")
        logger.error(f"Active orchestrator sessions: {list(active_sessions.keys())}")
        logger.error(f"Active UnifiedFastRTC handlers: {list(UnifiedFastRTCHandler.active_instances.keys())}")
        return  # Drop event
    
    # Now process safely
    session_data = active_sessions[orch_session_id]
    state_mgr = session_data["state_manager"]
    # ... handle event with correct session_manager
```

**Checklist:**
- [ ] Store `unified_handler` reference in `active_sessions[session_id]`
- [ ] Implement 3-rule routing in `handle_stt_event_from_redis()`
- [ ] Add debug logs showing mapping: `STT {fastrtc_XXX} → Orchestrator {orch_YYY}`
- [ ] Test: For each STT event, verify correct session_manager processes it

**Expected Outcome:** All STT events route to the correct session manager

---

### Priority 1.3: ELIMINATE STATE TRANSITION RACE CONDITIONS ⚠️
**Problem:** State can transition while a filler is still playing, causing invalid states
**Evidence:**
```
⚠️ Invalid transition: listening → speaking (state warnings appear)
🔴 SPEAKING → 🟢 IDLE (audio_complete)  # Too fast! 
🔵 IDLE → 🔵 LISTENING (timeout_complete)  # No wait!
```

**Root Cause:** State machine doesn't wait for audio playback to actually complete

**Fix Location:** `state_manager.py` lines ~150-250 (`transition` method)

```python
# BEFORE (raw transitions):
async def transition(self, new_state: State, trigger: str, data=None):
    old_state = self.state
    if new_state not in self.valid_transitions[old_state]:
        logger.warning(f"Invalid transition: {old_state.value} → {new_state.value}")
        return  # Silently ignore
    self.state = new_state  # Boom! Immediately in new state
    # ... rest of code

# AFTER (validate AND guard):
async def transition(self, new_state: State, trigger: str, data=None):
    old_state = self.state
    
    # Validate transition
    if new_state not in self.valid_transitions.get(old_state, []):
        # CRITICAL: Log why transition is invalid
        logger.error(
            f"[{self.session_id}] ❌ INVALID TRANSITION: "
            f"{old_state.value.upper()} → {new_state.value.upper()} "
            f"(trigger: {trigger})"
        )
        logger.error(
            f"[{self.session_id}] Valid next states from {old_state.value}: "
            f"{[s.value for s in self.valid_transitions.get(old_state, [])]}"
        )
        logger.error(f"[{self.session_id}] Rejecting transition with timestamp")
        return  # REJECT, don't proceed
    
    # Guard: Check preconditions before transition
    if new_state == State.SPEAKING:
        # Can only speak if we have text
        if not data or not data.get("response"):
            logger.error(f"[{self.session_id}] Cannot SPEAK without response text")
            return
    
    if new_state == State.LISTENING:
        # Must not be actively playing audio
        # (This is handled by TTS service, not here)
        pass
    
    # Perform transition with full logging
    timestamp = time.time()
    self.state = new_state
    self.context.state = new_state.value
    self.context.timestamps[f"{old_state.value}→{new_state.value}"] = timestamp
    
    # Emoji logging for Docker visibility
    state_emoji = {
        State.IDLE: "🟢",
        State.LISTENING: "🔵",
        State.THINKING: "🟡",
        State.SPEAKING: "🔴",
        State.INTERRUPT: "🟣",
    }
    
    logger.info(
        f"[{self.session_id}] "
        f"{state_emoji.get(old_state, '⚪')} {old_state.value.upper()} → "
        f"{state_emoji.get(new_state, '⚪')} {new_state.value.upper()} "
        f"({trigger})"
    )
    
    # Persist to Redis
    await self.save_state()
    
    # Execute side effects (handle state entry)
    handler = self.handlers.get(new_state)
    if handler:
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(trigger, data)
            else:
                handler(trigger, data)
        except Exception as e:
            logger.error(f"[{self.session_id}] State handler error: {e}", exc_info=True)
```

**Checklist:**
- [ ] Add validation check that rejects invalid transitions (don't silently ignore)
- [ ] Add precondition guards (e.g., SPEAKING needs response text)
- [ ] Add emoji logging for console visibility
- [ ] Add timestamp tracking for latency analysis
- [ ] Test: Try invalid transitions, verify they're rejected with clear logs

**Expected Outcome:** Only valid state transitions occur; logs clearly show rejected attempts

---

## PHASE 2: CONTROL FLOW UNIFICATION (2-3 hours)
### Goal: One clear execution path, no branching logic

### Priority 2.1: UNIFY STT→RAG→TTS PIPELINE ⚠️
**Problem:** Multiple code paths handle STT events (legacy + event-driven)
- Path 1: `app.py` WebSocket receives STT directly
- Path 2: `app.py` `listen_to_redis_events()` handles Redis STT events
- Path 3: `orchestrator_fsm.py` `OrchestratorFSM` consumes Redis Streams
- Result: Same event processed multiple times in different ways

**Fix:** ONE unified handler

**Fix Location:** Create new file `orchestrator/stt_event_handler.py`

```python
# orchestrator/stt_event_handler.py

class STTEventHandler:
    """
    Single point of truth for STT event processing.
    
    Receives STT events from:
    1. WebSocket (/orchestrate endpoint)
    2. Redis pub/sub (legacy)
    3. Redis Streams (event-driven)
    
    Routes to:
    1. Parallel Intent+RAG
    2. TTS streaming
    3. State transitions
    """
    
    def __init__(self, session_id: str, state_mgr: StateManager, config: OrchestratorConfig):
        self.session_id = session_id
        self.state_mgr = state_mgr
        self.config = config
    
    async def handle_stt_final(
        self,
        text: str,
        is_final: bool,
        confidence: float = 0.95,
        source: str = "unknown"  # "websocket", "redis_pubsub", "redis_stream"
    ) -> None:
        """
        Process final STT result.
        
        Single unified entry point for all STT sources.
        """
        # 1. Validate
        if not text or not text.strip():
            logger.warning(f"[{self.session_id}] Empty STT text, ignoring")
            return
        
        if not is_final:
            logger.debug(f"[{self.session_id}] Partial STT (ignoring): {text[:50]}...")
            return
        
        # 2. Log source for debugging
        logger.info(
            f"[{self.session_id}] STT FINAL (source={source}) | "
            f"text={text[:50]}... | confidence={confidence:.2f}"
        )
        
        # 3. Validate state (must be LISTENING)
        if self.state_mgr.state != State.LISTENING:
            logger.warning(
                f"[{self.session_id}] Cannot process STT in {self.state_mgr.state.value} state"
            )
            return
        
        # 4. Transition to THINKING
        await self.state_mgr.transition(State.THINKING, "stt_received", {"text": text})
        
        # 5. Gate microphone input during processing
        # (handled by state machine side effects)
        
        # 6. Parallel Intent+RAG (if configured)
        if self.config.rag_service_url:
            logger.info(f"[{self.session_id}] Starting parallel Intent+RAG processing...")
            start_time = time.time()
            
            try:
                result = await process_intent_rag_llm(
                    text=text,
                    session_id=self.session_id,
                    intent_url=self.config.intent_service_url if not self.config.skip_intent_service else None,
                    rag_url=self.config.rag_service_url,
                )
                
                processing_time_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"[{self.session_id}] Processing complete in {processing_time_ms:.0f}ms | "
                    f"response={result.get('response', '')[:50]}..."
                )
                
                # 7. Transition to SPEAKING
                await self.state_mgr.transition(
                    State.SPEAKING,
                    "response_ready",
                    {
                        "response": result.get("response", ""),
                        "intent": result.get("intent"),
                        "rag_results": result.get("rag"),
                    }
                )
                
                # 8. Return result to caller (for TTS)
                return result
                
            except Exception as e:
                logger.error(f"[{self.session_id}] Processing error: {e}", exc_info=True)
                await self.state_mgr.transition(State.LISTENING, "processing_error")
                return None
        else:
            logger.warning(f"[{self.session_id}] RAG service URL not configured")
            return None


# Usage in app.py:

# In /orchestrate WebSocket handler:
stt_handler = STTEventHandler(session_id, state_mgr, config)

if msg_type == "stt_fragment":
    text = message.get("text")
    is_final = message.get("is_final", False)
    result = await stt_handler.handle_stt_final(
        text=text,
        is_final=is_final,
        source="websocket"
    )
    if result:
        # Stream TTS
        await stream_tts_audio(session_id, result["response"], ...)

# In listen_to_redis_events():
async def redis_stt_handler(event_data):
    # Extract STT session and text
    stt_handler = STTEventHandler(session_id, state_mgr, config)
    result = await stt_handler.handle_stt_final(
        text=event_data["text"],
        is_final=event_data["is_final"],
        source="redis_pubsub"
    )
```

**Checklist:**
- [ ] Create `orchestrator/stt_event_handler.py` with unified handler
- [ ] Remove duplicate STT processing logic from `app.py` and `orchestrator_fsm.py`
- [ ] Update both WebSocket and Redis handlers to use `STTEventHandler`
- [ ] Add source logging to trace which input path each event took
- [ ] Test: Verify only one STT handler processes each event

**Expected Outcome:** Single code path handles all STT inputs; logs clearly show source

---

### Priority 2.2: SIMPLIFY STATE SIDE EFFECTS ⚠️
**Problem:** State transitions trigger multiple background tasks (fillers, timeouts, etc.) unpredictably

**Current chaos:**
```
IDLE → LISTENING: Start listening
LISTENING → THINKING: Immediately play filler? Cancel previous filler?
THINKING → SPEAKING: Start TTS
SPEAKING → LISTENING: Cancel TTS? Wait how long?
```

**Fix Location:** `state_manager.py` side effect handlers

```python
# BEFORE (implicit, hard to trace):
async def handle_thinking_state(self, trigger: str, data: dict):
    # Side effects buried in handler
    if trigger == "stt_received":
        # Maybe play a filler?
        if dialogue_manager:
            asyncio.create_task(...)  # Fire and forget

# AFTER (explicit state contract):
class StateContract:
    """Define what MUST happen when entering each state."""
    
    IDLE = {
        "microphone": "CLOSED",
        "audio_playback": "STOPPED",
        "side_effects": [],
    }
    
    LISTENING = {
        "microphone": "OPEN",
        "audio_playback": "STOPPED",
        "side_effects": ["cancel_previous_filler"],
    }
    
    THINKING = {
        "microphone": "GATED",
        "audio_playback": "STOPPED",
        "side_effects": ["play_immediate_filler"],
    }
    
    SPEAKING = {
        "microphone": "GATED",
        "audio_playback": "STREAMING",
        "side_effects": ["cancel_filler"],
    }
    
    INTERRUPT = {
        "microphone": "GATED",
        "audio_playback": "STOPPED",
        "side_effects": ["cancel_tts", "cancel_filler"],
    }


# Use contract to drive state handlers:
async def transition(self, new_state: State, trigger: str, data=None):
    # ... validation ...
    
    # Enter new state
    self.state = new_state
    await self.save_state()
    
    # Execute contract side effects
    contract = StateContract[new_state.value]
    
    # 1. Microphone control
    if contract["microphone"] == "OPEN":
        await self.open_microphone()
    elif contract["microphone"] == "GATED":
        await self.gate_microphone()
    elif contract["microphone"] == "CLOSED":
        await self.close_microphone()
    
    # 2. Audio playback control
    if contract["audio_playback"] == "STOPPED":
        await self.stop_audio_playback()
    elif contract["audio_playback"] == "STREAMING":
        pass  # Audio will be streamed by caller
    
    # 3. Side effects (in order)
    for side_effect in contract["side_effects"]:
        try:
            await self.execute_side_effect(side_effect, data)
        except Exception as e:
            logger.error(f"[{self.session_id}] Side effect {side_effect} failed: {e}")
    
    logger.info(f"[{self.session_id}] Entered {new_state.value} state")
```

**Checklist:**
- [ ] Create `StateContract` dict/dataclass
- [ ] Move all side effect logic into `execute_side_effect()` method
- [ ] Make side effects explicit and testable
- [ ] Remove implicit `asyncio.create_task()` calls
- [ ] Test: Verify side effects execute in correct order

**Expected Outcome:** Side effects are explicit, ordered, and traceable in logs

---

## PHASE 3: EVENT-DRIVEN CLEANUP (2 hours)
### Goal: Clean up event emission and consumption

### Priority 3.1: FIX EVENT BROKER INTEGRATION ⚠️
**Problem:** Multiple event systems (Redis pub/sub + Redis Streams) running simultaneously, both consuming same events

**Current state:**
- `listen_to_redis_events()` (pub/sub) ← running and consuming
- `OrchestratorFSM.run()` (Streams) ← running and consuming
- Both process same STT events → events processed twice!

**Fix Location:** `app.py` lifespan + `orchestrator_fsm.py`

```python
# Strategy: Use ONLY event-driven (Streams), remove legacy (pub/sub)

# In app.py lifespan:

# OLD:
# redis_listener_task = asyncio.create_task(listen_to_redis_events())

# NEW: Replace with event-driven FSM per session
# (No background global listener needed)

# In /orchestrate WebSocket handler:

# Create per-session event-driven FSM
fsm = OrchestratorFSM(
    session_id=session_id,
    redis_client=redis_client,
    broker=event_broker,
)

# Start FSM in background
fsm_task = asyncio.create_task(fsm.run())
active_sessions[session_id]["fsm_task"] = fsm_task

# When WebSocket receives STT event, emit to broker instead of processing directly:
if msg_type == "stt_fragment":
    text = message.get("text")
    is_final = message.get("is_final", False)
    
    # Emit to Redis Streams
    if event_broker:
        stt_event = VoiceEvent(
            event_type=EventTypes.STT_FINAL if is_final else EventTypes.STT_PARTIAL,
            session_id=session_id,
            source="orchestrator_websocket",
            payload={"text": text, "is_final": is_final}
        )
        await event_broker.publish(f"voice:stt:session:{session_id}", stt_event)
    
    # FSM will consume this event automatically from Streams
```

**Checklist:**
- [ ] Remove `listen_to_redis_events()` function (legacy pub/sub)
- [ ] Remove `redis_listener_task` from lifespan
- [ ] Verify `OrchestratorFSM` is created per session
- [ ] Test: Only FSM consumes events, logs show consumption

**Expected Outcome:** Only one event consumer (FSM) per session; no duplicate processing

---

### Priority 3.2: VALIDATE EVENT TYPES & PAYLOADS ⚠️
**Problem:** Events have inconsistent types and missing fields
**Evidence:** Logs show errors parsing event payloads

**Fix Location:** `shared/events.py` + validation in handlers

```python
# In shared/events.py:

class VoiceEvent:
    """Strongly-typed event model."""
    
    event_type: str  # Must be from EventTypes enum
    session_id: str  # Required
    source: str  # Where did this come from?
    timestamp: float  # When created?
    correlation_id: str  # For tracing?
    payload: Dict[str, Any]  # Event-specific data
    metadata: Dict[str, Any] = {}  # Execution context
    
    @validator("event_type")
    def validate_event_type(cls, v):
        valid_types = {e.value for e in EventTypes}
        if v not in valid_types:
            raise ValueError(f"Invalid event type: {v}. Must be one of {valid_types}")
        return v
    
    @validator("session_id")
    def validate_session_id(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("session_id must be non-empty string")
        return v
    
    # Payload validation per event type
    def validate_payload(self) -> None:
        """Validate payload matches event_type schema."""
        required_fields = {
            EventTypes.STT_FINAL: ["text", "is_final"],
            EventTypes.RAG_ANSWER_READY: ["text"],
            EventTypes.TTS_CHUNK_READY: ["audio_base64"],
            EventTypes.PLAYBACK_DONE: ["duration_ms"],
            EventTypes.BARGE_IN: ["reason"],
        }
        
        if self.event_type in required_fields:
            missing = set(required_fields[self.event_type]) - set(self.payload.keys())
            if missing:
                raise ValueError(
                    f"Event {self.event_type} missing fields: {missing}. "
                    f"Payload: {self.payload}"
                )


# In handlers, validate before processing:

async def handle_stt_final(self, event: VoiceEvent):
    # Validate
    try:
        event.validate_payload()
    except ValueError as e:
        logger.error(f"[{event.session_id}] Invalid STT event: {e}")
        return  # Drop invalid event
    
    # Safe to access fields
    text = event.payload["text"]
    is_final = event.payload["is_final"]
    # ... rest of handler
```

**Checklist:**
- [ ] Add Pydantic validators to `VoiceEvent`
- [ ] Add `validate_payload()` method with schema checks
- [ ] Call validation in every event handler
- [ ] Log validation errors clearly
- [ ] Test: Try emitting invalid events, verify rejection with clear error

**Expected Outcome:** Only valid events are processed; invalid events logged and dropped

---

## PHASE 4: MONITORING & OBSERVABILITY (1 hour)
### Goal: Clear visibility into what's happening

### Priority 4.1: STRUCTURED LOGGING ⚠️

Create a unified logging format:

```python
# orchestrator/structured_logger.py

import logging
import json
import time
from typing import Dict, Any

class StructuredLogger:
    """Emit structured JSON logs for ELK/CloudWatch parsing."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def event(
        self,
        session_id: str,
        event_type: str,
        message: str,
        level: str = "INFO",
        data: Dict[str, Any] = None,
    ):
        """Log structured event."""
        log_entry = {
            "timestamp": time.time(),
            "session_id": session_id,
            "event_type": event_type,
            "message": message,
            "level": level,
            "data": data or {},
        }
        
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(json.dumps(log_entry))
    
    # Convenience methods
    def state_transition(self, session_id, old_state, new_state, trigger):
        self.event(
            session_id,
            "state_transition",
            f"{old_state} → {new_state}",
            data={"old_state": old_state, "new_state": new_state, "trigger": trigger}
        )
    
    def event_received(self, session_id, event_type, payload):
        self.event(
            session_id,
            "event_received",
            f"Received {event_type}",
            data={"event_type": event_type, "payload": payload}
        )
    
    def latency_recorded(self, session_id, operation, duration_ms):
        self.event(
            session_id,
            "latency",
            f"{operation} took {duration_ms:.0f}ms",
            data={"operation": operation, "duration_ms": duration_ms}
        )


# Usage:
logger = StructuredLogger(logging.getLogger(__name__))

# In state transition:
logger.state_transition(session_id, "listening", "thinking", "stt_received")

# In event handler:
logger.event_received(session_id, "voice.stt.final", {"text": "hello"})

# Latency tracking:
start = time.time()
result = await process_intent_rag_llm(...)
duration_ms = (time.time() - start) * 1000
logger.latency_recorded(session_id, "intent_rag_llm", duration_ms)
```

**Checklist:**
- [ ] Create `structured_logger.py`
- [ ] Replace all `logger.info()` calls with structured logging
- [ ] Add latency tracking for all major operations
- [ ] Configure JSON output to stdout for container logging
- [ ] Test: Verify logs are valid JSON, parseable by ELK

**Expected Outcome:** Logs are machine-readable and analyzable

---

### Priority 4.2: LATENCY DASHBOARD ⚠️

Add latency tracking to StateManager:

```python
# In state_manager.py:

async def transition(self, new_state: State, trigger: str, data=None):
    # ... validation ...
    
    # Track latencies
    timestamp = time.time()
    old_state = self.state
    
    # Update state
    self.state = new_state
    
    # Record latency
    transition_key = f"{old_state.value}→{new_state.value}"
    if not hasattr(self, "latencies"):
        self.latencies = {}
    
    if transition_key not in self.latencies:
        self.latencies[transition_key] = []
    
    self.latencies[transition_key].append(timestamp)
    
    # Log latency
    logger.info(
        f"[{self.session_id}] Transition latency: {transition_key} "
        f"(avg: {sum(self.latencies[transition_key]) / len(self.latencies[transition_key]) * 1000:.0f}ms)"
    )


# Endpoint to check latencies:
@app.get("/latency/{session_id}")
async def get_latency(session_id: str):
    if session_id not in active_sessions:
        return {"error": "Session not found"}
    
    state_mgr = active_sessions[session_id]["state_manager"]
    return {
        "session_id": session_id,
        "latencies": getattr(state_mgr, "latencies", {}),
        "state": state_mgr.state.value,
    }

# Example response:
# {
#   "session_id": "auto_session_123",
#   "latencies": {
#     "listening→thinking": [0.100, 0.095, 0.102],
#     "thinking→speaking": [0.050, 0.055],
#     "speaking→listening": [0.200, 0.210],
#   },
#   "state": "listening"
# }
```

**Checklist:**
- [ ] Add latency tracking to state transitions
- [ ] Create `/latency/{session_id}` endpoint
- [ ] Add `/metrics` endpoint for all active sessions
- [ ] Test: Call endpoints, verify latency data is collected

**Expected Outcome:** Real-time latency visibility for each session

---

## VALIDATION CHECKLIST: End-to-End

After completing all 4 phases, validate:

```bash
# 1. Start orchestrator
docker-compose up orchestrator

# 2. Check logs for clean startup (no errors)
# Should see:
# ✅ Redis connected
# ✅ Services healthy
# ✅ Event broker initialized
# No duplicate task spawning messages

# 3. Connect client
# Open http://localhost:5204/fastrtc in browser
# Click "Record"

# 4. Send test message
# Say "Hello world"

# 5. Check logs for linear flow:
# [session_id] 🔵 IDLE → 🔵 LISTENING (client_connected)
# [session_id] STT FINAL | text=hello world | confidence=0.95
# [session_id] 🔵 LISTENING → 🟡 THINKING (stt_received)
# [session_id] Processing complete in 150ms | response=...
# [session_id] 🟡 THINKING → 🔴 SPEAKING (response_ready)
# [session_id] [TTS event] | chunks=5 | duration=2500ms
# [session_id] 🔴 SPEAKING → 🔵 LISTENING (playback_done)

# 6. Interrupt test
# Say "stop" while agent is speaking

# Should see:
# [session_id] 🔴 SPEAKING → 🟣 INTERRUPT (barge_in)
# [session_id] 🟣 INTERRUPT → 🔵 LISTENING (ready_after_interrupt)

# 7. Check latencies
curl http://localhost:5204/latency/auto_session_123

# Should return:
# {
#   "listening→thinking": 95ms,
#   "thinking→speaking": 50ms,
#   "speaking→listening": 200ms
# }

# 8. No errors in logs
# Grep for ERROR, CRITICAL, and verify count is 0
docker-compose logs orchestrator | grep -i error | wc -l
# Should be 0 (or very few expected errors)
```

---

## TESTING STRATEGY

Create test cases in `tests/test_orchestrator_fix.py`:

```python
# tests/test_orchestrator_fix.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_state_transition_valid():
    """Test valid state transitions are allowed."""
    state_mgr = StateManager("test_session", None)
    
    # IDLE → LISTENING (valid)
    await state_mgr.transition(State.LISTENING, "client_connected")
    assert state_mgr.state == State.LISTENING
    
    # LISTENING → THINKING (valid)
    await state_mgr.transition(State.THINKING, "stt_received", {"text": "hello"})
    assert state_mgr.state == State.THINKING


@pytest.mark.asyncio
async def test_state_transition_invalid():
    """Test invalid state transitions are rejected."""
    state_mgr = StateManager("test_session", None)
    state_mgr.state = State.SPEAKING
    
    # SPEAKING → THINKING (invalid)
    await state_mgr.transition(State.THINKING, "invalid_trigger")
    assert state_mgr.state == State.SPEAKING  # No change


@pytest.mark.asyncio
async def test_stt_routing_to_correct_session():
    """Test STT events route to correct orchestrator session."""
    # Create two sessions
    sess1_id = "auto_session_1"
    sess2_id = "auto_session_2"
    
    active_sessions[sess1_id] = {"state_manager": StateManager(sess1_id, None)}
    active_sessions[sess2_id] = {"state_manager": StateManager(sess2_id, None)}
    
    # Emit STT event for session 1
    stt_handler = STTEventHandler(sess1_id, active_sessions[sess1_id]["state_manager"], config)
    result = await stt_handler.handle_stt_final("hello", is_final=True, source="websocket")
    
    # Verify session 1 processed it
    assert active_sessions[sess1_id]["state_manager"].state == State.THINKING
    # Session 2 unchanged
    assert active_sessions[sess2_id]["state_manager"].state == State.IDLE


@pytest.mark.asyncio
async def test_no_duplicate_task_spawning():
    """Test only one task active per session."""
    session_id = "test_session"
    active_sessions[session_id] = {"current_task": None}
    
    # Spawn first task
    task1 = asyncio.create_task(asyncio.sleep(10))
    active_sessions[session_id]["current_task"] = task1
    
    # Spawn second task (should cancel first)
    if active_sessions[session_id]["current_task"] and not active_sessions[session_id]["current_task"].done():
        active_sessions[session_id]["current_task"].cancel()
    
    task2 = asyncio.create_task(asyncio.sleep(10))
    active_sessions[session_id]["current_task"] = task2
    
    # Verify first task is cancelled
    assert task1.cancelled()
    # Verify second task is active
    assert not task2.done()
```

---

## SUMMARY: EXPECTED OUTCOMES

After fixing these phases in order:

| Phase | Problem Fixed | Outcome |
|-------|---------------|---------| 
| 1 | Async chaos, routing bugs, race conditions | **Clean, linear logs** |
| 2 | Multiple execution paths | **Single unified flow** |
| 3 | Duplicate event processing | **Events processed once** |
| 4 | Hard to debug, no visibility | **Structured logs + latency metrics** |

**Total fix time:** 6-8 hours
**Complexity:** Medium (mostly refactoring, no new features)
**Risk:** Low (improvements are additive)

**Start with PHASE 1 Priority 1.1 (async chaos) - that's the biggest immediate win!**
