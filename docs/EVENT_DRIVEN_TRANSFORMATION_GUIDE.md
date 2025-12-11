# Complete Transformation Guide: Voice Conversational Agent → Event-Driven Architecture

## Executive Summary

Your current orchestrator uses **synchronous state functions** that strictly sequence operations. To unlock lower latency, better resilience, and true bidirectional voice synchronization with WebRTC, this guide outlines a **full architectural transformation** to **event-driven orchestration** with **Redis Streams** (or Kafka) as the central event broker.

**What changes:**
- STT → Direct event emission (not orchestrator-driven polling)
- TTS → Client playback events feed back to orchestrator via WebSocket
- State transitions → Triggered by event arrival, not function completion
- Orchestrator → Event consumer + conversation FSM engine
- WebRTC jitter buffer state → Explicitly fedback via playback events

**Target latencies:** 
- STT→Intent+RAG: Parallel (saves 50ms)
- RAG generation → TTS streaming: Overlapped (saves 100ms)
- Total E2E: <450ms (vs current ~550ms)

---

## Part 1: Current Architecture Analysis

### Current Flow (Synchronous State Functions)

```
User speaks
    ↓
WebRTC → Orchestrator (sync call)
    ↓
STT service (wait for response)
    ↓
Intent classify (wait)
    ↓
RAG query (wait)
    ↓
LLM generate (wait)
    ↓
TTS synthesize (wait for first chunk)
    ↓
Play audio to browser
    ↓
Wait for playback done (guessed via TTS completion)
    ↓
Ready for next turn
```

**Issues:**
1. **No parallelization** – Intent and RAG run sequentially (~80ms wasted)
2. **Weak WebRTC sync** – Orchestrator can't see what's actually playing in browser's jitter buffer
3. **Barge-in race condition** – Client sends new audio during TTS→STT transition, state machine confused
4. **State function coupling** – Each step waits for previous; hard to add retries or timeouts
5. **No event observability** – Can't trace why latency spiked

### Current Components (From Your Code)

| Component | File | Purpose |
|-----------|------|---------|
| **StateManager** | `state_manager.py` | FSM (IDLE→LISTENING→THINKING→SPEAKING→INTERRUPT) |
| **Orchestrator** | `app.py` | WebSocket endpoint, service calls, TTS streaming |
| **Parallel Pipeline** | `parallel_pipeline.py` | Intent+RAG calls (but sequential waits on final) |
| **Dialogue Manager** | `dialogue_manager.py` | Pre-scripted fillers, greetings, etc. |
| **Unified FastRTC** | `unified_fastrtc.py` | Browser ↔ STT/TTS media handler |
| **Redis** | (external) | Session state persistence (good!) |

**Strengths to preserve:**
- ✅ Clean Pydantic models (keep `STTFragment`, `VADEnd`, `UserSpeaking`)
- ✅ Redis session persistence (extend with event log)
- ✅ Parallel Intent+RAG skeleton in `parallel_pipeline.py`
- ✅ Dialogue Manager for fillers (integrate into event handlers)
- ✅ FastRTC state-aware gating (extend with orchestrator events)

---

## Part 2: Target Event-Driven Architecture

### Conceptual Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    REDIS STREAMS (Event Broker)                 │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ voice.stt.partial          → Session queue                   ││
│ │ voice.stt.final            → Session queue                   ││
│ │ voice.intent.detected      → Session queue                   ││
│ │ voice.rag.answer_ready     → Session queue                   ││
│ │ voice.tts.chunk_ready      → Session queue                   ││
│ │ voice.webrtc.playback_done → Session queue                   ││
│ │ voice.orchestrator.state   → Session queue (status)          ││
│ └──────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
         ↑              ↑              ↑              ↑
         │              │              │              │
    [STT Service]  [Intent Service] [RAG Service] [TTS Service]
    (emits events)  (consumes+emits)  (consumes+emits) (emits audio)
         ↑              │              │              ↓
         │              └──────────────┴──────→ [Orchestrator FSM]
         │                                      (consumes all events,
    [Client Browser]←─────────────────────── drives state machine)
         ↑                                          ↓
         └──────────────────────────────────────────┘
     WebRTC playback events (playback_started, playback_done, barge_in)
```

### Event Model (Unified Schema)

All events follow this structure:

```json
{
  "event_type": "voice.stt.final",
  "session_id": "uuid-abc-123",
  "correlation_id": "trace-xyz",
  "timestamp": 1702300000.123,
  "source": "stt-service-1",
  "payload": {
    "text": "hello world",
    "confidence": 0.95,
    "is_final": true,
    "duration_ms": 2500
  },
  "metadata": {
    "trace_id": "jaeger-123",
    "user_id": "user-456"
  }
}
```

### Event Topics (Redis Stream Keys)

| Stream Key | Source | Consumers | Purpose |
|------------|--------|-----------|---------|
| `voice:stt:session:{sid}` | STT Service | Orchestrator | Partial + final transcripts |
| `voice:intent:session:{sid}` | Intent Service | Orchestrator | Intent classification results |
| `voice:rag:session:{sid}` | RAG Service | Orchestrator | Knowledge base answers (streaming) |
| `voice:tts:session:{sid}` | TTS Service | Orchestrator + Browser | Audio chunk ready events |
| `voice:webrtc:session:{sid}` | Browser (WebSocket) | Orchestrator | Playback started/done, barge-in, mic events |
| `voice:orchestrator:session:{sid}` | Orchestrator | Browser + Monitoring | State transitions, metrics |

---

## Part 3: Component-by-Component Transformation

### 3.1 Orchestrator (app.py) → Event Consumer + FSM Engine

**Current:**
```python
# Synchronous call chain
stt_resp = await stt_service.transcribe(audio)
intent_resp = await intent_service.classify(stt_resp.text)
rag_resp = await rag_service.query(intent_resp.context)
tts_resp = await tts_service.synthesize(rag_resp.answer)
await stream_tts_audio(websocket, tts_resp)
```

**Target:**
```python
# Event-driven flow
class OrchestratorFSM:
    def __init__(self, session_id: str, redis: Redis, events_broker: RedisStreams):
        self.session_id = session_id
        self.redis = redis
        self.broker = events_broker  # NEW: Event broker
        self.state_mgr = StateManager(session_id, redis)
        self.pending_requests = {}  # Track in-flight requests
        
    async def run(self):
        """Main event loop – consumes from broker, updates FSM"""
        stream_key = f"voice:*:session:{self.session_id}"
        
        while True:
            # Block read from Redis Streams (new messages + existing)
            messages = await self.broker.xread(
                {stream_key: "$"},  # Read new events only
                block=100,  # 100ms timeout
                count=10    # Batch up to 10 events
            )
            
            for stream, message_list in messages:
                for message_id, data in message_list:
                    event = Event.from_dict(data)
                    await self.handle_event(event)
                    await self.broker.xack(stream, "orchestrator-group", message_id)
    
    async def handle_event(self, event: Event):
        """Dispatch event to appropriate handler"""
        handlers = {
            "voice.stt.final": self.on_stt_final,
            "voice.intent.detected": self.on_intent_detected,
            "voice.rag.answer_ready": self.on_rag_answer,
            "voice.tts.chunk_ready": self.on_tts_chunk,
            "voice.webrtc.playback_done": self.on_playback_done,
            "voice.webrtc.barge_in": self.on_barge_in,
        }
        handler = handlers.get(event.event_type)
        if handler:
            await handler(event)
    
    async def on_stt_final(self, event: Event):
        """Handle final STT event → trigger Intent+RAG in parallel"""
        text = event.payload["text"]
        
        # Transition state
        await self.state_mgr.transition(State.THINKING, "stt_final", {
            "text": text
        })
        
        # Fire parallel requests (as events, not waits!)
        await self.broker.xadd(
            f"voice:intent:session:{self.session_id}",
            {"text": text, "session_id": self.session_id, "request_id": uuid4()}
        )
        await self.broker.xadd(
            f"voice:rag:session:{self.session_id}",
            {"text": text, "session_id": self.session_id, "request_id": uuid4()}
        )
        
        # Don't block! Event handlers will fire when results arrive
        
    async def on_rag_answer(self, event: Event):
        """Handle RAG stream chunks → pass to TTS (overlapped)"""
        answer_chunk = event.payload.get("text", "")
        request_id = event.payload.get("request_id")
        
        if answer_chunk:
            # Queue for TTS streaming (non-blocking)
            await self.broker.xadd(
                f"voice:tts:session:{self.session_id}",
                {
                    "text": answer_chunk,
                    "source": "rag",
                    "request_id": request_id
                }
            )
        
        if event.payload.get("is_final"):
            # RAG complete
            await self.state_mgr.transition(State.SPEAKING, "rag_complete", {})
    
    async def on_playback_done(self, event: Event):
        """Handle WebRTC playback completion → back to LISTENING"""
        await self.state_mgr.transition(State.LISTENING, "playback_done", {})
    
    async def on_barge_in(self, event: Event):
        """Handle user interruption → cancel TTS, back to LISTENING"""
        # Cancel any pending TTS requests
        await self.broker.xadd(
            f"voice:tts:session:{self.session_id}",
            {"action": "cancel"}
        )
        
        # Transition to interrupt
        await self.state_mgr.transition(State.INTERRUPT, "barge_in", {})
        
        # After interrupt, back to LISTENING
        await asyncio.sleep(0.1)  # Brief pause
        await self.state_mgr.transition(State.LISTENING, "ready_for_next", {})
```

**Key changes:**
1. **Non-blocking calls** – Services emit events; orchestrator consumes
2. **Parallel execution** – Intent + RAG happen simultaneously
3. **Event multiplexing** – Redis Streams handles all message ordering
4. **Loose coupling** – Services don't know about each other
5. **Fault tolerance** – Unacked events can be replayed

---

### 3.2 STT Service → Event Emitter

**Current:** Returns transcript via WebSocket

**Target:** Emit events to Redis Stream

```python
# stt_service/main.py (or unified_fastrtc.py)

class STTEventEmitter:
    def __init__(self, redis: Redis):
        self.redis = redis
    
    async def emit_partial(self, session_id: str, text: str, confidence: float):
        """Emit partial transcription event"""
        event = {
            "event_type": "voice.stt.partial",
            "session_id": session_id,
            "timestamp": time.time(),
            "source": "stt-service",
            "payload": {
                "text": text,
                "confidence": confidence,
                "is_final": False
            }
        }
        
        stream_key = f"voice:stt:session:{session_id}"
        await self.redis.xadd(stream_key, {
            "data": json.dumps(event)
        })
        
        # Also send to WebSocket for UI feedback (keep realtime feel)
        await self.websocket.send_json({
            "type": "stt_partial",
            "text": text,
            "confidence": confidence
        })
    
    async def emit_final(self, session_id: str, text: str, confidence: float, duration_ms: float):
        """Emit final transcription event – THIS TRIGGERS ORCHESTRATION"""
        event = {
            "event_type": "voice.stt.final",
            "session_id": session_id,
            "timestamp": time.time(),
            "source": "stt-service",
            "payload": {
                "text": text,
                "confidence": confidence,
                "is_final": True,
                "duration_ms": duration_ms
            }
        }
        
        stream_key = f"voice:stt:session:{session_id}"
        await self.redis.xadd(stream_key, {
            "data": json.dumps(event)
        })
        
        # CRITICAL: Only this event triggers orchestrator state transition
        logger.info(f"[{session_id}] 🎤 STT FINAL emitted → orchestrator will consume")
```

**Integration points:**
- Keep WebSocket to STT service (media stream real-time)
- Add Redis Streams for event emission (control plane)
- Client still hears streaming STT via WebSocket (unchanged)

---

### 3.3 Intent + RAG Services → Event Consumers + Emitters

**Current:** Called synchronously by orchestrator

**Target:** Subscribe to STT events, emit results

```python
# intent_service/main.py

class IntentEventHandler:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.intent_classifier = IntentClassifier()
    
    async def run(self):
        """Subscribe to STT events, process, emit Intent events"""
        # Create consumer group
        try:
            await self.redis.xgroup_create(
                "voice:intent:*",
                "intent-service",
                id="$",
                mkstream=True
            )
        except:
            pass  # Group exists
        
        while True:
            # Read pending events from orchestrator
            messages = await self.redis.xreadgroup(
                groupname="intent-service",
                consumername=f"intent-worker-1",
                streams={"voice:intent:*": ">"},  # New messages
                block=1000,
                count=5
            )
            
            for stream, message_list in messages:
                for message_id, data in message_list:
                    request = json.loads(data.get("text", "{}"))
                    session_id = request.get("session_id")
                    text = request.get("text")
                    
                    # Process
                    intent_result = await self.intent_classifier.classify(text)
                    
                    # Emit result
                    event = {
                        "event_type": "voice.intent.detected",
                        "session_id": session_id,
                        "timestamp": time.time(),
                        "source": "intent-service",
                        "payload": intent_result
                    }
                    
                    await self.redis.xadd(
                        f"voice:intent:session:{session_id}",
                        {"data": json.dumps(event)}
                    )
                    
                    # Acknowledge
                    await self.redis.xack(stream, "intent-service", message_id)
```

**RAG Service similar pattern:**
```python
# rag_service/main.py

class RAGEventHandler:
    async def run(self):
        # Subscribe to voice:rag:session:* stream
        # For each event, stream chunks via voice:rag:session:{sid}
        
        async def handle_rag_request(session_id: str, text: str):
            async for chunk in self.rag_engine.stream_answer(text):
                event = {
                    "event_type": "voice.rag.answer_ready",
                    "session_id": session_id,
                    "payload": {
                        "text": chunk,
                        "is_final": False
                    }
                }
                await self.redis.xadd(
                    f"voice:rag:session:{session_id}",
                    {"data": json.dumps(event)}
                )
            
            # Final marker
            await self.redis.xadd(
                f"voice:rag:session:{session_id}",
                {"data": json.dumps({
                    "event_type": "voice.rag.answer_ready",
                    "payload": {"is_final": True}
                })}
            )
```

**Benefits:**
- Services independently processable (scale separately)
- Retry logic centralized in broker
- Can run multiple replicas (consumer groups)

---

### 3.4 TTS Service → Event Consumer + Chunk Emitter

**Current:** Streams audio directly to browser via WebSocket

**Target:** Listens for text events, emits audio chunks to Redis (browser pulls via WebSocket)

```python
# tts_service/main.py

class TTSEventHandler:
    async def run(self):
        """Consume text events, stream TTS audio, emit chunk events"""
        
        async def handle_tts_request(session_id: str, text_chunk: str):
            # Generate audio
            async for audio_chunk, is_final in self.tts_engine.stream(text_chunk):
                # Emit to Redis (for persistence + monitoring)
                event = {
                    "event_type": "voice.tts.chunk_ready",
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "payload": {
                        "audio_base64": base64.b64encode(audio_chunk).decode(),
                        "is_final": is_final
                    }
                }
                
                await self.redis.xadd(
                    f"voice:tts:session:{session_id}",
                    {"data": json.dumps(event)}
                )
                
                # ALSO push to orchestrator WebSocket (real-time audio)
                # (orchestrator forwards to browser)
                await self.orchestrator_ws.send_json(event)
```

---

### 3.5 Client Browser ↔ Orchestrator: PlaybackEvent Bridge

**CRITICAL for WebRTC Sync**

Current problem: Orchestrator doesn't know actual playback state. Client's jitter buffer can delay audio by 100-500ms.

Solution: Browser sends playback events back to orchestrator.

```javascript
// Client: unified_fastrtc.js (extended)

class OrchestratorBridge {
    constructor(websocket, audioElement) {
        this.ws = websocket;
        this.audio = audioElement;  // <audio> element or AudioContext
        this.audioQueue = [];
    }
    
    async onTTSChunkReady(event) {
        // Receive audio chunk from server
        const audioData = new Uint8Array(
            atob(event.payload.audio_base64).split('').map(c => c.charCodeAt(0))
        );
        
        // Queue for playback
        this.audioQueue.push(audioData);
        
        // Start playback if not playing
        if (!this.isPlaying) {
            await this.startPlayback();
        }
    }
    
    async startPlayback() {
        this.isPlaying = true;
        
        // Send playback started event to orchestrator
        this.ws.send(JSON.stringify({
            type: "webrtc_event",
            event_type: "voice.webrtc.playback_started",
            session_id: this.sessionId,
            timestamp: Date.now()
        }));
        
        // Play chunks in queue
        while (this.audioQueue.length > 0) {
            const chunk = this.audioQueue.shift();
            await this.playAudioChunk(chunk);
        }
        
        this.isPlaying = false;
        
        // Send playback done event to orchestrator
        this.ws.send(JSON.stringify({
            type: "webrtc_event",
            event_type: "voice.webrtc.playback_done",
            session_id: this.sessionId,
            timestamp: Date.now()
        }));
    }
    
    async playAudioChunk(audioData) {
        // Use Web Audio API or audio element
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(audioData.buffer);
        
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        
        return new Promise(resolve => {
            source.onended = resolve;
            source.start(0);
        });
    }
    
    // Barge-in: User speaks while agent is speaking
    onMicrophoneInput(audioChunk) {
        // Detect if TTS is currently playing
        if (this.isPlaying) {
            // Send barge-in event
            this.ws.send(JSON.stringify({
                type: "webrtc_event",
                event_type: "voice.webrtc.barge_in",
                session_id: this.sessionId,
                timestamp: Date.now()
            }));
            
            // Stop playback (browser doesn't continue current audio)
            this.audioQueue = [];  // Clear pending chunks
        }
        
        // Continue with STT
        this.sttWebSocket.send(audioChunk);
    }
}
```

**Server side (orchestrator):**
```python
# app.py: WebSocket message handler

@app.websocket("/ws/orchestrate")
async def orchestrate_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    while True:
        msg = await websocket.receive_json()
        
        if msg.get("type") == "webrtc_event":
            # Forward to orchestrator event stream
            event = {
                "event_type": msg["event_type"],
                "session_id": session_id,
                "timestamp": msg["timestamp"],
                "source": "webrtc-client",
                "payload": {}
            }
            
            await redis.xadd(
                f"voice:webrtc:session:{session_id}",
                {"data": json.dumps(event)}
            )
            
            # Orchestrator FSM will consume this
            logger.info(f"[{session_id}] WebRTC event forwarded: {event['event_type']}")
```

---

## Part 4: Event-Driven State Machine (Enhanced StateManager)

**Current:** Simple state transitions, Redis save

**Target:** Event-driven with side effects

```python
# state_manager.py (enhanced)

class StateManager:
    def __init__(self, session_id: str, redis: Redis, broker: RedisStreams):
        self.session_id = session_id
        self.redis = redis
        self.broker = broker  # NEW
        self.state = State.IDLE
        self.transitions = {
            State.IDLE: [State.LISTENING],
            State.LISTENING: [State.THINKING, State.INTERRUPT],
            State.THINKING: [State.SPEAKING, State.IDLE],
            State.SPEAKING: [State.LISTENING, State.INTERRUPT, State.IDLE],
            State.INTERRUPT: [State.LISTENING, State.IDLE],
        }
        
        # Event handlers per state
        self.handlers = {
            State.IDLE: self.handle_idle_state,
            State.LISTENING: self.handle_listening_state,
            State.THINKING: self.handle_thinking_state,
            State.SPEAKING: self.handle_speaking_state,
            State.INTERRUPT: self.handle_interrupt_state,
        }
    
    async def transition(self, new_state: State, trigger: str, data: dict = None):
        """Atomic state transition with event emission"""
        old_state = self.state
        
        if new_state not in self.transitions.get(old_state, []):
            logger.error(f"Invalid transition: {old_state} → {new_state}")
            return
        
        self.state = new_state
        
        # Persist to Redis
        await self.redis.hset(
            f"orchestrator:session:{self.session_id}",
            mapping={"state": new_state.value, "last_update": time.time()}
        )
        
        # Emit state change event
        event = {
            "event_type": "voice.orchestrator.state",
            "session_id": self.session_id,
            "timestamp": time.time(),
            "source": "orchestrator",
            "payload": {
                "old_state": old_state.value,
                "new_state": new_state.value,
                "trigger": trigger,
                "data": data or {}
            }
        }
        
        await self.broker.xadd(
            f"voice:orchestrator:session:{self.session_id}",
            {"data": json.dumps(event)}
        )
        
        logger.info(
            f"[{self.session_id}] {old_state.value.upper()} → {new_state.value.upper()} ({trigger})"
        )
        
        # Execute state-specific handlers
        handler = self.handlers.get(new_state)
        if handler:
            await handler(trigger, data)
    
    async def handle_listening_state(self, trigger: str, data: dict):
        """Execute side effects when entering LISTENING state"""
        # Gate microphone on (if it was gated during SPEAKING)
        await self.broker.xadd(
            f"voice:unified_fastrtc:session:{self.session_id}",
            {"action": "gate_off"}  # Open mic
        )
        
        # Ready for next STT fragment
        logger.info(f"[{self.session_id}] Microphone OPEN, listening for speech")
    
    async def handle_thinking_state(self, trigger: str, data: dict):
        """Side effects for THINKING state"""
        # Gate microphone off (prevent echo)
        # Play immediate filler
        if data and data.get("text"):
            text = data["text"]
            logger.info(f"[{self.session_id}] Processing: '{text[:50]}...'")
    
    async def handle_speaking_state(self, trigger: str, data: dict):
        """Side effects for SPEAKING state"""
        # Gate microphone off (important: prevents TTS from being fed into STT)
        await self.broker.xadd(
            f"voice:unified_fastrtc:session:{self.session_id}",
            {"action": "gate_on"}  # Close mic
        )
        
        logger.info(f"[{self.session_id}] Agent SPEAKING, microphone GATED")
    
    async def handle_interrupt_state(self, trigger: str, data: dict):
        """Side effects for INTERRUPT state (barge-in)"""
        logger.info(f"[{self.session_id}] User interrupted agent speech")
```

---

## Part 5: Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal:** Add Redis Streams infrastructure, minimal event emission

1. **Install Redis Streams client:**
   ```bash
   pip install redis[hiredis]  # Async support
   ```

2. **Create event broker wrapper:**
   ```python
   # services/shared/event_broker.py
   
   class RedisEventBroker:
       def __init__(self, redis_url: str):
           self.redis = redis.from_url(redis_url, decode_responses=True)
       
       async def xadd(self, stream_key: str, data: dict, max_len: int = 10000):
           """Add event to stream"""
           return await self.redis.xadd(stream_key, data, maxlen=max_len, approximate=True)
       
       async def xread(self, streams: dict, block: int = 0, count: int = 1):
           """Read events from stream(s)"""
           return await self.redis.xread(streams, block=block, count=count)
       
       async def xgroup_create(self, stream: str, group: str, id: str = "$", mkstream: bool = False):
           """Create consumer group"""
           try:
               return await self.redis.xgroup_create(stream, group, id=id, mkstream=mkstream)
           except redis.ResponseError:
               pass  # Group exists
       
       async def xreadgroup(self, groupname: str, consumername: str, streams: dict, block: int = 0, count: int = 1):
           """Read as consumer group"""
           return await self.redis.xreadgroup(groupname, consumername, streams, block=block, count=count)
   ```

3. **Emit STT final events:**
   ```python
   # unified_fastrtc.py: Add event emission on VAD end
   
   async def on_vad_end(self, session_id: str, confidence: float):
       event = {
           "event_type": "voice.stt.final",
           "session_id": session_id,
           "timestamp": time.time(),
           "source": "stt-service",
           "payload": {
               "text": self.transcript,
               "confidence": confidence,
               "is_final": True
           }
       }
       
       await self.broker.xadd(
           f"voice:stt:session:{session_id}",
           {"data": json.dumps(event)}
       )
   ```

4. **Create event consumer skeleton in orchestrator:**
   ```python
   # app.py
   
   async def orchestrator_event_loop(session_id: str):
       """Main event consumption loop"""
       stream_pattern = f"voice:*:session:{session_id}"
       
       while True:
           messages = await broker.xread({stream_pattern: "$"}, block=100)
           
           for stream, message_list in messages:
               for msg_id, data in message_list:
                   event = Event.from_dict(json.loads(data.get("data", "{}")))
                   await handle_event(event)
   ```

5. **Test:** Single STT event emitted, logged by orchestrator

---

### Phase 2: Parallel Processing (Week 2-3)

**Goal:** Intent + RAG events, parallel handling

1. **Orchestrator emits Intent + RAG requests on STT final:**
   ```python
   async def on_stt_final(event: Event):
       text = event.payload["text"]
       
       # Parallel requests
       await broker.xadd(
           f"voice:intent:session:{session_id}",
           {"text": text, "request_id": str(uuid4())}
       )
       
       await broker.xadd(
           f"voice:rag:session:{session_id}",
           {"text": text, "request_id": str(uuid4())}
       )
   ```

2. **Intent service listens + responds:**
   ```python
   # Create separate intent_service.py
   # Subscribes to voice:intent:session:* events
   # Classifies, emits voice.intent.detected events
   ```

3. **RAG service listens + streams:**
   ```python
   # Create separate rag_service.py
   # Subscribes to voice:rag:session:* events
   # Streams chunks as voice.rag.answer_ready events
   ```

4. **Test:** STT final → Intent + RAG events fired in parallel, latency reduced

---

### Phase 3: TTS + WebRTC Sync (Week 3-4)

**Goal:** Audio streaming, playback events, barge-in

1. **TTS service emits audio chunk events:**
   ```python
   async def on_rag_chunk(event: Event):
       text = event.payload["text"]
       
       async for audio_chunk in tts_engine.stream(text):
           event = {
               "event_type": "voice.tts.chunk_ready",
               "session_id": session_id,
               "payload": {
                   "audio_base64": base64.b64encode(audio_chunk).decode()
               }
           }
           
           await broker.xadd(...)
           await websocket.send_json(event)  # Real-time
   ```

2. **Client sends playback events:**
   ```javascript
   // Client code: When audio starts/stops playing
   ws.send(JSON.stringify({
       "type": "webrtc_event",
       "event_type": "voice.webrtc.playback_started",
       "session_id": sessionId
   }));
   
   ws.send(JSON.stringify({
       "type": "webrtc_event",
       "event_type": "voice.webrtc.playback_done",
       "session_id": sessionId
   }));
   ```

3. **Orchestrator consumes playback events:**
   ```python
   async def on_playback_done(event: Event):
       await state_mgr.transition(State.LISTENING, "playback_done")
   ```

4. **Test:** Full loop STT → Intent+RAG → TTS → Playback → LISTENING

---

### Phase 4: Resilience + Monitoring (Week 4-5)

**Goal:** Consumer groups, DLQ, tracing

1. **Implement consumer groups:**
   ```python
   # Each service uses consumer group (auto-acknowledge, replay on failure)
   
   await broker.xgroup_create(
       f"voice:intent:*",
       "intent-service",
       mkstream=True
   )
   
   messages = await broker.xreadgroup(
       groupname="intent-service",
       consumername="intent-1",
       streams={f"voice:intent:*": ">"}
   )
   ```

2. **Add dead-letter queue (DLQ) for failed events:**
   ```python
   async def handle_event_with_retry(event: Event, max_retries: int = 3):
       for attempt in range(max_retries):
           try:
               await process_event(event)
               return
           except Exception as e:
               if attempt == max_retries - 1:
                   # Send to DLQ
                   await broker.xadd(
                       f"voice:dlq:session:{session_id}",
                       {
                           "event": json.dumps(asdict(event)),
                           "error": str(e),
                           "attempts": max_retries
                       }
                   )
               else:
                   await asyncio.sleep(2 ** attempt)  # Exponential backoff
   ```

3. **Emit metrics events:**
   ```python
   # Latency tracking
   async def on_event(event: Event):
       start = time.time()
       await handle_event(event)
       latency_ms = (time.time() - start) * 1000
       
       await broker.xadd(
           f"voice:metrics:session:{session_id}",
           {
               "event_type": event.event_type,
               "latency_ms": latency_ms
           }
       )
   ```

4. **Test:** Kill intent service mid-request, verify retry + DLQ

---

### Phase 5: Production Hardening (Week 5-6)

**Goal:** Observability, scaling, deployment

1. **OpenTelemetry tracing:**
   ```python
   from opentelemetry import trace
   
   tracer = trace.get_tracer(__name__)
   
   async def handle_event(event: Event):
       with tracer.start_as_current_span(f"handle_{event.event_type}"):
           # Span includes session_id, correlation_id
           await process_event(event)
   ```

2. **Prometheus metrics:**
   ```python
   from prometheus_client import Counter, Histogram
   
   event_counter = Counter(
       "voice_events_total",
       "Total events processed",
       ["event_type", "status"]
   )
   
   event_latency = Histogram(
       "voice_event_latency_ms",
       "Event processing latency",
       ["event_type"]
   )
   ```

3. **Scaling considerations:**
   - Multiple orchestrator instances (Redis Streams supports parallel consumers)
   - Each service scales independently (more replicas = more consumers in group)
   - Redis Streams handles ordering per session

4. **Docker Compose updates:**
   ```yaml
   version: "3.8"
   services:
     redis:
       image: redis:7-alpine
       command: redis-server --appendonly yes
       ports: ["6379:6379"]
       volumes: ["redis_data:/data"]
     
     orchestrator:
       build: ./orchestrator
       depends_on: ["redis"]
       environment:
         REDIS_URL: redis://redis:6379/0
         BROKER_TYPE: redis_streams
       ports: ["8000:8000"]
     
     intent_service:
       build: ./intent_service
       depends_on: ["redis"]
       environment:
         REDIS_URL: redis://redis:6379/0
       # Listens to voice:intent:* events
     
     rag_service:
       build: ./rag_service
       depends_on: ["redis"]
       environment:
         REDIS_URL: redis://redis:6379/0
       # Listens to voice:rag:* events
     
     tts_service:
       build: ./tts_service
       depends_on: ["redis"]
       environment:
         REDIS_URL: redis://redis:6379/0
       # Listens to voice:tts:* events
   ```

---

## Part 6: Event Broker Choice: Redis vs Kafka

| Aspect | Redis Streams | Kafka |
|--------|---------------|-------|
| **Latency** | <1ms | 5-10ms |
| **Persistence** | Optional (RDB/AOF) | Always persisted |
| **Ordering** | Per stream (session) | Per partition (topic) |
| **Consumer Groups** | Yes, simple | Yes, sophisticated |
| **Operational** | Simple (single node) | Complex (brokers, ZK) |
| **Best for** | <500ms latency, <10K msg/sec | High throughput, archival |

**Recommendation:** Start with **Redis Streams** (already in your stack). Migrate to Kafka only if you exceed:
- >50K concurrent sessions
- >1M events/sec
- Need event replay across weeks

---

## Part 7: Critical Implementation Notes

### 7.1 Session ID Consistency

Every event must include `session_id`. Use UUID v4:

```python
import uuid

session_id = str(uuid.uuid4())  # e.g., "550e8400-e29b-41d4-a716-446655440000"

# All events keyed by session_id
await broker.xadd(f"voice:stt:session:{session_id}", {...})
```

### 7.2 Event Deduplication

Guard against duplicate event processing:

```python
# Use Redis Set for deduplication
message_id_key = f"event:processed:{session_id}:{event.message_id}"

if await redis.exists(message_id_key):
    logger.debug(f"Duplicate event {event.message_id}, skipping")
    return

# Process
await process_event(event)

# Mark as processed
await redis.setex(message_id_key, 3600, "1")  # 1 hour TTL
```

### 7.3 State Machine Transitions

Never assume events arrive in order. Validate state transitions:

```python
valid_next_states = {
    State.IDLE: [State.LISTENING],
    State.LISTENING: [State.THINKING, State.INTERRUPT],
    # ...
}

if new_state not in valid_next_states.get(current_state, []):
    logger.warning(f"Invalid transition {current_state} → {new_state}")
    return  # Silently ignore (event disorder)
```

### 7.4 WebRTC Jitter Buffer Synchronization

Client must accurately report when audio actually finishes:

```javascript
// Use AudioContext time for precision
const startTime = audioContext.currentTime;
const duration = audioBuffer.duration;
const expectedEndTime = startTime + duration;

// Poll for actual end (jitter buffer may add delay)
setInterval(() => {
    if (audioContext.currentTime > expectedEndTime + 0.5) {
        // Audio has ended, send event
        ws.send(JSON.stringify({
            event_type: "voice.webrtc.playback_done",
            actual_duration_ms: (audioContext.currentTime - startTime) * 1000
        }));
    }
}, 100);
```

### 7.5 Barge-In Handling

User can speak while agent is still speaking. Must:

1. Stop TTS playback (browser clears queue)
2. Cancel pending TTS requests (service stops generation)
3. Transition FSM to INTERRUPT → LISTENING

```python
# Orchestrator
async def on_barge_in(event: Event):
    # 1. Cancel TTS
    await broker.xadd(
        f"voice:tts:session:{session_id}",
        {"action": "cancel", "timestamp": time.time()}
    )
    
    # 2. Interrupt state
    await state_mgr.transition(State.INTERRUPT, "barge_in")
    
    # 3. Back to listening after brief pause
    await asyncio.sleep(0.05)  # Allow TTS cancellation
    await state_mgr.transition(State.LISTENING, "ready")
```

---

## Part 8: Testing Strategy

### Unit Tests

```python
# test_orchestrator.py

@pytest.mark.asyncio
async def test_stt_final_triggers_parallel_intent_rag():
    """Verify STT final event causes parallel Intent+RAG emission"""
    broker = FakeRedisStreams()
    orchestrator = OrchestratorFSM(session_id="test", broker=broker)
    
    # Emit STT final
    event = Event(
        event_type="voice.stt.final",
        session_id="test",
        payload={"text": "hello", "confidence": 0.95, "is_final": True}
    )
    
    await orchestrator.handle_event(event)
    
    # Verify both Intent + RAG requests emitted
    assert any(e.event_type == "voice.intent.request" for e in broker.emitted)
    assert any(e.event_type == "voice.rag.request" for e in broker.emitted)

@pytest.mark.asyncio
async def test_barge_in_cancels_tts():
    """Verify barge-in cancels pending TTS"""
    # ...
```

### Integration Tests

```python
# test_integration.py

@pytest.mark.asyncio
async def test_full_loop():
    """Full STT → Intent+RAG → TTS → Playback loop"""
    
    async with TestHarness() as harness:
        # 1. Send STT final
        await harness.stt_service.emit_final(
            session_id="test",
            text="hello",
            confidence=0.95
        )
        
        # 2. Verify orchestrator transitioned to THINKING
        await asyncio.sleep(0.1)
        assert harness.orchestrator.state == State.THINKING
        
        # 3. Verify Intent + RAG requests sent
        assert len(harness.broker.get_events("voice:intent:*")) > 0
        assert len(harness.broker.get_events("voice:rag:*")) > 0
        
        # 4. Emit RAG answer
        await harness.rag_service.emit_answer(
            session_id="test",
            text="Hello! How can I help?",
            is_final=True
        )
        
        # 5. Verify TTS request emitted
        await asyncio.sleep(0.1)
        assert len(harness.broker.get_events("voice:tts:*")) > 0
        
        # 6. Emit playback done
        await harness.webrtc_client.emit_playback_done(session_id="test")
        
        # 7. Verify orchestrator back to LISTENING
        await asyncio.sleep(0.1)
        assert harness.orchestrator.state == State.LISTENING
```

### Load Tests

```python
# test_load.py using Locust

from locust import HttpUser, task, between

class VoiceAgentUser(HttpUser):
    wait_time = between(1, 5)
    
    @task
    def full_conversation(self):
        """Simulate full conversation cycle"""
        session_id = str(uuid.uuid4())
        
        # 1. Connect WebSocket
        ws = self.client.websocket(f"/ws/orchestrate?session_id={session_id}")
        
        # 2. Send STT final
        ws.send_json({
            "type": "stt_final",
            "text": "hello world",
            "confidence": 0.95
        })
        
        # 3. Wait for TTS chunks
        chunks_received = 0
        timeout = time.time() + 5
        while time.time() < timeout:
            msg = ws.recv()
            if msg.get("type") == "tts_chunk":
                chunks_received += 1
            elif msg.get("type") == "orchestrator_state":
                if msg["payload"]["state"] == "LISTENING":
                    break
        
        ws.close()
        
        assert chunks_received > 0, "No TTS chunks received"
```

---

## Part 9: Migration Checklist

- [ ] **Week 1:** Install Redis Streams, create event broker wrapper, emit STT final events
- [ ] **Week 2:** Create Intent service event consumer, RAG service event consumer
- [ ] **Week 2:** Parallel processing in orchestrator (Intent+RAG simultaneously)
- [ ] **Week 3:** TTS service event listener, audio chunk emission
- [ ] **Week 3:** WebRTC playback events → orchestrator
- [ ] **Week 4:** Barge-in handling (cancellation, interrupt state)
- [ ] **Week 4:** Consumer groups, error handling, DLQ
- [ ] **Week 5:** Observability (OpenTelemetry, Prometheus)
- [ ] **Week 5:** Load testing (Locust)
- [ ] **Week 6:** Production hardening, rollout to staging
- [ ] **Week 6:** Monitor latency improvements, rollback plan

---

## Part 10: Expected Improvements

| Metric | Before | After | Gain |
|--------|--------|-------|------|
| Intent+RAG latency | 150ms (seq) | 80ms (parallel) | **47% faster** |
| Total E2E latency | 550ms | 400ms | **27% faster** |
| Perceived responsiveness | User hears 800ms after speak | User hears 300ms after speak | **63% faster** |
| Barge-in responsiveness | 300ms to interrupt | 100ms to respond | **67% faster** |
| Fault tolerance | Service down = failure | Automatic retry + DLQ | **Much better** |
| Scalability | Single orchestrator limit | Multi-instance horizontal scale | **Unlimited** |

---

## Summary

By transforming your orchestrator from **synchronous state functions** → **event-driven FSM**, you unlock:

1. **Low latency** (parallel processing, overlapped I/O)
2. **Resilience** (consumer groups, retries, DLQ)
3. **Scalability** (horizontal scaling of services)
4. **Observability** (event tracing, metrics per event)
5. **True bidirectional sync** (WebRTC playback events fed back)
6. **Natural barge-in** (interruption handling)

**Start with Redis Streams** (already in your stack) → Graduate to Kafka if you need archival/replay.

Good luck with the transformation! 🚀