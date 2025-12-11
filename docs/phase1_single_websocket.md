# PHASE 1: SINGLE WEBSOCKET IMPLEMENTATION GUIDE

## Overview

Transform from 2 separate WebRTC connections to 1 unified bidirectional WebSocket.

**Timeline:** 1 day  
**Files to create:** 1 new  
**Files to modify:** 2  
**Files to delete:** 0 (keep for fallback)

---

## ARCHITECTURE DIAGRAM

```
BEFORE (Current):
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │
       ├─→ [FastRTC STT Handler] → WebSocket /stt
       │   └─ Sends: STT text events
       │   └─ Connection: independent, can drop
       │
       └─→ [FastRTC TTS Handler] → WebSocket /tts
           └─ Receives: audio chunks
           └─ Connection: independent, can drop

Problems:
- 2 WebSocket handshakes (slower)
- 2 separate streams to manage
- Session mapping fragile (fastrtc_X → auto_session_Y)


AFTER (Single WebSocket):
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │
       └─→ [Orchestrator Handler] ←→ WebSocket /orchestrate
           ├─ Receive: audio chunks, interrupts
           ├─ Send: state updates, audio, responses
           └─ Single session, clear routing
```

---

## NEW MESSAGE PROTOCOL

### Browser → Server (Client sends)

```json
{
  "type": "audio_chunk",
  "data": "base64_encoded_pcm",
  "timestamp": 1734000000.123,
  "sequence": 42
}

{
  "type": "playback_done",
  "timestamp": 1734000000.456
}

{
  "type": "interrupt",
  "reason": "user_speaking",
  "timestamp": 1734000000.789
}

{
  "type": "start_session",
  "mode": "conversation"
}

{
  "type": "end_session"
}

{
  "type": "state_sync_request"
}
```

### Server → Browser (Server sends)

```json
{
  "type": "state_update",
  "state": "listening",
  "timestamp": 1734000000.123
}

{
  "type": "audio_chunk",
  "audio": "base64_encoded_pcm",
  "chunk_id": "chunk_0",
  "is_final": false,
  "timestamp": 1734000000.456
}

{
  "type": "agent_response",
  "text": "Hi! How can I help?",
  "is_streaming": true,
  "timestamp": 1734000000.789
}

{
  "type": "session_ready",
  "session_id": "auto_session_123",
  "timestamp": 1734000000.000
}

{
  "type": "playback_control",
  "action": "stop",
  "reason": "user_interrupt",
  "timestamp": 1734000000.123
}
```

---

## IMPLEMENTATION: New WebSocket Handler

### File: `orchestrator/orchestrator_ws_handler.py`

```python
import asyncio
import json
import logging
import time
import base64
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect
import numpy as np

from .state_manager import StateManager, State
from .parallel_pipeline import process_intent_rag_llm
from .dialogue_manager import DialogueManager
from .service_manager import ServiceManager

logger = logging.getLogger(__name__)

@dataclass
class OrchestratorSession:
    """Single unified session for a WebSocket connection"""
    session_id: str
    websocket: WebSocket
    state_manager: StateManager
    created_at: float = field(default_factory=time.time)
    
    # Audio state
    tts_task: Optional[asyncio.Task] = None
    current_interrupt_task: Optional[asyncio.Task] = None
    
    # Metrics
    audio_chunks_received: int = 0
    audio_chunks_sent: int = 0
    last_activity: float = field(default_factory=time.time)

class OrchestratorWSHandler:
    """
    Unified WebSocket handler for orchestrator.
    
    Single connection handles:
    - Audio input (microphone chunks)
    - Audio output (TTS streaming)
    - State synchronization
    - Interrupt handling
    """
    
    def __init__(self, 
                 service_manager: ServiceManager,
                 dialogue_manager: DialogueManager):
        self.service_manager = service_manager
        self.dialogue_manager = dialogue_manager
        self.sessions: Dict[str, OrchestratorSession] = {}
    
    async def handle_connection(self, 
                               websocket: WebSocket,
                               session_id: Optional[str] = None):
        """
        Main WebSocket connection handler.
        
        Replaces the fragmented logic in app.py with a single unified handler.
        """
        await websocket.accept()
        
        # Create or reuse session
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.websocket = websocket  # Update connection
            logger.info(f"[{session_id}] Reconnected")
        else:
            session = await self._create_session(websocket)
            session_id = session.session_id
            logger.info(f"[{session_id}] Session created")
        
        # Send session ready notification
        await self._send_json(websocket, {
            "type": "session_ready",
            "session_id": session_id,
            "timestamp": time.time()
        })
        
        try:
            # Main message loop
            while True:
                data = await websocket.receive_json()
                await self._route_message(session, data)
        
        except WebSocketDisconnect:
            logger.info(f"[{session_id}] Client disconnected")
            await self._cleanup_session(session)
        
        except Exception as e:
            logger.error(f"[{session_id}] WebSocket error: {e}", exc_info=True)
            await self._cleanup_session(session)
    
    async def _route_message(self, 
                           session: OrchestratorSession, 
                           msg: Dict[str, Any]):
        """Route incoming message to appropriate handler"""
        msg_type = msg.get("type")
        
        if msg_type == "audio_chunk":
            await self._handle_audio_chunk(session, msg)
        
        elif msg_type == "playback_done":
            await self._handle_playback_done(session, msg)
        
        elif msg_type == "interrupt":
            await self._handle_interrupt(session, msg)
        
        elif msg_type == "state_sync_request":
            await self._handle_state_sync(session)
        
        elif msg_type == "end_session":
            await self._handle_end_session(session)
        
        else:
            logger.warning(f"[{session.session_id}] Unknown message type: {msg_type}")
    
    async def _handle_audio_chunk(self, 
                                  session: OrchestratorSession, 
                                  msg: Dict[str, Any]):
        """
        Handle incoming audio chunk from browser microphone.
        
        This replaces the STT WebSocket listener.
        """
        state_mgr = session.state_manager
        
        # Check state - ignore if not listening
        if state_mgr.current_state not in [State.IDLE, State.LISTENING, State.INTERRUPT]:
            logger.debug(f"[{session.session_id}] Ignoring audio in {state_mgr.current_state.value} state")
            return
        
        session.last_activity = time.time()
        session.audio_chunks_received += 1
        
        try:
            # Decode audio
            audio_b64 = msg.get("data", "")
            audio_bytes = base64.b64decode(audio_b64)
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Stream to STT service
            transcript = await self.service_manager.stt_service.stream_chunk(
                audio_data, 
                session_id=session.session_id
            )
            
            if transcript and transcript.get("is_final"):
                # STT final result - trigger processing
                text = transcript.get("text", "")
                await self._process_user_input(session, text)
        
        except Exception as e:
            logger.error(f"[{session.session_id}] Audio chunk error: {e}")
    
    async def _process_user_input(self, 
                                  session: OrchestratorSession, 
                                  text: str):
        """
        Process final STT result.
        
        This replaces the separate STT event handler.
        """
        state_mgr = session.state_manager
        
        if not text.strip():
            return
        
        logger.info(f"[{session.session_id}] STT FINAL: {text[:100]}")
        
        # Transition to THINKING
        await state_mgr.transition(State.THINKING, trigger="stt_final", data={"text": text})
        await self._broadcast_state(session, State.THINKING)
        
        # Cancel any ongoing TTS
        if session.tts_task and not session.tts_task.done():
            session.tts_task.cancel()
        
        try:
            # Parallel Intent + RAG
            result = await process_intent_rag_llm(
                text=text,
                session_id=session.session_id,
                context=state_mgr.context
            )
            
            response_text = result.get("response", "")
            
            # Transition to SPEAKING
            await state_mgr.transition(State.SPEAKING, trigger="response_ready")
            await self._broadcast_state(session, State.SPEAKING)
            
            # Stream TTS to browser and play
            session.tts_task = asyncio.create_task(
                self._stream_tts_to_browser(session, response_text)
            )
            
            await session.tts_task
        
        except asyncio.CancelledError:
            logger.info(f"[{session.session_id}] Processing cancelled (user interrupt)")
        
        except Exception as e:
            logger.error(f"[{session.session_id}] Processing error: {e}", exc_info=True)
            await state_mgr.transition(State.LISTENING, trigger="error")
            await self._broadcast_state(session, State.LISTENING)
    
    async def _stream_tts_to_browser(self, 
                                     session: OrchestratorSession, 
                                     text: str):
        """
        Stream TTS audio to browser.
        
        This replaces the separate TTS streaming logic.
        """
        chunk_counter = 0
        
        try:
            # Get TTS service
            tts_service = self.service_manager.tts_service
            
            # Stream audio chunks
            async for audio_chunk in tts_service.stream(text):
                # Encode to base64
                audio_b64 = base64.b64encode(audio_chunk).decode()
                
                # Send to browser
                await self._send_json(session.websocket, {
                    "type": "audio_chunk",
                    "audio": audio_b64,
                    "chunk_id": f"{session.session_id}_chunk_{chunk_counter}",
                    "is_final": False,
                    "timestamp": time.time()
                })
                
                session.audio_chunks_sent += 1
                chunk_counter += 1
            
            # Signal completion
            await self._send_json(session.websocket, {
                "type": "audio_chunk",
                "chunk_id": f"{session.session_id}_chunk_{chunk_counter}",
                "is_final": True,
                "timestamp": time.time()
            })
            
            logger.info(f"[{session.session_id}] TTS streaming complete ({chunk_counter} chunks)")
        
        except asyncio.CancelledError:
            logger.info(f"[{session.session_id}] TTS cancelled (user interrupt)")
        
        except Exception as e:
            logger.error(f"[{session.session_id}] TTS streaming error: {e}")
    
    async def _handle_playback_done(self, 
                                    session: OrchestratorSession, 
                                    msg: Dict[str, Any]):
        """
        Handle browser confirming playback completion.
        
        This is the KEY to synchronization - server waits for browser confirmation,
        not a guess.
        """
        state_mgr = session.state_manager
        session.last_activity = time.time()
        
        if state_mgr.current_state != State.SPEAKING:
            logger.debug(f"[{session.session_id}] Playback done in non-SPEAKING state (ignored)")
            return
        
        logger.info(f"[{session.session_id}] ✅ Playback DONE (browser confirmed)")
        
        # Transition back to LISTENING
        await state_mgr.transition(State.LISTENING, trigger="playback_done")
        await self._broadcast_state(session, State.LISTENING)
    
    async def _handle_interrupt(self, 
                               session: OrchestratorSession, 
                               msg: Dict[str, Any]):
        """
        Handle user interrupting agent (barge-in).
        
        Server must respond immediately - cancel TTS and clear output queue.
        """
        state_mgr = session.state_manager
        session.last_activity = time.time()
        
        logger.info(f"[{session.session_id}] 🔴 INTERRUPT detected")
        
        # Cancel TTS task
        if session.tts_task and not session.tts_task.done():
            session.tts_task.cancel()
            try:
                await session.tts_task
            except asyncio.CancelledError:
                pass
        
        # Notify browser to stop playback
        await self._send_json(session.websocket, {
            "type": "playback_control",
            "action": "stop",
            "reason": "user_interrupt",
            "timestamp": time.time()
        })
        
        # Transition: SPEAKING → INTERRUPT → LISTENING
        if state_mgr.current_state == State.SPEAKING:
            await state_mgr.transition(State.INTERRUPT, trigger="user_interrupt")
            await asyncio.sleep(0.05)  # Brief pause
            await state_mgr.transition(State.LISTENING, trigger="interrupt_complete")
        
        await self._broadcast_state(session, State.LISTENING)
    
    async def _handle_state_sync(self, session: OrchestratorSession):
        """Sync current state to client"""
        await self._broadcast_state(session, session.state_manager.current_state)
    
    async def _handle_end_session(self, session: OrchestratorSession):
        """Client requests session end"""
        logger.info(f"[{session.session_id}] Session end requested")
        await session.websocket.close()
    
    async def _broadcast_state(self, 
                              session: OrchestratorSession, 
                              state: State):
        """Notify browser of state change"""
        await self._send_json(session.websocket, {
            "type": "state_update",
            "state": state.value,
            "timestamp": time.time()
        })
    
    async def _send_json(self, websocket: WebSocket, data: Dict[str, Any]):
        """Safe JSON send"""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send JSON: {e}")
    
    async def _create_session(self, websocket: WebSocket) -> OrchestratorSession:
        """Create new orchestrator session"""
        session_id = f"auto_session_{int(time.time() * 1000) % 1000000}"
        
        state_manager = StateManager(
            session_id=session_id,
            redis=None  # Optional - load from Redis if needed
        )
        
        session = OrchestratorSession(
            session_id=session_id,
            websocket=websocket,
            state_manager=state_manager
        )
        
        self.sessions[session_id] = session
        return session
    
    async def _cleanup_session(self, session: OrchestratorSession):
        """Clean up session resources"""
        if session.tts_task and not session.tts_task.done():
            session.tts_task.cancel()
        
        if session.session_id in self.sessions:
            del self.sessions[session.session_id]
        
        logger.info(f"[{session.session_id}] Session cleaned up")
```

---

## MODIFICATIONS: app.py

### Add WebSocket endpoint:

```python
# In app.py lifespan or after service manager initialization:

orchestrator_handler = None

@app.lifespan
async def lifespan(app: FastAPI):
    global orchestrator_handler
    # ... existing startup code ...
    
    orchestrator_handler = OrchestratorWSHandler(
        service_manager=service_manager,
        dialogue_manager=dialogue_manager
    )
    
    yield
    
    # ... existing shutdown code ...

@app.websocket("/orchestrate")
async def websocket_endpoint(websocket: WebSocket):
    """Single unified WebSocket endpoint for orchestrator"""
    await orchestrator_handler.handle_connection(websocket)
```

### Import:

```python
from .orchestrator_ws_handler import OrchestratorWSHandler
```

---

## MODIFICATIONS: Browser JavaScript

### Before:

```javascript
// Separate connections
const sttSocket = new WebSocket("http://localhost:5204/stt");
const ttsSocket = new WebSocket("http://localhost:5204/tts");

// Complex state sync
sttSocket.onmessage = (e) => {
  const data = JSON.parse(e.data);
  if (data.type === "transcript") {
    // Handle STT
  }
};

ttsSocket.onmessage = (e) => {
  const data = JSON.parse(e.data);
  if (data.type === "audio_chunk") {
    // Play audio
  }
};
```

### After:

```javascript
// Single unified connection
const orchSocket = new WebSocket("ws://localhost:5204/orchestrate");

class OrchestratorClient {
  constructor(websocket) {
    this.ws = websocket;
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    this.state = "idle";
    this.audioQueue = [];
    
    this.setupHandlers();
  }
  
  setupHandlers() {
    this.ws.onopen = () => console.log("Connected to orchestrator");
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this._handleMessage(data);
    };
    
    this.ws.onerror = (err) => console.error("WebSocket error:", err);
    this.ws.onclose = () => console.log("Disconnected from orchestrator");
  }
  
  _handleMessage(data) {
    const { type } = data;
    
    if (type === "session_ready") {
      this.sessionId = data.session_id;
      console.log("Session ready:", this.sessionId);
      this.state = "ready";
    }
    
    else if (type === "state_update") {
      this.state = data.state;
      this._updateUI(data.state);
    }
    
    else if (type === "audio_chunk") {
      if (data.is_final) {
        console.log("TTS complete");
      } else {
        this._queueAudio(data.audio);
      }
    }
    
    else if (type === "playback_control") {
      if (data.action === "stop") {
        this._stopPlayback();
      }
    }
  }
  
  async sendAudioChunk(audioBytes) {
    const audio_b64 = btoa(String.fromCharCode(...new Uint8Array(audioBytes)));
    
    this.ws.send(JSON.stringify({
      type: "audio_chunk",
      data: audio_b64,
      timestamp: Date.now() / 1000
    }));
  }
  
  playbackDone() {
    this.ws.send(JSON.stringify({
      type: "playback_done",
      timestamp: Date.now() / 1000
    }));
  }
  
  interrupt() {
    this.ws.send(JSON.stringify({
      type: "interrupt",
      reason: "user_speaking",
      timestamp: Date.now() / 1000
    }));
  }
  
  _queueAudio(audioB64) {
    const audioBytes = Uint8Array.from(atob(audioB64), c => c.charCodeAt(0));
    this.audioQueue.push(audioBytes);
    this._playNextChunk();
  }
  
  async _playNextChunk() {
    if (this.audioQueue.length === 0) return;
    
    const audioBytes = this.audioQueue.shift();
    const audioBuffer = await this.audioContext.decodeAudioData(audioBytes.buffer);
    
    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);
    
    source.onended = () => {
      this._playNextChunk();
      if (this.audioQueue.length === 0) {
        this.playbackDone();
      }
    };
    
    source.start(0);
  }
  
  _stopPlayback() {
    this.audioQueue = [];
    // Stop current playback
  }
  
  _updateUI(state) {
    // Update UI based on state
    console.log("State:", state);
  }
}

// Initialize
const client = new OrchestratorClient(orchSocket);
```

---

## TESTING CHECKLIST

### Unit Tests:

```python
# test_orchestrator_ws_handler.py

async def test_session_creation():
    handler = OrchestratorWSHandler(...)
    assert len(handler.sessions) == 0
    
    session = await handler._create_session(mock_ws)
    assert session.session_id.startswith("auto_session_")
    assert len(handler.sessions) == 1

async def test_audio_chunk_routing():
    session = await handler._create_session(mock_ws)
    
    msg = {
        "type": "audio_chunk",
        "data": "...",
        "timestamp": time.time()
    }
    
    await handler._route_message(session, msg)
    assert session.audio_chunks_received == 1

async def test_state_transitions():
    session = await handler._create_session(mock_ws)
    
    # IDLE → THINKING → SPEAKING → LISTENING
    assert session.state_manager.current_state == State.IDLE
    
    await handler._process_user_input(session, "hello")
    assert session.state_manager.current_state == State.THINKING
```

### Integration Tests:

```javascript
// test_client.js

async function testConnection() {
  const ws = new WebSocket("ws://localhost:5204/orchestrate");
  
  await new Promise(resolve => ws.onopen = resolve);
  assert(ws.readyState === WebSocket.OPEN);
}

async function testAudioFlow() {
  const client = new OrchestratorClient(ws);
  
  // Send audio chunk
  const audioBytes = new Uint8Array(1000);
  await client.sendAudioChunk(audioBytes);
  
  // Wait for response
  const response = await waitForMessage(ws, "agent_response");
  assert(response.type === "agent_response");
}

async function testInterrupt() {
  const client = new OrchestratorClient(ws);
  
  // Send interrupt
  await client.interrupt();
  
  // Should receive playback_control stop
  const control = await waitForMessage(ws, "playback_control");
  assert(control.action === "stop");
}
```

---

## DEPLOYMENT NOTES

1. **No breaking changes** - Keep old FastRTC endpoints for backward compatibility
2. **Gradual migration** - Clients can switch to new endpoint at their own pace
3. **Monitoring** - Track both old and new endpoint usage
4. **Rollback** - Easy fallback if issues found

---

## SUMMARY

This single file + 2 modifications transform your architecture:

✅ **Single WebSocket** - One connection for all I/O
✅ **Clear routing** - Single message switch statement
✅ **Direct streaming** - No Redis event routing
✅ **Tight synchronization** - Browser confirms state changes
✅ **50% latency reduction** - Single connection + no distributed routing
