# QUICK START: Audio Playback Tracking Fix

## The Problem in 10 Seconds

Your orchestrator thinks audio is done **before it actually finishes on the browser**. This causes:

```
Server: "Audio streaming complete!" → Transitions SPEAKING → LISTENING
Browser: "Still playing chunk 3..." 
Result: Microphone opens while agent speaking ❌
```

## The Solution in 10 Seconds

Browser tells server when audio actually finishes:

```
Browser: "Audio chunk 1 done" → Server: "Noted"
Browser: "Audio chunk 2 done" → Server: "Noted"
Browser: "Audio playback DONE" → Server: "OK, safe to open mic" ✅
```

---

## FILES YOU'LL MODIFY

```
app.py
├─ WebSocket handler
│  └─ Add: on_playback_event() handler
│  └─ Route: incoming playback_event messages
│
├─ TTS streaming function
│  └─ Add: chunk_id to each message
│  └─ Delete: Old completion triggers
│
└─ State transitions
   └─ Only SPEAKING→LISTENING on playback_done event
   └─ Delete: Guessing logic
```

---

## COPY-PASTE CODE SNIPPETS

### 1. Server: WebSocket Playback Event Handler

```python
# Add to app.py, in your WebSocket endpoint class

async def on_playback_event(self, session_id: str, event_data: dict):
    """Handle browser playback events"""
    import logging
    logger = logging.getLogger(__name__)
    
    event_type = event_data.get('event_type')
    session = self.active_sessions.get(session_id)
    
    if not session:
        return
    
    state_mgr = session['state_manager']
    
    if event_type == 'playback_done':
        # CRITICAL: Browser confirms audio finished
        logger.info(f"[{session_id}] ✅ Playback DONE (from browser)")
        
        # Only transition if currently SPEAKING
        if state_mgr.current_state.name == 'SPEAKING':
            await state_mgr.transition(
                State.LISTENING,
                trigger="playback_done",
                data={"source": "browser"}
            )
        
        state_mgr.update_last_activity()
    
    elif event_type == 'playback_started':
        logger.debug(f"[{session_id}] 🔊 Playback started")
        state_mgr.update_last_activity()
    
    elif event_type == 'chunk_complete':
        logger.debug(f"[{session_id}] ✅ Chunk: {event_data.get('chunk_id')}")
        state_mgr.update_last_activity()
```

### 2. Server: Route Playback Events in WebSocket

```python
# In your @app.websocket("/orchestrate") endpoint

@app.websocket("/orchestrate")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    session = self.active_sessions.get(session_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Route to handlers
            if data.get('type') == 'playback_event':
                await self.on_playback_event(session_id, data)
            
            elif data.get('type') == 'stt_event':
                await self.on_stt_event(session_id, data)
            
            # ... other handlers
    
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket error: {e}")
    finally:
        await websocket.close()
```

### 3. Server: Update TTS Streaming

```python
# In your TTS streaming function

async def stream_tts_to_browser(self, session_id: str, text: str, websocket):
    """Stream TTS with chunk IDs"""
    import base64
    
    chunk_counter = 0
    
    # Stream audio from TTS service
    async for audio_chunk in self.tts_service.stream(text):
        chunk_id = f"{session_id}_chunk_{chunk_counter}"
        
        # Encode to base64
        audio_b64 = base64.b64encode(audio_chunk).decode()
        
        # Send with chunk ID
        await websocket.send_json({
            "type": "tts_chunk",
            "chunk_id": chunk_id,            # NEW
            "audio_data": audio_b64,
            "chunk_sequence": chunk_counter,  # NEW
        })
        
        chunk_counter += 1
    
    # Signal TTS complete
    await websocket.send_json({
        "type": "tts_complete",
        "total_chunks": chunk_counter,
    })
```

### 4. Browser: JavaScript AudioPlaybackManager

```javascript
// Add to your HTML or external JS file

class AudioPlaybackManager {
    constructor(websocket) {
        this.ws = websocket;
        this.audioContext = null;
        this.audioQueue = [];
        this.isPlaying = false;
        this.chunkCounter = 0;
    }

    async init() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 24000,
        });
        console.log('🎵 Audio initialized | SR: 24000Hz');
    }

    async queueChunk(audioB64, chunkId) {
        const uint8 = new Uint8Array(atob(audioB64).split('').map(c => c.charCodeAt(0)));
        this.audioQueue.push({id: chunkId, data: uint8});
        
        if (!this.isPlaying) await this.playNextChunk();
    }

    async playNextChunk() {
        if (this.audioQueue.length === 0) {
            // No more chunks
            if (this.isPlaying) {
                this.isPlaying = false;
                this.sendEvent('playback_done');  // CRITICAL
            }
            return;
        }

        const chunk = this.audioQueue.shift();
        this.chunkCounter++;

        try {
            // Decode and play
            const audioBuffer = await this.audioContext.decodeAudioData(
                chunk.data.buffer.slice(0, chunk.data.byteLength)
            );
            
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            
            // When chunk finishes, play next
            source.onended = async () => {
                this.sendEvent('chunk_complete', {chunk_id: chunk.id});
                await this.playNextChunk();
            };
            
            if (this.chunkCounter === 1) {
                this.isPlaying = true;
                this.sendEvent('playback_started');
            }
            
            source.start(0);
        } catch (e) {
            console.error('Audio error:', e);
            await this.playNextChunk();
        }
    }

    sendEvent(eventType, data = {}) {
        this.ws.send(JSON.stringify({
            type: 'playback_event',
            event_type: eventType,
            timestamp: Date.now(),
            ...data,
        }));
    }
}
```

### 5. Browser: Setup in Your Interface

```javascript
// When page loads
let audioManager;

async function setupAudio(websocket) {
    audioManager = new AudioPlaybackManager(websocket);
    await audioManager.init();
    
    // Listen for TTS chunks from server
    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'tts_chunk') {
            audioManager.queueChunk(data.audio_data, data.chunk_id);
        }
    };
}
```

---

## DELETE THESE OLD PATTERNS

Search `app.py` and **DELETE** or **COMMENT OUT**:

```python
# ❌ OLD: Guessing audio duration
# await asyncio.sleep(expected_duration)
# await state_mgr.transition(State.LISTENING, ...)

# ❌ OLD: Counting chunks
# if chunks_sent == total_chunks:
#     await transition_to_listening(...)

# ❌ OLD: Timeout-based completion
# asyncio.create_task(audio_complete_timeout(10))
# await state_mgr.transition(...)

# ❌ OLD: Streaming completion without browser confirmation
# async for chunk in stream:
#     ...
# # DON'T transition here anymore!
# state_mgr.transition(State.LISTENING)
```

---

## VALIDATE WITH LOGS

After implementing, check these logs appear:

```
✅ [session] 📤 Sending: playback_started
✅ [session] ✅ Chunk complete: auto_session_XXXX_chunk_0
✅ [session] ✅ Chunk complete: auto_session_XXXX_chunk_1
✅ [session] ✅ Playback DONE on browser
✅ [session] 🔴 SPEAKING → 🔵 LISTENING (playback_done)
✅ [session] 🎤 Microphone OPEN
```

Don't see these? Check:
- [ ] JavaScript errors in browser console
- [ ] WebSocket connection still open
- [ ] Server is routing `playback_event` messages
- [ ] Old guessing logic is removed

---

## TIME & IMPACT

| Aspect | Before | After |
|--------|--------|-------|
| **Time to implement** | 2 hours | N/A |
| **Log clarity** | Chaotic | Clean |
| **State transitions** | Wrong | Correct |
| **Microphone timing** | During speech | After speech |
| **Reliability** | Guessing | Confirmed |

---

## NEXT: Test It

```bash
# 1. Implement all code changes above
# 2. Restart orchestrator: docker-compose up -d
# 3. Open browser: http://localhost:2004/fastrtc
# 4. Say something
# 5. Check logs: should see playback_done events
# 6. Repeat with interruptions
```

If logs show playback events and clean transitions → **✅ FIX COMPLETE**

If not → Check deleted old logic and WebSocket routing
