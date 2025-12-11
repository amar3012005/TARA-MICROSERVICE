# BROWSER CLIENT AUDIO PLAYBACK TRACKING FIX

## The Root Problem

Your orchestrator receives `audio_complete` events **TOO EARLY**. 

The flow is currently:
```
Server sends: audio chunks 1,2,3,4,5
Browser receives: chunk 1, 2, 3
Browser UI says: "streaming complete" → sends PLAYBACK_DONE to server
Browser still playing: 4, 5 (queued)
Server transitions: SPEAKING → LISTENING
Microphone opens while agent still speaking!
```

**Solution:** Browser must send playback events only when **audio actually finishes in browser**, not when streaming finishes.

---

## PART 1: Audio Context Tracking (Client-Side)

Add this to `unified_fastrtc.py` **WebRTC connection handler**:

```python
# In the UnifiedFastRTCHandler class or audio player section

class AudioPlaybackTracker:
    """Tracks when audio chunks are actually played, not just received"""
    
    def __init__(self):
        self.audio_context = None
        self.current_playback_id = None
        self.playback_start_time = None
        self.expected_end_time = None
        self.is_playing = False
        self.played_chunks = []  # Track which chunks completed
        self.playback_events_sent = set()  # Prevent duplicate events
        
    async def init_audio_context(self):
        """Initialize Web Audio API context"""
        # This runs on browser side (JavaScript)
        # But we track state server-side
        self.audio_context = {
            "sample_rate": 24000,  # Your output sample rate
            "channels": 1,
        }
        
    async def queue_audio_chunk(self, chunk_data: bytes, chunk_id: str) -> dict:
        """
        Queue audio chunk for playback.
        Returns: {"chunk_id": chunk_id, "duration_ms": duration, "queued_at": timestamp}
        """
        import time
        
        # Calculate expected duration based on chunk size
        # Formula: duration_ms = (chunk_size_bytes / 2) / (sample_rate / 1000)
        chunk_samples = len(chunk_data) // 2  # 16-bit = 2 bytes per sample
        duration_ms = (chunk_samples / self.audio_context["sample_rate"]) * 1000
        
        event = {
            "chunk_id": chunk_id,
            "size_bytes": len(chunk_data),
            "duration_ms": duration_ms,
            "queued_at": time.time(),
        }
        
        return event
    
    async def update_playback_status(self, status: str, chunk_id: str = None) -> dict:
        """
        Update playback status from browser.
        status: "started", "chunk_playing", "chunk_complete", "playback_done"
        """
        import time
        
        if status == "started":
            self.is_playing = True
            self.playback_start_time = time.time()
            return {
                "event": "playback_started",
                "timestamp": self.playback_start_time,
            }
        
        elif status == "chunk_playing":
            # Browser is actively playing this chunk
            return {
                "event": "chunk_playing",
                "chunk_id": chunk_id,
                "timestamp": time.time(),
            }
        
        elif status == "chunk_complete":
            # Browser finished playing specific chunk
            self.played_chunks.append(chunk_id)
            return {
                "event": "chunk_complete",
                "chunk_id": chunk_id,
                "timestamp": time.time(),
            }
        
        elif status == "playback_done":
            # ALL audio finished playing
            self.is_playing = False
            end_time = time.time()
            actual_duration_ms = (end_time - self.playback_start_time) * 1000 \
                if self.playback_start_time else 0
            
            self.playback_events_sent.add("playback_done")
            
            return {
                "event": "playback_done",
                "timestamp": end_time,
                "actual_duration_ms": actual_duration_ms,
                "chunks_played": len(self.played_chunks),
            }
```

---

## PART 2: JavaScript (Browser) - Accurate Playback Tracking

Replace audio playback in your Gradio FastRTC JavaScript with this:

```javascript
// In your unified_fastrtc.html or JavaScript handler

class AudioPlaybackManager {
  constructor(websocket) {
    this.ws = websocket;
    this.audioContext = null;
    this.audioQueue = [];
    this.isPlaying = false;
    this.gainNode = null;
    this.currentGain = 1.0;
    this.currentBuffer = null;
    this.currentSource = null;
    this.playbackStartTime = null;
    this.chunkCounter = 0;
    this.startPlaybackTimer = null;
  }

  async init() {
    // Initialize Web Audio API
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 24000,  // Match your TTS output
    });
    
    this.gainNode = this.audioContext.createGain();
    this.gainNode.connect(this.audioContext.destination);
    this.currentGain = 0.8;  // Reasonable default volume
    this.gainNode.gain.value = this.currentGain;
    
    console.log(`🎵 Audio Context initialized | Sample Rate: ${this.audioContext.sampleRate}Hz`);
  }

  async queueChunk(audioData, chunkId) {
    /**
     * Add audio chunk to queue.
     * audioData: Uint8Array or Base64 string
     * chunkId: Unique identifier for this chunk
     */
    
    // Decode base64 if needed
    let uint8Array;
    if (typeof audioData === 'string') {
      uint8Array = new Uint8Array(
        atob(audioData).split('').map(c => c.charCodeAt(0))
      );
    } else {
      uint8Array = audioData;
    }
    
    // Queue for playback
    this.audioQueue.push({
      id: chunkId,
      data: uint8Array,
      queuedAt: Date.now(),
    });
    
    console.log(`📦 Chunk queued: ${chunkId} (${uint8Array.length} bytes)`);
    
    // Start playback if not already playing
    if (!this.isPlaying && this.audioQueue.length > 0) {
      await this.playNextChunk();
    }
    
    return {
      chunk_id: chunkId,
      queued: true,
      timestamp: Date.now(),
    };
  }

  async playNextChunk() {
    /**
     * Play the next chunk in queue.
     * CRITICAL: Only send PLAYBACK_DONE when ALL chunks finish.
     */
    
    if (this.audioQueue.length === 0) {
      // Queue empty, playback complete
      if (this.isPlaying) {
        this.isPlaying = false;
        
        // Send final PLAYBACK_DONE event
        this.sendPlaybackEvent('playback_done', {
          actual_duration_ms: Date.now() - this.playbackStartTime,
          chunks_played: this.chunkCounter,
        });
      }
      return;
    }
    
    // Dequeue next chunk
    const chunk = this.audioQueue.shift();
    this.chunkCounter++;
    
    console.log(`▶️  Playing chunk: ${chunk.id} (#${this.chunkCounter})`);
    
    try {
      // Decode audio data
      const audioBuffer = await this.audioContext.decodeAudioData(
        chunk.data.buffer.slice(0, chunk.data.byteLength)
      );
      
      // Create playback source
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.gainNode);
      this.currentSource = source;
      
      // Set up playback completion handler
      source.onended = async () => {
        console.log(`✅ Chunk complete: ${chunk.id}`);
        
        // Send intermediate chunk complete event
        this.sendPlaybackEvent('chunk_complete', {
          chunk_id: chunk.id,
          buffer_duration_ms: audioBuffer.duration * 1000,
        });
        
        // Play next chunk
        await this.playNextChunk();
      };
      
      // If first chunk, send playback_started
      if (this.chunkCounter === 1) {
        this.isPlaying = true;
        this.playbackStartTime = Date.now();
        this.sendPlaybackEvent('playback_started');
      }
      
      // Start playback
      source.start(0);
      
    } catch (error) {
      console.error(`❌ Audio decode error: ${error}`);
      this.sendPlaybackEvent('playback_error', {
        chunk_id: chunk.id,
        error: error.message,
      });
      
      // Continue with next chunk
      await this.playNextChunk();
    }
  }

  sendPlaybackEvent(eventType, data = {}) {
    /**
     * Send playback event to orchestrator via WebSocket
     */
    
    const event = {
      type: 'playback_event',
      event_type: eventType,  // playback_started, playback_done, etc.
      timestamp: Date.now(),
      ...data,
    };
    
    console.log(`📤 Sending: ${eventType} →`, event);
    
    this.ws.send(JSON.stringify(event));
  }

  stop() {
    /**
     * Stop playback (e.g., for barge-in)
     */
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch (e) {
        console.warn(`Could not stop source: ${e}`);
      }
    }
    
    this.audioQueue = [];  // Clear queue
    this.isPlaying = false;
    this.chunkCounter = 0;
    
    console.log(`⛔ Audio playback stopped`);
    
    this.sendPlaybackEvent('playback_stopped');
  }

  setVolume(level) {
    /**
     * Set output volume (0.0 to 1.0)
     */
    this.currentGain = Math.max(0, Math.min(1, level));
    if (this.gainNode) {
      this.gainNode.gain.value = this.currentGain;
    }
    console.log(`🔊 Volume set to ${(this.currentGain * 100).toFixed(0)}%`);
  }
}

// ============================================================================
// USAGE IN YOUR GRADIO/FASTRTC INTERFACE
// ============================================================================

async function setupAudioPlayback(websocketConnection) {
  const audioManager = new AudioPlaybackManager(websocketConnection);
  await audioManager.init();
  
  // When server sends audio chunks
  websocketConnection.onmessage = async (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'tts_chunk') {
      // Server sent audio chunk
      const audioBase64 = data.audio_data;
      const chunkId = data.chunk_id || `chunk_${Date.now()}`;
      
      await audioManager.queueChunk(audioBase64, chunkId);
      
    } else if (data.type === 'stop_playback') {
      // Barge-in: stop current playback
      audioManager.stop();
    }
  };
  
  return audioManager;
}
```

---

## PART 3: Server-Side Event Handling (app.py)

Update your event handler to properly consume playback events:

```python
# In app.py, in the WebSocket handler

async def on_playback_event(self, session_id: str, event_data: dict):
    """
    Handle playback events from browser.
    These are more reliable than guessing completion.
    """
    
    event_type = event_data.get('event_type')
    timestamp = event_data.get('timestamp')
    
    session = self.active_sessions.get(session_id)
    if not session:
        logger.error(f"Playback event for unknown session: {session_id}")
        return
    
    state_mgr = session['state_manager']
    
    if event_type == 'playback_started':
        # Audio began playing on browser
        logger.info(f"[{session_id}] 🔊 Playback started on browser")
        state_mgr.update_last_activity()
        
    elif event_type == 'chunk_complete':
        # One chunk finished (intermediate)
        chunk_id = event_data.get('chunk_id')
        logger.debug(f"[{session_id}] ✅ Chunk complete: {chunk_id}")
        state_mgr.update_last_activity()
        
    elif event_type == 'playback_done':
        # **CRITICAL**: ALL audio finished on browser
        actual_duration_ms = event_data.get('actual_duration_ms', 0)
        chunks_played = event_data.get('chunks_played', 0)
        
        logger.info(
            f"[{session_id}] ✅ Playback DONE on browser | "
            f"Duration: {actual_duration_ms:.0f}ms | "
            f"Chunks: {chunks_played}"
        )
        
        # Only transition state when browser confirms playback done
        current_state = state_mgr.current_state
        
        if current_state == State.SPEAKING:
            # Expected: SPEAKING → LISTENING
            await state_mgr.transition(
                State.LISTENING,
                trigger="playback_done",
                data={"duration_ms": actual_duration_ms}
            )
        else:
            logger.warning(
                f"[{session_id}] Unexpected state for playback_done: "
                f"{current_state} (expected SPEAKING)"
            )
        
        state_mgr.update_last_activity()
        
    elif event_type == 'playback_stopped':
        # Browser stopped playback (barge-in or manual stop)
        logger.info(f"[{session_id}] ⛔ Playback stopped (barge-in?)")
        state_mgr.update_last_activity()
        
    elif event_type == 'playback_error':
        # Audio decode error
        error = event_data.get('error')
        chunk_id = event_data.get('chunk_id')
        logger.error(
            f"[{session_id}] ❌ Audio error on chunk {chunk_id}: {error}"
        )
```

---

## PART 4: Remove Old Audio Completion Triggers

**DELETE or DISABLE these in app.py:**

```python
# ❌ REMOVE: These were sending PLAYBACK_DONE too early

# OLD: Guessing based on TTS completion
# await send_playback_done_event(session_id)

# OLD: Timeout-based
# await asyncio.sleep(expected_tts_duration)
# await send_playback_done_event(session_id)

# OLD: Chunk counting
# if chunks_sent == total_chunks:
#     await send_playback_done_event(session_id)
```

---

## PART 5: TTS Streaming - Include Chunk IDs

Update TTS streaming to send chunk identifiers:

```python
# In stream_tts_audio() or TTS handler

async def stream_tts_audio(self, session_id: str, text: str, websocket):
    """Stream TTS audio with proper chunk tracking"""
    
    chunk_counter = 0
    
    async for audio_chunk in tts_service.stream(text):
        chunk_id = f"{session_id}_chunk_{chunk_counter}"
        chunk_counter += 1
        
        # Encode to base64 for WebSocket
        audio_b64 = base64.b64encode(audio_chunk).decode()
        
        # Send to browser with chunk ID
        await websocket.send_json({
            "type": "tts_chunk",
            "chunk_id": chunk_id,
            "audio_data": audio_b64,
            "chunk_sequence": chunk_counter,
            "is_final": False,
        })
        
        logger.debug(f"[{session_id}] Sent chunk {chunk_counter}")
        
        await asyncio.sleep(0)  # Yield to other tasks
    
    # Send final marker
    await websocket.send_json({
        "type": "tts_complete",
        "total_chunks": chunk_counter,
        "is_final": True,
    })
    
    logger.info(f"[{session_id}] TTS streaming complete ({chunk_counter} chunks)")
```

---

## PART 6: Testing

Test the playback tracking:

```bash
# 1. Start orchestrator
docker-compose up orchestrator

# 2. Open browser: http://localhost:2004/fastrtc

# 3. Check logs for proper sequence:

# Expected logs:
# [session] 📤 Sending: playback_started
# [session] ✅ Chunk complete: chunk_0
# [session] ✅ Chunk complete: chunk_1
# [session] ✅ Chunk complete: chunk_2
# [session] ✅ Playback DONE on browser | Duration: 2500ms | Chunks: 3
# [session] 🟡 SPEAKING → 🔵 LISTENING (playback_done)

# ❌ BAD logs (old behavior):
# [session] Transition to LISTENING (guessed)
# [session] Browser still playing chunk 3!
```

---

## KEY DIFFERENCES: BEFORE vs AFTER

| Aspect | BEFORE (Wrong) | AFTER (Fixed) |
|--------|---|---|
| **Playback tracking** | Server guesses | Browser reports actual |
| **State transition timing** | Too early | When audio actually finishes |
| **Events sent** | Based on chunk count | One final PLAYBACK_DONE |
| **Microphone opening** | During agent speech | After speech completes |
| **Latency** | Variable | Accurate (actual duration) |

---

## SUMMARY

1. **Add `AudioPlaybackManager` class** to JavaScript (handles actual playback)
2. **Remove guessing logic** from server (no more timeout-based transitions)
3. **Listen for `playback_done` events** from browser (this is the source of truth)
4. **Only transition SPEAKING→LISTENING** when browser sends `playback_done`
5. **Send chunk IDs** so browser can track which audio played

This makes the orchestrator follow **browser reality**, not server assumptions.
