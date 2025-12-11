# IMPLEMENTATION CHECKLIST: Audio Playback Tracking

## ✅ STEP 1: Create JavaScript AudioPlaybackManager (10 min)

**File:** `unified_fastrtc.py` → Gradio interface template, or external `.js`

**Task:**
- [ ] Copy `AudioPlaybackManager` class from guide (Part 2)
- [ ] Initialize with: `audioManager = new AudioPlaybackManager(websocket)`
- [ ] Verify audio context sample rate matches your TTS (24000Hz)
- [ ] Test browser console: no JavaScript errors

**Validation:**
```javascript
// In browser console
audioManager.init()
// Should see: 🎵 Audio Context initialized | Sample Rate: 24000Hz
```

---

## ✅ STEP 2: Update Server WebSocket Handler (15 min)

**File:** `app.py` → `@app.websocket("/orchestrate")`

**Task:**
- [ ] Find: `async def websocket_endpoint(websocket, session_id)`
- [ ] Add new handler: `async def on_playback_event()`
- [ ] Route incoming messages:
  ```python
  data = await websocket.receive_json()
  
  if data.get('type') == 'playback_event':
      await self.on_playback_event(session_id, data)
  ```
- [ ] Import `State` enum if not already

**Key implementation:**
```python
elif event_type == 'playback_done':
    # Only transition when BROWSER confirms
    if state_mgr.current_state == State.SPEAKING:
        await state_mgr.transition(
            State.LISTENING,
            trigger="playback_done"
        )
```

---

## ✅ STEP 3: Update TTS Streaming (10 min)

**File:** `app.py` → audio streaming function

**Task:**
- [ ] Find: function that sends TTS chunks to browser
- [ ] Add `chunk_id` to each message:
  ```python
  chunk_id = f"{session_id}_chunk_{counter}"
  
  await websocket.send_json({
      "type": "tts_chunk",
      "chunk_id": chunk_id,  # NEW
      "audio_data": base64_audio,
      "chunk_sequence": counter,
  })
  ```
- [ ] Increment counter for each chunk
- [ ] Send `"type": "tts_complete"` when done

---

## ✅ STEP 4: Remove Old Audio Completion Triggers (10 min)

**File:** `app.py` → search for these patterns

**Task:**
- [ ] Find and DELETE/COMMENT:
  ```python
  # OLD (REMOVE):
  await asyncio.sleep(tts_duration)
  state_mgr.transition(State.LISTENING, ...)  # Guessed
  ```

- [ ] Find and DELETE/COMMENT:
  ```python
  # OLD (REMOVE):
  if chunks_sent == total_chunks:
      await send_playback_done(...)  # Too early
  ```

- [ ] Find and DELETE/COMMENT:
  ```python
  # OLD (REMOVE):
  # Use only browser events, not heuristics
  # await state_mgr.transition(...)  # From stream completion
  ```

**Search strings:**
- `audio_complete`
- `playback_complete` (server-side guess)
- `asyncio.sleep` (before state transition)
- `chunks_sent == total`

---

## ✅ STEP 5: Test Each Component (30 min)

### Test 1: Audio Context Initialization
```bash
# Open browser: http://localhost:2004/fastrtc
# Open DevTools (F12) → Console

# Should show (no errors):
# 🎵 Audio Context initialized | Sample Rate: 24000Hz
```

### Test 2: Chunk Queueing
```bash
# Say something to trigger TTS
# Console should show:
# 📦 Chunk queued: auto_session_XXXX_chunk_0 (4096 bytes)
# 📦 Chunk queued: auto_session_XXXX_chunk_1 (4096 bytes)
```

### Test 3: Playback Events
```bash
# Console should show:
# 📤 Sending: playback_started
# ✅ Chunk complete: auto_session_XXXX_chunk_0
# ✅ Chunk complete: auto_session_XXXX_chunk_1
# 📤 Sending: playback_done
```

### Test 4: Server Logs
```bash
# docker-compose logs orchestrator | grep -i playback

# Should show:
# [session] 🔊 Playback started on browser
# [session] ✅ Chunk complete: ...
# [session] ✅ Playback DONE on browser | Duration: 2500ms | Chunks: 3
# [session] 🟡 SPEAKING → 🔵 LISTENING (playback_done)
```

### Test 5: State Transitions (Full Cycle)
```
Expected flow:

[session] 🟢 IDLE
[session] 🔵 LISTENING (client_connected)
[session] 🟡 THINKING (stt_final)
[session] 🔴 SPEAKING (response_ready)
[session] 🔊 Playback started on browser
[session] ✅ Chunk complete x5
[session] 🔵 LISTENING (playback_done) ← NOT during speaking
[session] Ready for next turn
```

---

## ✅ STEP 6: Integration Tests (20 min)

### Test Scenario 1: Normal Flow (No Interruption)
```
1. Browser connects
2. Say: "hello"
3. Listen to agent response completely
4. Say something else
5. Check logs for clean state transitions
```

**Expected:**
- No overlapping "SPEAKING" logs
- State transitions clean
- Microphone opens AFTER audio ends

### Test Scenario 2: Barge-in (Interruption)
```
1. Agent is speaking (SPEAKING state)
2. Start speaking yourself (during agent response)
3. Listen for smooth transition to LISTENING
```

**Expected:**
```
[session] 🔴 SPEAKING
[session] 🔊 Playback started
[session] ⛔ Playback stopped (barge-in)
[session] 🟣 INTERRUPT (wait 50ms)
[session] 🔵 LISTENING
```

### Test Scenario 3: Timeout (No Response)
```
1. Browser connects
2. Wait 10+ seconds without speaking
3. Agent should play timeout message
4. Back to LISTENING
```

**Expected:**
```
[session] 🔵 LISTENING (waiting)
[timeout] ⏱️ TIMEOUT DETECTED
[session] 🔴 SPEAKING (timeout message)
[session] ✅ Playback DONE
[session] 🔵 LISTENING
```

---

## ✅ FINAL VALIDATION

### Logs Should Look Like:
```
✅ Clean state transitions (no invalid ones)
✅ Playback events from browser (📤 Sending)
✅ Server acknowledges (🔊 Playback started)
✅ State changes only on PLAYBACK_DONE
✅ No "Cannot transition" errors
✅ No guessing/heuristic transitions
```

### Logs Should NOT Look Like:
```
❌ State transitions during audio playback
❌ "audio_complete" (server-side guess)
❌ "asyncio.sleep" before transitions
❌ Chunk counting for completion
❌ Microphone opening during SPEAKING
❌ "Invalid transition" warnings
```

---

## 🆘 DEBUGGING TIPS

**Problem: State still transitions early**
→ Check if old TTS completion logic still exists
→ Verify server is actually consuming `playback_done` events
→ Check WebSocket message format

**Problem: Browser doesn't send playback events**
→ Check browser console for JavaScript errors
→ Verify WebSocket connection is open
→ Check if `audioManager.sendPlaybackEvent()` is called

**Problem: Audio stuttering/gaps**
→ Audio queue is depleting too fast
→ Check chunk size and decode time
→ Monitor browser CPU usage

**Problem: Playback stops unexpectedly**
→ Check for errors in browser console
→ Verify audio buffer decode succeeds
→ Check gainNode is connected to destination

---

## TIME ESTIMATE

| Task | Time | Status |
|------|------|--------|
| Step 1: JavaScript | 10 min | |
| Step 2: Server handler | 15 min | |
| Step 3: TTS streaming | 10 min | |
| Step 4: Remove old logic | 10 min | |
| Step 5: Component tests | 30 min | |
| Step 6: Integration tests | 20 min | |
| **TOTAL** | **95 min** | |

**Expected Result:** Clean logs, accurate state transitions, microphone opens AFTER audio finishes.

---

## NEXT STEPS AFTER THIS FIX

Once playback tracking works:

1. **Phase 2:** Parallel Intent+RAG processing
2. **Phase 3:** Barge-in signal detection
3. **Phase 4:** Latency optimization (overlapped streaming)

All depend on correct state transitions (which this fix provides).
