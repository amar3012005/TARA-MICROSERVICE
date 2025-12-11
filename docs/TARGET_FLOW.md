# CORRECT ORCHESTRATOR FLOW (Target State)

## Session Lifecycle (Clean Flow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BROWSER CONNECTS                                    │
│                    (WebSocket /orchestrate)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ CREATE SESSION                                                              │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ active_sessions[session_id] = {                                       │   │
│ │     "state_manager": StateManager(session_id),                       │   │
│ │     "websocket": websocket,                                          │   │
│ │     "current_task": None,          ← Track ONE task per session      │   │
│ │     "unified_handler": None,       ← For session routing              │   │
│ │     "fsm_task": None,              ← Event-driven FSM                │   │
│ │ }                                                                      │   │
│ └───────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🟢 STATE: IDLE                                                              │
│ Side effect: Open microphone (ready to listen)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌───────────────────────────────────────────┐
          │ UNIFIED FASTRTC CONNECTS                  │
          │ (Handles BOTH STT + TTS for this session) │
          │ handler_id = fastrtc_1234567890          │
          │ Maps to: active_sessions[session_id]     │
          └───────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ┌─────────────────────┐      ┌──────────────────────┐
        │ STT STREAM ACTIVE   │      │ TTS STREAM READY     │
        │ (mic audio in)      │      │ (agent audio out)    │
        └─────────────────────┘      └──────────────────────┘
                    │                           │
                    │ User speaks "hello"       │
                    │                           │
                    ▼                           │
┌─────────────────────────────────────────────────────────────────────────────┐
│ STT SERVICE DETECTS SPEECH (VAD)                                            │
│ Streams partial results:                                                    │
│   "he" → emit STT_PARTIAL                                                   │
│   "hell" → emit STT_PARTIAL                                                 │
│   "hello" → emit STT_FINAL (is_final=True)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
        ┌───────────────────────┐    ┌──────────────────────┐
        │ REDIS PUBSUB (legacy) │    │ REDIS STREAMS (new)  │
        │ Channel: stt_events   │    │ Stream: voice:stt:.. │
        └───────────────────────┘    └──────────────────────┘
                    │                            │
        ┌───────────┴────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ROUTE TO CORRECT ORCHESTRATOR SESSION                                       │
│                                                                             │
│ Algorithm:                                                                  │
│ 1. Check if event.session_id matches any active_sessions key                │
│    → Direct match? Use it!                                                  │
│                                                                             │
│ 2. Check if event.session_id is in UnifiedFastRTC.active_instances          │
│    → Get handler from registry                                              │
│    → Find which orchestrator session has this handler stored                │
│    → Match found? Use it!                                                   │
│                                                                             │
│ 3. No match?                                                                │
│    → Log ERROR with details                                                │
│    → Drop event                                                             │
│                                                                             │
│ Result: state_mgr = active_sessions[correct_session_id]["state_manager"]   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STT EVENT HANDLER (UNIFIED)                                                 │
│ Class: STTEventHandler                                                      │
│                                                                             │
│ def handle_stt_final(text, is_final, source):                               │
│     1. Validate (not empty, is_final=True)                                  │
│     2. Check state is LISTENING                                             │
│     3. Transition: LISTENING → THINKING                                     │
│     4. Start parallel Intent+RAG                                            │
│     5. Wait for result                                                      │
│     6. Transition: THINKING → SPEAKING                                      │
│     7. Return result to caller                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🔵 STATE: LISTENING                                                          │
│ Side effect:                                                                │
│   - Open microphone                                                         │
│   - Cancel any previous TTS task                                            │
│   - Clear text buffer                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⏱️ LATENCY: STT Fragment → STT Final ≈ 500-2000ms                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🟡 STATE: THINKING                                                           │
│ Side effect:                                                                │
│   - Gate microphone (stop new input)                                        │
│   - Stop TTS playback if still going                                        │
│   - Play "thinking" filler (if configured)                                  │
│                                                                             │
│ ⏱️ Duration: 50-200ms (just state change)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PARALLEL PROCESSING                                                         │
│ ┌────────────────────────────────────────────────────────────────────────┐  │
│ │ Intent Service (if not skipped)  │  RAG Service                        │  │
│ │ POST /intent                      │  POST /query                       │  │
│ │ text: "hello"                     │  text: "hello"                     │  │
│ │ → intent: "greeting"              │  → answer: "Hi! How can I help?"   │  │
│ │ ← 150-300ms                       │  ← 200-500ms                       │  │
│ └────────────────────────────────────────────────────────────────────────┘  │
│                         ▲                       ▲                            │
│                         └───────────┬───────────┘                            │
│                                     │                                        │
│                           (HAPPENS IN PARALLEL)                             │
│                                     │                                        │
│                                     ▼                                        │
│                        ┌─────────────────────────┐                          │
│                        │ Result: {               │                          │
│                        │   "response": "...",    │                          │
│                        │   "intent": {...},      │                          │
│                        │   "rag": {...}          │                          │
│                        │ }                       │                          │
│                        │ Total time: max(150,200)│                          │
│                        │           = 200-500ms   │                          │
│                        └─────────────────────────┘                          │
│                                     │                                        │
│                         ⏱️ GAINS vs Sequential:                             │
│                         Sequential: 150+200 = 350ms                        │
│                         Parallel: max(150,200) = 200ms                     │
│                         SAVES 50% latency! ✅                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🔴 STATE: SPEAKING                                                           │
│ Side effect:                                                                │
│   - Gate microphone (prevent echo)                                          │
│   - Cancel any pending fillers                                              │
│                                                                             │
│ ⏱️ Duration: 50-100ms (just state change)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TTS SERVICE SYNTHESIZES & STREAMS AUDIO                                      │
│                                                                             │
│ flow:                                                                       │
│ 1. POST /stream?text="Hi! How can I help?"                                  │
│ 2. TTS generates audio chunks                                               │
│ 3. Each chunk: emit TTS_CHUNK_READY event                                   │
│ 4. WebSocket receives chunk                                                 │
│ 5. UnifiedFastRTC receives chunk via emit()                                 │
│ 6. Browser speaker plays audio                                              │
│ 7. When done: emit TTS_COMPLETE                                             │
│                                                                             │
│ ⏱️ Duration: 1000-3000ms (depends on response length)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
        ┌────────────────────────┐    ┌──────────────────────┐
        │ USER HEARS RESPONSE    │    │ ORCHESTRATOR WAITING  │
        │ Agent: "Hi! How can    │    │ for PLAYBACK_DONE     │
        │         I help?"       │    │ from Browser/TTS      │
        └────────────────────────┘    └──────────────────────┘
                    │                            │
                    │ (if user interrupts)       │
                    │ "Mmm, I want to change..."  │
                    ▼                            ▼
        ┌────────────────────────┐    ┌──────────────────────┐
        │ USER STARTS SPEAKING   │    │ TTS still playing    │
        │ (Barge-in)             │    │                      │
        └────────────────────────┘    └──────────────────────┘
                    │                            │
                    └────────────┬───────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ BARGE-IN DETECTED                                                           │
│ Browser or STT detects new speech during TTS playback                       │
│                                                                             │
│ Action:                                                                     │
│ 1. Emit BARGE_IN event to Redis                                             │
│ 2. Orchestrator receives event                                              │
│ 3. Cancel pending TTS task                                                  │
│ 4. Clear audio queue                                                        │
│ 5. Transition: SPEAKING → INTERRUPT                                         │
│ 6. Brief pause (50ms)                                                       │
│ 7. Transition: INTERRUPT → LISTENING                                        │
│ 8. Start new STT cycle for user's interruption                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌────────────────────────────────────────┐
        │ NO BARGE-IN? (User just listens)       │
        │ Wait for TTS to complete...            │
        │ Receive PLAYBACK_DONE event            │
        └────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🔵 STATE: LISTENING (BACK TO START)                                          │
│ Side effect:                                                                │
│   - Open microphone                                                         │
│   - Ready for next user input                                               │
│                                                                             │
│ ⏱️ FULL CYCLE TIME (example):                                                │
│   STT: 1000ms                                                               │
│   Intent+RAG: 250ms (parallel)                                              │
│   TTS: 2000ms                                                               │
│   ─────────────                                                             │
│   TOTAL: ~3250ms (vs 4250ms if sequential) ✅ 23% faster                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┘
                    │
                    │ User speaks again...
                    │ BACK TO STT EVENT HANDLER
                    ▼
```

---

## Execution Paths: BEFORE vs AFTER

### BEFORE (Chaotic - What's happening now)

```
User speaks "hello"
  │
  ├─→ WebSocket receives STT partial "hel"
  │     ├─→ State: LISTENING
  │     ├─→ Task 1: play_filler_1 spawned (fire & forget)
  │     └─→ Emit to Redis pub/sub
  │
  ├─→ STT partial "hell"
  │     ├─→ State: LISTENING
  │     └─→ (silently ignored, too frequent)
  │
  ├─→ STT final "hello" (VAD end)
  │     ├─→ WebSocket receives (Queue.recv)
  │     ├─→ Transition: LISTENING → THINKING ✓
  │     ├─→ Task 2: process_intent_rag spawned ✓
  │     │     ├─→ Intent service: POST /intent → 200ms
  │     │     │     └─→ But we don't wait, continue...
  │     │     └─→ RAG service: POST /query → 300ms
  │     │           └─→ But we don't wait, continue...
  │     │
  │     └─→ Meanwhile Redis pub/sub also received same event
  │           ├─→ Redis listener wakes up
  │           ├─→ Task 3: listen_to_redis_events handler
  │           ├─→ Tries to route to state_manager
  │           ├─→ Session ID mismatch! (fastrtc_XXX vs auto_session_YYY)
  │           ├─→ Task 4: play_filler_2 spawned (trying to recover)
  │           └─→ Process Intent+RAG AGAIN!? ⚠️
  │
  ├─→ Task 1 plays "thinking filler"
  │     ├─→ Stream audio chunks
  │     └─→ Complete
  │
  ├─→ Task 2 completes Intent+RAG
  │     ├─→ Gets response "Hello! How can I help?"
  │     ├─→ Transition: THINKING → SPEAKING
  │     ├─→ Task 5: stream_tts_audio spawned
  │     └─→ But Task 4 is ALSO trying to do this! ⚠️
  │
  ├─→ Task 3 (from Redis) also processes Intent+RAG
  │     ├─→ Duplicate processing! ⚠️
  │     └─→ Creates confusion in state
  │
  ├─→ Task 4 plays second filler (overlap!)
  │     └─→ Overlaps with Task 5 TTS
  │
  ├─→ State chaos:
  │     LISTENING → THINKING (Task 2)
  │     ↓
  │     WARNING: Invalid transition: LISTENING → THINKING (Task 3)
  │     ↓
  │     SPEAKING (Task 2)
  │     ↓
  │     SPEAKING (Task 5, redundant)
  │     ↓
  │     ⚠️ Invalid transition: SPEAKING → SPEAKING?
  │
  └─→ User hears overlapping audio
        + filler
        + TTS response
        + Unclear state
```

Result: **Chaotic logs, multiple tasks fighting, wrong sessions routing**

---

### AFTER (Clean - Target state)

```
User speaks "hello"
  │
  ├─→ WebSocket /orchestrate endpoint
  │     └─→ Session created: auto_session_12345
  │           state_manager = StateManager(auto_session_12345)
  │           current_task = None  ← Track ONE task
  │
  ├─→ STT service sends partial "he", "hell", "hello"
  │     ├─→ Unified FastRTC receives all
  │     └─→ Only emits final when VAD detects silence
  │           └─→ Event: STT_FINAL (is_final=True, text="hello")
  │
  ├─→ Redis receives event
  │     └─→ Route to correct session
  │           1. Check if "auto_session_12345" in active_sessions → YES!
  │           2. Get state_manager for this session
  │           3. Call STTEventHandler.handle_stt_final("hello", ...)
  │
  ├─→ STTEventHandler.handle_stt_final()
  │     ├─→ Validate: text="hello" (not empty), is_final=True ✓
  │     ├─→ Check state: LISTENING ✓
  │     ├─→ Transition: LISTENING → THINKING
  │     │     └─→ Side effect: gate_microphone()
  │     │
  │     ├─→ Parallel INT+RAG
  │     │     ├─→ Intent: POST /intent ("hello") → 200ms
  │     │     ├─→ RAG: POST /query ("hello") → 300ms
  │     │     └─→ Max = 300ms (parallel! not sequential)
  │     │           Result: {"response": "Hello! How can I help?", ...}
  │     │
  │     ├─→ Transition: THINKING → SPEAKING
  │     │     └─→ Side effect: ensure_mic_gated()
  │     │
  │     └─→ Return result to caller
  │           (TTS streaming begins)
  │
  ├─→ TTS Service receives request
  │     ├─→ Streams audio chunks to WebSocket
  │     └─→ Each chunk: emit TTS_CHUNK_READY
  │
  ├─→ WebSocket sends TTS chunks to browser
  │     └─→ Browser speaker plays: "Hello! How can I help?"
  │
  ├─→ After TTS complete
  │     ├─→ Browser emits PLAYBACK_DONE
  │     ├─→ Redis receives event
  │     ├─→ Route to correct session: auto_session_12345
  │     ├─→ Transition: SPEAKING → LISTENING
  │     │     └─→ Side effect: open_microphone()
  │     └─→ Back to start, ready for next user input
  │
  └─→ If user interrupts (barge-in)
        ├─→ User starts speaking
        ├─→ STT detects new speech
        ├─→ Browser/STT emits BARGE_IN event
        ├─→ Redis receives event
        ├─→ Cancel current TTS task
        │     (old_task = active_sessions[sid]["current_task"]
        │      if old_task: old_task.cancel())
        ├─→ Transition: SPEAKING → INTERRUPT
        ├─→ Pause 50ms
        ├─→ Transition: INTERRUPT → LISTENING
        └─→ Start new STT cycle
```

Result: **Clean logs, one task per session, clear session routing, predictable behavior**

---

## Key Differences

| Aspect | BEFORE | AFTER |
|--------|--------|-------|
| **Tasks per session** | 4-5 running simultaneously | 1 (current_task) |
| **State transitions** | Multiple paths to same state | Single unified path |
| **Session routing** | Fragile guessing | 3-rule validation |
| **Event processing** | Duplicate events (pub/sub + Streams) | Events processed once (Streams only) |
| **Logs** | Chaotic, overlapping | Linear, sequential |
| **Debugging** | Hard to follow | Easy to trace |
| **Latency** | Variable, unpredictable | Consistent, measurable |
| **Barge-in** | Unreliable | Predictable cancellation |

---

## Testing the Fix

### Test Case 1: Normal Flow
```
1. Browser connects → Session created
2. User says "hello"
3. STT emits FINAL
4. Intent+RAG runs in parallel
5. TTS streams response
6. User hears reply
7. Playback completes
8. Back to LISTENING

Expected logs (linear, no overlaps):
[auto_session_12345] 🟢 IDLE
[auto_session_12345] 🔵 LISTENING (client_connected)
[auto_session_12345] STT FINAL | text=hello
[auto_session_12345] 🟡 THINKING
[auto_session_12345] Processing complete in 300ms
[auto_session_12345] 🔴 SPEAKING
[auto_session_12345] TTS stream | chunks=10
[auto_session_12345] 🔵 LISTENING (playback_done)
✅ PASS
```

### Test Case 2: Barge-in Flow
```
1. Agent is speaking (TTS playing)
2. User starts speaking (interrupt)
3. STT detects new input
4. Browser emits BARGE_IN
5. TTS task cancelled
6. State: INTERRUPT → LISTENING
7. New STT cycle begins

Expected logs:
[auto_session_12345] 🔴 SPEAKING
[auto_session_12345] [TTS] Playing...
[auto_session_12345] 🔴 User interrupted (barge_in)
[auto_session_12345] Cancelling current task
[auto_session_12345] 🟣 INTERRUPT
[auto_session_12345] 🔵 LISTENING
[auto_session_12345] STT FINAL | text=stop  (new user input)
✅ PASS
```

### Test Case 3: Session Isolation
```
1. Browser A connects → Session A created
2. Browser B connects → Session B created
3. User A says "hello"
4. User B says "goodbye"
5. Each processes in parallel but independently

Expected logs:
[auto_session_A] STT FINAL | text=hello
[auto_session_A] 🟡 THINKING
[auto_session_B] STT FINAL | text=goodbye
[auto_session_B] 🟡 THINKING
[auto_session_A] 🔴 SPEAKING
[auto_session_B] 🔴 SPEAKING
(Both can speak simultaneously without interference)
✅ PASS
```
