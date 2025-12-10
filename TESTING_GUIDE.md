# Unified FastRTC Testing Guide

## Current Status

✅ **Completed & Verified:**
- **Unified FastRTC UI**: Successfully mounted at `http://localhost:2004/fastrtc`
- **Handler Logic**: Refactored with state-aware gating and echo prevention
- **Orchestrator Integration**: State broadcasting and audio routing fully wired
- **Dependencies**: Fixed version conflicts by aligning with `tts_sarvam` configuration

## How to Test

### 1. Unified Interface (Recommended)

1. **Start Services:**
   ```bash
   docker --context desktop-linux compose -f docker-compose-tara-task.yml up -d
   ```

2. **Open Browser:**
   - Go to: http://localhost:2004/fastrtc
   - Allow microphone permissions.

3. **Connect & Record:**
   - Click the "Record" button.
   - This establishes a single WebSocket connection that handles BOTH STT (input) and TTS (output).

4. **Trigger Intro:**
   - Open a terminal and run:
     ```bash
     curl -X POST http://localhost:2004/start
     ```
   - You should hear the intro greeting through the browser.
   - Speak after the greeting.

5. **Verify State:**
   - Check the status endpoint to see the active handler state:
     ```bash
     curl -s http://localhost:2004/status | python3 -m json.tool
     ```
   - Look for `"unified_fastrtc": { "active_handlers": ... }`

### 2. Architecture Verification

✅ **State Management:**
- When Orchestrator transitions to `SPEAKING`/`THINKING`, the handler gates microphone input.
- When Orchestrator transitions to `LISTENING`/`IDLE`, the handler opens microphone input.

✅ **Audio Routing:**
- **TTS**: Audio from `tts_sarvam` -> Orchestrator (`app.py`) -> `UnifiedFastRTCHandler` -> Browser.
- **STT**: Audio from Browser -> `UnifiedFastRTCHandler` -> `stt_vad` Service.

✅ **Error Handling:**
- Automatic WebSocket reconnection if STT/TTS services drop.
- Robust state tracking prevents "closed socket" errors.

## Service URLs

- **Orchestrator API:** http://localhost:2004
- **Unified FastRTC UI:** http://localhost:2004/fastrtc ✅
- **STT FastRTC UI:** http://localhost:2012 (Still available for isolated testing)
- **TTS FastRTC UI:** http://localhost:2007/fastrtc (Still available for isolated testing)
