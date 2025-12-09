# Gemini Live API Compatibility Fix

## Issue Summary

The microservice code was using outdated Gemini Live API patterns that don't match the current API version.

## Errors Encountered

```
⚠️ Failed to send end-of-turn: AsyncSession.send_realtime_input() got an unexpected keyword argument 'end_of_turn'

❌ Capture error: 'LiveServerContent' object has no attribute 'input_audio_transcription'
```

## Root Cause

The Gemini Live API structure differs from what was assumed:

### WRONG (Old/Assumed API):
```python
# ❌ Response field doesn't exist
response.server_content.input_audio_transcription

# ❌ Parameter not supported
await session.send_realtime_input(end_of_turn=True)
```

### CORRECT (Current API):
```python
# ✅ Correct response field
response.server_content.input_transcription

# ✅ No end_of_turn parameter needed
await session.send_realtime_input(
    audio=types.Blob(data=audio_chunk, mime_type="audio/pcm;rate=16000")
)
# Auto-finalizes when audio stops flowing
```

## Fixes Applied

### 1. Response Field Name ✅
**File**: `vad_manager.py` (line ~230)

```python
# BEFORE:
if response.server_content.input_audio_transcription:
    final_text = response.server_content.input_audio_transcription.text

# AFTER:
if response.server_content.input_transcription:
    final_text = response.server_content.input_transcription.text
```

### 2. Removed End-of-Turn Signal ✅
**File**: `vad_manager.py` (line ~175)

```python
# BEFORE:
finally:
    await session.send_realtime_input(end_of_turn=True)  # ❌ Not supported

# AFTER:
finally:
    # NOTE: Gemini Live API auto-finalizes when audio stops
    # No explicit end_of_turn signal needed
    logger.debug(f"Audio send complete: {chunk_count} chunks")
```

## Session Config (Still Correct)

The session configuration is **correct** and unchanged:

```python
session_config = {
    "response_modalities": ["TEXT"],  # TEXT only for transcription
    "input_audio_transcription": {}   # Enable user speech transcription (CONFIG field)
}
```

**Key distinction**:
- **Config field**: `input_audio_transcription: {}` ✅ (enables transcription)
- **Response field**: `response.server_content.input_transcription` ✅ (receives text)

## Reference Implementation

The correct API usage is demonstrated in the working `leibniz_agent/leibniz_vad.py`:

```python
# Lines 534-540: Audio sending (NO end_of_turn)
await session.send_realtime_input(
    audio=types.Blob(
        data=audio_chunk,
        mime_type=f"audio/pcm;rate={self.config.sample_rate}"
    )
)

# Lines 571-575: Response handling (input_transcription)
if response.server_content and response.server_content.input_transcription:
    text = response.server_content.input_transcription.text
```

## Testing After Fix

Run the test again:
```powershell
# Restart service with fixes
docker compose -f docker-compose.leibniz.yml restart stt-vad

# Run audio test
python test_stt_simple.py
```

**Expected behavior**:
- ✅ No `end_of_turn` error
- ✅ No `input_audio_transcription` attribute error
- ✅ Transcripts captured successfully
- ✅ Auto-finalization when audio stops

## Compatibility Notes

**Gemini Live API Version**: google-genai >= 1.33.0

The API uses:
- **Config**: `input_audio_transcription: {}` to enable
- **Response**: `input_transcription.text` to receive
- **Finalization**: Automatic (no explicit signal)

This matches the production-tested `leibniz_vad.py` implementation.
