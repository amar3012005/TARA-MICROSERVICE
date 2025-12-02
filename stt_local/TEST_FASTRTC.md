# Test STT Local with FastRTC - Real-time Terminal Transcripts

## Quick Start

### Option 1: Using Test Script (Easiest)
```bash
cd services/stt_local
bash run_test.sh
```

### Option 2: Manual Setup
```bash
cd services
source venv/bin/activate
cd stt_local

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_after.txt  # Heavy deps (optional for CPU mode)

# Run service
python3 run_local_fastrtc.py
```

## Testing Steps

1. **Start the service** (see above)
2. **Open browser**: `http://localhost:7861/fastrtc`
3. **Click "Start"** in the FastRTC UI
4. **Speak into your microphone**
5. **Watch terminal** for real-time transcripts!

## Expected Terminal Output

You'll see real-time transcripts appearing in the terminal:

```
======================================================================
📝 [🔄 PARTIAL] STT Transcript
   Text: 'Hello how are you'
   Session: fastrtc_1234567890
======================================================================
======================================================================
📝 [✅ FINAL] STT Transcript
   Text: 'Hello how are you today'
   Session: fastrtc_1234567890
   Chunks processed: 150
======================================================================
```

## Features

- **🔄 PARTIAL**: Updates every 500ms while speaking (streaming)
- **✅ FINAL**: Complete transcript when speech ends
- **Real-time**: Ultra-low latency transcription
- **Terminal visibility**: All transcripts logged prominently

## Pipeline Flow

```
Browser Microphone
    ↓
FastRTC (WebRTC)
    ↓
FastRTC Handler (resample 48kHz→16kHz)
    ↓
STT Manager
    ↓
Silero VAD (speech detection)
    ↓
Faster Whisper (transcription)
    ↓
Terminal Logs (real-time)
```

## Troubleshooting

- **No audio**: Check browser microphone permissions
- **No transcripts**: Ensure heavy deps are installed (`pip install -r requirements_after.txt`)
- **CUDA errors**: Service auto-falls back to CPU
- **Port conflicts**: Change ports in `run_local_fastrtc.py`

## Comparison with stt-vad

| Feature | stt-vad (Gemini) | stt-local (Whisper) |
|---------|------------------|---------------------|
| Model | Cloud (Gemini Live) | Local (Faster Whisper) |
| VAD | Gemini built-in | Silero VAD |
| Latency | ~200-300ms | ~300-500ms |
| Accuracy | 99% | 99% |
| GPU Required | No | Optional (CPU fallback) |
| Internet | Required | Not required |



