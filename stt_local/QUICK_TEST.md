# Quick Test Guide - STT Local with FastRTC

## Local Testing (Before Docker)

### 1. Install Dependencies
```bash
cd services/stt_local
source ../venv/bin/activate  # Or create venv: python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_after.txt  # Heavy deps (torch, faster-whisper)
```

### 2. Run Service Locally
```bash
cd services/stt_local
python3 run_local_fastrtc.py
```

### 3. Open FastRTC UI
- Open browser: `http://localhost:7861/fastrtc`
- Click "Start" to begin audio streaming
- Speak into your microphone
- **Watch terminal for real-time transcripts!**

## CLI Microphone Smoke Test

Use the lightweight CLI demo when you want to sanity-check the STT stack
without opening the FastRTC UI.

```bash
cd services/stt_local
python3 test_mic_realtime.py --list-devices  # optional helper
python3 test_mic_realtime.py --device <input_device_index>
# or use the convenience wrapper (supports --skip-warmup and forwards args)
bash run_whisper_mic_demo.sh --skip-warmup -- --device <input_device_index>
```

- Defaults to 100 ms chunks at the configured sample rate.
- Prints partial/final transcripts inline.
- Press `Ctrl+C` to stop the stream.

## Expected Terminal Output

You should see transcripts appearing in real-time:

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
======================================================================
```

## Docker Testing

### 1. Build and Start
```bash
cd services
docker-compose build stt-local-service
docker-compose up -d stt-local-service
```

### 2. Install Heavy Dependencies (in container)
```bash
docker exec -it stt-local-service bash
cd /app/stt_local
bash install_heavy_deps.sh
exit
```

### 3. Restart Service
```bash
docker-compose restart stt-local-service
```

### 4. View Logs (Terminal Transcripts)
```bash
docker-compose logs -f stt-local-service
```

### 5. Open FastRTC UI
- Browser: `http://localhost:7863/fastrtc`
- Click "Start" and speak
- **Watch Docker logs for real-time transcripts!**

## Troubleshooting

- **No transcripts**: Check microphone permissions in browser
- **CUDA errors**: Service auto-falls back to CPU
- **Port conflicts**: Change ports in docker-compose.yml
- **Model download**: First run downloads models (~150MB for base)



