# STT Local Service - Testing Guide

## Quick Start

### Option 1: Using Virtual Environment (Recommended)
```bash
cd services/stt_local
bash setup_test_env.sh
source venv/bin/activate
python3 test_local.py
```

### Option 2: System-wide Installation
```bash
cd services/stt_local
pip3 install --user -r requirements.txt
python3 test_local.py
```

### Option 3: Direct Docker Build (Fastest)
```bash
cd services
docker-compose build stt-local-service
docker-compose up -d stt-local-service
```

This will test:
- ✅ Basic imports (numpy, fastapi, etc.)
- ✅ Configuration loading
- ✅ Utility functions
- ⚠️ VAD utilities (requires torch - will skip if not installed)
- ⚠️ Whisper service (requires faster-whisper - will skip if not installed)
- ⚠️ STT Manager (requires all deps - will skip if not installed)
- ✅ FastAPI app structure

### 3. Install Heavy Dependencies (Optional - for full testing)
```bash
pip install -r requirements_after.txt
```

Then run tests again:
```bash
python3 test_local.py
```

## Docker Testing

### Quick Build (Lightweight)
```bash
cd services
docker-compose build stt-local-service
```

### Start Container
```bash
docker-compose up -d stt-local-service
```

### Install Heavy Dependencies in Container
```bash
# Enter container
docker exec -it stt-local-service bash

# Install heavy deps
cd /app/stt_local
bash install_heavy_deps.sh

# Or manually:
pip install -r requirements_after.txt
```

### Start Service in Container
```bash
# Inside container
python3 -u -m uvicorn app:app --host 0.0.0.0 --port 8006 --workers 1
```

### Test Endpoints
- Health: `curl http://localhost:8014/health`
- FastRTC UI: `http://localhost:7863/fastrtc`
- WebSocket: `ws://localhost:8014/api/v1/transcribe/stream?session_id=test`

## Expected Test Results

### With Lightweight Dependencies Only
- ✅ All basic imports pass
- ✅ Config and utils pass
- ⚠️ VAD/Whisper tests will skip (expected)

### With All Dependencies
- ✅ All tests should pass
- ✅ Service ready to run

## Troubleshooting

### CUDA Not Available
If CUDA is not available, the service will automatically fall back to CPU mode. Tests will still pass but will be slower.

### Model Downloads
- Silero VAD model downloads automatically on first use (~50MB)
- Faster Whisper model downloads automatically on first use (~150MB for base model)

### Port Conflicts
If ports 8014 or 7863 are in use, modify `docker-compose.yml` to use different ports.

