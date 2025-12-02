# Quick Start Guide

## Local Testing (Before Docker)

### 1. Setup Test Environment
```bash
cd services/stt_local
bash setup_test_env.sh
source venv/bin/activate
```

### 2. Run Tests
```bash
python3 test_local.py
```

Expected: Basic tests pass, heavy deps tests skip (normal)

### 3. Install Heavy Dependencies (Optional - for full testing)
```bash
pip install -r requirements_after.txt
python3 test_local.py  # Run again - all tests should pass
```

## Docker Testing (Production)

### 1. Quick Build (Lightweight - Fast)
```bash
cd services
docker-compose build stt-local-service
```

### 2. Start Container
```bash
docker-compose up -d stt-local-service
```

### 3. Install Heavy Dependencies in Container
```bash
# Enter container
docker exec -it stt-local-service bash

# Install heavy deps (takes 5-10 minutes)
cd /app/stt_local
bash install_heavy_deps.sh

# Exit container
exit
```

### 4. Restart Service
```bash
docker-compose restart stt-local-service
```

### 5. Test Service
```bash
# Health check
curl http://localhost:8014/health

# Open FastRTC UI
# Browser: http://localhost:7863/fastrtc
```

## Troubleshooting

- **Import errors**: Install dependencies first
- **CUDA not available**: Service auto-falls back to CPU
- **Port conflicts**: Change ports in docker-compose.yml
