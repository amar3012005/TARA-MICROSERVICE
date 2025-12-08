# Orchestrator Master Controller Mode

## Overview

The Orchestrator has been enhanced to act as a **Master Controller** that automatically starts and manages all dependent microservices. Simply start the orchestrator, and it will:

1. ✅ Start all services in order (Redis → STT → RAG → TTS)
2. ✅ Check health of each service
3. ✅ Display STT/TTS FastRTC links when ready
4. ✅ Wait for browser connections (detected via Redis events)
5. ✅ Wait for `/start` trigger to begin workflow

## Features

### Auto-Start Services
- **Enabled by default** via `AUTO_START_SERVICES=true`
- Starts services in dependency order
- Skips Intent and Appointment services (TARA mode)
- Health checks with retries and timeouts

### Service Manager
- New `service_manager.py` module handles:
  - Docker container lifecycle (start/stop)
  - Health check polling
  - Network management
  - Service URL discovery

### Connection Detection
- Automatically detects STT/TTS browser connections via Redis events
- Displays connection status in logs
- Auto-creates session when both services connect

### Start Trigger
- Waits for explicit `/start` POST request
- Triggers intro greeting sequence
- Supports both WebSocket and auto-created sessions

## Configuration

### Environment Variables

```bash
# Enable auto-start (default: true)
AUTO_START_SERVICES=true

# Docker compose file path (optional)
DOCKER_COMPOSE_FILE=/app/docker-compose-tara.yml

# Docker context (default: desktop-linux)
DOCKER_CONTEXT=desktop-linux

# Service URLs (auto-discovered if not set)
STT_GRADIO_URL=http://localhost:5212
TTS_GRADIO_URL=http://localhost:5205/fastrtc

# Skip services
SKIP_INTENT_SERVICE=true
SKIP_APPOINTMENT_SERVICE=true
```

## Usage

### Start Orchestrator Only

```bash
docker context use desktop-linux
docker-compose -f docker-compose-tara.yml up -d orchestrator-tara
```

The orchestrator will:
1. Start Redis, STT, RAG, and TTS services automatically
2. Wait for all services to be healthy
3. Display FastRTC links
4. Wait for browser connections
5. Wait for `/start` trigger

### Manual Start (Disable Auto-Start)

```bash
# Set AUTO_START_SERVICES=false
docker-compose -f docker-compose-tara.yml up -d orchestrator-tara
```

### Start Workflow

Once services are ready and connected:

```bash
curl -X POST http://localhost:5204/start
```

## Service Startup Sequence

1. **Redis** (port 5200)
   - Foundation service
   - No HTTP health endpoint (container check only)

2. **STT-VAD** (ports 5202, 5212)
   - Health: `http://localhost:5202/health`
   - FastRTC: `http://localhost:5212`

3. **RAG** (port 5203)
   - Health: `http://localhost:5203/health`
   - Knowledge base indexing

4. **TTS Sarvam** (port 5205)
   - Health: `http://localhost:5205/health`
   - FastRTC: `http://localhost:5205/fastrtc`

## Logs Output

```
======================================================================
🚀 Starting StateManager Orchestrator
======================================================================
📋 Configuration loaded
🎛️ MASTER CONTROLLER MODE: Auto-starting services...
======================================================================
🚀 Starting services in order: Redis → STT → RAG → TTS
✅ redis is ready
✅ stt is ready
✅ rag is ready
✅ tts is ready
✅ All services started successfully!
🏥 Checking health of all services...
✅ All dependent services are READY
✅ StateManager Orchestrator Ready
======================================================================
📋 SERVICE LINKS:
   🔗 STT FastRTC UI: http://localhost:5212
   🔗 TTS FastRTC UI: http://localhost:5205/fastrtc
   🔗 Orchestrator API: http://localhost:5204
======================================================================
⏳ WAITING FOR CONNECTIONS:
   1. Open STT FastRTC UI in browser: http://localhost:5212
   2. Open TTS FastRTC UI in browser: http://localhost:5205/fastrtc
   3. Connections will be detected automatically via Redis events
   4. Once both STT and TTS connect, workflow will be ready
   5. Send POST /start to trigger: curl -X POST http://localhost:5204/start
======================================================================
```

## Architecture

```
┌─────────────────────────────────────────┐
│   Orchestrator (Master Controller)      │
│   - Service Manager                      │
│   - Health Checker                      │
│   - Connection Monitor                  │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
   ┌───▼───┐      ┌─────▼─────┐
   │ Redis │      │  Docker   │
   └───┬───┘      │  Socket   │
       │          └───────────┘
       │
   ┌───▼──────────┬──────────┬──────────┐
   │              │          │          │
┌──▼──┐    ┌─────▼──┐  ┌────▼──┐  ┌────▼──┐
│ STT │    │  RAG   │  │  TTS  │  │Intent│
└─────┘    └────────┘  └───────┘  └──────┘
           (skipped)
```

## Requirements

- Docker socket mounted: `/var/run/docker.sock`
- Docker compose file mounted (optional)
- Docker CLI installed in orchestrator container
- Services on same Docker network (`tara-network`)

## Troubleshooting

### Services Not Starting
- Check Docker socket permissions
- Verify Docker context is set correctly
- Check logs: `docker logs tara-orchestrator`

### Health Checks Failing
- Verify services are accessible on expected ports
- Check network connectivity: `docker network inspect tara-network`
- Increase timeout in `service_manager.py`

### Connections Not Detected
- Ensure STT/TTS FastRTC UIs are open in browser
- Check Redis events: `docker exec tara-redis redis-cli MONITOR`
- Verify Redis connection in orchestrator logs

## Future Enhancements

- [ ] Service restart on failure
- [ ] Graceful shutdown of services
- [ ] Service dependency graph visualization
- [ ] Health check metrics dashboard
- [ ] Service scaling support




