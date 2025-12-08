# TARA X TASK - Complete Microservices Stack

## Overview

Complete Telugu customer service agent stack organized in Docker Desktop as **"tara-task"** project folder.

## Quick Start

### Option 1: Start Everything (Recommended)

```bash
cd TARA-MICROSERVICE
./start-tara-task.sh
```

### Option 2: Start Orchestrator Only (Auto-starts others)

```bash
docker context use desktop-linux
docker-compose -f docker-compose-tara-task.yml up -d orchestrator
```

### Option 3: Start All Services Manually

```bash
docker context use desktop-linux
docker-compose -f docker-compose-tara-task.yml up -d
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Docker Desktop: tara-task               │
│  ┌───────────────────────────────────────────┐  │
│  │      tara-task-network (bridge)           │  │
│  │                                            │  │
│  │  ┌──────────┐  ┌──────────┐  ┌────────┐ │  │
│  │  │  Redis   │  │   STT    │  │  RAG   │ │  │
│  │  │  :6000   │  │  :6001   │  │ :6003  │ │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬───┘ │  │
│  │       │              │             │     │  │
│  │       └──────────────┼─────────────┘     │  │
│  │                      │                    │  │
│  │              ┌───────▼───────┐           │  │
│  │              │ Orchestrator  │           │  │
│  │              │   :6004       │           │  │
│  │              │ (Master Ctrl) │           │  │
│  │              └───────┬───────┘           │  │
│  │                      │                    │  │
│  │              ┌───────▼───────┐           │  │
│  │              │     TTS       │           │  │
│  │              │    :6005      │           │  │
│  │              └───────────────┘           │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Services

| Service | Container Name | Ports | Description |
|---------|---------------|-------|-------------|
| **Redis** | `tara-task-redis` | 6000 | Data store & pub/sub |
| **Orchestrator** | `tara-task-orchestrator` | 6004 | Master controller |
| **STT-VAD** | `tara-task-stt-vad` | 6001, 6012 | Speech-to-text |
| **RAG** | `tara-task-rag` | 6003 | Knowledge base |
| **TTS Sarvam** | `tara-task-tts-sarvam` | 6005 | Text-to-speech |

## Service URLs

- **Orchestrator API**: http://localhost:6004
- **STT FastRTC UI**: http://localhost:6012
- **TTS FastRTC UI**: http://localhost:6005/fastrtc
- **RAG API**: http://localhost:6003
- **Redis**: localhost:6000

## Workflow

1. **Start Orchestrator** → Auto-starts all services
2. **Check Health** → All services become healthy
3. **Open FastRTC UIs** → STT and TTS in browser
4. **Wait for Connections** → Detected via Redis events
5. **Trigger Start** → `curl -X POST http://localhost:6004/start`

## Commands

### View Logs

```bash
# All services
docker-compose -f docker-compose-tara-task.yml logs -f

# Specific service
docker logs tara-task-orchestrator -f
docker logs tara-task-stt-vad -f
docker logs tara-task-rag -f
docker logs tara-task-tts-sarvam -f
```

### Check Status

```bash
# Service status
docker-compose -f docker-compose-tara-task.yml ps

# Orchestrator status
curl http://localhost:6004/status | jq

# Health checks
curl http://localhost:6001/health  # STT
curl http://localhost:6003/health  # RAG
curl http://localhost:6005/health  # TTS
```

### Stop Services

```bash
# Stop all
docker-compose -f docker-compose-tara-task.yml down

# Stop with volumes (clean slate)
docker-compose -f docker-compose-tara-task.yml down -v
```

### Restart Services

```bash
# Restart all
docker-compose -f docker-compose-tara-task.yml restart

# Restart specific service
docker-compose -f docker-compose-tara-task.yml restart orchestrator
```

## Docker Desktop Organization

All services appear under **"tara-task"** folder in Docker Desktop:

```
Docker Desktop
└── tara-task (project)
    ├── redis
    ├── orchestrator
    ├── stt-vad
    ├── rag
    └── tts-sarvam
```

## Environment Variables

Key environment variables (set in `docker-compose-tara-task.yml`):

- `AUTO_START_SERVICES=true` - Orchestrator auto-starts services
- `TARA_MODE=true` - Telugu TASK mode
- `SKIP_INTENT_SERVICE=true` - Skip Intent service
- `SKIP_APPOINTMENT_SERVICE=true` - Skip Appointment service
- `DOCKER_PROJECT_NAME=tara-task` - Project name

## Volumes

- `tara-task-redis-data` - Redis persistence
- `tara-task-rag-index` - FAISS vector index

## Network

- `tara-task-network` - Bridge network for all services

## Troubleshooting

### Port Conflicts

```bash
# Check what's using ports
lsof -i :6000
lsof -i :6004

# Stop conflicting containers
docker stop $(docker ps -q --filter "publish=6000")
```

### Services Not Starting

```bash
# Check logs
docker-compose -f docker-compose-tara-task.yml logs

# Rebuild services
docker-compose -f docker-compose-tara-task.yml build --no-cache
```

### Network Issues

```bash
# Inspect network
docker network inspect tara-task-network

# Recreate network
docker network rm tara-task-network
docker-compose -f docker-compose-tara-task.yml up -d
```

## Cleanup

```bash
# Remove all containers, networks, volumes
docker-compose -f docker-compose-tara-task.yml down -v

# Remove images
docker-compose -f docker-compose-tara-task.yml down --rmi all
```

## Next Steps

1. Start the stack: `./start-tara-task.sh`
2. Monitor logs: `docker logs tara-task-orchestrator -f`
3. Open FastRTC UIs when ready
4. Trigger workflow: `curl -X POST http://localhost:6004/start`

