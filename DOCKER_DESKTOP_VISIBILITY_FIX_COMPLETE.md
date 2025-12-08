# Docker Desktop Visibility - Complete Fix Guide

## ✅ Current Status

All 6 containers are **running and healthy**:
- ✅ tara-task-redis (healthy)
- ✅ tara-task-orchestrator (healthy)
- ✅ tara-task-rag (healthy)
- ✅ tara-task-stt-vad (healthy)
- ✅ tara-task-stt-sarvam (healthy)
- ✅ tara-task-tts-sarvam (healthy)

## Why Containers Might Not Appear in Docker Desktop

Docker Desktop groups containers by the `com.docker.compose.project` label. If containers don't appear:

### Solution 1: Refresh Docker Desktop (Most Common Fix)

1. **Press F5** in Docker Desktop to refresh
2. Or **restart Docker Desktop** completely
3. Go to **Containers** tab
4. Look for **"tara-task"** folder/project

### Solution 2: Check Docker Desktop Filters

1. In Docker Desktop, go to **Containers** tab
2. Make sure filter shows **"All"** containers (not just running)
3. Check if there's a search/filter box - clear it
4. Look for containers starting with `tara-task-`

### Solution 3: Verify Docker Context

Docker Desktop only shows containers from the active Docker context:

```bash
# Check current context
docker context show

# Should be: desktop-linux
# If not, switch:
docker context use desktop-linux
```

### Solution 4: Restart Containers with Explicit Project Name

```bash
cd TARA-MICROSERVICE
docker context use desktop-linux
export COMPOSE_PROJECT_NAME=tara-task

# Stop and restart
docker-compose -f docker-compose-tara-task.yml down
docker-compose -f docker-compose-tara-task.yml --project-name tara-task up -d
```

### Solution 5: Check Container Labels

Verify containers have the correct project label:

```bash
docker inspect tara-task-redis --format '{{index .Config.Labels "com.docker.compose.project"}}'
# Should output: tara-task
```

If it's empty, restart with explicit project name (Solution 4).

## Quick Fix Script

Run the provided fix script:

```bash
cd TARA-MICROSERVICE
./fix-docker-desktop-visibility.sh
```

Then:
1. **Refresh Docker Desktop** (F5)
2. Check **Containers** tab
3. Look for **"tara-task"** folder

## Verify Containers Are Running

Even if Docker Desktop doesn't show them, verify they're running:

```bash
# Check all containers
docker-compose -f docker-compose-tara-task.yml ps

# Check specific container
docker ps --filter "name=tara-task"

# Check logs
docker logs tara-task-orchestrator
```

## Docker Desktop Alternative Views

If containers don't appear in the Containers tab:

1. **Images Tab**: Check if images are there (`tara-task-*`)
2. **Volumes Tab**: Check if volumes exist (`tara-task-redis-data`, `tara-task-rag-index`)
3. **Networks Tab**: Check if network exists (`tara-task-network`)

## Manual Verification

All containers are accessible via their ports:

- **Redis**: `localhost:6000`
- **Orchestrator**: `localhost:6004`
- **STT-VAD**: `localhost:6001` (API), `localhost:6012` (FastRTC)
- **STT-Sarvam**: `localhost:6002` (API), `localhost:6013` (FastRTC)
- **RAG**: `localhost:6003`
- **TTS-Sarvam**: `localhost:6005`

Test with:
```bash
curl http://localhost:6004/health  # Orchestrator
curl http://localhost:6003/health  # RAG
curl http://localhost:6001/health  # STT-VAD
```

## Summary

**Containers ARE running and healthy!** The issue is likely:
1. Docker Desktop needs refresh (F5)
2. Docker Desktop filter/view settings
3. Docker context mismatch

Try refreshing Docker Desktop first - that's the most common solution.




