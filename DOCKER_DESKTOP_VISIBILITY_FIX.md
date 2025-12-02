# Docker Desktop Not Showing TARA Containers

## Issue
TARA containers are running but not visible in Docker Desktop GUI.

## Root Cause
Docker Desktop may filter out or hide containers that use `network_mode: host` because they don't show port mappings in the standard way.

## Solution: Check Docker Desktop Settings

### Option 1: Check Container Filters
1. In Docker Desktop, look at the "Containers" tab
2. Check if "Only show running containers" toggle is ON
3. Try turning it OFF to see all containers (including stopped ones)
4. Check the search/filter box - make sure it's not filtering out "tara" containers

### Option 2: Verify Containers Are Actually Running
Run this command to verify:
```bash
docker context use desktop-linux
docker ps --filter "name=tara"
```

If containers show up here but not in Docker Desktop, it's a Docker Desktop display issue.

### Option 3: Restart Docker Desktop
Sometimes Docker Desktop needs a refresh:
1. Click Docker Desktop menu → Quit Docker Desktop
2. Wait 10 seconds
3. Restart Docker Desktop
4. Check Containers tab again

### Option 4: Use Docker CLI Instead
If Docker Desktop still doesn't show them, you can manage containers via CLI:
```bash
# View all TARA containers
docker context use desktop-linux
docker ps --filter "name=tara"

# View logs
docker logs tara-redis
docker logs tara-orchestrator-service

# Stop/Start
docker-compose -p tara-microservice -f docker-compose-tara.yml stop
docker-compose -p tara-microservice -f docker-compose-tara.yml start
```

## Current Container Status
Run this to see what's actually running:
```bash
docker context use desktop-linux
docker-compose -p tara-microservice -f docker-compose-tara.yml ps
```

## Why This Happens
- Containers with `network_mode: host` don't have port mappings visible to Docker Desktop
- Docker Desktop may filter them out from the main view
- Some Docker Desktop versions have bugs with host networking display

## Workaround
The containers ARE running and functional, even if Docker Desktop doesn't show them. You can:
1. Use `docker ps` to verify they're running
2. Access services directly via their ports (localhost:6382, localhost:8020, etc.)
3. Use Docker CLI for management instead of GUI




