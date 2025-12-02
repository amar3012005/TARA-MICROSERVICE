# Docker Context Issue - Containers Not Visible in Docker Desktop

## Problem
Containers are running in the `default` Docker context, but Docker Desktop shows containers from the `desktop-linux` context.

## Why This Happens

Docker has multiple contexts:
- **`default`**: Uses `unix:///var/run/docker.sock` (system Docker daemon)
- **`desktop-linux`**: Uses `unix:///home/prometheus/.docker/desktop/docker.sock` (Docker Desktop)

**Docker Desktop only shows containers from the `desktop-linux` context.**

## Current Situation

Your containers are running in the `default` context:
```bash
docker context use default
docker ps  # Shows tara-* containers
```

But Docker Desktop shows the `desktop-linux` context:
```bash
docker context use desktop-linux
docker ps  # No tara-* containers (they're in default context)
```

## Solution Options

### Option 1: Use desktop-linux Context (Recommended for Docker Desktop)

Stop containers in default context and restart in desktop-linux:

```bash
# Stop containers in default context
docker context use default
docker-compose -p tara-microservice -f docker-compose-tara.yml down

# Start in desktop-linux context (visible in Docker Desktop)
docker context use desktop-linux
docker-compose -p tara-microservice -f docker-compose-tara.yml up -d --build
```

### Option 2: Always Use desktop-linux Context

Set desktop-linux as default:
```bash
docker context use desktop-linux
# Or add to your shell profile:
echo 'docker context use desktop-linux' >> ~/.bashrc
```

### Option 3: View Containers in Default Context

If you want to keep using default context:
```bash
docker context use default
docker ps  # See your containers
# But they won't appear in Docker Desktop GUI
```

## Quick Fix

Run this to move containers to desktop-linux context:

```bash
cd /home/prometheus/leibniz_agent/services

# Stop in default context
docker context use default
docker-compose -p tara-microservice -f docker-compose-tara.yml down

# Start in desktop-linux context (visible in Docker Desktop)
docker context use desktop-linux
docker-compose -p tara-microservice -f docker-compose-tara.yml up -d
```

## Verify

After switching to desktop-linux context:
```bash
docker context use desktop-linux
docker ps | grep tara
# Should show containers, and they'll be visible in Docker Desktop
```

