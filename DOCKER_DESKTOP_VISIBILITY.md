# Docker Desktop Visibility Fix

## Issue
Containers not appearing in Docker Desktop GUI under the "tara-task" project folder.

## Solution

Docker Desktop groups containers by the `com.docker.compose.project` label. This label is set when using `docker-compose` with the `COMPOSE_PROJECT_NAME` environment variable.

### Method 1: Use Environment Variable (Recommended)

```bash
docker context use desktop-linux
export COMPOSE_PROJECT_NAME=tara-task
docker-compose -f docker-compose-tara-task.yml up -d
```

### Method 2: Use --project-name Flag

```bash
docker context use desktop-linux
docker-compose -f docker-compose-tara-task.yml --project-name tara-task up -d
```

### Method 3: Use .env File

Create `.env` file in TARA-MICROSERVICE directory:
```
COMPOSE_PROJECT_NAME=tara-task
```

Then:
```bash
docker context use desktop-linux
docker-compose -f docker-compose-tara-task.yml up -d
```

## Verification

Check if containers have the correct project label:

```bash
docker ps --format "{{.Names}}\t{{.Label \"com.docker.compose.project\"}}" | grep tara-task
```

All containers should show `tara-task` as the project name.

## Docker Desktop GUI

After starting with the correct project name:
1. Open Docker Desktop
2. Go to Containers tab
3. Look for "tara-task" folder/project
4. All containers should be grouped under it

## Troubleshooting

If containers still don't appear:

1. **Refresh Docker Desktop**: Press F5 or restart Docker Desktop
2. **Check Filters**: Make sure "All" containers are shown (not just running)
3. **Verify Context**: `docker context show` should show `desktop-linux`
4. **Check Labels**: Run verification script: `./verify-docker-desktop.sh`

## Updated Startup Script

The `start-tara-task.sh` script now sets `COMPOSE_PROJECT_NAME=tara-task` automatically.




