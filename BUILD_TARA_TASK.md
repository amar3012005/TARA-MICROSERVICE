# Building TARA-TASK Microservices for Docker Desktop Linux

This guide explains how to build all microservices for the tara-task cluster specifically for Docker Desktop Linux.

## Quick Start

### Option 1: Use the Build Script (Recommended)

```bash
cd TARA-MICROSERVICE
./build-tara-task.sh
```

This script will:
1. Set Docker context to `desktop-linux`
2. Build all services in parallel
3. Verify the build completed successfully
4. Show you the built images

### Option 2: Manual Build

```bash
# Set Docker context
docker context use desktop-linux

# Set project name (for Docker Desktop organization)
export COMPOSE_PROJECT_NAME=tara-task

# Build all services
docker-compose -f docker-compose-tara-task.yml build --parallel
```

### Option 3: Build Individual Services

```bash
docker context use desktop-linux
export COMPOSE_PROJECT_NAME=tara-task

# Build specific service
docker-compose -f docker-compose-tara-task.yml build orchestrator
docker-compose -f docker-compose-tara-task.yml build stt-vad
docker-compose -f docker-compose-tara-task.yml build rag
docker-compose -f docker-compose-tara-task.yml build tts-sarvam
```

## Verify Build

After building, verify all images are created:

```bash
./verify-tara-task-build.sh
```

Or manually:

```bash
export COMPOSE_PROJECT_NAME=tara-task
docker-compose -f docker-compose-tara-task.yml images
```

## Services Being Built

| Service | Image Name | Description | Build Time |
|---------|-----------|-------------|------------|
| **Redis** | `redis:7-alpine` | Official Redis image (pulled, not built) | ~30s |
| **Orchestrator** | `tara-task-orchestrator` | Master controller service | ~3-5 min |
| **STT-VAD** | `tara-task-stt-vad` | Speech-to-text service | ~5-8 min |
| **RAG** | `tara-task-rag` | Knowledge base service | ~10-15 min |
| **TTS Sarvam** | `tara-task-tts-sarvam` | Text-to-speech service | ~3-5 min |

**Total build time (first time)**: ~20-35 minutes  
**Subsequent builds**: ~2-5 minutes (Docker cache used)

## Docker Desktop Visibility

For images to appear correctly in Docker Desktop:

1. **Set Docker Context**: Must use `desktop-linux` context
   ```bash
   docker context use desktop-linux
   ```

2. **Set Project Name**: Use `COMPOSE_PROJECT_NAME=tara-task`
   ```bash
   export COMPOSE_PROJECT_NAME=tara-task
   ```

3. **Build with docker-compose**: This ensures proper labeling
   ```bash
   docker-compose -f docker-compose-tara-task.yml build
   ```

After building, images will appear in Docker Desktop:
- **Images tab**: Look for `tara-task-*` images
- **Containers tab**: When running, containers grouped under "tara-task" project

## Troubleshooting

### Build Fails with "Context Not Found"

```bash
# List available contexts
docker context ls

# If desktop-linux doesn't exist, create it
docker context create desktop-linux --docker "host=unix:///var/run/docker.sock"
docker context use desktop-linux
```

### Images Not Appearing in Docker Desktop

1. **Verify context**:
   ```bash
   docker context show
   # Should output: desktop-linux
   ```

2. **Check project name**:
   ```bash
   echo $COMPOSE_PROJECT_NAME
   # Should output: tara-task
   ```

3. **Refresh Docker Desktop**: Press F5 or restart Docker Desktop

4. **Check images manually**:
   ```bash
   docker images | grep tara-task
   ```

### Build Takes Too Long

- **First build**: Normal, downloading all dependencies
- **Subsequent builds**: Should be fast due to Docker cache
- **Force rebuild**: Use `--no-cache` flag (slower but clean)
  ```bash
  docker-compose -f docker-compose-tara-task.yml build --no-cache
  ```

### Out of Disk Space

Docker images can be large. Check space:

```bash
docker system df
```

Clean up unused images:

```bash
docker image prune -a
```

### Network Issues During Build

If package downloads fail:

1. Check internet connection
2. Try building with verbose output:
   ```bash
   docker-compose -f docker-compose-tara-task.yml build --progress=plain
   ```
3. Check Docker daemon is running:
   ```bash
   docker info
   ```

## Build Details

### Orchestrator Service
- **Dockerfile**: `orchestrator/Dockerfile`
- **Dependencies**: Python 3.11, Docker CLI, FastAPI, Uvicorn
- **Size**: ~500MB

### STT-VAD Service
- **Dockerfile**: `stt_vad/Dockerfile`
- **Dependencies**: Python 3.11, Gradio, FastRTC, Gemini API
- **Size**: ~1.2GB

### RAG Service
- **Dockerfile**: `rag/Dockerfile.tara`
- **Dependencies**: Python 3.11, PyTorch, Sentence Transformers, FAISS
- **Size**: ~8GB (includes pre-built FAISS index)
- **Note**: Builds FAISS index from `task_knowledge_base/` during build

### TTS Sarvam Service
- **Dockerfile**: `tts_sarvam/Dockerfile`
- **Dependencies**: Python 3.11, FastAPI, Sarvam API client
- **Size**: ~1.2GB

## Next Steps

After successful build:

1. **Start services**:
   ```bash
   ./start-tara-task.sh
   ```

2. **Check logs**:
   ```bash
   docker logs tara-task-orchestrator -f
   ```

3. **Verify health**:
   ```bash
   curl http://localhost:6004/health
   ```

## Additional Resources

- [README-TARA-TASK.md](README-TARA-TASK.md) - Complete service documentation
- [DOCKER_DESKTOP_VISIBILITY.md](DOCKER_DESKTOP_VISIBILITY.md) - Docker Desktop setup
- [start-tara-task.sh](start-tara-task.sh) - Startup script




