# Docker Setup Guide - Leibniz Agent Services

This guide covers building and running all microservices using Docker.

## Prerequisites

- Docker Engine 20.10+ and Docker Compose 2.0+
- Knowledge base directory `leibniz_knowledge_base/` at repository root
- Gemini API key

## Quick Start

### 1. Set Environment Variables

Create a `.env` file in the `services/` directory (or export variables):

```bash
export GEMINI_API_KEY=your_api_key_here
```

### 2. Start All Services

```bash
cd services
docker-compose up -d
```

This will start:
- **Redis** (port 6379) - Caching service
- **STT/VAD Service** (port 8001) - Speech-to-text and voice activity detection
- **Intent Service** (port 8002) - Intent classification
- **RAG Service** (port 8003) - Knowledge base queries

### 3. Verify Services

```bash
# Check all services are running
docker-compose ps

# Check logs
docker-compose logs -f

# Test individual services
curl http://localhost:8001/health  # STT/VAD
curl http://localhost:8002/health  # Intent
curl http://localhost:8003/health  # RAG
```

## Building Individual Services

### Build Script

```bash
cd services
./build_docker_services.sh
```

This builds:
- `leibniz-stt-vad:latest`
- `leibniz-intent:latest`
- `leibniz-rag:latest`

### Manual Build

```bash
# From services/ directory
docker build -f stt_vad/Dockerfile -t leibniz-stt-vad:latest .
docker build -f intent/Dockerfile -t leibniz-intent:latest .
docker build -f rag/Dockerfile -t leibniz-rag:latest .
```

**Note**: RAG service build takes ~8-10 minutes first time (downloads torch ~2GB and builds FAISS index).

## Service Details

### STT/VAD Service (Port 8001)

- **Build time**: ~5 minutes
- **Image size**: ~1.5GB
- **Dependencies**: Gradio, FastRTC, Gemini Live API

### Intent Service (Port 8002)

- **Build time**: ~3 minutes
- **Image size**: ~800MB
- **Dependencies**: DistilBERT model, Spacy, Gemini API

### RAG Service (Port 8003)

- **Build time**: ~8-10 minutes (first time)
- **Image size**: ~2.5GB
- **Dependencies**: torch, sentence-transformers, FAISS, Gemini API
- **Index build**: Happens during Docker build (FAISS index from knowledge base)

## Docker Compose Configuration

See `services/docker-compose.yml` for:
- Service definitions
- Environment variables
- Network configuration
- Volume mounts
- Health checks

## Troubleshooting

### Services Won't Start

1. Check logs: `docker-compose logs [service-name]`
2. Verify environment variables are set
3. Check Redis is healthy: `docker-compose ps redis`
4. Verify knowledge base exists: `ls -la ../leibniz_knowledge_base/`

### RAG Service Issues

- **Index build failed**: Service will build index on first query (slower startup)
- **Missing knowledge base**: Ensure `leibniz_knowledge_base/` exists at repo root
- **Redis connection**: Check `LEIBNIZ_REDIS_HOST=redis` in environment

### Port Conflicts

If ports are already in use, modify `docker-compose.yml`:

```yaml
ports:
  - "8001:8001"  # Change first number to available port
```

## Development Workflow

### Rebuild After Code Changes

```bash
# Rebuild specific service
docker-compose build rag-service

# Rebuild and restart
docker-compose up -d --build rag-service
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f rag-service
```

### Execute Commands in Container

```bash
# Open shell in container
docker-compose exec rag-service bash

# Run Python script
docker-compose exec rag-service python -m leibniz_agent.services.rag.index_builder --rebuild
```

## Production Deployment

### Environment Variables

Set production values in `docker-compose.yml` or use `.env` file:

```bash
GEMINI_API_KEY=production_key
LEIBNIZ_REDIS_HOST=redis-production
LOG_LEVEL=INFO
```

### Resource Limits

Add to `docker-compose.yml`:

```yaml
services:
  rag-service:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Health Checks

All services include health checks. Monitor with:

```bash
docker-compose ps
# Check "State" column for "healthy"
```

## Network Architecture

```
Client → [STT/VAD:8001] → [Intent:8002] → [RAG:8003]
                              ↓
                         [Redis:6379]
```

All services communicate via `leibniz-network` bridge network.

## Next Steps

- See individual service READMEs for API documentation
- Test endpoints using curl or Postman
- Monitor performance via `/metrics` endpoints
- Set up logging aggregation for production







