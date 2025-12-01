# RAG Service - Docker Quick Start

## Prerequisites

- Docker and Docker Compose installed
- Knowledge base directory `leibniz_knowledge_base/` at repository root
- Gemini API key (set in environment or `.env` file)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# From repository root
cd services

# Set your Gemini API key
export GEMINI_API_KEY=your_api_key_here

# Start all services (Redis, STT/VAD, Intent, RAG)
docker-compose up -d

# Check RAG service logs
docker-compose logs -f rag-service

# Test the service
curl http://localhost:8003/health
```

### Option 2: Build and Run Manually

```bash
# From repository root
cd services

# Build the image (this will take 8-10 minutes first time)
docker build -f rag/Dockerfile -t leibniz-rag:latest .

# Run the container
docker run -d \
  --name rag-service \
  -p 8003:8003 \
  -e GEMINI_API_KEY=your_api_key_here \
  -e LEIBNIZ_REDIS_HOST=redis \
  --network leibniz-network \
  leibniz-rag:latest

# Check logs
docker logs -f rag-service
```

## Building the Image

The Dockerfile uses a multi-stage build:

1. **Builder stage**: Installs all Python dependencies including torch and sentence-transformers
2. **Indexer stage**: Builds the FAISS index from the knowledge base
3. **Runtime stage**: Creates the final production image

**Build time**: ~8-10 minutes (first time, includes downloading torch ~2GB)

**Subsequent builds**: ~2-3 minutes (if only code changed, Docker cache used)

## Environment Variables

Key environment variables (set in docker-compose.yml or docker run):

```bash
GEMINI_API_KEY=your_key_here              # Required
GEMINI_MODEL=gemini-2.0-flash-lite        # Optional, default shown
LEIBNIZ_REDIS_HOST=redis                  # Redis hostname
LEIBNIZ_REDIS_PORT=6379                   # Redis port
LEIBNIZ_RAG_ENABLE_HYBRID_SEARCH=true     # Enable hybrid search
```

## Health Check

```bash
# Check service health
curl http://localhost:8003/health

# Expected response:
{
  "status": "healthy",
  "index_loaded": true,
  "index_size": 1234,
  "cache_hit_rate": 0.0,
  "redis_connected": true,
  "gemini_available": true,
  "uptime_seconds": 123.45
}
```

## Testing the Service

```bash
# Query the knowledge base
curl -X POST http://localhost:8003/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the office hours?",
    "context": {
      "user_goal": "asking about office hours",
      "key_entities": {"department": "admissions"}
    }
  }'

# Expected response:
{
  "answer": "...",
  "sources": ["contact_information.md"],
  "confidence": 0.85,
  "timing_breakdown": {...},
  "cached": false
}
```

## Troubleshooting

### Index Build Failed During Docker Build

If the index build fails during `docker build`, the service will still start but will build the index on first query. To rebuild manually:

```bash
docker exec -it rag-service python -m leibniz_agent.services.rag.index_builder \
  --knowledge-base /app/leibniz_knowledge_base \
  --output /app/index \
  --rebuild
```

### Service Won't Start

Check logs:
```bash
docker logs rag-service
```

Common issues:
- Missing `GEMINI_API_KEY`: Set it in environment
- Redis not accessible: Check `LEIBNIZ_REDIS_HOST` and network connectivity
- Knowledge base missing: Ensure `leibniz_knowledge_base/` exists at repo root

### Index Not Loading

Verify index files exist:
```bash
docker exec rag-service ls -la /app/index/
# Should show: index.faiss, metadata.json, texts.json
```

Rebuild index if needed (see above).

## Performance

- **Cold start**: ~30 seconds (loads FAISS index into memory)
- **Query latency**: 200-800ms (depends on pattern detection and Gemini response time)
- **Hybrid search**: 50-70% token reduction for common queries (office hours, contact info, etc.)

## Production Notes

- The image includes all dependencies (~2.5GB) including torch
- FAISS index is built at image build time (faster startup)
- Use 1 worker (FAISS index is not thread-safe)
- Redis caching reduces duplicate query costs
- Health check ensures service is ready before accepting traffic



