# Leibniz RAG Microservice

Standalone FastAPI service providing context-aware retrieval-augmented generation (RAG) for the Leibniz University Agent using FAISS vector search, Gemini 2.0 Flash Lite, and Redis caching.

## Overview

This microservice extracts RAG functionality from the monolithic `leibniz_rag.py` (1778 lines) into a scalable, independently deployable service. It provides intelligent document chunking, semantic search, entity-aware retrieval, response humanization, and quality validation.

**Key Features**:
- **FAISS Vector Search**: Pre-built IndexFlatL2 with 384-dim embeddings (all-MiniLM-L6-v2)
- **Redis Distributed Cache**: 1-hour TTL with md5-based cache keys
- **Gemini 2.0 Flash Lite**: Response generation with streaming support
- **Intelligent Chunking**: FAQ Q&A pairs, markdown sections, semantic paragraphs
- **Entity Boosting**: Category-based document ranking (admission, student_services, academic, contact)
- **Response Humanization**: Conversational starters, formal prefix removal, quality validation
- **Docker Multi-Stage Build**: Pre-built index at build time, minimal runtime image (~500MB)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Leibniz Orchestrator                        │
│                 (leibniz_pro.py or leibniz_vad.py)               │
└────────────────┬────────────────────────────────────────────────┘
                 │ POST /api/v1/query
                 │ {"query": "...", "context": {...}}
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RAG Microservice (Port 8003)                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ FastAPI App (app.py)                                       │  │
│  │  • Lifespan: Load config, create engine, connect Redis   │  │
│  │  • POST /api/v1/query (Redis cache: rag:{md5})           │  │
│  │  • GET /health (status, cache hit rate, uptime)          │  │
│  │  • GET /metrics (performance stats, index stats)         │  │
│  │  • POST /api/v1/admin/rebuild_index (rebuild, clear cache)│  │
│  └───────────────┬───────────────────────────────────────────┘  │
│                  │                                               │
│  ┌───────────────▼───────────────────────────────────────────┐  │
│  │ RAG Engine (rag_engine.py)                                │  │
│  │  1. Extract query from context (extracted_meaning > user_goal)│
│  │  2. Enrich with entities from context                     │  │
│  │  3. Embed query (HuggingFace embeddings)                  │  │
│  │  4. FAISS search (top_k=8, similarity >0.3)               │  │
│  │  5. Entity boosting (admission +10, services +10, ...)    │  │
│  │  6. Select top_n=5 documents                              │  │
│  │  7. Gemini generation (temperature=0.7, max_tokens=600)   │  │
│  │  8. Humanize response (conversational starters, casing)   │  │
│  │  9. Validate quality (detect formal language, length)     │  │
│  └───────────────┬───────────────────────────────────────────┘  │
│                  │                                               │
│  ┌───────────────▼───────────────────────────────────────────┐  │
│  │ FAISS Index (Pre-built at Docker build time)              │  │
│  │  • index.faiss (IndexFlatL2, 384-dim)                     │  │
│  │  • metadata.json (categories, filenames, priorities)      │  │
│  │  • texts.json (chunk content)                             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Redis (Port 6379)                            │
│  Cache keys: rag:{md5(query)}                                    │
│  TTL: 3600s (1 hour)                                             │
└─────────────────────────────────────────────────────────────────┘
```

**Processing Flow**:
1. **Request arrives**: Orchestrator sends POST /api/v1/query with query + optional context
2. **Cache check**: Generate md5 cache key, check Redis (hit → return cached response)
3. **Query processing**: Extract query from context, enrich with entities, embed with HuggingFace
4. **FAISS retrieval**: Search top_k=8 documents, filter by similarity >0.3
5. **Entity boosting**: Boost category-relevant documents (admission queries → boost admission docs)
6. **Response generation**: Build prompt with top_n=5 docs, call Gemini 2.0 Flash Lite
7. **Humanization**: Remove formal prefixes, add conversational starters, enforce length
8. **Quality validation**: Check for formal language, jargon, length issues, unhelpfulness
9. **Cache storage**: Store response in Redis with 1-hour TTL
10. **Return**: Send structured QueryResponse with answer, sources, confidence, timing

## API Endpoints

### POST /api/v1/query

Process RAG query with optional context and caching.

**Request**:
```json
{
  "query": "What are the admission requirements?",
  "context": {
    "extracted_meaning": "admission process and requirements",
    "user_goal": "learn about university admission",
    "key_entities": ["admission", "requirements", "application"]
  },
  "stream": false
}
```

**Response**:
```json
{
  "answer": "To apply to Leibniz University, you'll need to submit your high school transcripts, proof of English proficiency (TOEFL/IELTS), and a completed application form by January 15th. For undergraduate programs, a minimum GPA of 3.0 is recommended.",
  "sources": [
    {"text": "Admission requirements include...", "metadata": {"category": "01_admission", "filename": "admission_faq.md"}},
    {"text": "Application deadlines...", "metadata": {"category": "01_admission", "filename": "deadlines.md"}}
  ],
  "confidence": 0.85,
  "timing_breakdown": {
    "embedding_time": 0.045,
    "retrieval_time": 0.012,
    "generation_time": 1.234,
    "total_time": 1.291
  },
  "cached": false,
  "metadata": {
    "quality_score": 0.9,
    "humanization_applied": true,
    "response_length": 186,
    "top_k_retrieved": 8,
    "top_n_used": 5
  }
}
```

**Parameters**:
- `query` (string, required): User question or statement
- `context` (object, optional): Extracted context from intent parser
  - `extracted_meaning`: Cleaned query semantics (highest priority for retrieval)
  - `user_goal`: User's intent description
  - `key_entities`: List of relevant entities to boost retrieval
- `stream` (boolean, optional): Enable streaming response (default: false)

**Curl Example**:
```bash
curl -X POST http://localhost:8003/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Where is the university located?",
    "context": {
      "extracted_meaning": "university location",
      "key_entities": ["location", "address", "campus"]
    }
  }'
```

---

### GET /health

Health check with service status and metrics.

**Response**:
```json
{
  "status": "healthy",
  "index_loaded": true,
  "index_size": 342,
  "cache_hit_rate": 0.42,
  "redis_connected": true,
  "gemini_available": true,
  "uptime_seconds": 3642
}
```

**Status Levels**:
- `healthy`: All services operational (index loaded, Redis connected, Gemini available)
- `degraded`: Partial functionality (e.g., Redis unavailable, cache disabled)
- `unhealthy`: Critical failure (index not loaded, Gemini API error)

**Curl Example**:
```bash
curl http://localhost:8003/health
```

---

### GET /metrics

Performance metrics and statistics.

**Response**:
```json
{
  "rag_engine": {
    "total_queries": 1543,
    "average_response_time": 1.287,
    "vector_store_size": 342,
    "embeddings_available": true,
    "gemini_available": true
  },
  "cache": {
    "hits": 648,
    "misses": 895,
    "hit_rate": 0.42
  },
  "index": {
    "total_documents": 342,
    "categories": ["01_admission", "02_student_services", "03_academic", "04_contact"],
    "embedding_dimension": 384
  },
  "uptime_seconds": 3642
}
```

**Curl Example**:
```bash
curl http://localhost:8003/metrics
```

---

### POST /api/v1/admin/rebuild_index

Rebuild FAISS index and clear cache (admin only).

**Request**:
```json
{
  "knowledge_base_path": "/app/leibniz_knowledge_base",
  "output_path": "/app/index"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Index rebuilt successfully",
  "index_stats": {
    "total_documents": 342,
    "categories": ["01_admission", "02_student_services", "03_academic", "04_contact"],
    "embedding_dimension": 384
  },
  "cache_cleared": true,
  "build_time_seconds": 12.45
}
```

**Curl Example**:
```bash
curl -X POST http://localhost:8003/api/v1/admin/rebuild_index \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_base_path": "/app/leibniz_knowledge_base",
    "output_path": "/app/index"
  }'
```

## Knowledge Base Structure

Knowledge base organized by category prefixes for entity boosting:

```
leibniz_knowledge_base/
├── 01_admission/               # Admission and enrollment (+10 boost)
│   ├── admission_faq.md
│   ├── application_process.md
│   └── deadlines.md
├── 02_student_services/        # Student support and services (+10 boost)
│   ├── housing.md
│   ├── financial_aid.md
│   └── counseling.md
├── 03_academic/                # Programs and courses (+6 boost)
│   ├── undergraduate_programs.md
│   ├── graduate_programs.md
│   └── course_catalog.md
└── 04_contact/                 # Contact information (+10 boost)
    ├── department_contacts.md
    └── office_hours.md
```

**Document Format** (Markdown):
```markdown
# Section Title

## Question 1
Answer to question 1 with detailed information.

## Question 2
Answer to question 2 with detailed information.

---

**Note**: Additional notes or disclaimers.
```

**Chunking Strategy**:
1. **FAQ Q&A pairs**: Detects `Q1:`, `Q2:`, etc. and splits into paired chunks
2. **Markdown sections**: Splits by headers (`##`, `###`) for structural chunks
3. **Semantic paragraphs**: Splits by double newlines for semantic coherence
4. **Sentence-based**: Fallback with overlap (100 chars) for large paragraphs

**Chunk Size Constraints**:
- Min: 500 chars
- Max: 800 chars
- Overlap: 100 chars (for sentence-based chunking)

## FAISS Index

**Build Process** (Docker build time):
1. Scan knowledge base directories (01_admission, 02_student_services, ...)
2. Apply intelligent chunking (FAQ → sections → semantic → sentences)
3. Assign priorities (category keywords, university terms, length)
4. Generate embeddings (all-MiniLM-L6-v2, 384-dim, normalized)
5. Build FAISS IndexFlatL2 (L2 distance, converted to cosine similarity)
6. Save index.faiss, metadata.json, texts.json to `/app/index`

**Index Files**:
- `index.faiss`: FAISS IndexFlatL2 binary (dimension=384)
- `metadata.json`: Per-document metadata (category, filename, priority, source)
- `texts.json`: Chunk content strings

**Retrieval Process**:
1. Embed query with all-MiniLM-L6-v2 (384-dim)
2. FAISS search top_k=8 documents
3. Convert L2 distances to cosine similarity: `1 - (distance / 2)`
4. Filter by similarity threshold (default: 0.3)
5. Apply entity boosting (admission +10, services +10, academic +6, contact +10)
6. Sort by boosted scores, select top_n=5
7. Return documents with metadata

**Dynamic Embedding Dimension**:
```python
# Reads dimension from index.d (FAISS attribute) or infers from embeddings
def get_index_stats():
    if self.index:
        dimension = self.index.d  # From FAISS index
    else:
        # Infer from embeddings
        test_embedding = self.embeddings.embed_query("test")
        dimension = len(test_embedding)
    return {"embedding_dimension": dimension, ...}
```

## Configuration

All settings via environment variables in `.env.leibniz`:

### Core Settings
```bash
# Gemini API
GEMINI_API_KEY=your_api_key_here
LEIBNIZ_RAG_GEMINI_MODEL=gemini-2.0-flash-exp  # Or gemini-2.0-flash-lite

# Service
LEIBNIZ_RAG_SERVICE_PORT=8003
LEIBNIZ_RAG_WORKERS=1  # Single worker for persistent services

# Redis
LEIBNIZ_RAG_REDIS_HOST=redis
LEIBNIZ_RAG_REDIS_PORT=6379
LEIBNIZ_RAG_CACHE_TTL=3600  # 1 hour
```

### Paths
```bash
LEIBNIZ_RAG_KNOWLEDGE_BASE_PATH=./leibniz_knowledge_base
LEIBNIZ_RAG_INDEX_DIR=./index
LEIBNIZ_RAG_LOG_DIR=./logs
```

### Retrieval Parameters
```bash
LEIBNIZ_RAG_TOP_K=8              # Retrieve 8 documents from FAISS
LEIBNIZ_RAG_TOP_N=5              # Use top 5 for generation
LEIBNIZ_RAG_SIMILARITY_THRESHOLD=0.3  # Filter documents below 0.3 similarity
```

### Chunking Parameters
```bash
LEIBNIZ_RAG_CHUNK_SIZE_MIN=500   # Minimum chunk size (chars)
LEIBNIZ_RAG_CHUNK_SIZE_MAX=800   # Maximum chunk size (chars)
LEIBNIZ_RAG_CHUNK_OVERLAP=100    # Overlap for sentence-based chunking
```

### Response Parameters
```bash
LEIBNIZ_RAG_RESPONSE_STYLE=friendly_casual  # Or formal_professional
LEIBNIZ_RAG_MAX_RESPONSE_LENGTH=500         # Max response length (chars)
LEIBNIZ_RAG_ENABLE_HUMANIZATION=true        # Apply humanization
LEIBNIZ_RAG_ENABLE_CACHE=true               # Enable Redis caching
LEIBNIZ_RAG_ENABLE_STREAMING=true           # Support streaming responses
```

## Local Development

### Prerequisites
- Python 3.9+
- Redis server (or Docker)
- Gemini API key

### Setup
```bash
# Clone repository
cd SINDH-Orchestra-Complete/leibniz_agent/services/rag

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.leibniz.example .env.leibniz
# Edit .env.leibniz with your GEMINI_API_KEY

# Build FAISS index (one-time)
python -m leibniz_agent.services.rag.index_builder \
  --knowledge-base ../../leibniz_knowledge_base \
  --output ./index \
  --rebuild

# Start Redis (if not running)
docker run -d -p 6379:6379 redis:7-alpine

# Run service
python -m leibniz_agent.services.rag.app
# or
uvicorn leibniz_agent.services.rag.app:app --reload --port 8003
```

### Testing
```bash
# Run all tests
pytest leibniz_agent/services/rag/tests/ -v

# Run unit tests only
pytest leibniz_agent/services/rag/tests/test_unit.py -v -m unit

# Run integration tests (requires Redis)
pytest leibniz_agent/services/rag/tests/test_integration.py -v -m integration

# Run service tests (requires running service)
pytest leibniz_agent/services/rag/tests/test_rag_engine.py -v -m requires_service

# Skip slow tests
pytest -v -m "not slow"
```

### Rebuild Index
```bash
# Via CLI
python -m leibniz_agent.services.rag.index_builder \
  --knowledge-base ./leibniz_knowledge_base \
  --output ./index \
  --rebuild

# Via API (requires running service)
curl -X POST http://localhost:8003/api/v1/admin/rebuild_index \
  -H "Content-Type: application/json" \
  -d '{"knowledge_base_path": "./leibniz_knowledge_base", "output_path": "./index"}'
```

## Docker Deployment

> **🚀 Quick Start**: Use the helper scripts for fast development!
> - **PowerShell**: `.\rag-dev.ps1 build; .\rag-dev.ps1 start; .\rag-dev.ps1 setup; .\rag-dev.ps1 run`
> - **Bash**: `./rag-dev.sh build; ./rag-dev.sh start; ./rag-dev.sh setup; ./rag-dev.sh run`
> 
> See [QUICK_START.md](./QUICK_START.md) for detailed workflow or [DOCKER_DEV_GUIDE.md](./DOCKER_DEV_GUIDE.md) for comprehensive documentation.

### Development vs Production Modes

The Dockerfile is optimized for **fast development cycles** by separating lightweight dependencies (installed during build) from heavy dependencies like PyTorch (installed interactively).

#### Development Mode (Default) - Fast Rebuilds ⚡

**Benefits**:
- **Build time**: ~2 minutes (excludes torch/sentence-transformers)
- **Code changes**: Instant (volume-mounted, auto-reload)
- **Setup**: One-time 5-min interactive installation

**Workflow**:
```powershell
# 1. Build lightweight image (2 min)
docker-compose -f docker-compose.leibniz.yml build rag

# 2. Start container in dev mode
docker-compose -f docker-compose.leibniz.yml up -d rag

# 3. Install heavy deps (one-time, 5 min)
docker exec -it leibniz-rag /app/setup_heavy_deps.sh

# 4. Start service with hot-reload
docker exec -it leibniz-rag uvicorn leibniz_agent.services.rag.app:app \
    --host 0.0.0.0 --port 8003 --reload
```

**Why faster?**  
Skips ~2GB PyTorch download during build → 6-8 minutes saved per rebuild. Install once, edit code freely!

#### Production Mode - Auto-Setup 🚀

**Benefits**:
- **First start**: ~8 minutes (installs torch + builds index)
- **Subsequent starts**: ~2 seconds (everything pre-installed)
- **No manual steps**: Fully automated

**Workflow**:
```powershell
# One command starts everything
docker-compose -f docker-compose.leibniz.yml --profile production up -d rag-prod
```

Auto-runs `setup_heavy_deps.sh` on first start, then starts service. Subsequent restarts are instant.

### Multi-Stage Build (Technical Details)
```dockerfile
# Stage 1: Builder (install dependencies)
FROM python:3.9-slim AS builder
RUN apt-get update && apt-get install -y gcc g++
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Indexer (build FAISS index at build time)
FROM python:3.9-slim AS indexer
COPY --from=builder /root/.local /root/.local
COPY leibniz_agent/services/rag /app/leibniz_agent/services/rag
COPY leibniz_knowledge_base /app/leibniz_knowledge_base
WORKDIR /app
RUN python -m leibniz_agent.services.rag.index_builder \
    --knowledge-base /app/leibniz_knowledge_base \
    --output /app/index

# Stage 3: Runtime (minimal image with pre-built index)
FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
COPY --from=indexer /app/index /app/index
COPY leibniz_agent/services/rag /app/leibniz_agent/services/rag
WORKDIR /app
EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health')"
CMD ["uvicorn", "leibniz_agent.services.rag.app:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
```

### Build and Run
```bash
# Build image
docker build -f leibniz_agent/services/rag/Dockerfile -t leibniz-rag:latest .

# Run container
docker run -d \
  --name leibniz-rag \
  -p 8003:8003 \
  -e GEMINI_API_KEY=your_api_key_here \
  -e LEIBNIZ_RAG_REDIS_HOST=redis \
  -v $(pwd)/leibniz_knowledge_base:/app/leibniz_knowledge_base:ro \
  leibniz-rag:latest

# Check logs
docker logs -f leibniz-rag

# Check health
curl http://localhost:8003/health
```

### Docker Compose
```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  rag:
    build:
      context: .
      dockerfile: leibniz_agent/services/rag/Dockerfile
    ports:
      - "8003:8003"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - LEIBNIZ_RAG_REDIS_HOST=redis
      - LEIBNIZ_RAG_REDIS_PORT=6379
      - LEIBNIZ_RAG_TOP_K=8
      - LEIBNIZ_RAG_TOP_N=5
      - LEIBNIZ_RAG_SIMILARITY_THRESHOLD=0.3
      - LEIBNIZ_RAG_ENABLE_CACHE=true
      - LEIBNIZ_RAG_CACHE_TTL=3600
    volumes:
      - ./leibniz_knowledge_base:/app/leibniz_knowledge_base:ro
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8003/health')"]
      interval: 30s
      timeout: 5s
      retries: 3
```

**Start services**:
```bash
docker-compose -f docker-compose.leibniz.yml up -d rag
```

## Performance

### Benchmarks
- **Cache hit**: 1-5ms (200-600x speedup)
- **Cache miss**: 500-2000ms
  - Embedding: 40-60ms
  - FAISS retrieval: 10-20ms
  - Gemini generation: 800-1500ms
  - Humanization: 5-10ms
- **Memory**: ~300MB (FAISS index + models)
- **Startup**: ~5s (load index, connect Redis, initialize Gemini)

### Expected Hit Rates
- Query cache: 40-60% (depends on query diversity)
- Similar queries benefit from md5 caching (exact match only)

### Optimization Tips
1. **Pre-build index at Docker build time** (avoid runtime overhead)
2. **Use Redis for distributed caching** (share cache across replicas)
3. **Tune top_k/top_n** (higher values = more context but slower generation)
4. **Enable streaming** (reduces perceived latency for long responses)
5. **Monitor cache hit rate** (low rate → adjust TTL or cache strategy)

## Monitoring

### Health Checks
```bash
# Basic health
curl http://localhost:8003/health

# Expected: {"status": "healthy", "index_loaded": true, ...}

# Metrics
curl http://localhost:8003/metrics

# Expected: {"rag_engine": {"total_queries": 1543, ...}, ...}
```

### Logs
```bash
# Docker logs
docker logs -f leibniz-rag

# Local logs (if configured)
tail -f logs/rag_service.log
```

### Key Metrics
- **cache_hit_rate**: Should be >0.4 (40%) for good performance
- **average_response_time**: Should be <2s for cache misses
- **index_loaded**: Must be true for service to function
- **redis_connected**: False = cache disabled (degraded mode)
- **gemini_available**: False = generation fails (unhealthy)

### Alerts
- **cache_hit_rate < 0.3**: Review query patterns, consider increasing TTL
- **average_response_time > 3s**: Check Gemini API latency, optimize retrieval
- **index_loaded = false**: Rebuild index, check knowledge base path
- **redis_connected = false**: Check Redis service, network connectivity

## Troubleshooting

### Issue: Index not loaded
**Symptoms**: Health status "unhealthy", index_loaded=false

**Causes**:
- Index files missing (index.faiss, metadata.json, texts.json)
- Incorrect LEIBNIZ_RAG_INDEX_DIR path
- Permissions issue (Docker volume mount)

**Solutions**:
```bash
# Check index files exist
ls -la /app/index/
# Expected: index.faiss, metadata.json, texts.json

# Rebuild index
python -m leibniz_agent.services.rag.index_builder \
  --knowledge-base /app/leibniz_knowledge_base \
  --output /app/index \
  --rebuild

# Check permissions (Docker)
docker exec leibniz-rag ls -la /app/index/
```

---

### Issue: Redis connection failed
**Symptoms**: Health status "degraded", redis_connected=false, cache disabled

**Causes**:
- Redis service not running
- Incorrect LEIBNIZ_RAG_REDIS_HOST/PORT
- Network connectivity issue

**Solutions**:
```bash
# Check Redis running
docker ps | grep redis
# or
redis-cli ping  # Expected: PONG

# Check connection from container
docker exec leibniz-rag ping -c 3 redis

# Check environment variables
docker exec leibniz-rag env | grep REDIS
```

---

### Issue: Gemini API errors
**Symptoms**: Health status "unhealthy", gemini_available=false, 500 errors on /query

**Causes**:
- Invalid GEMINI_API_KEY
- API quota exceeded
- Model name incorrect (LEIBNIZ_RAG_GEMINI_MODEL)

**Solutions**:
```bash
# Verify API key
echo $GEMINI_API_KEY  # Should be valid key

# Test API directly
curl https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY

# Check quota
# Visit: https://aistudio.google.com/app/apikey

# Use alternative model
export LEIBNIZ_RAG_GEMINI_MODEL=gemini-2.0-flash-lite
```

---

### Issue: Low cache hit rate (<30%)
**Symptoms**: cache_hit_rate <0.3, high average_response_time

**Causes**:
- High query diversity (unique queries)
- Low TTL (cache expires too quickly)
- md5 cache keys (no semantic matching)

**Solutions**:
```bash
# Increase TTL to 2 hours
export LEIBNIZ_RAG_CACHE_TTL=7200

# Monitor query patterns
curl http://localhost:8003/metrics | jq '.cache'

# Consider semantic cache keys (future enhancement)
```

---

### Issue: Responses too formal/too long
**Symptoms**: Responses use formal language ("pursuant to", "aforementioned"), exceed max_length

**Causes**:
- Humanization disabled (LEIBNIZ_RAG_ENABLE_HUMANIZATION=false)
- Quality validation thresholds too low
- Response style misconfigured

**Solutions**:
```bash
# Enable humanization
export LEIBNIZ_RAG_ENABLE_HUMANIZATION=true

# Set casual style
export LEIBNIZ_RAG_RESPONSE_STYLE=friendly_casual

# Reduce max length
export LEIBNIZ_RAG_MAX_RESPONSE_LENGTH=400

# Check validation logic in rag_engine.py (validate_response_quality)
```

---

### Issue: Poor retrieval accuracy
**Symptoms**: Irrelevant sources in response, low confidence scores

**Causes**:
- Similarity threshold too low (retrieves irrelevant docs)
- Entity boosting not working (category mismatch)
- Knowledge base incomplete

**Solutions**:
```bash
# Increase similarity threshold
export LEIBNIZ_RAG_SIMILARITY_THRESHOLD=0.5

# Review entity boosting (rag_engine.py apply_entity_boosting)
# Ensure categories match: admission, student_services, academic, contact

# Add/update knowledge base documents
# Rebuild index after changes
curl -X POST http://localhost:8003/api/v1/admin/rebuild_index
```

## Integration with Orchestrator

### From Leibniz Agent (leibniz_pro.py)
```python
import httpx
from leibniz_agent.services.rag.app import QueryRequest

async def process_rag_query(query: str, context: dict = None):
    """Query RAG microservice."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8003/api/v1/query",
            json={
                "query": query,
                "context": context,
                "stream": False
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

# Usage
context = {
    "extracted_meaning": "admission requirements for undergraduate",
    "user_goal": "learn about admission process",
    "key_entities": ["admission", "requirements", "undergraduate"]
}
result = await process_rag_query("What do I need to apply?", context)
print(result['answer'])
```

### From Intent Parser (leibniz_intent_parser.py)
```python
# Extract context for RAG
intent_result = await parse_intent(user_input)

if intent_result['should_use_rag']:
    # Build context from intent parsing
    context = {
        "extracted_meaning": intent_result.get('extracted_meaning'),
        "user_goal": intent_result.get('user_goal'),
        "key_entities": intent_result.get('key_entities', [])
    }
    
    # Query RAG with enriched context
    rag_result = await process_rag_query(user_input, context)
    response = rag_result['answer']
else:
    # Handle non-RAG intents (appointment scheduling, etc.)
    response = await handle_intent(intent_result)
```

### Docker Compose Integration
```yaml
services:
  orchestrator:
    build: .
    environment:
      - LEIBNIZ_RAG_SERVICE_URL=http://rag:8003
    depends_on:
      - rag
  
  rag:
    build:
      context: .
      dockerfile: leibniz_agent/services/rag/Dockerfile
    # ... (see Docker Deployment section)
```

---

## Development Roadmap

### Phase 5 (Future Enhancements)
- [ ] Semantic cache keys (similarity-based caching)
- [ ] Hybrid search (FAISS + keyword matching)
- [ ] Multi-modal support (image-based retrieval)
- [ ] A/B testing for response styles
- [ ] Prometheus metrics export
- [ ] Automated index rebuilding (file watchers)
- [ ] Query analytics dashboard
- [ ] Response feedback loop (user ratings)

---

## License

MIT License - See repository root for details.

## Support

For issues or questions:
- GitHub Issues: [SINDH-Orchestra-Complete](https://github.com/yourusername/SINDH-Orchestra-Complete)
- Documentation: `leibniz_agent/services/rag/README.md`
- Integration Guide: `leibniz_agent/INTEGRATION_GUIDE.md`
