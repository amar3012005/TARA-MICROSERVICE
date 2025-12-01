# RAG Service - Docker Development Guide

## 🎯 Overview

This Dockerfile is optimized for **fast development cycles** by separating lightweight dependencies (installed during build) from heavy dependencies like PyTorch (installed interactively).

### Build Strategy

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Builder (~2 min)                                    │
│ - Install lightweight deps: FastAPI, Redis, FAISS, numpy    │
│ - Skip torch, sentence-transformers (~2GB, 3-5 min)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Indexer (placeholder)                               │
│ - Create empty index directory                               │
│ - Real index built after torch installation                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Runtime (ready for development)                     │
│ - Lightweight image ready in ~2 minutes                      │
│ - Install heavy deps interactively when needed               │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Interactive Development (Recommended for Dev)

Build and start the container in interactive mode:

```powershell
# Build the image (fast - ~2 minutes)
docker-compose -f docker-compose.leibniz.yml build rag

# Start container in interactive mode (keeps running without starting service)
docker-compose -f docker-compose.leibniz.yml run --rm rag /bin/bash
```

Inside the container, run the setup script:

```bash
# Install heavy dependencies and build FAISS index (one-time, 3-5 min)
./setup_heavy_deps.sh

# Start the service manually
uvicorn leibniz_agent.services.rag.app:app --host 0.0.0.0 --port 8003 --reload
```

### Option 2: Keep Container Running (Background Development)

Start container without auto-starting the service:

```powershell
# Override CMD to keep container alive
docker-compose -f docker-compose.leibniz.yml run -d --name rag-dev rag tail -f /dev/null

# Exec into container
docker exec -it rag-dev /bin/bash

# Run setup (inside container)
./setup_heavy_deps.sh

# Start service manually
uvicorn leibniz_agent.services.rag.app:app --host 0.0.0.0 --port 8003 --reload
```

### Option 3: Production Mode (Pre-installed Heavy Deps)

For production, modify `docker-compose.leibniz.yml` to run setup on startup:

```yaml
services:
  rag:
    # ... existing config ...
    command: >
      bash -c "
        if [ ! -f /app/.setup_complete ]; then
          /app/setup_heavy_deps.sh && touch /app/.setup_complete
        fi &&
        uvicorn leibniz_agent.services.rag.app:app --host 0.0.0.0 --port 8003 --workers 1
      "
```

Then start normally:

```powershell
docker-compose -f docker-compose.leibniz.yml up -d rag
```

## 📦 What Gets Installed When

### During Build (Fast - ~2 min)
- **System packages**: curl, vim
- **Lightweight Python deps** (~50MB total):
  - FastAPI + Uvicorn (web server)
  - Pydantic (validation)
  - Redis client (asyncio)
  - FAISS-CPU (vector search)
  - NumPy (arrays)
  - google-genai (Gemini API)
  - httpx (HTTP client)

### Interactively (One-time - 3-5 min)
- **Heavy Python deps** (~2GB total):
  - PyTorch (~1.5GB) - Deep learning framework
  - sentence-transformers (~500MB) - Embedding models
  - langchain-huggingface - Model integration

### After Heavy Deps (One-time - ~30s)
- **FAISS index build**: Processes knowledge base, creates vector index

## 🔧 Manual Installation (Alternative to Script)

If you prefer manual control:

```bash
# 1. Install torch and sentence-transformers
pip install torch sentence-transformers>=2.2.0 langchain-huggingface>=0.0.1

# 2. Build FAISS index
python -m leibniz_agent.services.rag.index_builder \
    --knowledge-base /app/leibniz_knowledge_base \
    --output /app/index

# 3. Verify installation
python -c "from leibniz_agent.services.rag.rag_engine import RAGEngine; print('✅ RAG ready')"

# 4. Start service
uvicorn leibniz_agent.services.rag.app:app --host 0.0.0.0 --port 8003
```

## 🐛 Troubleshooting

### Issue: Service fails to start with "ModuleNotFoundError: torch"

**Cause**: Heavy dependencies not installed yet.

**Fix**: Run `./setup_heavy_deps.sh` inside the container.

---

### Issue: FAISS index not found

**Cause**: Index not built yet.

**Fix**: 
```bash
python -m leibniz_agent.services.rag.index_builder \
    --knowledge-base /app/leibniz_knowledge_base \
    --output /app/index
```

---

### Issue: Build takes too long

**Cause**: Trying to install torch during build.

**Fix**: Use this Dockerfile - it skips heavy deps during build. Install interactively.

---

### Issue: Container exits immediately

**Cause**: Service tries to start but torch is missing.

**Fix**: Override CMD to keep container running:
```powershell
docker-compose run rag tail -f /dev/null
```

## 📊 Performance Comparison

| Approach | Build Time | First Start | Rebuild Frequency |
|----------|-----------|-------------|-------------------|
| **Old (torch in build)** | ~8-10 min | ~2s | Every code change |
| **New (interactive torch)** | ~2 min | ~5 min (one-time) | Only when deps change |

**Development Win**: 6-8 minutes saved per rebuild! After first setup, subsequent starts are instant.

## 🎯 Recommended Workflow

1. **First time setup** (one-time, ~7 min total):
   ```powershell
   # Build image (2 min)
   docker-compose -f docker-compose.leibniz.yml build rag
   
   # Start and setup (5 min)
   docker-compose run --rm rag /bin/bash
   ./setup_heavy_deps.sh
   ```

2. **Daily development** (instant):
   ```powershell
   # Start container in background
   docker-compose run -d --name rag-dev rag tail -f /dev/null
   
   # Exec in and work
   docker exec -it rag-dev /bin/bash
   uvicorn leibniz_agent.services.rag.app:app --host 0.0.0.0 --port 8003 --reload
   ```

3. **Code changes** (instant):
   - Edit files on host (volume-mounted)
   - Uvicorn auto-reloads (if using `--reload` flag)
   - No rebuild needed!

4. **Dependency changes** (~2 min):
   ```powershell
   # Rebuild only if requirements.txt changes
   docker-compose -f docker-compose.leibniz.yml build rag
   ```

## 🔍 Verifying Installation

Check what's installed:

```bash
# Check lightweight deps (should be present)
pip list | grep -E "fastapi|redis|faiss|numpy"

# Check heavy deps (installed after running setup script)
pip list | grep -E "torch|sentence-transformers"

# Check FAISS index
ls -lh /app/index/
# Should show: faiss_index, metadata.json, stats.json
```

## 🌐 Environment Variables

All environment variables are pre-configured in `docker-compose.leibniz.yml`. Override as needed:

```yaml
environment:
  - GEMINI_API_KEY=${GEMINI_API_KEY}
  - LEIBNIZ_RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
  - LEIBNIZ_RAG_TOP_K=8
  - LOG_LEVEL=DEBUG  # For development
```

## 📝 Notes

- **Volume mounts**: Knowledge base is read-only, index is persistent
- **Redis dependency**: Ensure Redis is running (started automatically in compose)
- **Health checks**: Graceful - allows service to start even without index (for development)
- **Single worker**: Optimal for development (auto-reload works, less memory)

## 🚀 Next Steps

After setup is complete:
1. Test endpoints: `curl http://localhost:8003/health`
2. Run tests: `pytest leibniz_agent/services/rag/tests/`
3. Check metrics: `curl http://localhost:8003/metrics`
4. Query RAG: `POST http://localhost:8003/api/v1/query`
