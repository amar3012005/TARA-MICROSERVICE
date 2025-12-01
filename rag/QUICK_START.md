# RAG Service - Quick Start Reference

## 🎯 TL;DR - Get Running in 5 Commands

```powershell
# 1. Build image (2 min - lightweight only)
docker-compose -f docker-compose.leibniz.yml build rag

# 2. Start container in dev mode
docker-compose -f docker-compose.leibniz.yml up -d rag

# 3. Exec into container
docker exec -it leibniz-rag /bin/bash

# 4. Install heavy deps and build index (one-time, 5 min)
./setup_heavy_deps.sh

# 5. Start service with auto-reload
uvicorn leibniz_agent.services.rag.app:app --host 0.0.0.0 --port 8003 --reload
```

**Access**: http://localhost:8003/health

---

## 🔄 Daily Development Workflow

Once setup is complete (step 4 above), use this every day:

```powershell
# Start container (instant)
docker-compose -f docker-compose.leibniz.yml up -d rag

# Exec in and start service with hot-reload
docker exec -it leibniz-rag uvicorn leibniz_agent.services.rag.app:app --host 0.0.0.0 --port 8003 --reload
```

Edit files on host → Service auto-reloads → Test immediately!

---

## 🚀 Production Deployment (One Command)

```powershell
# Builds with heavy deps pre-installed, auto-starts service
docker-compose -f docker-compose.leibniz.yml --profile production up -d rag-prod
```

First start: ~8 min (installs torch + builds index)  
Subsequent starts: ~2s

---

## 📋 Common Commands

### Check Status
```powershell
# Container logs
docker logs -f leibniz-rag

# Health check
curl http://localhost:8003/health

# Metrics
curl http://localhost:8003/metrics
```

### Rebuild Index
```bash
# Inside container
python -m leibniz_agent.services.rag.index_builder \
    --knowledge-base /app/leibniz_knowledge_base \
    --output /app/index
```

### Clear Redis Cache
```bash
# Inside container or from host
docker exec leibniz-redis redis-cli FLUSHDB
```

### Update Knowledge Base
```powershell
# Edit files in leibniz_knowledge_base/ on host
# Then rebuild index (inside container):
docker exec -it leibniz-rag python -m leibniz_agent.services.rag.index_builder \
    --knowledge-base /app/leibniz_knowledge_base \
    --output /app/index

# Restart service to reload
docker restart leibniz-rag
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| ModuleNotFoundError: torch | Run `./setup_heavy_deps.sh` inside container |
| FAISS index not found | Run index builder (see "Rebuild Index" above) |
| Service won't start | Check logs: `docker logs leibniz-rag` |
| Slow queries | Check Redis: `docker exec leibniz-redis redis-cli PING` |
| Container exits immediately | Switch to dev profile (see Quick Start above) |

---

## 🎓 What's Different from Old Dockerfile?

| Old Approach | New Approach | Benefit |
|--------------|--------------|---------|
| Install torch in build | Install torch interactively | **6-8 min faster** rebuilds |
| Pre-build FAISS index | Build index on demand | Flexibility for KB updates |
| Auto-start service | Keep container running | Full dev control |
| Single mode | Dev + Production profiles | Best of both worlds |

---

## 📚 Full Documentation

- **Detailed Guide**: [DOCKER_DEV_GUIDE.md](./DOCKER_DEV_GUIDE.md)
- **Architecture**: [README.md](./README.md)
- **API Reference**: http://localhost:8003/docs (when running)

---

## ⚡ Performance Tips

1. **Use Redis caching**: Default TTL is 1 hour - queries with identical text hit cache (1-5ms vs 500-1500ms)
2. **Single worker**: Already configured - each worker loads ~500KB index
3. **Hot reload in dev**: Use `--reload` flag for instant code changes
4. **Pre-warm in production**: First query after cold start takes ~1s (loads models), subsequent queries ~200-500ms

---

## 🔐 Environment Variables (Optional Overrides)

Add to `.env` file:

```bash
# API Keys
GEMINI_API_KEY=your_key_here

# RAG Tuning
LEIBNIZ_RAG_TOP_K=8              # Retrieve top 8 chunks
LEIBNIZ_RAG_TOP_N=5              # Return top 5 after reranking
LEIBNIZ_RAG_SIMILARITY_THRESHOLD=0.3  # Min similarity score
LEIBNIZ_RAG_CACHE_TTL=3600       # Cache for 1 hour

# Logging
LOG_LEVEL=DEBUG                  # For development
```

---

## 🎯 Next Steps

1. ✅ Complete Quick Start (above)
2. Test endpoint: `curl -X POST http://localhost:8003/api/v1/query -H "Content-Type: application/json" -d '{"query":"What is Leibniz University?"}'`
3. Run tests: `docker exec leibniz-rag pytest leibniz_agent/services/rag/tests/`
4. Add knowledge: Edit `leibniz_knowledge_base/*.md`, rebuild index
5. Monitor: Check `/metrics` endpoint regularly

**Questions?** See [DOCKER_DEV_GUIDE.md](./DOCKER_DEV_GUIDE.md) or [README.md](./README.md)
