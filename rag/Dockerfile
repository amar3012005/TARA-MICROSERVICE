# ============================================================================
# RAG Service Dockerfile - Production Build
# ============================================================================
# Multi-stage build: Install all dependencies and build FAISS index at build time

# ============================================================================
# Builder Stage - Install all Python dependencies
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies needed for torch, numpy, faiss, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY services/rag/requirements.txt .
COPY services/rag/requirements_after.txt .

# Install ALL dependencies during build (includes heavy packages)
RUN pip install --default-timeout=1000 --user --no-cache-dir -r requirements.txt && \
    echo "✅ All dependencies installed (including sentence-transformers, torch)"

# ============================================================================
# Index Builder Stage - Build FAISS index
# ============================================================================
FROM python:3.11-slim as indexer

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Set PYTHONPATH for imports
ENV PYTHONPATH=/app

# Copy service code (needed for index builder)
COPY services/rag /app/leibniz_agent/services/rag
COPY services/shared /app/leibniz_agent/services/shared

# Copy install script
COPY services/rag/install_heavy_packages.sh /app/install_heavy_packages.sh
RUN chmod +x /app/install_heavy_packages.sh

# Create package structure
RUN mkdir -p /app/leibniz_agent/services && \
    echo "" > /app/leibniz_agent/__init__.py && \
    echo "" > /app/leibniz_agent/services/__init__.py

# Copy knowledge base (from repository root)
COPY leibniz_knowledge_base /app/leibniz_knowledge_base

# Set environment variables for index builder
ENV LEIBNIZ_RAG_KNOWLEDGE_BASE_PATH=/app/leibniz_knowledge_base
ENV LEIBNIZ_RAG_VECTOR_STORE_PATH=/app/index
ENV LEIBNIZ_RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# Dummy API key for index build (not used, but required by config validation)
ENV GEMINI_API_KEY=dummy_for_build

# Build FAISS index during Docker build (with better error handling)
RUN echo "📦 Building FAISS index from knowledge base..." && \
    python -m leibniz_agent.services.rag.index_builder \
        --knowledge-base /app/leibniz_knowledge_base \
        --output /app/index \
        --rebuild && \
    ls -la /app/index/ && \
    echo "✅ FAISS index built successfully!" || \
    (echo "⚠️  Index build failed - service will build at runtime" && \
     mkdir -p /app/index)

# ============================================================================
# Runtime Stage - Final production image
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Set PYTHONPATH for absolute imports
ENV PYTHONPATH=/app

# Copy service code
COPY services/rag /app/leibniz_agent/services/rag
COPY services/shared /app/leibniz_agent/services/shared

# Copy install script for heavy packages (for manual installation)
COPY services/rag/install_heavy_packages.sh /app/install_heavy_packages.sh
RUN chmod +x /app/install_heavy_packages.sh

# Create package structure
RUN mkdir -p /app/leibniz_agent/services && \
    echo "" > /app/leibniz_agent/__init__.py && \
    echo "" > /app/leibniz_agent/services/__init__.py

# Copy knowledge base
COPY leibniz_knowledge_base /app/leibniz_knowledge_base

# Copy pre-built FAISS index from indexer stage
COPY --from=indexer /app/index /app/index

# Set Python environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Environment variables (defaults, override in docker-compose)
ENV GEMINI_API_KEY="AIzaSyBiKXVYsQ0UcxRicctFI1U5dEpQct2ieOA"
ENV GEMINI_MODEL="gemini-2.0-flash-lite"
ENV LEIBNIZ_REDIS_HOST="redis"
ENV LEIBNIZ_REDIS_PORT="6379"
ENV LEIBNIZ_REDIS_DB="0"
ENV LEIBNIZ_RAG_KNOWLEDGE_BASE_PATH="/app/leibniz_knowledge_base"
ENV LEIBNIZ_RAG_VECTOR_STORE_PATH="/app/index"
ENV LEIBNIZ_RAG_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
ENV LEIBNIZ_RAG_TOP_K="8"
ENV LEIBNIZ_RAG_TOP_N="5"
ENV LEIBNIZ_RAG_SIMILARITY_THRESHOLD="0.3"
ENV LEIBNIZ_RAG_CACHE_TTL="3600"
ENV LEIBNIZ_RAG_ENABLE_HYBRID_SEARCH="true"
ENV PORT="8003"
ENV LOG_LEVEL="INFO"

# Expose service port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health', timeout=5).read()" || exit 1

# Run service with uvicorn
CMD ["python", "-m", "uvicorn", "leibniz_agent.services.rag.app:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
