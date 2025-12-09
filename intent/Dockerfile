# Intent Classification Service - TARA Task Setup
# Multi-stage build following tara-task pattern

# ============================================================================
# Builder Stage
# ============================================================================
FROM python:3.11-slim AS builder
WORKDIR /build

# Install build dependencies (gcc, g++, make) needed for spacy/numpy/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
# Context is project root
# Requirements.txt is optimized: HEAVY packages (torch, transformers) first
# This ensures heavy packages are cached when lighter packages change
COPY intent/requirements.txt /build/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 --user -r requirements.txt

# ============================================================================
# Runtime Stage
# ============================================================================
FROM python:3.11-slim

# Install OS runtime packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-installed Python packages from builder
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Copy service code
# Context is project root
COPY intent /app/intent
COPY shared /app/shared

# Set working directory to service code
WORKDIR /app/intent
ENV PYTHONPATH=/app:/app/shared:$PYTHONPATH

# Copy DistilBERT model for Layer 2 (SLM)
COPY intent/leibniz_distilbert_intent_v2 /app/intent/leibniz_distilbert_intent_v2

# Verify model presence
RUN if [ -d "/app/intent/leibniz_distilbert_intent_v2" ]; then \
        echo "✅ DistilBERT model copied - Layer 2 (SLM) will be enabled"; \
    else \
        echo "⚠️ DistilBERT model copy failed - Layer 2 (SLM) will be disabled"; \
    fi

# Download Spacy model for semantic context extraction
RUN python -m spacy download en_core_web_sm || echo "⚠️ Spacy model download failed - will use regex fallback"

# CRITICAL: Unbuffered Python output for real-time logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Configure environment variables
ENV GEMINI_API_KEY="AIzaSyC6cvyEl4FNjIQCV_p5_2wJkOa1cUObFHU"
ENV LEIBNIZ_REDIS_HOST="redis"
ENV LEIBNIZ_REDIS_PORT="6379"
ENV LEIBNIZ_REDIS_DB="0"
ENV PORT=8002

# Expose service port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health', timeout=5)" || exit 1

# Run with 1 worker to maintain consistent metrics
CMD ["python", "-u", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
