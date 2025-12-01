# Multi-stage build for Intent Classification Service
# Phase 3 - Cloud Transformation

# ============================================================================
# Builder Stage
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies (gcc, g++, make) needed for spacy/numpy/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies (none needed for pure Python service)
# All dependencies are pure Python (no system libs required unlike STT/VAD)

# Copy requirements and install dependencies
# Context is leibniz_agent root
COPY services/intent/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================================================
# Runtime Stage
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Update PATH to use installed packages
ENV PATH=/root/.local/bin:$PATH

# Set PYTHONPATH for absolute imports
ENV PYTHONPATH=/app

# Copy service code
# Context is leibniz_agent root
COPY services/intent /app/leibniz_agent/services/intent
COPY services/shared /app/leibniz_agent/services/shared
RUN mkdir -p /app/leibniz_agent/services && \
    echo "" > /app/leibniz_agent/__init__.py && \
    echo "" > /app/leibniz_agent/services/__init__.py

# Copy DistilBERT model for Layer 2 (SLM)
# We need to copy it from the build context (project root) into the image
COPY leibniz_distilbert_intent_v2 /app/leibniz_agent/services/intent/leibniz_distilbert_intent_v2

# Verify model presence
RUN if [ -d "/app/leibniz_agent/services/intent/leibniz_distilbert_intent_v2" ]; then \
        echo "✅ DistilBERT model copied - Layer 2 (SLM) will be enabled"; \
    else \
        echo "⚠️ DistilBERT model copy failed - Layer 2 (SLM) will be disabled"; \
    fi

# Download Spacy model for semantic context extraction
RUN python -m spacy download en_core_web_sm || echo "⚠️ Spacy model download failed - will use regex fallback"

# Expose service port
EXPOSE 8002

# Health check (using stdlib urllib - no external dependencies needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health', timeout=5)" || exit 1

# Run with 1 worker to maintain consistent metrics and avoid state fragmentation
CMD ["python", "-m", "uvicorn", "leibniz_agent.services.intent.app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
