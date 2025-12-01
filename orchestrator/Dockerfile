# Multi-stage build for StateManager Orchestrator Service
# Phase 4 - Cloud Transformation

# ============================================================================
# Builder Stage
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
# Context is services/ directory (matching docker-compose.yml context: .)
COPY orchestrator/requirements.txt .
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
# Context is services/ directory (matching docker-compose.yml context: .)
COPY orchestrator /app/leibniz_agent/services/orchestrator
COPY shared /app/leibniz_agent/services/shared
RUN mkdir -p /app/leibniz_agent/services && \
    echo "" > /app/leibniz_agent/__init__.py && \
    echo "" > /app/leibniz_agent/services/__init__.py

# Expose service port
EXPOSE 8004

# Unbuffered Python for real-time logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8004/health', timeout=5)" || exit 1

# Run with 1 worker for state consistency
CMD ["python", "-m", "uvicorn", "leibniz_agent.services.orchestrator.app:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]

