# Stage 1: Builder - Install ALL dependencies (CACHED)
FROM python:3.11-slim AS builder
WORKDIR /build

# ========== CRITICAL: Copy ONLY requirements first for caching ==========
# This layer is cached unless requirements.txt changes
COPY tts_sarvam/requirements.txt /build/requirements.txt

# Install ALL Python packages (CACHED - only rebuilds if requirements.txt changes)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 --user -r requirements.txt && \
    echo "✅ All packages installed (cached for future builds)"

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Install OS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
## SARVAM API - Configure SARVAM_API_KEY environment variable
# Copy artifacts and code
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Copy service code (context is TARA-MICROSERVICE/ directory)
COPY tts_sarvam /app/leibniz_agent/services/tts_streaming
COPY shared /app/leibniz_agent/services/shared
RUN mkdir -p /app/leibniz_agent/services && \
    echo "" > /app/leibniz_agent/__init__.py && \
    echo "" > /app/leibniz_agent/services/__init__.py && \
    echo "" > /app/leibniz_agent/services/shared/__init__.py

# CRITICAL: Unbuffered Python output for real-time logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Configure env vars and ports - no hardcoded keys
ENV TTS_STREAMING_PORT=8025
ENV LEIBNIZ_TTS_CACHE_DIR=/app/audio_cache
# TARA mode defaults - Telugu language for TASK organization
ENV LEIBNIZ_SARVAM_LANGUAGE=te-IN
ENV LEIBNIZ_TTS_LANGUAGE_CODE=te-IN
EXPOSE 8025

# Add HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8025/health', timeout=5.0).raise_for_status()"

# CRITICAL: Run with -u flag for unbuffered output
CMD ["python", "-u", "-m", "uvicorn", "leibniz_agent.services.tts_streaming.app:app", "--host", "0.0.0.0", "--port", "8025", "--workers", "1"]

