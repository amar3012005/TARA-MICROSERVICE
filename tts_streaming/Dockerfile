# Stage 1: Builder
FROM python:3.11-slim AS builder
WORKDIR /build
COPY tts_streaming/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Install OS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy artifacts and code
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Copy service code (context is services/ directory)
COPY tts_streaming /app/leibniz_agent/services/tts_streaming
RUN mkdir -p /app/leibniz_agent/services && \
    echo "" > /app/leibniz_agent/__init__.py && \
    echo "" > /app/leibniz_agent/services/__init__.py

# CRITICAL: Unbuffered Python output for real-time logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Configure env vars and ports
ENV LEMONFOX_API_KEY=""
ENV TTS_STREAMING_PORT=8005
ENV LEIBNIZ_TTS_CACHE_DIR=/app/audio_cache
EXPOSE 8005

# Add HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8005/health', timeout=5.0).raise_for_status()"

# CRITICAL: Run with -u flag for unbuffered output
CMD ["python", "-u", "-m", "uvicorn", "leibniz_agent.services.tts_streaming.app:app", "--host", "0.0.0.0", "--port", "8005", "--workers", "1"]

