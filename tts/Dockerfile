# Leibniz TTS Service Dockerfile
# Minimal build - install heavy packages later inside running container

FROM python:3.11-slim

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Set PYTHONPATH for absolute imports
ENV PYTHONPATH=/app

# Copy and install PRODUCTION dependencies (includes ElevenLabs + Gemini)
COPY leibniz_agent/services/tts/requirements-production.txt /app/requirements-production.txt
RUN pip install --no-cache-dir -r requirements-production.txt

# Copy application code
COPY leibniz_agent/services/tts /app/leibniz_agent/services/tts
COPY leibniz_agent/services/shared /app/leibniz_agent/services/shared
RUN mkdir -p /app/leibniz_agent/services && \
    echo "" > /app/leibniz_agent/__init__.py && \
    echo "" > /app/leibniz_agent/services/__init__.py

# Create cache directory
RUN mkdir -p /app/audio_cache

# Copy full requirements for optional heavy packages (XTTS, Google Cloud)
COPY leibniz_agent/services/tts/requirements.txt /app/requirements-full.txt

# Expose port
EXPOSE 8004

# Environment variables (override with docker-compose or -e flags)
ENV LEIBNIZ_TTS_PROVIDER=elevenlabs \
    LEIBNIZ_TTS_FALLBACK_PROVIDER=gemini \
    LEIBNIZ_TTS_CACHE_ENABLED=true \
    LEIBNIZ_TTS_CACHE_DIR=/app/audio_cache \
    LEIBNIZ_TTS_SERVICE_PORT=8004

# Health check (relaxed for initial startup)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run with single worker to avoid cache index duplication
CMD ["python", "-m", "uvicorn", "leibniz_agent.services.tts.app:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]
