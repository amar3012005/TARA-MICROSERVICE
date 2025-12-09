# Stage 1: Builder - Install ALL dependencies (CACHED)
FROM python:3.11-slim AS builder
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ========== CRITICAL: Copy ONLY requirements first for caching ==========
# This layer is cached unless requirements.txt changes
COPY gemini_live_experiment/requirements.txt /build/requirements.txt

# Install ALL Python packages (CACHED - only rebuilds if requirements.txt changes)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 --user -r requirements.txt && \
    echo "✅ All packages installed (cached for future builds)"

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Install OS runtime packages (CACHED - rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-installed Python packages from builder (LARGE but CACHED)
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# ========== Source code - changes here are FAST rebuilds ==========
# Copy ONLY the service code (not entire directory)
COPY gemini_live_experiment /app/gemini_live_experiment
COPY shared /app/shared

# Set working directory to service code
WORKDIR /app/gemini_live_experiment
ENV PYTHONPATH=/app:/app/shared:$PYTHONPATH

# CRITICAL: Unbuffered Python output for real-time logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Configure env vars and ports
ENV GEMINI_API_KEY=${GEMINI_API_KEY:-AIzaSyCXz2gMP-1zXqA4xyJO0L_4PP3zHhsBCCg}
ENV STT_VAD_HOST=0.0.0.0
ENV STT_VAD_PORT=8001
ENV FAST_RTC_PORT=7860
ENV DOCKER_ENVIRONMENT=true
ENV TARA_REDIS_HOST=tara-task-redis
ENV TARA_REDIS_PORT=6381
ENV TARA_REDIS_DB=0
ENV RAG_SERVICE_URL=http://tara-task-rag:8003
EXPOSE 8001
EXPOSE 7860

# Add HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health', timeout=5.0).raise_for_status()"

# CRITICAL: Run with -u flag for unbuffered output
# FastRTC is now integrated directly into app.py, so we only need to run uvicorn
# The FastRTC UI will be available at /fastrtc endpoint
CMD ["python", "-u", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
