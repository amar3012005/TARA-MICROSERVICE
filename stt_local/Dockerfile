# Stage 1: Builder
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04 AS builder

# Install Python and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY stt_local/requirements.txt .
COPY stt_local/requirements_after.txt .

# Install lightweight Python dependencies only (heavy packages installed later)
RUN pip3 install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04 AS runtime

# Install OS packages and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libsndfile1 \
    portaudio19-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy artifacts and code
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
COPY stt_local/requirements_after.txt /app/stt_local/
COPY stt_local/install_heavy_deps.sh /app/stt_local/
WORKDIR /app/stt_local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:/app/shared:$PYTHONPATH
RUN chmod +x /app/stt_local/install_heavy_deps.sh

# CRITICAL: Unbuffered Python output for real-time logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CUDA environment
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Configure env vars and ports
ENV STT_LOCAL_HOST=0.0.0.0
ENV STT_LOCAL_PORT=8006
ENV FAST_RTC_PORT=7861
ENV DOCKER_ENVIRONMENT=true
ENV LEIBNIZ_REDIS_HOST="redis"
ENV LEIBNIZ_REDIS_PORT="6379"
ENV LEIBNIZ_REDIS_DB="0"

# Whisper model configuration
ENV WHISPER_MODEL_SIZE="base"
ENV WHISPER_DEVICE="cuda"
ENV WHISPER_COMPUTE_TYPE="float16"

EXPOSE 8006
EXPOSE 7861

# Add HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import httpx; httpx.get('http://localhost:8006/health', timeout=5.0).raise_for_status()"

# CRITICAL: Run with -u flag for unbuffered output
# FastRTC is integrated directly into app.py
CMD ["python3", "-u", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8006", "--workers", "1"]

