# Stage 1: Builder
FROM python:3.11-slim AS builder
WORKDIR /build
COPY stt_sarvam/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Install OS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy artifacts and code
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
WORKDIR /app/stt_sarvam
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:/app/shared:$PYTHONPATH

# CRITICAL: Unbuffered Python output for real-time logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Configure env vars and ports
# NOTE: SARVAM_API_SUBSCRIPTION_KEY should be passed at runtime via docker-compose or -e
ENV STT_VAD_HOST=0.0.0.0
ENV STT_VAD_PORT=8001
ENV FAST_RTC_PORT=7860
ENV DOCKER_ENVIRONMENT=true
ENV LEIBNIZ_REDIS_HOST="redis"
ENV LEIBNIZ_REDIS_PORT="6379"
ENV LEIBNIZ_REDIS_DB="0"
EXPOSE 8001
EXPOSE 7860

# Add HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health', timeout=5.0).raise_for_status()"

# CRITICAL: Run with -u flag for unbuffered output
# FastRTC is now integrated directly into app.py, so we only need to run uvicorn
# The FastRTC UI will be available at /fastrtc endpoint
CMD ["python", "-u", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
