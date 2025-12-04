FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS base-deps

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY tts_xtts_v2/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Heavy deps (patched Coqui TTS + shared libs) live in the base layer so we can reuse it
COPY TTS /app/leibniz_agent/services/TTS
RUN pip install --no-cache-dir --editable /app/leibniz_agent/services/TTS

COPY shared /app/leibniz_agent/services/shared
RUN mkdir -p /app/leibniz_agent/services && \
    touch /app/leibniz_agent/__init__.py && \
    touch /app/leibniz_agent/services/__init__.py && \
    touch /app/leibniz_agent/services/shared/__init__.py

ENV PYTHONPATH=/app \
    XTTS_MODEL_DIR=/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2 \
    LEIBNIZ_TTS_CACHE_DIR=/app/audio_cache \
    TTS_STREAMING_PORT=8005


FROM base-deps AS runtime

COPY tts_xtts_v2 /app/leibniz_agent/services/tts_xtts_v2
RUN touch /app/leibniz_agent/services/tts_xtts_v2/__init__.py

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8005/health', timeout=5.0).raise_for_status()"

CMD ["python", "-u", "-m", "uvicorn", "leibniz_agent.services.tts_xtts_v2.app:app", "--host", "0.0.0.0", "--port", "8005"]

