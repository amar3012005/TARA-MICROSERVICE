# Multi-stage build for Leibniz Appointment FSM Service

# ============================================================================
# Builder Stage
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Copy requirements and install dependencies
COPY leibniz_agent/services/appointment/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================================================
# Runtime Stage
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/app/.local

# Update PATH to use installed packages
ENV PATH=/home/app/.local/bin:$PATH

# Set PYTHONPATH for absolute imports
ENV PYTHONPATH=/app

# Copy service code
COPY leibniz_agent/services/appointment /app/leibniz_agent/services/appointment
COPY leibniz_agent/services/shared /app/leibniz_agent/services/shared
RUN mkdir -p /app/leibniz_agent/services && \
    echo "" > /app/leibniz_agent/__init__.py && \
    echo "" > /app/leibniz_agent/services/__init__.py

# Copy app.py to root for simpler import
COPY leibniz_agent/services/appointment/app.py /app/app.py

# Create non-root user and set permissions
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /home/app \
    && chown -R app:app /app
USER app

# Expose service port
EXPOSE 8005

# Health check (using stdlib urllib - no external dependencies needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8005/health', timeout=5)" || exit 1

# Run with 1 worker to prevent state fragmentation in Redis session management
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005", "--workers", "1"]