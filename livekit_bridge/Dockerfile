FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY livekit_bridge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY livekit_bridge/agent.py .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Run the agent (start worker in production mode)
CMD ["python", "agent.py", "start"]
