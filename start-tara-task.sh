#!/bin/bash
# TARA X TASK - Master Startup Script
# Starts all services in organized Docker Desktop folder

set -e

echo "======================================================================"
echo "🚀 TARA X TASK - Starting Complete Microservices Stack"
echo "======================================================================"

# Set Docker context (CRITICAL for Docker Desktop visibility)
echo "📋 Setting Docker context to desktop-linux..."
docker context use desktop-linux

# Verify context is set
CURRENT_CONTEXT=$(docker context show)
if [ "$CURRENT_CONTEXT" != "desktop-linux" ]; then
    echo "❌ Error: Failed to set Docker context to desktop-linux"
    echo "   Current context: $CURRENT_CONTEXT"
    exit 1
fi
echo "✅ Docker context set to: $CURRENT_CONTEXT"

# Navigate to project directory
cd "$(dirname "$0")"

# Check if docker-compose file exists
COMPOSE_FILE="docker-compose-tara-task.yml"
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ Error: $COMPOSE_FILE not found!"
    exit 1
fi

# Stop any existing containers
echo ""
echo "🧹 Cleaning up existing containers..."
docker-compose -f "$COMPOSE_FILE" down 2>/dev/null || true

# Start only orchestrator (it will start others automatically)
echo ""
echo "🎛️ Starting Master Controller (Orchestrator)..."
echo "   The orchestrator will automatically start all other services"
echo "   Using Docker context: $(docker context show)"
echo ""

# Use docker-compose with explicit context and project name
docker context use desktop-linux
export COMPOSE_PROJECT_NAME=tara-task
docker-compose -f "$COMPOSE_FILE" up -d orchestrator

# Wait a bit for orchestrator to start
echo "⏳ Waiting for orchestrator to initialize..."
sleep 5

# Show status
echo ""
echo "======================================================================"
echo "📊 Service Status"
echo "======================================================================"
docker-compose -f "$COMPOSE_FILE" ps

echo ""
echo "======================================================================"
echo "📋 Next Steps"
echo "======================================================================"
echo "1. Check orchestrator logs:"
echo "   docker logs tara-task-orchestrator -f"
echo ""
echo "2. Once services are ready, open FastRTC UIs:"
echo "   - STT: http://localhost:6012"
echo "   - TTS: http://localhost:6005/fastrtc"
echo ""
echo "3. Start the workflow:"
echo "   curl -X POST http://localhost:6004/start"
echo ""
echo "======================================================================"
echo "✅ TARA X TASK stack starting!"
echo "======================================================================"

