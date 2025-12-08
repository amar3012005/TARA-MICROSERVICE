#!/bin/bash
# TARA X TASK - Start using main docker-compose.yml
# Builds and starts services one by one with desktop-linux context

set -e

echo "======================================================================"
echo "🚀 TARA X TASK - Starting with main docker-compose.yml"
echo "======================================================================"

# Set Docker context
echo "📋 Setting Docker context to desktop-linux..."
docker context use desktop-linux

# Verify context
CURRENT_CONTEXT=$(docker context show)
if [ "$CURRENT_CONTEXT" != "desktop-linux" ]; then
    echo "❌ Error: Failed to set Docker context to desktop-linux"
    echo "   Current context: $CURRENT_CONTEXT"
    exit 1
fi
echo "✅ Docker context: $CURRENT_CONTEXT"
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

COMPOSE_FILE="docker-compose.yml"

# Check if docker-compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ Error: $COMPOSE_FILE not found!"
    exit 1
fi

# Set project name for Docker Desktop visibility
export COMPOSE_PROJECT_NAME=tara-task

echo "🔨 Building services one by one..."
echo ""

# Step 1: Redis (foundation)
echo "Step 1/5: Building Redis..."
docker-compose -f "$COMPOSE_FILE" build redis
echo "✅ Redis built"
echo ""

# Step 2: STT-VAD
echo "Step 2/5: Building STT-VAD..."
docker-compose -f "$COMPOSE_FILE" build tara-stt-vad-service
echo "✅ STT-VAD built"
echo ""

# Step 3: RAG
echo "Step 3/5: Building RAG..."
docker-compose -f "$COMPOSE_FILE" build rag-service
echo "✅ RAG built"
echo ""

# Step 4: Orchestrator
echo "Step 4/5: Building Orchestrator..."
docker-compose -f "$COMPOSE_FILE" build orchestrator
echo "✅ Orchestrator built"
echo ""

# Step 5: TTS Sarvam
echo "Step 5/5: Building TTS Sarvam..."
docker-compose -f "$COMPOSE_FILE" build tts-sarvam-service
echo "✅ TTS Sarvam built"
echo ""

echo "======================================================================"
echo "🚀 Starting services..."
echo "======================================================================"

# Start services in dependency order
echo "Starting Redis..."
docker-compose -f "$COMPOSE_FILE" up -d redis

echo "Waiting for Redis to be healthy..."
sleep 5

echo "Starting STT-VAD..."
docker-compose -f "$COMPOSE_FILE" up -d tara-stt-vad-service

echo "Starting RAG..."
docker-compose -f "$COMPOSE_FILE" up -d rag-service

echo "Starting TTS Sarvam..."
docker-compose -f "$COMPOSE_FILE" up -d tts-sarvam-service

echo "Starting Orchestrator..."
docker-compose -f "$COMPOSE_FILE" up -d orchestrator

echo ""
echo "======================================================================"
echo "📊 Service Status"
echo "======================================================================"
docker-compose -f "$COMPOSE_FILE" ps redis tara-stt-vad-service rag-service orchestrator tts-sarvam-service

echo ""
echo "======================================================================"
echo "📋 Service URLs"
echo "======================================================================"
echo "   Redis:         localhost:6381"
echo "   STT API:       localhost:8026"
echo "   STT FastRTC:   localhost:7861"
echo "   RAG:           localhost:8023"
echo "   Orchestrator:  localhost:8004"
echo "   TTS Sarvam:    localhost:8025"
echo ""
echo "======================================================================"
echo "📋 Next Steps"
echo "======================================================================"
echo "1. Check logs:"
echo "   docker logs orchestrator-service -f"
echo ""
echo "2. Open FastRTC UIs:"
echo "   - STT: http://localhost:7861"
echo "   - TTS: http://localhost:8025/fastrtc"
echo ""
echo "3. Start workflow:"
echo "   curl -X POST http://localhost:8004/start"
echo ""
echo "======================================================================"
echo "✅ TARA X TASK stack started!"
echo "======================================================================"




