#!/bin/bash
# Test script for Unified FastRTC Architecture

set -e

ORCHESTRATOR_PORT=2004
ORCHESTRATOR_URL="http://localhost:${ORCHESTRATOR_PORT}"
UNIFIED_FASTRTC_URL="${ORCHESTRATOR_URL}/fastrtc"

echo "=========================================="
echo "🧪 Testing Unified FastRTC Architecture"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if service is healthy
check_health() {
    local service_name=$1
    local url=$2
    echo -n "Checking ${service_name}... "
    
    if curl -s -f "${url}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ Not responding${NC}"
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=0
    
    echo "Waiting for ${service_name} to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if check_health "${service_name}" "${url}"; then
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    echo -e "${RED}✗ ${service_name} failed to start after ${max_attempts} attempts${NC}"
    return 1
}

echo "Step 1: Building Docker images..."
echo "-----------------------------------"
docker --context desktop-linux compose -f docker-compose-tara-task.yml build orchestrator
echo ""

echo "Step 2: Starting services..."
echo "-----------------------------------"
docker --context desktop-linux compose -f docker-compose-tara-task.yml up -d redis stt-vad rag tts-labs orchestrator
echo ""

echo "Step 3: Waiting for services to be healthy..."
echo "-----------------------------------"
wait_for_service "Redis" "http://localhost:2006"
wait_for_service "STT-VAD" "http://localhost:2001"
wait_for_service "RAG" "http://localhost:2003"
wait_for_service "TTS-Labs" "http://localhost:2006"
wait_for_service "Orchestrator" "${ORCHESTRATOR_URL}"
echo ""

echo "Step 4: Checking Orchestrator status..."
echo "-----------------------------------"
STATUS_RESPONSE=$(curl -s "${ORCHESTRATOR_URL}/status")
echo "$STATUS_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$STATUS_RESPONSE"
echo ""

echo "Step 5: Checking Unified FastRTC availability..."
echo "-----------------------------------"
if curl -s -f "${UNIFIED_FASTRTC_URL}" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Unified FastRTC UI is accessible at: ${UNIFIED_FASTRTC_URL}${NC}"
else
    echo -e "${YELLOW}⚠ Unified FastRTC UI may not be ready yet${NC}"
    echo "   URL: ${UNIFIED_FASTRTC_URL}"
fi
echo ""

echo "Step 6: Testing Unified FastRTC handler state..."
echo "-----------------------------------"
HANDLER_STATES=$(curl -s "${ORCHESTRATOR_URL}/status" | python3 -c "import sys, json; data=json.load(sys.stdin); print(json.dumps(data.get('unified_fastrtc', {}), indent=2))" 2>/dev/null)
if [ -n "$HANDLER_STATES" ]; then
    echo "$HANDLER_STATES"
else
    echo "No unified FastRTC handlers active yet (this is normal - handlers are created when browser connects)"
fi
echo ""

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo "   1. Open browser: ${UNIFIED_FASTRTC_URL}"
echo "   2. Click 'Record' button to connect"
echo "   3. Trigger workflow: curl -X POST ${ORCHESTRATOR_URL}/start"
echo "   4. Check status: curl ${ORCHESTRATOR_URL}/status"
echo ""
echo "🔍 Monitor logs:"
echo "   docker --context desktop-linux compose -f docker-compose-tara-task.yml logs -f orchestrator"
echo ""
echo "🛑 Stop services:"
echo "   docker --context desktop-linux compose -f docker-compose-tara-task.yml down"
echo ""
