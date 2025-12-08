#!/bin/bash
# Comprehensive Sarvam TTS FastRTC Streaming Test Script

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
SERVICE_URL="http://localhost:8025"
PASSED=0
FAILED=0

# Test texts
declare -a TEST_TEXTS=(
    "Testing Sarvam TTS"
    "Hello! This is a comprehensive test of the Sarvam AI text to speech system."
    "The quick brown fox jumps over the lazy dog."
    "Testing FastRTC streaming with real-time audio playback."
)

echo -e "${BOLD}============================================================================${NC}"
echo -e "${BOLD}             Sarvam TTS FastRTC Streaming Test Suite${NC}"
echo -e "${BOLD}============================================================================${NC}"
echo ""

# Function to print status
print_status() {
    local message=$1
    local status=$2
    local timestamp=$(date +%H:%M:%S)
    
    case $status in
        "SUCCESS")
            echo -e "${GREEN}[${timestamp}] [SUCCESS]${NC} ${message}"
            ;;
        "ERROR")
            echo -e "${RED}[${timestamp}] [ERROR]${NC} ${message}"
            ;;
        "WARNING")
            echo -e "${YELLOW}[${timestamp}] [WARNING]${NC} ${message}"
            ;;
        *)
            echo -e "${BLUE}[${timestamp}] [INFO]${NC} ${message}"
            ;;
    esac
}

# Check service health
print_status "Checking service health..." "INFO"
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" "${SERVICE_URL}/health")
HEALTH_CODE=$(echo "$HEALTH_RESPONSE" | tail -n 1)
HEALTH_BODY=$(echo "$HEALTH_RESPONSE" | head -n -1)

if [ "$HEALTH_CODE" = "200" ]; then
    print_status "Service is healthy" "SUCCESS"
    echo "$HEALTH_BODY" | python3 -m json.tool 2>/dev/null | sed 's/^/    /'
    echo ""
else
    print_status "Health check failed (HTTP $HEALTH_CODE)" "ERROR"
    echo "$HEALTH_BODY"
    exit 1
fi

# Test FastRTC synthesis endpoint
echo -e "${BOLD}Testing FastRTC Synthesis Endpoint${NC}"
echo "--------------------------------------------------------------------"

for i in "${!TEST_TEXTS[@]}"; do
    TEXT="${TEST_TEXTS[$i]}"
    TEST_NUM=$((i + 1))
    
    echo ""
    print_status "Test ${TEST_NUM}/${#TEST_TEXTS[@]}: Synthesizing text" "INFO"
    print_status "  Text (${#TEXT} chars): ${TEXT:0:80}${TEXT:80+:...}" "INFO"
    
    # Create JSON payload
    JSON_PAYLOAD=$(cat <<EOF
{
  "text": "$TEXT",
  "emotion": "helpful"
}
EOF
)
    
    # Send request
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
        "${SERVICE_URL}/api/v1/fastrtc/synthesize" \
        -H "Content-Type: application/json" \
        -d "$JSON_PAYLOAD" 2>&1)
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
    BODY=$(echo "$RESPONSE" | head -n -1)
    
    if [ "$HTTP_CODE" = "200" ]; then
        print_status "✅ Synthesis successful!" "SUCCESS"
        
        # Parse response
        SENTENCES=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('sentences', 0))" 2>/dev/null)
        PLAYED=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('sentences_played', 0))" 2>/dev/null)
        DURATION=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total_duration_ms', 0))" 2>/dev/null)
        
        print_status "  Sentences: ${SENTENCES}" "INFO"
        print_status "  Sentences played: ${PLAYED}" "INFO"
        print_status "  Duration: ${DURATION}ms" "INFO"
        
        if [ "$SENTENCES" = "$PLAYED" ]; then
            print_status "  ✅ All sentences played successfully" "SUCCESS"
            ((PASSED++))
        else
            print_status "  ⚠️ Not all sentences played (${PLAYED}/${SENTENCES})" "WARNING"
            ((FAILED++))
        fi
    else
        print_status "❌ Synthesis failed (HTTP $HTTP_CODE)" "ERROR"
        echo "$BODY"
        ((FAILED++))
    fi
    
    sleep 0.5
done

# Test regular HTTP synthesis endpoint
echo ""
echo -e "${BOLD}Testing Regular HTTP Synthesis Endpoint${NC}"
echo "--------------------------------------------------------------------"

TEST_TEXT="${TEST_TEXTS[1]}"
print_status "Synthesizing: ${TEST_TEXT:0:60}..." "INFO"

JSON_PAYLOAD=$(cat <<EOF
{
  "text": "$TEST_TEXT",
  "emotion": "helpful"
}
EOF
)

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    "${SERVICE_URL}/api/v1/synthesize" \
    -H "Content-Type: application/json" \
    -d "$JSON_PAYLOAD" 2>&1)

HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
BODY=$(echo "$RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    SUCCESS=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null)
    
    if [ "$SUCCESS" = "True" ]; then
        print_status "✅ Regular synthesis successful!" "SUCCESS"
        
        SAMPLE_RATE=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('sample_rate', 0))" 2>/dev/null)
        DURATION=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('duration_ms', 0))" 2>/dev/null)
        SENTENCES=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('sentences', 0))" 2>/dev/null)
        AUDIO_LEN=$(echo "$BODY" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('audio_data', '')))" 2>/dev/null)
        
        print_status "  Sample rate: ${SAMPLE_RATE}Hz" "INFO"
        print_status "  Duration: ${DURATION}ms" "INFO"
        print_status "  Sentences: ${SENTENCES}" "INFO"
        print_status "  Audio data size: ${AUDIO_LEN} chars (base64)" "INFO"
        ((PASSED++))
    else
        ERROR=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('error', 'Unknown error'))" 2>/dev/null)
        print_status "❌ Synthesis failed: ${ERROR}" "ERROR"
        ((FAILED++))
    fi
else
    print_status "❌ HTTP request failed (HTTP $HTTP_CODE)" "ERROR"
    echo "$BODY"
    ((FAILED++))
fi

# Summary
echo ""
echo -e "${BOLD}============================================================================${NC}"
echo -e "${BOLD}                          Test Summary${NC}"
echo -e "${BOLD}============================================================================${NC}"

TOTAL=$((PASSED + FAILED))
echo -e "Total Tests:    ${BOLD}${TOTAL}${NC}"
echo -e "Passed:         ${GREEN}${BOLD}${PASSED}${NC}"
echo -e "Failed:         ${RED}${BOLD}${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✅ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}❌ SOME TESTS FAILED${NC}"
    exit 1
fi
