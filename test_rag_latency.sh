#!/bin/bash

# RAG Service Latency Test Script
# Tests first chunk latency (TTFB) and overall response time

RAG_URL="http://localhost:2003"
ENDPOINT="${RAG_URL}/api/v1/stream_query"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🚀 RAG Service First Chunk Latency Test"
echo "=========================================="
echo ""

# Test query
QUERY="what is task"
echo "📝 Test Query: \"${QUERY}\""
echo ""

# Create JSON payload
JSON_PAYLOAD=$(cat <<EOF
{
  "query": "${QUERY}",
  "context": {
    "language": "te-mixed",
    "organization": "T.A.S.K"
  },
  "enable_streaming": true
}
EOF
)

echo "⏱️  Measuring latency..."
echo ""

# Measure time to first byte (TTFB) and total time
START_TIME=$(date +%s.%N)

# Use curl with timing and stream the response
RESPONSE=$(curl -s -w "\n%{time_total}\n%{time_starttransfer}\n%{time_connect}\n" \
  -X POST \
  -H "Content-Type: application/json" \
  -d "${JSON_PAYLOAD}" \
  "${ENDPOINT}")

END_TIME=$(date +%s.%N)

# Extract timing info (last 3 lines)
TOTAL_TIME=$(echo "$RESPONSE" | tail -n 3 | head -n 1)
TTFB=$(echo "$RESPONSE" | tail -n 2 | head -n 1)
CONNECT_TIME=$(echo "$RESPONSE" | tail -n 1)

# Extract actual response (everything except last 3 lines)
ACTUAL_RESPONSE=$(echo "$RESPONSE" | head -n -3)

# Find first chunk
FIRST_CHUNK=$(echo "$ACTUAL_RESPONSE" | head -n 1)

# Calculate first chunk latency (approximate)
FIRST_CHUNK_TIME=$(echo "$TTFB * 1000" | bc)

echo "=========================================="
echo "📊 Latency Results"
echo "=========================================="
printf "${GREEN}Time to First Byte (TTFB):${NC} %.2f ms\n" "$(echo "$TTFB * 1000" | bc)"
printf "${GREEN}Connection Time:${NC} %.2f ms\n" "$(echo "$CONNECT_TIME * 1000" | bc)"
printf "${GREEN}Total Response Time:${NC} %.2f ms\n" "$(echo "$TOTAL_TIME * 1000" | bc)"
echo ""

# Check if response is valid
if [ -z "$FIRST_CHUNK" ]; then
    echo "${RED}❌ No response received${NC}"
    exit 1
fi

# Parse first chunk JSON
FIRST_TEXT=$(echo "$FIRST_CHUNK" | grep -o '"text":"[^"]*"' | cut -d'"' -f4)
IS_FINAL=$(echo "$FIRST_CHUNK" | grep -o '"is_final":[^,}]*' | cut -d':' -f2)

echo "=========================================="
echo "📦 First Chunk Content"
echo "=========================================="
echo "Text: ${FIRST_TEXT:0:100}..."
echo "Is Final: $IS_FINAL"
echo ""

# Show full response (first 5 chunks)
echo "=========================================="
echo "📄 Response Preview (First 5 chunks)"
echo "=========================================="
echo "$ACTUAL_RESPONSE" | head -n 5 | while IFS= read -r line; do
    TEXT=$(echo "$line" | grep -o '"text":"[^"]*"' | cut -d'"' -f4)
    FINAL=$(echo "$line" | grep -o '"is_final":[^,}]*' | cut -d':' -f2)
    echo "  [Final: $FINAL] ${TEXT:0:80}..."
done

echo ""
echo "=========================================="

# Performance assessment
TTFB_MS=$(echo "$TTFB * 1000" | bc | cut -d'.' -f1)
if [ "$TTFB_MS" -lt 500 ]; then
    echo "${GREEN}✅ EXCELLENT: TTFB < 500ms${NC}"
elif [ "$TTFB_MS" -lt 1000 ]; then
    echo "${YELLOW}⚠️  GOOD: TTFB < 1000ms${NC}"
else
    echo "${RED}❌ NEEDS IMPROVEMENT: TTFB >= 1000ms${NC}"
fi

echo ""




