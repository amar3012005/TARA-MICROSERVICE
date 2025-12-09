#!/bin/bash

# Test RAG Service - Show Complete Response
# This shows the full streaming response with all chunks

echo "🚀 Testing RAG Service - Complete Response"
echo "=========================================="
echo ""

# Store response in temp file
TEMP_FILE=$(mktemp)

# Run curl and capture response with timing
curl -w "\nTIMING_START\nTTFB:%{time_starttransfer}\nCONNECT:%{time_connect}\nTOTAL:%{time_total}\nTIMING_END\n" \
  -s \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is task",
    "context": {
      "language": "te-mixed",
      "organization": "T.A.S.K"
    },
    "enable_streaming": true
  }' \
  http://localhost:2003/api/v1/stream_query > "$TEMP_FILE"

# Extract all chunks (excluding timing lines)
ALL_CHUNKS=$(grep -v "TIMING" "$TEMP_FILE")

# Extract timing info
TTFB=$(grep "TTFB:" "$TEMP_FILE" | cut -d':' -f2)
CONNECT=$(grep "CONNECT:" "$TEMP_FILE" | cut -d':' -f2)
TOTAL=$(grep "TOTAL:" "$TEMP_FILE" | cut -d':' -f2)

# Convert to milliseconds
TTFB_MS=$(echo "$TTFB * 1000" | bc)
CONNECT_MS=$(echo "$CONNECT * 1000" | bc)
TOTAL_MS=$(echo "$TOTAL * 1000" | bc)

echo "=========================================="
echo "📄 COMPLETE STREAMING RESPONSE"
echo "=========================================="
echo ""

# Display all chunks with formatting
CHUNK_NUM=1
COMPLETE_TEXT=""
while IFS= read -r line; do
    TEXT=$(echo "$line" | grep -o '"text":"[^"]*"' | cut -d'"' -f4)
    FINAL=$(echo "$line" | grep -o '"is_final":[^,}]*' | cut -d':' -f2 | tr -d ' ')
    
    if [ -n "$TEXT" ] || [ "$FINAL" = "true" ]; then
        if [ "$FINAL" = "true" ]; then
            echo "[Chunk $CHUNK_NUM - FINAL]"
        else
            echo "[Chunk $CHUNK_NUM] $TEXT"
            COMPLETE_TEXT="${COMPLETE_TEXT}${TEXT}"
            CHUNK_NUM=$((CHUNK_NUM + 1))
        fi
    fi
done <<< "$ALL_CHUNKS"

echo ""
echo "=========================================="
echo "📊 COMPLETE RESPONSE TEXT"
echo "=========================================="
echo "$COMPLETE_TEXT"
echo ""
echo ""

echo "=========================================="
echo "⏱️  LATENCY METRICS"
echo "=========================================="
printf "Time to First Byte (TTFB): %.2f ms\n" "$TTFB_MS"
printf "Connection Time:           %.2f ms\n" "$CONNECT_MS"
printf "Total Response Time:       %.2f ms\n" "$TOTAL_MS"
echo "=========================================="

# Cleanup
rm "$TEMP_FILE"

