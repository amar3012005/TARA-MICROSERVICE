#!/bin/bash

# Test RAG Service with Cold Query (No Cache)
# Uses a unique query each time to avoid cache hits

echo "🧪 Testing RAG Service - COLD QUERY (No Cache)"
echo "=========================================="
echo ""

# Generate unique query with timestamp to avoid cache
UNIQUE_QUERY="what is task organization $(date +%s)"

echo "📝 Query: \"$UNIQUE_QUERY\""
echo ""

# Store response in temp file
TEMP_FILE=$(mktemp)

# Run curl and capture timing
curl -w "\nTIMING_START\nTTFB:%{time_starttransfer}\nCONNECT:%{time_connect}\nTOTAL:%{time_total}\nTIMING_END\n" \
  -s \
  -X POST \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"$UNIQUE_QUERY\",
    \"context\": {
      \"language\": \"te-mixed\",
      \"organization\": \"T.A.S.K\"
    },
    \"enable_streaming\": true
  }" \
  http://localhost:2003/api/v1/stream_query > "$TEMP_FILE"

# Extract all chunks (excluding timing lines)
ALL_CHUNKS=$(grep -v "TIMING" "$TEMP_FILE")

# Extract first chunk
FIRST_CHUNK=$(echo "$ALL_CHUNKS" | head -n 1)

# Extract timing info
TTFB=$(grep "TTFB:" "$TEMP_FILE" | cut -d':' -f2)
CONNECT=$(grep "CONNECT:" "$TEMP_FILE" | cut -d':' -f2)
TOTAL=$(grep "TOTAL:" "$TEMP_FILE" | cut -d':' -f2)

# Convert to milliseconds
TTFB_MS=$(echo "$TTFB * 1000" | bc)
CONNECT_MS=$(echo "$CONNECT * 1000" | bc)
TOTAL_MS=$(echo "$TOTAL * 1000" | bc)

echo "📦 First Chunk:"
echo "$FIRST_CHUNK"
echo ""
echo "=========================================="
echo "📄 COMPLETE RESPONSE"
echo "=========================================="
# Extract and display all text chunks
COMPLETE_TEXT=""
while IFS= read -r line; do
    TEXT=$(echo "$line" | grep -o '"text":"[^"]*"' | cut -d'"' -f4)
    if [ -n "$TEXT" ]; then
        COMPLETE_TEXT="${COMPLETE_TEXT}${TEXT}"
    fi
done <<< "$ALL_CHUNKS"
echo "$COMPLETE_TEXT"
echo ""
echo ""
echo "=========================================="
echo "⏱️  COLD QUERY LATENCY METRICS"
echo "=========================================="
printf "Time to First Byte (TTFB): %.2f ms\n" "$TTFB_MS"
printf "Connection Time:           %.2f ms\n" "$CONNECT_MS"
printf "Total Response Time:       %.2f ms\n" "$TOTAL_MS"
echo "=========================================="
echo ""

# Performance assessment
TTFB_INT=$(echo "$TTFB_MS" | cut -d'.' -f1)
if [ "$TTFB_INT" -lt 500 ]; then
    echo "✅ EXCELLENT: TTFB < 500ms (Target met!)"
elif [ "$TTFB_INT" -lt 1000 ]; then
    echo "⚠️  GOOD: TTFB < 1000ms"
else
    echo "❌ NEEDS IMPROVEMENT: TTFB >= 1000ms"
fi

# Cleanup
rm "$TEMP_FILE"




