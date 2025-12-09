#!/bin/bash

# Quick RAG Latency Test - Simple curl command with timing

echo "🚀 Testing RAG Service - Complete Response..."
echo ""

# Simple curl with timing - shows complete response
curl -w "\n\n⏱️  Time to First Byte: %{time_starttransfer}s\n⏱️  Total Time: %{time_total}s\n" \
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
  http://localhost:2003/api/v1/stream_query

echo ""
echo "✅ Test complete!"

