#!/bin/bash
set -e

# Ensure we are in the services directory
cd "$(dirname "$0")"
echo "📂 Working directory: $(pwd)"

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: docker could not be found. Please install docker first."
    exit 1
fi

echo "🚀 Building Leibniz Service Containers"
echo "======================================="

# Build STT/VAD Service
echo "🔊 Building STT/VAD Service..."
echo "   Context: services/"
echo "   Dockerfile: services/stt_vad/Dockerfile"
docker build -f stt_vad/Dockerfile -t leibniz-stt-vad:latest .
echo "✅ STT/VAD Service built successfully!"
echo ""

# Build Intent Service
echo "🧠 Building Intent Service..."
echo "   Context: services/"
echo "   Dockerfile: services/intent/Dockerfile"
docker build -f intent/Dockerfile -t leibniz-intent:latest .
echo "✅ Intent Service built successfully!"
echo ""

# Build RAG Service
echo "📚 Building RAG Service..."
echo "   Context: services/"
echo "   Dockerfile: services/rag/Dockerfile"
echo "   Note: This will install torch and sentence-transformers (~2GB), and build FAISS index"
docker build -f rag/Dockerfile -t leibniz-rag:latest .
echo "✅ RAG Service built successfully!"
echo ""

echo "🎉 All requested services built!"
echo "   - leibniz-stt-vad:latest"
echo "   - leibniz-intent:latest"
echo "   - leibniz-rag:latest"

