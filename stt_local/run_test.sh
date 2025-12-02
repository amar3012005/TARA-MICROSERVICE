#!/bin/bash
# Quick test script for STT Local Service with FastRTC

set -e

echo "=========================================="
echo "STT Local Service - FastRTC Test"
echo "=========================================="

# Check if venv exists
if [ ! -d "../venv" ]; then
    echo "Creating virtual environment..."
    cd ..
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    cd stt_local
fi

# Activate venv
cd ..
source venv/bin/activate
cd stt_local

# Install dependencies if needed
echo "Checking dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing lightweight dependencies..."
    pip install -r requirements.txt
fi

# Check if heavy deps are installed
if ! python3 -c "import faster_whisper" 2>/dev/null; then
    echo ""
    echo "⚠️  Heavy dependencies (faster-whisper, torch) not installed."
    echo "   The service will work but Whisper tests will skip."
    echo "   Install with: pip install -r requirements_after.txt"
    echo ""
fi

# Set environment for CPU mode (safer for local testing)
export LEIBNIZ_STT_LOCAL_WHISPER_DEVICE=cpu
export LEIBNIZ_STT_LOCAL_USE_GPU=false

echo ""
echo "=========================================="
echo "Starting STT Local Service..."
echo "=========================================="
echo ""
echo "📊 FastRTC UI: http://localhost:7861/fastrtc"
echo "📊 API: http://localhost:8006"
echo "📊 Health: http://localhost:8006/health"
echo ""
echo "🎤 Instructions:"
echo "   1. Open http://localhost:7861/fastrtc in browser"
echo "   2. Click 'Start' to begin streaming"
echo "   3. Speak into microphone"
echo "   4. Watch this terminal for real-time transcripts!"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Run the service
python3 run_local_fastrtc.py



