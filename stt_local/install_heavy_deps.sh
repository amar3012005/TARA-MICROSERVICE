#!/bin/bash
# Install heavy dependencies after container starts
# Run this manually in the container: bash install_heavy_deps.sh

echo "=========================================="
echo "Installing Heavy Dependencies"
echo "=========================================="
echo "This will install:"
echo "  - PyTorch (CUDA) ~2GB"
echo "  - Faster Whisper ~500MB"
echo "  - Audio libraries"
echo ""
echo "This may take 5-10 minutes..."
echo "=========================================="

pip3 install --no-cache-dir --user -r requirements_after.txt

echo ""
echo "=========================================="
echo "✅ Heavy dependencies installed!"
echo "=========================================="
echo "You can now start the service with:"
echo "  python3 -u -m uvicorn app:app --host 0.0.0.0 --port 8006 --workers 1"
echo "=========================================="

