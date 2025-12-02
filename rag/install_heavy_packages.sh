#!/bin/bash
# Script to install heavy packages after container is built
# Usage: Run this inside the container terminal

set -e

echo "=========================================="
echo "📦 Installing Heavy Packages"
echo "=========================================="
echo ""
echo "This will install:"
echo "  - sentence-transformers (~500MB)"
echo "  - torch (~900MB)"
echo "  - tokenizers (~3MB)"
echo "  - transformers (~100MB)"
echo "  - huggingface-hub (~50MB)"
echo ""
echo "Total download size: ~1.5GB"
echo "Estimated time: 5-10 minutes"
echo ""
echo "Starting installation..."
echo ""

# Install heavy packages
pip install --default-timeout=1000 --user --no-cache-dir -r /app/leibniz_agent/services/rag/requirements_after.txt

echo ""
echo "=========================================="
echo "✅ Heavy packages installed successfully!"
echo "=========================================="
echo ""
echo "You can now use the RAG service with full ML capabilities."
echo "To verify, run: python -c 'import sentence_transformers; print(\"OK\")'"




