#!/bin/bash
# Setup test environment for STT Local Service
# Creates a virtual environment and installs dependencies

set -e

echo "=========================================="
echo "Setting up STT Local Test Environment"
echo "=========================================="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install lightweight dependencies
echo "Installing lightweight dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  python3 test_local.py"
echo ""
echo "To install heavy dependencies (optional):"
echo "  pip install -r requirements_after.txt"
echo "=========================================="

