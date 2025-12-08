#!/bin/bash
# TARA X TASK - Build Script for Docker Desktop Linux
# Builds all microservices for the tara-task cluster

set -e

echo "======================================================================"
echo "🔨 TARA X TASK - Building Microservices for Docker Desktop Linux"
echo "======================================================================"

# Set Docker context (CRITICAL for Docker Desktop visibility)
echo ""
echo "📋 Setting Docker context to desktop-linux..."
docker context use desktop-linux

# Verify context is set
CURRENT_CONTEXT=$(docker context show)
if [ "$CURRENT_CONTEXT" != "desktop-linux" ]; then
    echo "❌ Error: Failed to set Docker context to desktop-linux"
    echo "   Current context: $CURRENT_CONTEXT"
    echo "   Available contexts:"
    docker context ls
    exit 1
fi
echo "✅ Docker context set to: $CURRENT_CONTEXT"

# Navigate to project directory
cd "$(dirname "$0")"
echo "📂 Working directory: $(pwd)"

# Check if docker-compose file exists
COMPOSE_FILE="docker-compose-tara-task.yml"
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ Error: $COMPOSE_FILE not found!"
    exit 1
fi

# Set project name for Docker Desktop organization
export COMPOSE_PROJECT_NAME=tara-task

echo ""
echo "======================================================================"
echo "🏗️  Building Services"
echo "======================================================================"
echo "   Project: $COMPOSE_PROJECT_NAME"
echo "   Context: $(docker context show)"
echo "   Compose file: $COMPOSE_FILE"
echo ""

# Build all services
echo "🔨 Building all services (this may take several minutes)..."
echo ""

docker-compose -f "$COMPOSE_FILE" build --parallel

# Check build status
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ Build Complete!"
    echo "======================================================================"
    echo ""
    echo "📦 Built Images:"
    docker images --filter "reference=tara-task*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo ""
    echo "======================================================================"
    echo "📋 Next Steps"
    echo "======================================================================"
    echo "1. Start the services:"
    echo "   ./start-tara-task.sh"
    echo ""
    echo "2. Or start manually:"
    echo "   docker context use desktop-linux"
    echo "   export COMPOSE_PROJECT_NAME=tara-task"
    echo "   docker-compose -f $COMPOSE_FILE up -d"
    echo ""
    echo "3. View in Docker Desktop:"
    echo "   - Open Docker Desktop"
    echo "   - Go to Images tab"
    echo "   - Look for 'tara-task-*' images"
    echo "   - Go to Containers tab"
    echo "   - Look for 'tara-task' project folder"
    echo ""
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "❌ Build Failed!"
    echo "======================================================================"
    echo "Check the error messages above for details."
    echo ""
    echo "Common issues:"
    echo "1. Docker context not set correctly"
    echo "2. Missing dependencies or files"
    echo "3. Network issues downloading packages"
    echo ""
    echo "Try rebuilding with verbose output:"
    echo "   docker-compose -f $COMPOSE_FILE build --progress=plain"
    exit 1
fi




