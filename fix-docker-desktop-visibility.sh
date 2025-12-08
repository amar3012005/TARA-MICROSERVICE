#!/bin/bash
# Fix Docker Desktop Visibility for TARA-TASK containers
# Ensures containers appear in Docker Desktop GUI under "tara-task" folder

set -e

echo "======================================================================"
echo "🔧 Fixing Docker Desktop Visibility for TARA-TASK"
echo "======================================================================"

cd "$(dirname "$0")"

# Set Docker context
docker context use desktop-linux
export COMPOSE_PROJECT_NAME=tara-task

echo ""
echo "📋 Current Docker context: $(docker context show)"
echo "📋 Project name: $COMPOSE_PROJECT_NAME"
echo ""

# Stop and remove existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose-tara-task.yml down 2>/dev/null || true

# Wait a moment
sleep 2

# Start containers with explicit project name
echo ""
echo "🚀 Starting containers with proper project labels..."
docker-compose -f docker-compose-tara-task.yml --project-name tara-task up -d

# Wait for containers to start
echo ""
echo "⏳ Waiting for containers to initialize..."
sleep 5

# Verify labels
echo ""
echo "======================================================================"
echo "✅ Verifying Container Labels"
echo "======================================================================"
docker ps --filter "name=tara-task" --format "table {{.Names}}\t{{.Label \"com.docker.compose.project\"}}\t{{.Label \"com.tara.project\"}}" || \
docker ps --filter "name=tara-task" --format "{{.Names}}"

echo ""
echo "======================================================================"
echo "📋 Next Steps"
echo "======================================================================"
echo "1. Open Docker Desktop"
echo "2. Go to Containers tab"
echo "3. Look for 'tara-task' folder/project"
echo "4. If not visible, press F5 to refresh Docker Desktop"
echo ""
echo "To verify containers are running:"
echo "   docker ps --filter 'name=tara-task'"
echo ""
echo "To check logs:"
echo "   docker-compose -f docker-compose-tara-task.yml logs -f"
echo "======================================================================"




