#!/bin/bash
# Verify Docker Desktop visibility for TARA X TASK

echo "======================================================================"
echo "🔍 Verifying Docker Desktop Visibility"
echo "======================================================================"

# Set context
docker context use desktop-linux
CURRENT_CONTEXT=$(docker context show)
echo "✅ Docker Context: $CURRENT_CONTEXT"
echo ""

# Check if containers exist
echo "📋 Checking containers..."
CONTAINERS=$(docker ps -a --filter "name=tara-task" --format "{{.Names}}")
if [ -z "$CONTAINERS" ]; then
    echo "❌ No tara-task containers found!"
    exit 1
fi

echo "Found containers:"
docker ps --filter "name=tara-task" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Check project labels
echo "📋 Checking project labels..."
docker ps --filter "name=tara-task" --format "{{.Names}}: {{.Label \"com.docker.compose.project\"}}" | while read line; do
    echo "   $line"
done
echo ""

# Check network
echo "📋 Checking network..."
docker network ls --filter "name=tara-task" --format "table {{.Name}}\t{{.Driver}}"
echo ""

# Check volumes
echo "📋 Checking volumes..."
docker volume ls --filter "label=com.tara.project=tara-task" --format "table {{.Name}}\t{{.Driver}}"
echo ""

echo "======================================================================"
echo "💡 Docker Desktop Tips:"
echo "======================================================================"
echo "1. Open Docker Desktop GUI"
echo "2. Look for 'tara-task' in the Containers list"
echo "3. If not visible, try:"
echo "   - Refresh Docker Desktop (F5 or restart)"
echo "   - Check Filters (make sure 'All' is selected)"
echo "   - Verify context: docker context show"
echo ""
echo "4. Containers should appear under project name: tara-task"
echo "======================================================================"




