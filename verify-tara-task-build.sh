#!/bin/bash
# TARA X TASK - Build Verification Script
# Verifies that all required images are built and ready

set -e

echo "======================================================================"
echo "🔍 TARA X TASK - Build Verification"
echo "======================================================================"

# Check Docker context
echo ""
echo "📋 Checking Docker context..."
CURRENT_CONTEXT=$(docker context show)
echo "   Current context: $CURRENT_CONTEXT"

if [ "$CURRENT_CONTEXT" != "desktop-linux" ]; then
    echo "   ⚠️  Warning: Context is not 'desktop-linux'"
    echo "   Run: docker context use desktop-linux"
else
    echo "   ✅ Context is correct"
fi

# Check required images
echo ""
echo "📦 Checking required images..."
export COMPOSE_PROJECT_NAME=tara-task

REQUIRED_SERVICES=(
    "redis"
    "orchestrator"
    "stt-vad"
    "rag"
    "tts-sarvam"
)

MISSING_IMAGES=()
EXISTING_IMAGES=()

# Check Redis (official image)
if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^redis:7-alpine$"; then
    EXISTING_IMAGES+=("redis")
    SIZE=$(docker images --format "{{.Size}}" "redis:7-alpine" | head -1)
    echo "   ✅ redis ($SIZE)"
else
    MISSING_IMAGES+=("redis")
    echo "   ⚠️  redis:7-alpine (will be pulled automatically when starting)"
fi

# Check built images using docker-compose (most reliable)
COMPOSE_FILE="docker-compose-tara-task.yml"
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "   ❌ Error: $COMPOSE_FILE not found!"
    exit 1
fi

for service in "${REQUIRED_SERVICES[@]}"; do
    if [ "$service" = "redis" ]; then
        continue  # Already checked above
    fi
    
    # Use docker-compose to check if image exists
    # docker-compose images shows images even if they're stored with SHA256 tags
    IMAGE_CHECK=$(docker-compose -f "$COMPOSE_FILE" images "$service" 2>/dev/null | grep -v "CONTAINER" | grep -v "^$" | tail -n +2 | head -1)
    
    if [ -n "$IMAGE_CHECK" ] && echo "$IMAGE_CHECK" | grep -q "tara-task"; then
        EXISTING_IMAGES+=("$service")
        SIZE=$(echo "$IMAGE_CHECK" | awk '{print $6}')
        echo "   ✅ $service ($SIZE)"
    else
        # Fallback: check if image ID exists (docker-compose might store with SHA256)
        IMAGE_ID=$(docker-compose -f "$COMPOSE_FILE" images -q "$service" 2>/dev/null | head -1)
        if [ -n "$IMAGE_ID" ]; then
            EXISTING_IMAGES+=("$service")
            SIZE=$(docker images --format "{{.Size}}" "$IMAGE_ID" 2>/dev/null | head -1)
            echo "   ✅ $service ($SIZE) - found by ID"
        else
            MISSING_IMAGES+=("$service")
            echo "   ❌ $service (missing)"
        fi
    fi
done

# Summary
echo ""
echo "======================================================================"
echo "📊 Summary"
echo "======================================================================"
echo "   Found: ${#EXISTING_IMAGES[@]}/${#REQUIRED_SERVICES[@]} images"

if [ ${#MISSING_IMAGES[@]} -eq 0 ]; then
    echo ""
    echo "✅ All required images are built!"
    echo ""
    echo "To start the services:"
    echo "   ./start-tara-task.sh"
    exit 0
else
    echo ""
    echo "❌ Missing images:"
    for img in "${MISSING_IMAGES[@]}"; do
        echo "   - $img"
    done
    echo ""
    echo "To build missing images:"
    echo "   ./build-tara-task.sh"
    exit 1
fi

