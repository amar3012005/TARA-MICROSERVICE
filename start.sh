#!/bin/bash

# TARA Microservices Startup Script
# Forces execution in 'desktop-linux' context using explicit flags.

set -e

# Configuration
REQUIRED_CONTEXT="desktop-linux"
COMPOSE_FILE="docker-compose-tara-task.yml"
PROJECT_NAME="tara-task"

echo "=================================================="
echo "🚀 TARA Microservices Launcher (Context Enforced)"
echo "=================================================="

# 1. Explicitly Clean 'default' context
echo "🧹 Cleaning 'default' context..."
# We use the docker CLI with --context flag to be absolutely sure where commands go
#docker --context default rm -f $(docker --context default ps -aq --filter name=${PROJECT_NAME}) 2>/dev/null || true
echo "   ✅ Default context clean."

# 2. Explicitly Clean 'desktop-linux' context
echo "🧹 Cleaning '$REQUIRED_CONTEXT' context..."
#docker --context $REQUIRED_CONTEXT rm -f $(docker --context $REQUIRED_CONTEXT ps -aq --filter name=${PROJECT_NAME}) 2>/dev/null || true
echo "   ✅ Desktop-linux context clean."

# 3. Set Environment
export COMPOSE_PROJECT_NAME=$PROJECT_NAME

# 4. Start Fresh using 'docker compose' V2 with explicit context
echo "🚀 Starting TARA Microservices in '$REQUIRED_CONTEXT'..."

# Check if 'docker compose' is available, otherwise fall back to 'docker-compose' with env var
if docker compose version >/dev/null 2>&1; then
    echo "   Using 'docker compose' (V2)..."
    # This is the key fix: passing --context directly to the docker command
    docker --context $REQUIRED_CONTEXT compose -f $COMPOSE_FILE up -d --build --force-recreate
else
    echo "   Using 'docker-compose' (Legacy) - attempting to force context via env..."
    export DOCKER_CONTEXT=$REQUIRED_CONTEXT
    docker-compose -f $COMPOSE_FILE up -d --build --force-recreate
fi

# 5. Verify
echo "=================================================="
echo "📊 Service Status (Context: $REQUIRED_CONTEXT):"
docker --context $REQUIRED_CONTEXT ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep $PROJECT_NAME
echo "=================================================="