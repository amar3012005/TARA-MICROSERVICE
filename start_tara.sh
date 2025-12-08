#!/bin/bash
#
# TARA - Telugu TASK Customer Service Agent
# Startup Script
#

set -e

echo "=============================================="
echo "🇮🇳 TARA - Telugu TASK Customer Service Agent"
echo "=============================================="
echo ""

# Check for required environment variables
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  GEMINI_API_KEY not set. Using default key."
    echo "   For production, set: export GEMINI_API_KEY=your_key"
fi

if [ -z "$SARVAM_API_KEY" ]; then
    echo "⚠️  SARVAM_API_KEY not set. Using default key."
    echo "   For production, set: export SARVAM_API_KEY=your_key"
fi

# Navigate to TARA-MICROSERVICE directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "📁 Working directory: $SCRIPT_DIR"
echo ""

# Check if knowledge base exists
if [ ! -d "./task_knowledge_base" ]; then
    echo "⚠️  TASK knowledge base not found. Creating sample structure..."
    mkdir -p ./task_knowledge_base/{services,contact,faq,procedures}
    echo "   Created: ./task_knowledge_base/"
fi

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker first."
        exit 1
    fi
    echo "✅ Docker is running"
}

# Function to build services
build_services() {
    echo ""
    echo "🔨 Building TARA services..."
    docker-compose -f docker-compose-tara.yml build --parallel
    echo "✅ Build complete"
}

# Function to start services
start_services() {
    echo ""
    echo "🚀 Starting TARA services..."
    docker-compose -f docker-compose-tara.yml up -d
    echo ""
    echo "✅ Services started"
}

# Function to wait for services
wait_for_services() {
    echo ""
    echo "⏳ Waiting for services to be ready..."
    
    # Wait for Redis
    echo -n "   Redis: "
    for i in {1..30}; do
        if docker exec tara-redis redis-cli ping > /dev/null 2>&1; then
            echo "✅ Ready"
            break
        fi
        sleep 1
        echo -n "."
    done
    
    # Wait for RAG service
    echo -n "   RAG Service: "
    for i in {1..60}; do
        if curl -s http://localhost:8022/health > /dev/null 2>&1; then
            echo "✅ Ready"
            break
        fi
        sleep 2
        echo -n "."
    done
    
    # Wait for Orchestrator
    echo -n "   Orchestrator: "
    for i in {1..30}; do
        if curl -s http://localhost:8023/health > /dev/null 2>&1; then
            echo "✅ Ready"
            break
        fi
        sleep 1
        echo -n "."
    done
    
    # Wait for TTS Sarvam
    echo -n "   TTS Sarvam: "
    for i in {1..30}; do
        if curl -s http://localhost:8024/health > /dev/null 2>&1; then
            echo "✅ Ready"
            break
        fi
        sleep 1
        echo -n "."
    done
}

# Function to show service info
show_info() {
    echo ""
    echo "=============================================="
    echo "🎉 TARA is ready!"
    echo "=============================================="
    echo ""
    echo "📡 Service URLs:"
    echo "   Orchestrator:    http://localhost:8023"
    echo "   STT FastRTC UI:  http://localhost:7870"
    echo "   RAG Service:     http://localhost:8022"
    echo "   TTS Sarvam:      http://localhost:8024"
    echo "   Redis:           localhost:6382"
    echo ""
    echo "🎤 Telugu Intro Greeting:"
    echo "   నమస్కారం అండి! నేను TARA, TASK యొక్క కస్టమర్ సర్వీస్ ఏజెంట్."
    echo "   మీకు ఎలా సహాయం చేయగలను?"
    echo ""
    echo "📝 Quick Commands:"
    echo "   Start workflow:  curl -X POST http://localhost:8023/start"
    echo "   Check status:    curl http://localhost:8023/status"
    echo "   View logs:       docker-compose -f docker-compose-tara.yml logs -f"
    echo "   Stop TARA:       docker-compose -f docker-compose-tara.yml down"
    echo ""
    echo "=============================================="
}

# Main execution
case "${1:-start}" in
    build)
        check_docker
        build_services
        ;;
    start)
        check_docker
        start_services
        wait_for_services
        show_info
        ;;
    stop)
        echo "🛑 Stopping TARA services..."
        docker-compose -f docker-compose-tara.yml down
        echo "✅ Services stopped"
        ;;
    restart)
        echo "🔄 Restarting TARA services..."
        docker-compose -f docker-compose-tara.yml restart
        wait_for_services
        show_info
        ;;
    logs)
        docker-compose -f docker-compose-tara.yml logs -f
        ;;
    status)
        echo "📊 TARA Service Status:"
        docker-compose -f docker-compose-tara.yml ps
        echo ""
        echo "🔍 Health Checks:"
        echo -n "   Orchestrator: "
        curl -s http://localhost:8023/health | jq -r '.status' 2>/dev/null || echo "❌ Not responding"
        echo -n "   RAG Service:  "
        curl -s http://localhost:8022/health | jq -r '.status' 2>/dev/null || echo "❌ Not responding"
        echo -n "   TTS Sarvam:   "
        curl -s http://localhost:8024/health | jq -r '.status' 2>/dev/null || echo "❌ Not responding"
        ;;
    rebuild-index)
        echo "🔄 Rebuilding RAG index..."
        curl -X POST http://localhost:8022/api/v1/admin/rebuild_index
        echo ""
        echo "✅ Index rebuild triggered"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|build|logs|status|rebuild-index}"
        echo ""
        echo "Commands:"
        echo "  start         Start TARA services"
        echo "  stop          Stop TARA services"
        echo "  restart       Restart TARA services"
        echo "  build         Build Docker images"
        echo "  logs          View service logs"
        echo "  status        Check service status"
        echo "  rebuild-index Rebuild RAG knowledge base index"
        exit 1
        ;;
esac





