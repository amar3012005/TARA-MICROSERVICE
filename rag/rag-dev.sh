#!/bin/bash
# ============================================================================
# RAG Service - Development Helper Script
# ============================================================================
# Usage: 
#   ./rag-dev.sh build    - Build development image
#   ./rag-dev.sh start    - Start in development mode
#   ./rag-dev.sh setup    - Run setup (install heavy deps)
#   ./rag-dev.sh run      - Start service with hot-reload
#   ./rag-dev.sh logs     - Show container logs
#   ./rag-dev.sh shell    - Open interactive shell
#   ./rag-dev.sh test     - Run tests
#   ./rag-dev.sh rebuild  - Rebuild FAISS index
#   ./rag-dev.sh prod     - Start in production mode
#   ./rag-dev.sh stop     - Stop container
# ============================================================================

set -e

COMPOSE_FILE="docker-compose.leibniz.yml"
SERVICE_NAME="rag"
CONTAINER_NAME="leibniz-rag"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

case "$1" in
    build)
        log_info "Building RAG service image (lightweight, ~2 min)..."
        docker-compose -f "$COMPOSE_FILE" build "$SERVICE_NAME"
        log_success "Build complete!"
        log_info "Next step: ./rag-dev.sh start"
        ;;
    
    start)
        log_info "Starting RAG service in development mode..."
        docker-compose -f "$COMPOSE_FILE" up -d "$SERVICE_NAME"
        log_success "Container started!"
        log_warning "Heavy dependencies not installed yet"
        log_info "Next step: ./rag-dev.sh setup"
        ;;
    
    setup)
        log_info "Installing heavy dependencies (torch, sentence-transformers)..."
        log_warning "This will take 3-5 minutes on first run"
        docker exec -it "$CONTAINER_NAME" /app/setup_heavy_deps.sh
        log_success "Setup complete! Service ready to run."
        log_info "Next step: ./rag-dev.sh run"
        ;;
    
    run)
        log_info "Starting RAG service with hot-reload..."
        log_info "Edit files on host - service will auto-reload!"
        log_info "Access: http://localhost:8003/health"
        log_info "API Docs: http://localhost:8003/docs"
        log_info "Press Ctrl+C to stop"
        docker exec -it "$CONTAINER_NAME" uvicorn leibniz_agent.services.rag.app:app \
            --host 0.0.0.0 \
            --port 8003 \
            --reload
        ;;
    
    logs)
        log_info "Showing container logs (Ctrl+C to exit)..."
        docker logs -f "$CONTAINER_NAME"
        ;;
    
    shell)
        log_info "Opening interactive shell..."
        docker exec -it "$CONTAINER_NAME" /bin/bash
        ;;
    
    test)
        log_info "Running RAG service tests..."
        docker exec -it "$CONTAINER_NAME" pytest leibniz_agent/services/rag/tests/ -v
        ;;
    
    rebuild)
        log_info "Rebuilding FAISS index from knowledge base..."
        docker exec -it "$CONTAINER_NAME" python -m leibniz_agent.services.rag.index_builder \
            --knowledge-base /app/leibniz_knowledge_base \
            --output /app/index
        log_success "Index rebuilt!"
        log_warning "Restart service to reload: docker restart $CONTAINER_NAME"
        ;;
    
    prod)
        log_info "Starting RAG service in PRODUCTION mode..."
        log_warning "First start will take ~8 minutes (installs torch + builds index)"
        docker-compose -f "$COMPOSE_FILE" --profile production up -d rag-prod
        log_success "Production service starting..."
        log_info "Check logs: docker logs -f leibniz-rag-prod"
        ;;
    
    stop)
        log_info "Stopping RAG service..."
        docker-compose -f "$COMPOSE_FILE" stop "$SERVICE_NAME"
        log_success "Container stopped"
        ;;
    
    restart)
        log_info "Restarting RAG service..."
        docker restart "$CONTAINER_NAME"
        log_success "Container restarted"
        ;;
    
    clean)
        log_warning "Removing container and volumes..."
        docker-compose -f "$COMPOSE_FILE" down -v
        log_success "Cleanup complete"
        ;;
    
    health)
        log_info "Checking RAG service health..."
        if curl -f http://localhost:8003/health 2>/dev/null; then
            log_success "Service is healthy!"
        else
            log_error "Service is not responding"
            exit 1
        fi
        ;;
    
    metrics)
        log_info "Fetching service metrics..."
        curl -s http://localhost:8003/metrics | python -m json.tool
        ;;
    
    query)
        if [ -z "$2" ]; then
            log_error "Usage: ./rag-dev.sh query \"Your question here\""
            exit 1
        fi
        log_info "Querying RAG service..."
        curl -X POST http://localhost:8003/api/v1/query \
            -H "Content-Type: application/json" \
            -d "{\"query\": \"$2\"}" | python -m json.tool
        ;;
    
    *)
        echo "RAG Service - Development Helper"
        echo ""
        echo "Usage: ./rag-dev.sh COMMAND"
        echo ""
        echo "Development Workflow:"
        echo "  build      - Build development image (~2 min)"
        echo "  start      - Start container in dev mode"
        echo "  setup      - Install heavy deps (one-time, ~5 min)"
        echo "  run        - Start service with hot-reload"
        echo "  shell      - Open interactive shell"
        echo ""
        echo "Testing & Debugging:"
        echo "  logs       - Show container logs"
        echo "  test       - Run pytest tests"
        echo "  health     - Check service health"
        echo "  metrics    - Show service metrics"
        echo "  query      - Test RAG query (usage: query \"question\")"
        echo ""
        echo "Maintenance:"
        echo "  rebuild    - Rebuild FAISS index"
        echo "  restart    - Restart container"
        echo "  stop       - Stop container"
        echo "  clean      - Remove container and volumes"
        echo ""
        echo "Production:"
        echo "  prod       - Start in production mode (~8 min first time)"
        echo ""
        echo "Quick Start:"
        echo "  1. ./rag-dev.sh build"
        echo "  2. ./rag-dev.sh start"
        echo "  3. ./rag-dev.sh setup"
        echo "  4. ./rag-dev.sh run"
        echo ""
        exit 1
        ;;
esac
