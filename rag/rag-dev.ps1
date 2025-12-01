# ============================================================================
# RAG Service - Development Helper Script (PowerShell)
# ============================================================================
# Usage: 
#   .\rag-dev.ps1 build    - Build development image
#   .\rag-dev.ps1 start    - Start in development mode
#   .\rag-dev.ps1 setup    - Run setup (install heavy deps)
#   .\rag-dev.ps1 run      - Start service with hot-reload
#   .\rag-dev.ps1 logs     - Show container logs
#   .\rag-dev.ps1 shell    - Open interactive shell
#   .\rag-dev.ps1 test     - Run tests
#   .\rag-dev.ps1 rebuild  - Rebuild FAISS index
#   .\rag-dev.ps1 prod     - Start in production mode
#   .\rag-dev.ps1 stop     - Stop container
# ============================================================================

param(
    [Parameter(Mandatory=$false, Position=0)]
    [string]$Command = "help",
    
    [Parameter(Mandatory=$false, Position=1)]
    [string]$Arg = ""
)

$COMPOSE_FILE = "docker-compose.leibniz.yml"
$SERVICE_NAME = "rag"
$CONTAINER_NAME = "leibniz-rag"

function Log-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

function Log-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Log-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Log-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

switch ($Command.ToLower()) {
    "build" {
        Log-Info "Building RAG service image (lightweight, ~2 min)..."
        docker-compose -f $COMPOSE_FILE build $SERVICE_NAME
        Log-Success "Build complete!"
        Log-Info "Next step: .\rag-dev.ps1 start"
    }
    
    "start" {
        Log-Info "Starting RAG service in development mode..."
        docker-compose -f $COMPOSE_FILE up -d $SERVICE_NAME
        Log-Success "Container started!"
        Log-Warning "Heavy dependencies not installed yet"
        Log-Info "Next step: .\rag-dev.ps1 setup"
    }
    
    "setup" {
        Log-Info "Installing heavy dependencies (torch, sentence-transformers)..."
        Log-Warning "This will take 3-5 minutes on first run"
        docker exec -it $CONTAINER_NAME /app/setup_heavy_deps.sh
        Log-Success "Setup complete! Service ready to run."
        Log-Info "Next step: .\rag-dev.ps1 run"
    }
    
    "run" {
        Log-Info "Starting RAG service with hot-reload..."
        Log-Info "Edit files on host - service will auto-reload!"
        Log-Info "Access: http://localhost:8003/health"
        Log-Info "API Docs: http://localhost:8003/docs"
        Log-Info "Press Ctrl+C to stop"
        docker exec -it $CONTAINER_NAME uvicorn leibniz_agent.services.rag.app:app `
            --host 0.0.0.0 `
            --port 8003 `
            --reload
    }
    
    "logs" {
        Log-Info "Showing container logs (Ctrl+C to exit)..."
        docker logs -f $CONTAINER_NAME
    }
    
    "shell" {
        Log-Info "Opening interactive shell..."
        docker exec -it $CONTAINER_NAME /bin/bash
    }
    
    "test" {
        Log-Info "Running RAG service tests..."
        docker exec -it $CONTAINER_NAME pytest leibniz_agent/services/rag/tests/ -v
    }
    
    "rebuild" {
        Log-Info "Rebuilding FAISS index from knowledge base..."
        docker exec -it $CONTAINER_NAME python -m leibniz_agent.services.rag.index_builder `
            --knowledge-base /app/leibniz_knowledge_base `
            --output /app/index
        Log-Success "Index rebuilt!"
        Log-Warning "Restart service to reload: docker restart $CONTAINER_NAME"
    }
    
    "prod" {
        Log-Info "Starting RAG service in PRODUCTION mode..."
        Log-Warning "First start will take ~8 minutes (installs torch + builds index)"
        docker-compose -f $COMPOSE_FILE --profile production up -d rag-prod
        Log-Success "Production service starting..."
        Log-Info "Check logs: docker logs -f leibniz-rag-prod"
    }
    
    "stop" {
        Log-Info "Stopping RAG service..."
        docker-compose -f $COMPOSE_FILE stop $SERVICE_NAME
        Log-Success "Container stopped"
    }
    
    "restart" {
        Log-Info "Restarting RAG service..."
        docker restart $CONTAINER_NAME
        Log-Success "Container restarted"
    }
    
    "clean" {
        Log-Warning "Removing container and volumes..."
        docker-compose -f $COMPOSE_FILE down -v
        Log-Success "Cleanup complete"
    }
    
    "health" {
        Log-Info "Checking RAG service health..."
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8003/health" -Method Get -UseBasicParsing
            Log-Success "Service is healthy!"
            $response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
        }
        catch {
            Log-Error "Service is not responding"
            exit 1
        }
    }
    
    "metrics" {
        Log-Info "Fetching service metrics..."
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8003/metrics" -Method Get -UseBasicParsing
            $response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
        }
        catch {
            Log-Error "Failed to fetch metrics"
        }
    }
    
    "query" {
        if ([string]::IsNullOrEmpty($Arg)) {
            Log-Error "Usage: .\rag-dev.ps1 query `"Your question here`""
            exit 1
        }
        Log-Info "Querying RAG service..."
        $body = @{ query = $Arg } | ConvertTo-Json
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8003/api/v1/query" `
                -Method Post `
                -ContentType "application/json" `
                -Body $body `
                -UseBasicParsing
            $response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
        }
        catch {
            Log-Error "Query failed: $_"
        }
    }
    
    default {
        Write-Host ""
        Write-Host "RAG Service - Development Helper" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Usage: .\rag-dev.ps1 COMMAND" -ForegroundColor White
        Write-Host ""
        Write-Host "Development Workflow:" -ForegroundColor Yellow
        Write-Host "  build      - Build development image (~2 min)"
        Write-Host "  start      - Start container in dev mode"
        Write-Host "  setup      - Install heavy deps (one-time, ~5 min)"
        Write-Host "  run        - Start service with hot-reload"
        Write-Host "  shell      - Open interactive shell"
        Write-Host ""
        Write-Host "Testing & Debugging:" -ForegroundColor Yellow
        Write-Host "  logs       - Show container logs"
        Write-Host "  test       - Run pytest tests"
        Write-Host "  health     - Check service health"
        Write-Host "  metrics    - Show service metrics"
        Write-Host "  query      - Test RAG query (usage: query `"question`")"
        Write-Host ""
        Write-Host "Maintenance:" -ForegroundColor Yellow
        Write-Host "  rebuild    - Rebuild FAISS index"
        Write-Host "  restart    - Restart container"
        Write-Host "  stop       - Stop container"
        Write-Host "  clean      - Remove container and volumes"
        Write-Host ""
        Write-Host "Production:" -ForegroundColor Yellow
        Write-Host "  prod       - Start in production mode (~8 min first time)"
        Write-Host ""
        Write-Host "Quick Start:" -ForegroundColor Green
        Write-Host "  1. .\rag-dev.ps1 build"
        Write-Host "  2. .\rag-dev.ps1 start"
        Write-Host "  3. .\rag-dev.ps1 setup"
        Write-Host "  4. .\rag-dev.ps1 run"
        Write-Host ""
    }
}
