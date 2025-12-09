# TARA GPU Container - Complete Microservices Stack

## Overview

This is the TARA X TASK microservices stack, a complete Telugu customer service agent system designed for GPU-accelerated operations. The stack includes speech-to-text, text-to-speech, RAG (Retrieval-Augmented Generation), and orchestration services, with the RAG service utilizing GPU-capable machine learning libraries for embeddings and processing.

**Key Features:**
- GPU-accelerated RAG service using PyTorch and Sentence Transformers
- Real-time speech processing with WebRTC integration
- Telugu language support with mixed-script processing
- Docker-based microservices architecture
- Redis-backed caching and session management
- LiveKit WebRTC server for real-time communication

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Docker Desktop: tara-task               │
│  ┌───────────────────────────────────────────┐  │
│  │      tara-task-network (bridge)           │  │
│  │                                            │  │
│  │  ┌──────────┐  ┌──────────┐  ┌────────┐ │  │
│  │  │  Redis   │  │   STT    │  │  RAG   │ │  │
│  │  │  :2006   │  │  :2001   │  │ :2003  │ │  │
│  │  │          │  │  :2012   │  │        │ │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬───┘ │  │
│  │       │              │             │     │  │
│  │       └──────────────┼─────────────┘     │  │
│  │                      │                    │  │
│  │              ┌───────▼───────┐           │  │
│  │              │ Orchestrator  │           │  │
│  │              │   :2004       │           │  │
│  │              │ (Master Ctrl) │           │  │
│  │              └───────┬───────┘           │  │
│  │                      │                    │  │
│  │              ┌───────▼───────┐           │  │
│  │              │     TTS       │           │  │
│  │              │   :2005       │           │  │
│  │              └───────────────┘           │  │
│  │                                            │  │
│  │  ┌──────────┐  ┌──────────┐               │  │
│  │  │ LiveKit  │  │ LiveKit  │               │  │
│  │  │ :7880-82 │  │ Bridge   │               │  │
│  │  └──────────┘  └──────────┘               │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Services

### 1. Redis (Port 2006)
- **Purpose**: Caching, session storage, and inter-service communication
- **Image**: redis:7-alpine
- **Data Persistence**: tara-task-redis-data volume
- **Health Check**: Built-in Redis ping

### 2. Orchestrator (Port 2004)
- **Purpose**: Master controller coordinating all services
- **Features**: 
  - Auto-starts dependent services
  - WebSocket orchestration
  - Session management (TTL: 3600s, max concurrent: 1000)
  - Telugu greeting and responses
- **Environment**: TARA_MODE=true, RESPONSE_LANGUAGE=te-mixed

### 3. STT-VAD Service (Ports 2001, 2012)
- **Purpose**: Speech-to-text with voice activity detection
- **Providers**: Gemini API + FastRTC streaming
- **Features**: Real-time transcription, barge-in detection
- **API**: /api/v1/transcribe/stream (WebSocket)

### 4. RAG Service (Port 2003) - **GPU ACCELERATED**
- **Purpose**: Knowledge base retrieval and generation
- **GPU Libraries**: 
  - PyTorch >= 2.0.0 (CUDA-enabled if GPU available)
  - Sentence Transformers (GPU embeddings)
  - FAISS-CPU (vector search)
- **Model**: Gemini 2.0 Flash Lite
- **Features**:
  - Telugu knowledge base processing
  - Vector embeddings with GPU acceleration
  - Prewarming and caching for performance
  - Embedding batch size: 32
- **Knowledge Base**: TASK organization data
- **Index Volume**: tara-task-rag-index

### 5. TTS Sarvam Service (Port 2005)
- **Purpose**: Text-to-speech synthesis
- **Provider**: Sarvam AI (bulbul:v2 model)
- **Voice**: Anushka (te-IN)
- **Features**: Streaming TTS, caching enabled
- **Sample Rate**: 22050 Hz

### 6. LiveKit Server (Ports 7880-7882)
- **Purpose**: WebRTC signaling server
- **Mode**: Development (--dev)
- **Features**: Real-time audio/video streaming

### 7. LiveKit Bridge
- **Purpose**: Connects LiveKit to STT and Orchestrator
- **WebSocket Connections**:
  - STT: ws://tara-task-stt-vad:8001/api/v1/transcribe/stream
  - Orchestrator: ws://tara-task-orchestrator:8004/orchestrate

## GPU Configuration

### RAG Service GPU Support
The RAG service is designed to leverage GPU acceleration for:
- **Embeddings Generation**: Sentence Transformers can use CUDA if available
- **PyTorch Operations**: Automatic GPU detection and utilization
- **Batch Processing**: Optimized for embedding batch size of 32

**To enable GPU acceleration:**
1. Ensure NVIDIA Docker runtime is installed
2. Add GPU resources to RAG service in docker-compose.yml:
```yaml
rag:
  # ... existing config ...
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **VRAM**: Minimum 4GB for embeddings processing
- **CUDA**: Version 11.8+ compatible with PyTorch 2.0+
- **Docker**: NVIDIA Container Toolkit installed

## Quick Start

### Prerequisites
```bash
# Install NVIDIA Docker support (if using GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -nvidia-docker2
sudo systemctl restart docker

# Set Docker context
docker context use desktop-linux
```

### Start the Stack
```bash
cd /home/prometheus/leibniz_agent/TARA-MICROSERVICE

# Option 1: Use the convenience script
./start-tara-task.sh

# Option 2: Manual start
docker-compose -f docker-compose-tara-task.yml up -d

# Option 3: Start orchestrator only (auto-starts others)
docker-compose -f docker-compose-tara-task.yml up -d orchestrator
```

### Verify Startup
```bash
# Check container status
docker ps --format "table {{.Names}}\t{{.Ports}}"

# Check service health
curl http://localhost:2004/health
curl http://localhost:2003/health
curl http://localhost:2001/health
curl http://localhost:2005/health
```

## Environment Variables

### Required API Keys
```bash
# Gemini API (for STT and RAG)
export GEMINI_API_KEY="your-gemini-api-key"

# Sarvam API (for TTS)
export SARVAM_API_KEY="your-sarvam-api-key"
export SARVAM_API_SUBSCRIPTION_KEY="your-subscription-key"

# LiveKit (optional, defaults provided)
export LIVEKIT_API_KEY="your-livekit-key"
export LIVEKIT_API_SECRET="your-livekit-secret"
```

### Service Configuration
- **Redis**: Host tara-task-redis, Port 6379, DB 0
- **Response Language**: te-mixed (Telugu mixed script)
- **Organization**: TASK
- **Agent Name**: TARA
- **Session TTL**: 3600 seconds
- **Max Concurrent Sessions**: 1000

## Monitoring and Logs

### View Logs
```bash
# All services
docker-compose -f docker-compose-tara-task.yml logs -f

# Specific service
docker-compose -f docker-compose-tara-task.yml logs -f rag
docker-compose -f docker-compose-tara-task.yml logs -f orchestrator
```

### GPU Monitoring
```bash
# GPU usage
nvidia-smi

# PyTorch GPU detection (inside RAG container)
docker exec tara-task-rag python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Performance Metrics
- **RAG Service**: Embedding batch size 32, prewarming enabled
- **STT Service**: Real-time processing with barge-in detection
- **TTS Service**: Streaming with 22kHz sample rate
- **Orchestrator**: Handles up to 1000 concurrent sessions

## Development and Testing

### Test Scripts
```bash
# Health check all services
python test_service_health.py

# Test streaming flow
python test_streaming_flow.py

# Test RAG functionality
python test_streaming_integration.py

# Test TTS performance
python test_ultra_low_latency.py
```

### Build Individual Services
```bash
# Build all services
./build-tara-task.sh

# Build specific service
docker build -t tara-rag -f rag/Dockerfile.tara .
```

### Debug Mode
```bash
# Start with debug logging
LOG_LEVEL=DEBUG docker-compose -f docker-compose-tara-task.yml up -d
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   - Ensure NVIDIA drivers are installed
   - Check CUDA compatibility
   - Verify nvidia-docker runtime

2. **Service Health Checks Failing**
   - Check logs: `docker-compose logs <service>`
   - Verify API keys are set
   - Ensure network connectivity between services

3. **High Latency**
   - Monitor GPU usage with nvidia-smi
   - Check Redis connection
   - Verify embedding batch size settings

4. **Memory Issues**
   - RAG service may need more VRAM for large knowledge bases
   - Consider increasing embedding_batch_size or using CPU-only FAISS

### Reset and Cleanup
```bash
# Stop all services
docker-compose -f docker-compose-tara-task.yml down

# Remove volumes (WARNING: deletes data)
docker-compose -f docker-compose-tara-task.yml down -v

# Clean up unused images
docker system prune -f
```

## Deployment Notes

- **Production**: Set production API keys and disable dev mode
- **Scaling**: Increase Redis memory limits for high concurrency
- **Backup**: Regularly backup tara-task-rag-index volume
- **Updates**: Rebuild services when updating requirements.txt

## API Endpoints

- **Orchestrator**: http://localhost:2004
  - `/health` - Health check
  - `/orchestrate` - WebSocket orchestration

- **RAG**: http://localhost:2003
  - `/health` - Health check
  - `/query` - Knowledge base queries

- **STT**: http://localhost:2001
  - `/health` - Health check
  - `/api/v1/transcribe/stream` - WebSocket transcription

- **TTS**: http://localhost:2005
  - `/health` - Health check
  - `/tts` - Text-to-speech synthesis

## File Structure
```
TARA-MICROSERVICE/
├── docker-compose-tara-task.yml    # Main compose file
├── orchestrator/                   # Master controller
├── rag/                           # GPU-accelerated RAG service
│   ├── Dockerfile.tara            # GPU-enabled build
│   └── requirements.txt           # PyTorch, transformers
├── stt_vad/                       # Speech-to-text
├── tts_sarvam/                    # Text-to-speech
├── livekit_bridge/                # WebRTC bridge
├── task_knowledge_base/           # TASK-specific knowledge
├── start-tara-task.sh            # Quick start script
└── test_*.py                      # Testing scripts
```

This README ensures you have all the details to operate and maintain the TARA GPU container stack effectively.</content>
<parameter name="filePath">/home/prometheus/leibniz_agent/TARA-MICROSERVICE/GPU_CONTAINER_README.md