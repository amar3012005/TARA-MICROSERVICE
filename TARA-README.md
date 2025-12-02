# TARA Microservices Cluster

This is a separate Docker Compose setup for the TARA microservices cluster, isolated from the main Leibniz services.

## Quick Start

### 1. Build and Start All Services

```bash
cd /home/prometheus/leibniz_agent/services
docker context use desktop-linux
docker-compose -p tara-microservice -f docker-compose-tara.yml up -d --build
```

### 2. Check Service Status

```bash
docker-compose -p tara-microservice -f docker-compose-tara.yml ps
```

### 3. View Logs

```bash
# All services
docker-compose -p tara-microservice -f docker-compose-tara.yml logs -f

# Specific service
docker-compose -p tara-microservice -f docker-compose-tara.yml logs -f orchestrator-tara
```

### 4. Access the Unified Client

Open your browser and navigate to:
```
http://localhost:8023
```

Or directly:
```
http://localhost:8023/static/client.html
```

## Service Ports (TARA Cluster)

| Service | Container Name | External Port | Internal Port |
|---------|---------------|---------------|---------------|
| Redis | tara-redis | 6382 | 6379 |
| STT-VAD | tara-stt-vad-service | 8020 | 8001 |
| Intent | tara-intent-service | 8021 | 8002 |
| RAG | tara-rag-service | 8022 | 8003 |
| Orchestrator | tara-orchestrator-service | 8023 | 8004 |
| TTS Streaming | tara-tts-streaming-service | 8024 | 8005 |
| FastRTC (STT) | tara-stt-vad-service | 7870 | 7860 |

## Network

All services run on the `tara-network` Docker network, isolated from the main `leibniz-network`.

## Environment Variables

You can customize the intro greeting by setting:
```bash
export INTRO_GREETING="Your custom greeting here"
docker-compose -p tara-microservice -f docker-compose-tara.yml up -d
```

## Stopping Services

```bash
docker-compose -p tara-microservice -f docker-compose-tara.yml down
```

## Removing Everything (Including Volumes)

```bash
docker-compose -p tara-microservice -f docker-compose-tara.yml down -v
```

## Architecture Flow

1. **User connects** → Orchestrator plays intro greeting via TTS
2. **User speaks** → STT-VAD processes audio → Publishes to Redis (`leibniz:events:stt`)
3. **Orchestrator receives event** → Triggers Intent+RAG pipeline
4. **Response generated** → Orchestrator streams TTS audio to client
5. **Client plays audio** → Barge-in detection stops audio if user speaks

## Troubleshooting

### Check if services are running:
```bash
docker ps | grep tara
```

### Check Redis connectivity:
```bash
docker exec tara-redis redis-cli ping
```

### Check service health:
```bash
curl http://localhost:8023/health
curl http://localhost:8020/health
curl http://localhost:8024/health
```

### View specific service logs:
```bash
docker logs tara-orchestrator-service -f
docker logs tara-stt-vad-service -f
docker logs tara-tts-streaming-service -f
```




