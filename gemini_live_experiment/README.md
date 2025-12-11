# Gemini Live RAG Experiment

Experimental microservice that combines Gemini Live's bidirectional conversational capabilities with RAG (Retrieval-Augmented Generation) for domain-specific knowledge.

## Architecture

This service extends the `stt_vad` service to:

1. **Receive user audio** via FastRTC (browser microphone)
2. **Send audio to Gemini Live** configured for:
   - Audio output (TTS)
   - Tool use (function calling)
   - Input transcription (STT)
3. **Send partial transcripts to RAG** incrementally to warm up context
4. **Handle tool calls** - When Gemini calls `query_knowledge_base`, fetch answer from RAG service
5. **Stream audio responses** - Gemini's TTS audio is sent back to the browser via FastRTC

## Flow

```
Browser (FastRTC) 
  → Audio Input 
  → Gemini Live (STT + Tool Use) 
  → Partial Transcripts → RAG (incremental)
  → Tool Call → RAG (final query)
  → Tool Response → Gemini Live
  → Audio Output (TTS) 
  → FastRTC → Browser
```

## Running the Experiment

### Prerequisites

- Docker and Docker Compose
- Access to Gemini API (GEMINI_API_KEY)
- RAG service running (or use the included docker-compose)

### Quick Start

```bash
# Build and start the experiment service
docker-compose -f docker-compose.experiment.yml up --build

# Access the FastRTC UI
# Open browser to: http://localhost:7862
```

### Environment Variables

- `GEMINI_API_KEY`: Your Gemini API key
- `RAG_SERVICE_URL`: RAG service URL (default: `http://tara-task-rag:8003`)
- `TARA_REDIS_HOST`: Redis host (default: `tara-task-redis`)
- `TARA_REDIS_PORT`: Redis port (default: `6381`)

## Features

- **Ultra-low latency** bidirectional conversation
- **Domain-specific knowledge** via RAG integration
- **Incremental RAG** - partial transcripts warm up context
- **Tool-based RAG** - Gemini decides when to query knowledge base
- **Audio streaming** - Real-time TTS output

## Differences from stt_vad

1. **Response modalities**: `["AUDIO", "TEXT"]` instead of `["TEXT"]`
2. **Tool configuration**: Added `query_knowledge_base` function
3. **RAG client**: Integrated for incremental and tool-based queries
4. **Audio output**: FastRTC handler queues and emits audio from Gemini
5. **System instruction**: Updated for conversational assistant role

## Testing

1. Start the service: `docker-compose -f docker-compose.experiment.yml up`
2. Open browser to `http://localhost:7862`
3. Click "Start" and speak into microphone
4. Ask questions about T.A.S.K (the knowledge base domain)
5. Listen for audio responses from Gemini

## Notes

- This is an **experimental** service - not production-ready
- Tool response format may need adjustment based on Gemini Live API version
- Audio format conversion assumes 16kHz mono PCM
- RAG service must be running and accessible



