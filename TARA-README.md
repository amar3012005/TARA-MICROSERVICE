# 🇮🇳 TARA - Telugu TASK Customer Service Agent

**TARA (Telangana Automated Response Assistant)** is a Telugu-speaking AI customer service agent for the TASK (Telangana Academy for Skill and Knowledge) organization.

## Overview

```
User (Telugu) → STT_VAD (Gemini Live) → RAG (TASK KB) → TTS_Sarvam (Telugu)
                                         ↑
                               Skip Intent Service
```

### Key Features
- **Telugu Language Support**: Native Telugu speech recognition and synthesis
- **Mixed Telugu-English Responses**: Tenglish style for natural customer service
- **TASK Knowledge Base**: Dedicated to TASK organization information
- **No Barge-in Interference**: Ignores STT while agent is speaking

## Quick Start

### 1. Start TARA Services

```bash
cd TARA-MICROSERVICE
docker-compose -f docker-compose-tara.yml up -d
```

### 2. Check Service Health

```bash
# Check all services
docker-compose -f docker-compose-tara.yml ps

# Check orchestrator health
curl http://localhost:8023/health

# Check RAG service
curl http://localhost:8022/health
```

### 3. Access Services

| Service | Port | URL |
|---------|------|-----|
| Orchestrator | 8023 | http://localhost:8023 |
| STT/VAD FastRTC | 7870 | http://localhost:7870 |
| RAG Service | 8022 | http://localhost:8022 |
| TTS Sarvam | 8024 | http://localhost:8024 |
| Redis | 6382 | localhost:6382 |

## Architecture

### Service Flow

1. **User speaks in Telugu** → Browser microphone captures audio
2. **STT_VAD Service** → Gemini Live transcribes Telugu speech
3. **Orchestrator** → Routes transcript directly to RAG (skips Intent)
4. **RAG Service** → Retrieves from TASK KB, generates Telugu response via Gemini
5. **TTS_Sarvam** → Synthesizes Telugu audio using Sarvam API
6. **User hears Telugu response** → Audio plays in browser

### Key Configuration

```yaml
# TARA Mode Settings
TARA_MODE: true
SKIP_INTENT_SERVICE: true
RESPONSE_LANGUAGE: te-mixed
ORGANIZATION_NAME: TASK
IGNORE_STT_WHILE_SPEAKING: true

# Telugu TTS (Sarvam)
LEIBNIZ_SARVAM_SPEAKER: anushka
LEIBNIZ_SARVAM_LANGUAGE: te-IN
LEIBNIZ_SARVAM_MODEL: bulbul:v2
```

## Telugu Intro Greeting

When TARA starts, it greets users with:

```
నమస్కారం అండి! నేను TARA, TASK యొక్క కస్టమర్ సర్వీస్ ఏజెంట్. మీకు ఎలా సహాయం చేయగలను?
```

Translation: "Hello! I am TARA, TASK's customer service agent. How can I help you?"

## Knowledge Base

The TASK knowledge base is located at `./task_knowledge_base/` with:

```
task_knowledge_base/
├── services/          # TASK services information
├── contact/           # Contact details & locations
├── faq/               # Frequently asked questions
└── procedures/        # Registration & enrollment processes
```

### Adding Content
1. Add markdown files to appropriate folders
2. Rebuild the RAG index:
   ```bash
   curl -X POST http://localhost:8022/api/v1/admin/rebuild_index
   ```

## Environment Variables

Create a `.env` file:

```bash
# API Keys
GEMINI_API_KEY=your_gemini_api_key
SARVAM_API_KEY=your_sarvam_api_key

# Optional: Custom knowledge base path
TASK_KNOWLEDGE_BASE=./task_knowledge_base
```

## API Endpoints

### Orchestrator (Port 8023)

```bash
# Start workflow
POST /start

# Check status
GET /status

# Reset workflow
POST /reset

# Health check
GET /health
```

### RAG Service (Port 8022)

```bash
# Query knowledge base
POST /api/v1/query
{
  "query": "TASK లో ఎలా register చేయాలి?",
  "context": {"language": "te", "organization": "TASK"}
}

# Rebuild index
POST /api/v1/admin/rebuild_index
```

## Testing

### Test Telugu Query

```bash
curl -X POST http://localhost:8022/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "TASK సేవలు ఏమిటి?"}'
```

### Expected Response (Mixed Telugu-English)

```
TASK అనేక skill development programs అందిస్తుంది. IT training, communication skills, 
certifications అన్నీ available. Registration free గా చేసుకోవచ్చు. 
More details కోసం 040-2345-6789 కు call చేయండి.
```

## Troubleshooting

### Services Not Starting
```bash
# Check logs
docker-compose -f docker-compose-tara.yml logs orchestrator-tara
docker-compose -f docker-compose-tara.yml logs rag-service-tara

# Restart services
docker-compose -f docker-compose-tara.yml restart
```

### STT Not Working
- Ensure Gemini API key is valid
- Check browser microphone permissions
- Verify STT service is running on port 7870

### TTS Not Speaking Telugu
- Verify Sarvam API key is valid
- Check TTS service logs for errors
- Ensure `LEIBNIZ_SARVAM_LANGUAGE=te-IN`

### RAG Not Finding Answers
- Verify knowledge base files exist
- Rebuild index after adding content
- Check similarity threshold settings

## Stopping Services

```bash
docker-compose -f docker-compose-tara.yml down

# Remove volumes (clears data)
docker-compose -f docker-compose-tara.yml down -v
```

## Support

For issues or questions:
- Check service logs: `docker-compose logs <service-name>`
- Verify environment variables
- Ensure all API keys are valid

---

**TARA** - Bringing Telugu customer service to TASK organization 🇮🇳
