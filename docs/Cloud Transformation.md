# Leibniz Agent - Cloud Transformation & Microservices Architecture

## 📋 Executive Summary

**Transformation Goal**: Migrate monolithic Leibniz Agent to containerized microservices for cloud deployment with independent scaling, fault isolation, and distributed caching.

**Architecture**: 6 microservices + 3 infrastructure services (Redis, MinIO, MongoDB)

**Timeline**: 4-6 weeks (code refactoring → containerization → K8s deployment)

**Expected Performance**: 20x concurrent user capacity, 10x cache speed, 5-10x faster deployments

---

## 🎯 Current Architecture Analysis (Monolithic)


### Monolithic Components

**Primary Files**:
- `leibniz_pro.py` - Main orchestrator (1000+ lines)
- `leibniz_persistent_services.py` - Singleton service manager
- `leibniz_vad.py` - Bidirectional VAD with Gemini Live
- `leibniz_stt.py` - Speech-to-text wrapper
- `leibniz_intent_parser.py` - Intent classification
- `leibniz_rag.py` - FAISS-based knowledge retrieval
- `leibniz_tts.py` - Multi-provider TTS (ElevenLabs/Google/Gemini)
- `leibniz_appointment_fsm.py` - Appointment state machine

**Shared State Problems**:
1. **Global Singleton**: `LeibnizPersistentServices` in `leibniz_persistent_services.py`
2. **File-based Cache**: `rag_cache/` directory for query/intent caching (1-5ms latency)
3. **Local Audio Storage**: `audio_archive/` for TTS cache (limited disk space)
4. **In-memory Sessions**: Conversation state in `leibniz_pro.py` (lost on restart)
5. **Gemini Live Session**: Persistent session in `LeibnizBidirectionalVAD` (not shared across instances)

**Data Flow (Monolithic)**:
```python
# Current synchronous pipeline in leibniz_pro.py
User Speech → leibniz_vad.capture_speech_bidirectional()
    ↓ Returns: (transcript: str, confidence: float)
    
Intent Parser → classify_intent(text, context_dict)
    ↓ Returns: {
        'intent': str,              # e.g., 'APPOINTMENT_SCHEDULING'
        'confidence': float,        # 0.0-1.0
        'user_context': str,        # Enriched query
        'entities': dict,           # Extracted data
        'extracted_meaning': str    # Semantic interpretation
    }
    
RAG System → process_rag_query(query, context)
    ↓ Returns: {
        'answer': str,
        'sources': list,
        'timing_breakdown': dict,
        'confidence': float
    }
    
TTS → synthesize_to_file(text, emotion, cache_name)
    ↓ Returns: (audio_path: str, duration_ms: int)
    
Audio Playback → background_player.play_audio(audio_path)
```

**Dependencies**:
- `google-genai>=1.33.0` - Gemini Live API (VAD + STT + LLM)
- `elevenlabs>=0.2.0` - Premium TTS provider
- `faiss-cpu>=1.7.0` - Vector similarity search
- `sentence-transformers>=2.2.0` - Embedding model
- `fastapi>=0.100.0` - HTTP server (not currently used)
- `redis>=4.5.0` - NOT CURRENTLY USED (will add)
- `minio>=7.1.0` - NOT CURRENTLY USED (will add)

---

## 🏗️ Target Microservices Architecture

### Service Overview

| Service | Port | Responsibility | Scaling Strategy |
|---------|------|----------------|------------------|
| **STT/VAD Service** | 8001 | Real-time speech capture + transcription | Horizontal (WebSocket connections) |
| **Intent Service** | 8002 | Intent classification + entity extraction | Horizontal (stateless) |
| **RAG Service** | 8003 | Knowledge base retrieval + answer generation | Horizontal (read-only FAISS) |
| **TTS Service** | 8004 | Multi-provider TTS with caching | Horizontal (stateless) |
| **Appointment FSM** | 8005 | Appointment slot filling state machine | Vertical (stateful sessions) |
| **Orchestrator** | 8000 | API gateway + conversation flow | Horizontal (session in Redis) |
| **Redis** | 6379 | Distributed cache + session store | Vertical (master-replica) |
| **MinIO** | 9000 | Audio file storage (TTS cache) | Horizontal (distributed storage) |
| **MongoDB** | 27017 | Appointment data persistence | Vertical (replica set) |

**Inter-Service Communication**:
```
Client (WebSocket) → Orchestrator (8000)
    ↓ WebSocket → STT Service (8001)
    ↓ HTTP POST → Intent Service (8002)
    ↓ HTTP POST → RAG Service (8003) OR Appointment FSM (8005)
    ↓ HTTP POST → TTS Service (8004)
    ↓ WebSocket → Client (audio stream)
```

**Data Flow (Microservices)**:
```json
// 1. WebSocket connection to Orchestrator
WS /api/v1/conversation?session_id=abc123

// 2. Orchestrator proxies audio to STT Service
POST http://stt-service:8001/api/v1/transcribe/stream
Response: {
    "transcript": "I need to schedule an appointment",
    "confidence": 0.95,
    "language": "en-US",
    "duration_ms": 3200
}

// 3. Orchestrator calls Intent Service
POST http://intent-service:8002/api/v1/classify
Body: {
    "text": "I need to schedule an appointment",
    "context": {"previous_intent": null, "session_id": "abc123"}
}
Response: {
    "intent": "APPOINTMENT_SCHEDULING",
    "confidence": 0.98,
    "user_context": "User wants to book a meeting with advisor",
    "entities": {"action": "schedule", "service_type": "appointment"},
    "extracted_meaning": "Schedule appointment request"
}

// 4a. If intent == RAG_QUERY → RAG Service
POST http://rag-service:8003/api/v1/query
Body: {
    "query": "What are your office hours?",
    "context": {"session_id": "abc123"}
}
Response: {
    "answer": "Office hours are Mon-Fri 9 AM to 5 PM.",
    "sources": ["office_hours.md", "contact_info.md"],
    "confidence": 0.92,
    "timing_breakdown": {"retrieval_ms": 45, "generation_ms": 320}
}

// 4b. If intent == APPOINTMENT_SCHEDULING → Appointment FSM
POST http://appointment-service:8005/api/v1/session/create
Body: {"session_id": "abc123"}
Response: {
    "fsm_state": "INITIAL_GREETING",
    "next_prompt": "When would you like to schedule?",
    "required_slots": ["preferred_date", "preferred_time", "purpose"]
}

// 5. Orchestrator calls TTS Service
POST http://tts-service:8004/api/v1/synthesize
Body: {
    "text": "Office hours are Mon-Fri 9 AM to 5 PM.",
    "emotion": "neutral",
    "cache_key": "office_hours_query_v1"
}
Response: {
    "audio_url": "http://minio:9000/leibniz-audio/abc123_1234567890.mp3",
    "duration_ms": 4200,
    "provider": "elevenlabs",
    "cache_hit": true
}

// 6. Orchestrator streams audio back to client via WebSocket
WS → Binary audio chunks (50ms each)
```

---

## 📦 Service 1: STT/VAD Service (Gemini Live)

### Specifications


**Port**: 8001  
**Protocol**: WebSocket (real-time streaming) + HTTP (health/metrics)  
**Extracted Components**:
- `leibniz_vad.py` → `services/stt_vad/vad_manager.py`
- `leibniz_stt.py` → `services/stt_vad/stt_client.py`
- `sindh_bidirectional_vad.py` → Reference patterns for session management

**Key Classes**:
```python
# services/stt_vad/vad_manager.py
class LeibnizVADManager:
    """Manages Gemini Live persistent session and audio streaming"""
    def __init__(self, config: VADConfig, redis_client: Redis):
        self.session = None  # Gemini Live session
        self.redis = redis_client  # Session state storage
        self.is_listening = False
        self._capture_lock = asyncio.Lock()
    
    async def capture_speech_streaming(
        self, 
        session_id: str,
        streaming_callback: Callable[[str, bool], None]
    ) -> Dict[str, Any]:
        """
        Capture speech with real-time fragment callbacks
        Returns: {
            'transcript': str,
            'confidence': float,
            'duration_ms': int,
            'language': str
        }
        """
        pass

# services/stt_vad/app.py
class STTVADService:
    """FastAPI service for STT/VAD"""
    def __init__(self):
        self.vad_manager = LeibnizVADManager(...)
        self.app = FastAPI()
        self._register_routes()
    
    async def websocket_transcribe(self, websocket: WebSocket):
        """WebSocket endpoint for streaming transcription"""
        pass
```

**API Endpoints**:
```python
# WebSocket endpoint for real-time transcription
WS /api/v1/transcribe/stream?session_id=<uuid>
    Input: Binary PCM audio chunks (16kHz, mono, 50ms)
    Output: JSON fragments {
        "transcript": str,
        "is_final": bool,
        "confidence": float,
        "timestamp_ms": int
    }

# HTTP endpoint for health checks
GET /health
    Response: {
        "status": "healthy",
        "gemini_session": "active",
        "uptime_seconds": 3600
    }

# HTTP endpoint for metrics
GET /metrics
    Response: Prometheus format
```

**Module Structure**:
```
services/
└── stt_vad/
    ├── app.py                  # FastAPI application entrypoint
    ├── vad_manager.py          # LeibnizVADManager class
    ├── stt_client.py           # Gemini Live client wrapper
    ├── config.py               # VADConfig dataclass
    ├── requirements.txt        # google-genai, sounddevice, fastapi
    ├── Dockerfile              # Multi-stage build
    └── tests/
        ├── test_vad_manager.py
        └── test_streaming.py
```

**Dockerfile** (`services/stt_vad/Dockerfile`):
```dockerfile
# Multi-stage build for STT/VAD service
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim AS runtime

# Install audio dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV GEMINI_API_KEY=""
ENV REDIS_URL="redis://redis:6379/0"

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health').raise_for_status()"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
```

**Environment Variables**:
```bash
GEMINI_API_KEY=<your_gemini_api_key>
REDIS_URL=redis://redis:6379/0
VAD_MODEL_NAME=gemini-2.0-flash-exp
VAD_TIMEOUT_INITIAL=10.0
VAD_TIMEOUT_GREETING=12.0
VAD_TIMEOUT_DECISION=15.0
VAD_LANGUAGE=en-US
LOG_LEVEL=INFO
```

**Docker Run Command**:
```powershell
docker build -t leibniz-stt-vad:latest -f services/stt_vad/Dockerfile services/stt_vad

docker run -d `
  --name leibniz-stt-vad `
  -p 8001:8001 `
  -e GEMINI_API_KEY=$env:GEMINI_API_KEY `
  -e REDIS_URL=redis://host.docker.internal:6379/0 `
  --restart unless-stopped `
  leibniz-stt-vad:latest
```

---

## 📦 Service 2: Intent Classification Service

### Specifications


**Port**: 8002  
**Protocol**: HTTP REST  
**Extracted Components**:
- `leibniz_intent_parser.py` → `services/intent/intent_classifier.py`
- `leibniz_persistent_services.py` (intent portion) → `services/intent/intent_cache.py`

**Key Classes**:
```python
# services/intent/intent_classifier.py
class LeibnizIntentClassifier:
    """Fast pattern-based + Gemini fallback intent classification"""
    
    def __init__(self, gemini_client, redis_client: Redis):
        self.gemini_client = gemini_client
        self.redis = redis_client
        self.pattern_rules = self._load_pattern_rules()
    
    async def classify_intent(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify user intent with semantic enrichment
        Returns: {
            'intent': str,              # APPOINTMENT_SCHEDULING | RAG_QUERY | GREETING | etc.
            'confidence': float,        # 0.0-1.0
            'user_context': str,        # Enriched query for RAG
            'entities': dict,           # Extracted key entities
            'extracted_meaning': str,   # Semantic interpretation
            'method': str,              # 'pattern' or 'gemini_fallback'
            'response_time_ms': float
        }
        """
        # 1. Check Redis cache
        cache_key = f"intent:{hash(text)}"
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # 2. Try fast pattern matching
        pattern_result = self._match_patterns(text)
        if pattern_result and pattern_result['confidence'] > 0.85:
            await self.redis.setex(cache_key, 1800, json.dumps(pattern_result))
            return pattern_result
        
        # 3. Fallback to Gemini classification
        gemini_result = await self._classify_with_gemini(text, context)
        await self.redis.setex(cache_key, 1800, json.dumps(gemini_result))
        return gemini_result

# services/intent/app.py
class IntentService:
    """FastAPI service for intent classification"""
    def __init__(self):
        self.classifier = LeibnizIntentClassifier(...)
        self.app = FastAPI()
        self._register_routes()
```

**API Endpoints**:
```python
# POST endpoint for intent classification
POST /api/v1/classify
    Body: {
        "text": str,                          # User transcript
        "context": {                          # Optional context
            "previous_intent": str,
            "session_id": str,
            "conversation_turn": int
        }
    }
    Response: {
        "intent": "APPOINTMENT_SCHEDULING",
        "confidence": 0.95,
        "user_context": "User wants to book advisor meeting",
        "entities": {
            "action": "schedule",
            "service_type": "appointment"
        },
        "extracted_meaning": "Schedule appointment request",
        "method": "pattern",
        "response_time_ms": 12.5
    }

# GET endpoint for health checks
GET /health
    Response: {"status": "healthy", "cache_hit_rate": 0.78}
```

**Module Structure**:
```
services/
└── intent/
    ├── app.py                      # FastAPI application
    ├── intent_classifier.py        # LeibnizIntentClassifier
    ├── intent_cache.py             # Redis cache wrapper
    ├── pattern_rules.json          # Pattern matching rules
    ├── config.py                   # IntentConfig dataclass
    ├── requirements.txt            # google-genai, redis, fastapi
    ├── Dockerfile
    └── tests/
        └── test_classifier.py
```

**Dockerfile** (`services/intent/Dockerfile`):
```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV GEMINI_API_KEY=""
ENV REDIS_URL="redis://redis:6379/0"

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8002/health').raise_for_status()"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "4"]
```

**Docker Run Command**:
```powershell
docker build -t leibniz-intent:latest -f services/intent/Dockerfile services/intent

docker run -d `
  --name leibniz-intent `
  -p 8002:8002 `
  -e GEMINI_API_KEY=$env:GEMINI_API_KEY `
  -e REDIS_URL=redis://host.docker.internal:6379/0 `
  --restart unless-stopped `
  leibniz-intent:latest
```

---

## 📦 Service 3: RAG Service (Knowledge Base)

### Specifications


**Port**: 8003  
**Protocol**: HTTP REST  
**Extracted Components**:
- `leibniz_rag.py` → `services/rag/rag_engine.py`
- `rag_cache.py` → Migrate to Redis (remove file-based cache)
- `rag_speculative.py` → Optional (disable for simplicity)
- `leibniz_knowledge_base/` → Mount as Docker volume

**Critical RAG Specifications**:
```python
# Current configuration (leibniz_rag.py)
VECTOR_STORE: "FAISS (CPU-based, faiss-cpu>=1.7.0)"
EMBEDDING_MODEL: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE: 800 characters
CHUNK_OVERLAP: 150 characters
RETRIEVAL_TOP_K: 5 documents
GENERATION_MODEL: "gemini-2.0-flash-exp"
CACHE_TTL: 3600 seconds (1 hour)
INDEX_PATH: "leibniz_knowledge_base/leibniz_faiss.index"

# System prompt for answer generation (leibniz_rag.py lines 150-180)
SYSTEM_PROMPT = """You are a helpful assistant for Leibniz University Institute.
Use the provided context to answer questions accurately and concisely.
If the context doesn't contain relevant information, say so politely.
Keep answers friendly, professional, and academically appropriate.
Always cite sources when providing specific information."""
```

**Key Classes**:
```python
# services/rag/rag_engine.py
class LeibnizRAGEngine:
    """FAISS-based RAG with Redis caching and async retrieval"""
    
    def __init__(
        self,
        faiss_index_path: str,
        embedding_model_name: str,
        gemini_client,
        redis_client: Redis
    ):
        self.index = faiss.read_index(faiss_index_path)  # Load at startup
        self.embedder = SentenceTransformer(embedding_model_name)
        self.gemini_client = gemini_client
        self.redis = redis_client
        self.system_prompt = SYSTEM_PROMPT
    
    async def process_rag_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve + generate answer with caching
        Returns: {
            'answer': str,
            'sources': list[str],       # Source document names
            'confidence': float,
            'timing_breakdown': {
                'cache_check_ms': float,
                'embedding_ms': float,
                'retrieval_ms': float,
                'generation_ms': float,
                'total_ms': float
            }
        }
        """
        # 1. Check Redis cache (key: hash of query)
        cache_key = f"rag:{hashlib.md5(query.encode()).hexdigest()}"
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # 2. Embed query
        query_embedding = self.embedder.encode([query])[0]
        
        # 3. FAISS retrieval (top-k=5)
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k=5
        )
        
        # 4. Load retrieved documents
        retrieved_docs = self._load_documents(indices[0])
        
        # 5. Generate answer with Gemini
        answer = await self._generate_answer(query, retrieved_docs)
        
        # 6. Cache result (1 hour TTL)
        result = {
            'answer': answer,
            'sources': [doc['source'] for doc in retrieved_docs],
            'confidence': self._calculate_confidence(distances[0]),
            'timing_breakdown': {...}
        }
        await self.redis.setex(cache_key, 3600, json.dumps(result))
        
        return result

# services/rag/app.py
class RAGService:
    """FastAPI service for knowledge base queries"""
    def __init__(self):
        self.rag_engine = LeibnizRAGEngine(...)
        self.app = FastAPI()
```

**API Endpoints**:
```python
# POST endpoint for RAG queries
POST /api/v1/query
    Body: {
        "query": str,                   # User question
        "context": {                    # Optional enriched context
            "user_context": str,        # From intent service
            "session_id": str
        }
    }
    Response: {
        "answer": "Office hours are Monday to Friday, 9 AM to 5 PM.",
        "sources": ["office_hours.md", "contact_info.md"],
        "confidence": 0.92,
        "timing_breakdown": {
            "cache_check_ms": 0.8,
            "embedding_ms": 45.2,
            "retrieval_ms": 12.5,
            "generation_ms": 320.8,
            "total_ms": 379.3
        }
    }

# GET endpoint for health checks
GET /health
    Response: {
        "status": "healthy",
        "index_loaded": true,
        "index_size": 1234,
        "cache_hit_rate": 0.65
    }

# POST endpoint for index rebuild (admin only)
POST /api/v1/admin/rebuild_index
    Body: {"knowledge_base_path": "/app/knowledge_base"}
    Response: {"status": "success", "documents_indexed": 45}
```

**Module Structure**:
```
services/
└── rag/
    ├── app.py                          # FastAPI application
    ├── rag_engine.py                   # LeibnizRAGEngine
    ├── document_loader.py              # Markdown → chunks
    ├── index_builder.py                # Build FAISS index
    ├── config.py                       # RAGConfig
    ├── requirements.txt                # faiss-cpu, sentence-transformers
    ├── Dockerfile
    ├── knowledge_base/                 # VOLUME MOUNT (read-only)
    │   ├── office_hours.md
    │   ├── courses.md
    │   └── ... (all KB docs)
    └── tests/
        └── test_rag_engine.py
```

**Dockerfile** (`services/rag/Dockerfile`):
```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

# Pre-build FAISS index at image build time
COPY knowledge_base /app/knowledge_base
RUN python index_builder.py --input /app/knowledge_base --output /app/leibniz_faiss.index

ENV PATH=/root/.local/bin:$PATH
ENV GEMINI_API_KEY=""
ENV REDIS_URL="redis://redis:6379/0"
ENV FAISS_INDEX_PATH="/app/leibniz_faiss.index"

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8003/health').raise_for_status()"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "2"]
```

**Docker Run Command**:
```powershell
docker build -t leibniz-rag:latest -f services/rag/Dockerfile services/rag

docker run -d `
  --name leibniz-rag `
  -p 8003:8003 `
  -e GEMINI_API_KEY=$env:GEMINI_API_KEY `
  -e REDIS_URL=redis://host.docker.internal:6379/0 `
  -v ${PWD}\leibniz_agent\leibniz_knowledge_base:/app/knowledge_base:ro `
  --restart unless-stopped `
  leibniz-rag:latest
```

---

## 📦 Service 4: TTS Service (Multi-Provider)

### Specifications

**Port**: 8004  
**Protocol**: HTTP REST  
**Extracted Components**:
- `leibniz_tts.py` → `services/tts/tts_synthesizer.py`
- `audio_archive/` → Migrate to MinIO object storage

**Key Classes**:
```python
# services/tts/tts_synthesizer.py
class LeibnizTTSSynthesizer:
    """Multi-provider TTS with MinIO caching"""
    
    async def synthesize_to_url(
        self,
        text: str,
        emotion: str = "neutral",
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Returns: {'audio_url': str, 'duration_ms': int, 'provider': str, 'cache_hit': bool}"""
        pass

# services/tts/app.py
class TTSService:
    """FastAPI service for TTS synthesis"""
    pass
```

**API Endpoints**:
```python
POST /api/v1/synthesize
    Body: {"text": str, "emotion": str, "cache_key": str}
    Response: {"audio_url": str, "duration_ms": int, "provider": str, "cache_hit": bool}

GET /health
```

**Module Structure**:
```
services/tts/
├── app.py, tts_synthesizer.py, providers/, config.py
├── Dockerfile, requirements.txt, tests/
```

**Dockerfile**: Multi-stage with ffmpeg for audio processing

**Docker Run**:
```powershell
docker build -t leibniz-tts:latest -f services/tts/Dockerfile services/tts
docker run -d --name leibniz-tts -p 8004:8004 -e ELEVENLABS_API_KEY=$env:ELEVENLABS_API_KEY leibniz-tts:latest
```

---

## 📦 Service 5: Appointment FSM Service

### Specifications

**Port**: 8005  
**Protocol**: HTTP REST  
**Extracted Components**:
- `leibniz_appointment_fsm.py` → `services/appointment/fsm_manager.py`

**Key Classes**:
```python
# services/appointment/fsm_manager.py
class LeibnizAppointmentFSM:
    """Appointment slot-filling state machine with Redis persistence"""
    
    STATES = [
        "INITIAL_GREETING",
        "WAITING_FOR_DATE",
        "WAITING_FOR_TIME", 
        "WAITING_FOR_PURPOSE",
        "CONFIRMATION",
        "COMPLETED"
    ]
    
    def __init__(self, session_id: str, redis_client: Redis, mongodb_client=None):
        self.session_id = session_id
        self.redis = redis_client
        self.mongodb = mongodb_client
        self.state = "INITIAL_GREETING"
        self.slots = {
            "preferred_date": None,
            "preferred_time": None,
            "purpose": None,
            "user_name": None
        }
    
    async def process_user_input(
        self,
        user_input: str
    ) -> Dict[str, Any]:
        """
        Process user input and advance FSM state
        Returns: {
            'current_state': str,
            'next_prompt': str,
            'slots_filled': dict,
            'slots_remaining': list[str],
            'is_complete': bool
        }
        """
        # Extract entities from input
        entities = self._extract_entities(user_input)
        
        # Update slots
        for slot_name, value in entities.items():
            if slot_name in self.slots:
                self.slots[slot_name] = value
        
        # Advance state based on filled slots
        self._advance_state()
        
        # Persist to Redis
        await self._save_to_redis()
        
        # Generate next prompt
        next_prompt = self._generate_next_prompt()
        
        return {
            'current_state': self.state,
            'next_prompt': next_prompt,
            'slots_filled': {k: v for k, v in self.slots.items() if v is not None},
            'slots_remaining': [k for k, v in self.slots.items() if v is None],
            'is_complete': self.state == "COMPLETED"
        }
    
    async def _save_to_redis(self):
        """Persist FSM state to Redis with 30min TTL"""
        state_data = {
            'state': self.state,
            'slots': self.slots,
            'timestamp': time.time()
        }
        await self.redis.setex(
            f"appointment_fsm:{self.session_id}",
            1800,  # 30 minutes
            json.dumps(state_data)
        )
    
    @classmethod
    async def load_from_redis(
        cls,
        session_id: str,
        redis_client: Redis,
        mongodb_client=None
    ):
        """Load existing FSM session from Redis"""
        key = f"appointment_fsm:{session_id}"
        data = await redis_client.get(key)
        if not data:
            return cls(session_id, redis_client, mongodb_client)
        
        state_data = json.loads(data)
        instance = cls(session_id, redis_client, mongodb_client)
        instance.state = state_data['state']
        instance.slots = state_data['slots']
        return instance

# services/appointment/app.py
class AppointmentService:
    """FastAPI service for appointment scheduling"""
    
    def __init__(self):
        self.redis_client = Redis(...)
        self.mongodb_client = MongoClient(...)  # Optional
        self.app = FastAPI()
        self._register_routes()
    
    async def create_session(self, session_id: str):
        """Create new FSM session"""
        fsm = LeibnizAppointmentFSM(
            session_id, self.redis_client, self.mongodb_client
        )
        return {
            'session_id': session_id,
            'current_state': fsm.state,
            'next_prompt': "When would you like to schedule your appointment?"
        }
    
    async def process_turn(self, session_id: str, user_input: str):
        """Process user turn in existing session"""
        fsm = await LeibnizAppointmentFSM.load_from_redis(
            session_id, self.redis_client, self.mongodb_client
        )
        result = await fsm.process_user_input(user_input)
        
        # If completed, save to MongoDB
        if result['is_complete']:
            await self._save_appointment_to_db(session_id, fsm.slots)
        
        return result
```

**API Endpoints**:
```python
# POST endpoint to create FSM session
POST /api/v1/session/create
    Body: {"session_id": str}
    Response: {
        "session_id": "abc123",
        "current_state": "INITIAL_GREETING",
        "next_prompt": "When would you like to schedule your appointment?",
        "slots_filled": {},
        "slots_remaining": ["preferred_date", "preferred_time", "purpose", "user_name"]
    }

# POST endpoint to process user input
POST /api/v1/session/process
    Body: {
        "session_id": str,
        "user_input": str
    }
    Response: {
        "current_state": "WAITING_FOR_TIME",
        "next_prompt": "What time works best for you?",
        "slots_filled": {"preferred_date": "2025-11-05"},
        "slots_remaining": ["preferred_time", "purpose", "user_name"],
        "is_complete": false
    }

# GET endpoint to retrieve FSM state
GET /api/v1/session/{session_id}
    Response: {
        "session_id": "abc123",
        "current_state": "WAITING_FOR_TIME",
        "slots_filled": {"preferred_date": "2025-11-05"},
        "slots_remaining": ["preferred_time", "purpose", "user_name"]
    }

# DELETE endpoint to cancel session
DELETE /api/v1/session/{session_id}
    Response: {"status": "deleted"}

# GET endpoint for health checks
GET /health
    Response: {
        "status": "healthy",
        "active_sessions": 12,
        "mongodb_connected": true
    }
```

**Module Structure**:
```
services/
└── appointment/
    ├── app.py                      # FastAPI application
    ├── fsm_manager.py              # LeibnizAppointmentFSM
    ├── entity_extractor.py         # Extract date/time/purpose
    ├── config.py                   # AppointmentConfig
    ├── requirements.txt            # redis, motor (async MongoDB)
    ├── Dockerfile
    └── tests/
        └── test_fsm_manager.py
```

**Dockerfile** (`services/appointment/Dockerfile`):
```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV REDIS_URL="redis://redis:6379/0"
ENV MONGODB_URI="mongodb://mongodb:27017/leibniz"

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8005/health').raise_for_status()"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005", "--workers", "2"]
```

**Docker Run Command**:
```powershell
docker build -t leibniz-appointment:latest -f services/appointment/Dockerfile services/appointment

docker run -d `
  --name leibniz-appointment `
  -p 8005:8005 `
  -e REDIS_URL=redis://host.docker.internal:6379/0 `
  -e MONGODB_URI=mongodb://host.docker.internal:27017/leibniz `
  --restart unless-stopped `
  leibniz-appointment:latest
```


**Port**: 8000 (main entry point for clients)  
**Protocol**: WebSocket (conversation) + HTTP REST (management)  
**Extracted Components**:
- `leibniz_pro.py` → Refactored into `services/orchestrator/orchestrator.py`
- NEW: Service discovery and routing logic

**Key Classes**:
```python
# services/orchestrator/orchestrator.py
class LeibnizOrchestrator:
    """Main conversation flow coordinator"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.service_clients = {
            'stt': HTTPClient('http://stt-service:8001'),
            'intent': HTTPClient('http://intent-service:8002'),
            'rag': HTTPClient('http://rag-service:8003'),
            'tts': HTTPClient('http://tts-service:8004'),
            'appointment': HTTPClient('http://appointment-service:8005')
        }
        self.app = FastAPI()
        self._register_routes()
    
    async def handle_conversation_turn(
        self,
        websocket: WebSocket,
        session_id: str,
        audio_chunk: bytes
    ):
        """
        Orchestrate full conversation turn:
        1. STT (transcribe audio)
        2. Intent (classify + enrich)
        3. RAG or Appointment (generate response)
        4. TTS (synthesize audio)
        5. Stream audio back to client
        """
        # 1. Transcribe audio via STT service
        transcript_response = await self.service_clients['stt'].post(
            '/api/v1/transcribe/stream',
            data=audio_chunk
        )
        transcript = transcript_response['transcript']
        
        # 2. Classify intent
        intent_response = await self.service_clients['intent'].post(
            '/api/v1/classify',
            json={'text': transcript, 'context': {'session_id': session_id}}
        )
        
        # 3. Route based on intent
        if intent_response['intent'] == 'APPOINTMENT_SCHEDULING':
            # Use Appointment FSM
            fsm_response = await self.service_clients['appointment'].post(
                '/api/v1/session/process',
                json={'session_id': session_id, 'user_input': transcript}
            )
            answer_text = fsm_response['next_prompt']
        else:
            # Use RAG for knowledge queries
            rag_response = await self.service_clients['rag'].post(
                '/api/v1/query',
                json={'query': intent_response['user_context'], 'context': {...}}
            )
            answer_text = rag_response['answer']
        
        # 4. Synthesize TTS
        tts_response = await self.service_clients['tts'].post(
            '/api/v1/synthesize',
            json={'text': answer_text, 'emotion': 'neutral'}
        )
        
        # 5. Stream audio back to client
        audio_url = tts_response['audio_url']
        await self._stream_audio_to_websocket(websocket, audio_url)
        
        # 6. Save conversation turn to Redis
        await self._save_turn_to_session(session_id, transcript, answer_text)

# services/orchestrator/app.py
class OrchestratorApp:
    """FastAPI application for orchestrator"""
    pass
```

**API Endpoints**:
```python
# WebSocket endpoint for full conversation
WS /api/v1/conversation?session_id=<uuid>
    Input: Binary PCM audio chunks OR JSON control messages
    Output: Binary audio chunks + JSON metadata

# HTTP endpoint to create session
POST /api/v1/session/create
    Response: {"session_id": str, "created_at": int}

# HTTP endpoint to get session history
GET /api/v1/session/{session_id}/history
    Response: {
        "session_id": str,
        "turns": [
            {"user": "What are office hours?", "agent": "Mon-Fri 9 AM to 5 PM", "timestamp": 1234567890},
            ...
        ]
    }

# HTTP endpoint for health checks
GET /health
    Response: {
        "status": "healthy",
        "services": {
            "stt": "healthy",
            "intent": "healthy",
            "rag": "healthy",
            "tts": "healthy",
            "appointment": "healthy"
        }
    }
```

**Module Structure**:
```
services/
└── orchestrator/
    ├── app.py                      # FastAPI application
    ├── orchestrator.py             # LeibnizOrchestrator
    ├── service_clients.py          # HTTP client wrappers
    ├── session_manager.py          # Redis session persistence
    ├── config.py                   # OrchestratorConfig
    ├── requirements.txt            # fastapi, httpx, redis
    ├── Dockerfile
    └── tests/
        └── test_orchestrator.py
```

**Dockerfile** (`services/orchestrator/Dockerfile`):
```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV REDIS_URL="redis://redis:6379/0"
ENV STT_SERVICE_URL="http://stt-service:8001"
ENV INTENT_SERVICE_URL="http://intent-service:8002"
ENV RAG_SERVICE_URL="http://rag-service:8003"
ENV TTS_SERVICE_URL="http://tts-service:8004"
ENV APPOINTMENT_SERVICE_URL="http://appointment-service:8005"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Docker Run Command**:
```powershell
docker build -t leibniz-orchestrator:latest -f services/orchestrator/Dockerfile services/orchestrator

docker run -d `
  --name leibniz-orchestrator `
  -p 8000:8000 `
  -e REDIS_URL=redis://host.docker.internal:6379/0 `
  -e STT_SERVICE_URL=http://host.docker.internal:8001 `
  -e INTENT_SERVICE_URL=http://host.docker.internal:8002 `
  -e RAG_SERVICE_URL=http://host.docker.internal:8003 `
  -e TTS_SERVICE_URL=http://host.docker.internal:8004 `
  -e APPOINTMENT_SERVICE_URL=http://host.docker.internal:8005 `
  --restart unless-stopped `
  leibniz-orchestrator:latest
```

---

## 🐳 Complete Docker Compose Setup

**File**: `docker-compose.yml` (project root)

```yaml
version: '3.8'

services:
  # Infrastructure Services
  redis:
    image: redis:7-alpine
    container_name: leibniz-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    restart: unless-stopped

  minio:
    image: minio/minio:latest
    container_name: leibniz-minio
    ports:
      - "9000:9000"
      - "9001:9001"  # Console UI
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  mongodb:
    image: mongo:7
    container_name: leibniz-mongodb
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_DATABASE: leibniz
    volumes:
      - mongodb_data:/data/db
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  # Microservices
  stt-service:
    build:
      context: ./services/stt_vad
      dockerfile: Dockerfile
    container_name: leibniz-stt-vad
    ports:
      - "8001:8001"
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      REDIS_URL: redis://redis:6379/0
      VAD_MODEL_NAME: gemini-2.0-flash-exp
      LOG_LEVEL: INFO
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

  intent-service:
    build:
      context: ./services/intent
      dockerfile: Dockerfile
    container_name: leibniz-intent
    ports:
      - "8002:8002"
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      REDIS_URL: redis://redis:6379/0
      LOG_LEVEL: INFO
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

  rag-service:
    build:
      context: ./services/rag
      dockerfile: Dockerfile
    container_name: leibniz-rag
    ports:
      - "8003:8003"
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      REDIS_URL: redis://redis:6379/0
      FAISS_INDEX_PATH: /app/leibniz_faiss.index
      LOG_LEVEL: INFO
    volumes:
      - ./leibniz_agent/leibniz_knowledge_base:/app/knowledge_base:ro
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

  tts-service:
    build:
      context: ./services/tts
      dockerfile: Dockerfile
    container_name: leibniz-tts
    ports:
      - "8004:8004"
    environment:
      ELEVENLABS_API_KEY: ${ELEVENLABS_API_KEY}
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      MINIO_ENDPOINT: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
      TTS_PRIMARY_PROVIDER: elevenlabs
      LOG_LEVEL: INFO
    depends_on:
      minio:
        condition: service_healthy
    restart: unless-stopped

  appointment-service:
    build:
      context: ./services/appointment
      dockerfile: Dockerfile
    container_name: leibniz-appointment
    ports:
      - "8005:8005"
    environment:
      REDIS_URL: redis://redis:6379/0
      MONGODB_URI: mongodb://mongodb:27017/leibniz
      LOG_LEVEL: INFO
    depends_on:
      redis:
        condition: service_healthy
      mongodb:
        condition: service_healthy
    restart: unless-stopped

  orchestrator:
    build:
      context: ./services/orchestrator
      dockerfile: Dockerfile
    container_name: leibniz-orchestrator
    ports:
      - "8000:8000"
    environment:
      REDIS_URL: redis://redis:6379/0
      STT_SERVICE_URL: http://stt-service:8001
      INTENT_SERVICE_URL: http://intent-service:8002
      RAG_SERVICE_URL: http://rag-service:8003
      TTS_SERVICE_URL: http://tts-service:8004
      APPOINTMENT_SERVICE_URL: http://appointment-service:8005
      LOG_LEVEL: INFO
    depends_on:
      - redis
      - stt-service
      - intent-service
      - rag-service
      - tts-service
      - appointment-service
    restart: unless-stopped

volumes:
  redis_data:
  minio_data:
  mongodb_data:
```

**Environment File** (`.env`):
```bash
GEMINI_API_KEY=your_gemini_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

**Docker Compose Commands**:
```powershell
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v

# Rebuild and restart specific service
docker-compose up -d --build stt-service
```

---

## 🧪 Testing & Validation

### End-to-End Test Script

**File**: `test_microservices_e2e.py`

```python
import asyncio
import httpx
import websockets
import json

async def test_full_conversation_flow():
    """Test complete conversation through orchestrator"""
    
    # 1. Create session
    async with httpx.AsyncClient() as client:
        response = await client.post('http://localhost:8000/api/v1/session/create')
        session_id = response.json()['session_id']
        print(f"✅ Session created: {session_id}")
    
    # 2. Connect to WebSocket
    uri = f"ws://localhost:8000/api/v1/conversation?session_id={session_id}"
    async with websockets.connect(uri) as websocket:
        # 3. Send mock audio (simulating user saying "What are office hours?")
        mock_audio = b'\x00' * 16000  # 1 second of silence
        await websocket.send(mock_audio)
        
        # 4. Receive response
        response = await websocket.recv()
        print(f"✅ Received audio response: {len(response)} bytes")
    
    # 5. Check session history
    async with httpx.AsyncClient() as client:
        response = await client.get(f'http://localhost:8000/api/v1/session/{session_id}/history')
        history = response.json()
        print(f"✅ Session history: {len(history['turns'])} turns")

if __name__ == "__main__":
    asyncio.run(test_full_conversation_flow())
```

### Health Check Script

**File**: `check_services_health.py`

```python
import httpx
import asyncio

SERVICES = {
    "Orchestrator": "http://localhost:8000/health",
    "STT/VAD": "http://localhost:8001/health",
    "Intent": "http://localhost:8002/health",
    "RAG": "http://localhost:8003/health",
    "TTS": "http://localhost:8004/health",
    "Appointment": "http://localhost:8005/health"
}

async def check_all_services():
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in SERVICES.items():
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    print(f"✅ {name:20s} - HEALTHY")
                else:
                    print(f"❌ {name:20s} - UNHEALTHY (status: {response.status_code})")
            except Exception as e:
                print(f"❌ {name:20s} - UNREACHABLE ({str(e)})")

if __name__ == "__main__":
    asyncio.run(check_all_services())
```

**Run Health Checks**:
```powershell
python check_services_health.py
```

---

## 📊 Performance Benchmarks

### Expected Latency Breakdown (Microservices)

| Operation | Monolithic | Microservices | Notes |
|-----------|-----------|---------------|-------|
| **STT (Transcription)** | 500-1500ms | 500-1500ms | Same (Gemini Live API) |
| **Intent Classification (Cache Hit)** | 1-5ms | 5-10ms | +Network overhead |
| **Intent Classification (Cache Miss)** | 200-800ms | 220-850ms | +Network overhead |
| **RAG Query (Cache Hit)** | 1-5ms | 5-10ms | File → Redis (10x faster) |
| **RAG Query (Cache Miss)** | 500-2000ms | 550-2100ms | +Network overhead |
| **TTS Synthesis (Cache Hit)** | 10-50ms | 15-60ms | File → MinIO |
| **TTS Synthesis (Cache Miss)** | 1000-4000ms | 1050-4100ms | +Network overhead |
| **Total (Cached)** | 512-1560ms | 525-1580ms | Minimal impact |
| **Total (Uncached)** | 2200-7300ms | 2320-7550ms | +5-10% latency |

### Scaling Advantages

| Metric | Monolithic | Microservices (3x replicas) |
|--------|-----------|----------------------------|
| **Concurrent Users** | 1-5 (shared state) | 100+ (stateless services) |
| **Memory per Instance** | 500MB | 150MB avg (per service) |
| **Deploy Time (Full)** | 5-10 min | 1-2 min (parallel) |
| **Deploy Time (Single Service)** | 5-10 min | 30-60s |
| **Fault Tolerance** | Total failure | Graceful degradation |
| **Cache Hit Rate (Redis)** | 40-60% | 70-85% (shared cache) |

---

## 🚀 Deployment Roadmap

### Phase 1: Code Refactoring (Weeks 1-2)

**Week 1**: Service Extraction
- [ ] Extract STT/VAD service (`leibniz_vad.py` → `services/stt_vad/`)
- [ ] Extract Intent service (`leibniz_intent_parser.py` → `services/intent/`)
- [ ] Extract RAG service (`leibniz_rag.py` → `services/rag/`)
- [ ] Extract TTS service (`leibniz_tts.py` → `services/tts/`)
- [ ] Extract Appointment FSM (`leibniz_appointment_fsm.py` → `services/appointment/`)
- [ ] Create Orchestrator service (NEW: `services/orchestrator/`)

**Week 2**: Dependency Migration
- [ ] Replace file-based cache with Redis (rag_cache.py → Redis client)
- [ ] Migrate audio storage to MinIO (audio_archive/ → MinIO buckets)
- [ ] Remove global singletons (convert to FastAPI dependency injection)
- [ ] Add health check endpoints to all services
- [ ] Add Prometheus metrics to all services

### Phase 2: Containerization (Week 3)

- [ ] Write Dockerfiles for all 6 services (multi-stage builds)
- [ ] Create docker-compose.yml with all services + infrastructure
- [ ] Test local deployment: `docker-compose up`
- [ ] Verify service-to-service communication
- [ ] Test end-to-end conversation flow
- [ ] Optimize Docker images (reduce size, layer caching)

### Phase 3: Testing (Week 4)

- [ ] Unit tests for each service (pytest)
- [ ] Integration tests (service-to-service calls)
- [ ] Load testing (Locust: 100 concurrent users)
- [ ] Latency profiling (measure each service hop)
- [ ] Chaos engineering (kill random services, test recovery)

### Phase 4: Production Deployment (Weeks 5-6)

**Week 5**: Kubernetes Setup
- [ ] Write Kubernetes manifests (Deployment, Service, Ingress)
- [ ] Create Helm chart for easy installation
- [ ] Set up CI/CD pipeline (GitHub Actions → Docker Hub → K8s)
- [ ] Configure auto-scaling (HPA based on CPU/memory)
- [ ] Set up monitoring (Prometheus + Grafana dashboards)

**Week 6**: Production Hardening
- [ ] Set up centralized logging (ELK Stack or Loki)
- [ ] Configure secrets management (Kubernetes Secrets + Vault)
- [ ] Implement API authentication (JWT tokens)
- [ ] Add rate limiting (per-user quotas)
- [ ] Set up alerts (PagerDuty/Slack for service failures)
- [ ] Load balancer configuration (NGINX Ingress)
- [ ] SSL/TLS certificates (Let's Encrypt)

---

## ⚠️ Critical Considerations

### 1. Latency Overhead
**Problem**: Network calls between services add 5-50ms per hop  
**Mitigation**:
- Deploy all services in same VPC/Kubernetes cluster
- Use HTTP/2 for multiplexing (keep-alive connections)
- Implement circuit breakers (fail fast on service outages)
- Add request timeouts (5s for STT, 1s for Intent, 3s for RAG, 2s for TTS)

### 2. State Management
**Problem**: Conversation state must be externalized (no in-memory sessions)  
**Solution**:
- Store session data in Redis with 30-minute TTL
- Use session_id as key: `session:{uuid}` → JSON state
- FSM state persisted per-turn in Appointment service

### 3. FAISS Index Replication
**Problem**: FAISS index cannot be shared across RAG service replicas  
**Solution**:
- Each RAG replica loads read-only index from shared volume at startup
- Index rebuild is manual (admin endpoint: POST /api/v1/admin/rebuild_index)
- Pre-build index at Docker image build time (RUN python index_builder.py)

### 4. Audio Streaming
**Problem**: Binary data over WebSocket needs buffering  
**Solution**:
- Use chunked transfer encoding (50ms audio chunks)
- Implement backpressure handling (slow client detection)
- TTS service returns presigned URLs (1-hour expiry) for large files

### 5. Cost Optimization
**Problem**: Running 9 containers (6 services + 3 infra) 24/7 is expensive  
**Solutions**:
- **Auto-scaling**: Scale down to 0 replicas during off-hours (HPA + KEDA)
- **Spot Instances**: Use AWS Spot or GCP Preemptible VMs (70% cost savings)
- **Cold Start Optimization**: Keep 1 "warm" replica per service, scale on demand
- **Shared Infrastructure**: Use managed Redis/MongoDB (AWS ElastiCache, Atlas)

---

## 📚 Additional Resources

**Documentation**:
- FastAPI: https://fastapi.tiangolo.com/
- Docker Multi-Stage Builds: https://docs.docker.com/build/building/multi-stage/
- Kubernetes Deployments: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- Prometheus Metrics: https://prometheus.io/docs/practices/instrumentation/

**Tools**:
- **Locust** (load testing): https://locust.io/
- **K6** (performance testing): https://k6.io/
- **Helm** (Kubernetes package manager): https://helm.sh/
- **Lens** (Kubernetes IDE): https://k8slens.dev/

---

## 🎯 Success Criteria

**Minimum Viable Microservices (MVP)**:
- ✅ All 6 services running independently via Docker Compose
- ✅ End-to-end conversation flow working (STT → Intent → RAG → TTS)
- ✅ Redis caching functional (70%+ hit rate)
- ✅ MinIO audio storage operational
- ✅ Health checks passing for all services

**Production Ready**:
- ✅ Kubernetes deployment with 2+ replicas per service
- ✅ Auto-scaling configured (HPA triggers at 70% CPU)
- ✅ Monitoring dashboards live (Grafana + Prometheus)
- ✅ Load testing validated (100 concurrent users, <3s p95 latency)
- ✅ CI/CD pipeline operational (automated builds + deployments)
- ✅ Centralized logging functional (searchable logs for past 7 days)
- ✅ SSL/TLS enabled with valid certificates

---

**Last Updated**: 2025-10-31  
**Version**: 2.0 (Streamlined with detailed specifications)  
**Maintainer**: Leibniz Agent Development Team
Cost: Running 6+ containers 24/7 is expensive

Solution: Auto-scale based on load, use spot instances, implement cold start optimization