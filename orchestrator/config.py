"""
Configuration for StateManager Orchestrator

Loads configuration from environment variables for Docker deployment.
Supports TARA mode for Telugu TASK organization customer service agent.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

# TARA Telugu intro greeting for TASK organization
TARA_INTRO_GREETING = "నమస్కారం అండి! నేను TARA, TASK యొక్క కస్టమర్ సర్వీస్ ఏజెంట్. మీకు ఎలా సహాయం చేయగలను?"

# Default English intro greeting for TASK
DEFAULT_INTRO_GREETING = "Hello! I'm TARA, the customer service agent for TASK. How can I help you today?"


@dataclass
class OrchestratorConfig:
    """Configuration for StateManager Orchestrator"""
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://tara-task-redis:6381/0")
    redis_host: str = os.getenv("TARA_REDIS_HOST", os.getenv("LEIBNIZ_REDIS_HOST", "tara-task-redis"))
    redis_port: int = int(os.getenv("TARA_REDIS_PORT", os.getenv("LEIBNIZ_REDIS_PORT", "6381")))
    redis_db: int = int(os.getenv("TARA_REDIS_DB", os.getenv("LEIBNIZ_REDIS_DB", "0")))
    
    # Service URLs (defaults configured for TARA mode)
    intent_service_url: str = os.getenv("INTENT_SERVICE_URL", "http://intent-service:8002")  # Not used in TARA mode
    rag_service_url: Optional[str] = os.getenv("RAG_SERVICE_URL", "http://rag-service:8003")
    stt_service_url: str = os.getenv("STT_SERVICE_URL", "http://tara-stt-vad-service:8001")
    tts_service_url: str = os.getenv("TTS_SERVICE_URL", "http://tts-sarvam-service:8025")  # Telugu TTS
    appointment_service_url: str = os.getenv("APPOINTMENT_SERVICE_URL", "http://appointment-service:8006")
    
    # TARA Mode Configuration (Telugu TASK Customer Service Agent)
    tara_mode: bool = os.getenv("TARA_MODE", "true").lower() == "true"  # Default: TARA enabled
    skip_intent_service: bool = os.getenv("SKIP_INTENT_SERVICE", "true").lower() == "true"  # Default: Skip intent
    skip_appointment_service: bool = os.getenv("SKIP_APPOINTMENT_SERVICE", "true").lower() == "true"  # Default: Skip appointment
    response_language: str = os.getenv("RESPONSE_LANGUAGE", "te-mixed")  # Default: Telugu mixed
    organization_name: str = os.getenv("ORGANIZATION_NAME", "TASK")
    
    # Incremental RAG - pre-buffer documents during partial STT for faster response
    enable_incremental_rag: bool = os.getenv("ENABLE_INCREMENTAL_RAG", "true").lower() == "true"
    
    # Intro greeting - use TARA Telugu greeting by default (TARA_MODE defaults to true)
    intro_greeting: str = os.getenv(
        "INTRO_GREETING",
        TARA_INTRO_GREETING if os.getenv("TARA_MODE", "true").lower() == "true" else DEFAULT_INTRO_GREETING
    )
    
    # Ignore STT while speaking (prevents barge-in interference)
    ignore_stt_while_speaking: bool = os.getenv("IGNORE_STT_WHILE_SPEAKING", "true").lower() == "true"
    
    # TTS Streaming Mode:
    # - "buffered": Wait for complete sentences before sending to TTS (better prosody)
    # - "continuous": Stream text chunks immediately (ultra-low latency, for ElevenLabs)
    tts_streaming_mode: str = os.getenv("TTS_STREAMING_MODE", "buffered")
    
    # ElevenLabs TTS Configuration (for production ultra-low latency)
    # When USE_ELEVENLABS_TTS=true, orchestrator uses direct ElevenLabs streaming
    use_elevenlabs_tts: bool = os.getenv("USE_ELEVENLABS_TTS", "false").lower() == "true"
    elevenlabs_tts_url: str = os.getenv("ELEVENLABS_TTS_URL", "http://tara-task-tts-labs:8006")
    elevenlabs_prewarm_on_vad: bool = os.getenv("ELEVENLABS_PREWARM_ON_VAD", "true").lower() == "true"
    
    # LLM (for future use)
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")
    
    # TTS (for future use)
    tts_provider: str = os.getenv("TTS_PROVIDER", "elevenlabs")
    tts_api_key: str = os.getenv("TTS_API_KEY", "")
    tts_voice_id: str = os.getenv("TTS_VOICE_ID", "default")
    
    # Performance
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "3600"))  # 1 hour
    max_concurrent_sessions: int = int(os.getenv("MAX_CONCURRENT_SESSIONS", "1000"))
    buffer_size: int = int(os.getenv("BUFFER_SIZE", "200"))
    
    # Dialogue Manager Configuration
    timeout_seconds: int = int(os.getenv("TIMEOUT_SECONDS", "10"))  # 10 seconds timeout for no response
    exit_keywords: List[str] = field(default_factory=lambda: os.getenv(
        "EXIT_KEYWORDS", 
        "bye,exit,quit,goodbye,see you,thank you,thanks,stop,end"
    ).split(","))  # Keywords that trigger exit dialogue
    
    # Latency targets (ms)
    intent_timeout_ms: int = int(os.getenv("INTENT_TIMEOUT_MS", "100"))
    rag_timeout_ms: int = int(os.getenv("RAG_TIMEOUT_MS", "150"))
    llm_timeout_ms: int = int(os.getenv("LLM_TIMEOUT_MS", "300"))
    tts_timeout_ms: int = int(os.getenv("TTS_TIMEOUT_MS", "200"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    @staticmethod
    def from_env() -> "OrchestratorConfig":
        """Load configuration from environment variables"""
        return OrchestratorConfig()
    
    def __post_init__(self):
        """Validate configuration"""
        if self.session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be positive")
        if self.max_concurrent_sessions <= 0:
            raise ValueError("max_concurrent_sessions must be positive")
        
        rag_info = self.rag_service_url if self.rag_service_url else "not configured"
        
        # Log base configuration
        logger.info(
            f"OrchestratorConfig loaded: "
            f"intent_url={self.intent_service_url}, "
            f"rag_url={rag_info}, "
            f"redis_host={self.redis_host}:{self.redis_port}"
        )
        
        # Log TARA mode configuration
        if self.tara_mode:
            logger.info("=" * 70)
            logger.info("🇮🇳 TARA MODE ENABLED - Telugu TASK Customer Service Agent")
            logger.info(f"   Organization: {self.organization_name}")
            logger.info(f"   Language: {self.response_language}")
            logger.info(f"   Skip Intent Service: {self.skip_intent_service}")
            logger.info(f"   Skip Appointment Service: {self.skip_appointment_service}")
            logger.info(f"   Ignore STT While Speaking: {self.ignore_stt_while_speaking}")
            logger.info(f"   TTS Streaming Mode: {self.tts_streaming_mode}")
            logger.info(f"   Intro: {self.intro_greeting[:50]}...")
            logger.info("=" * 70)
        
        # Log ElevenLabs TTS configuration
        if self.use_elevenlabs_tts:
            logger.info("=" * 70)
            logger.info("🎙️ ELEVENLABS TTS ENABLED - Ultra-low latency streaming")
            logger.info(f"   TTS URL: {self.elevenlabs_tts_url}")
            logger.info(f"   Prewarm on VAD: {self.elevenlabs_prewarm_on_vad}")
            logger.info(f"   Streaming Mode: {self.tts_streaming_mode}")
            logger.info("=" * 70)


