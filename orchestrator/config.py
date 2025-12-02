"""
Configuration for StateManager Orchestrator

Loads configuration from environment variables for Docker deployment.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for StateManager Orchestrator"""
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    redis_host: str = os.getenv("LEIBNIZ_REDIS_HOST", "redis")
    redis_port: int = int(os.getenv("LEIBNIZ_REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("LEIBNIZ_REDIS_DB", "0"))
    
    # Service URLs
    intent_service_url: str = os.getenv("INTENT_SERVICE_URL", "http://intent-service:8002")
    rag_service_url: Optional[str] = os.getenv("RAG_SERVICE_URL", None)  # Optional - can be None
    stt_service_url: str = os.getenv("STT_SERVICE_URL", "http://stt-vad-service:8001")
    tts_service_url: str = os.getenv("TTS_SERVICE_URL", "http://tts-streaming-service-new:8005")
    
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
        logger.info(
            f"OrchestratorConfig loaded: "
            f"intent_url={self.intent_service_url}, "
            f"rag_url={rag_info}, "
            f"redis_host={self.redis_host}:{self.redis_port}"
        )


