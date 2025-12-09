"""
TTS Streaming Configuration

Configuration dataclass with hardcoded LemonFox settings.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TTSStreamingConfig:
    """Configuration for TTS Streaming microservice with Sarvam AI settings"""
    
    # Sarvam AI API (hardcoded endpoint and key)
    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "sk_2d8w6udi_gaPItmPcCEsf3CoON7RBzqPr")
    sarvam_model: str = os.getenv("SARVAM_TTS_MODEL", "bulbul:v2")
    sarvam_speaker: str = os.getenv("SARVAM_TTS_SPEAKER", "anushka")
    sarvam_language: str = os.getenv("SARVAM_TTS_LANGUAGE", "en-IN")
    sarvam_api_endpoint: str = "https://api.sarvam.ai/text-to-speech"  # Hardcoded
    
    # Voice control parameters (NEW - Sarvam specific)
    sarvam_pitch: float = float(os.getenv("SARVAM_TTS_PITCH", "0.0"))        # -0.75 to 0.75
    sarvam_pace: float = float(os.getenv("SARVAM_TTS_PACE", "1.0"))          # 0.3 to 3.0
    sarvam_loudness: float = float(os.getenv("SARVAM_TTS_LOUDNESS", "1.0"))  # 0.1 to 3.0
    sarvam_preprocessing: bool = os.getenv("SARVAM_TTS_PREPROCESSING", "false").lower() == "true"
    
    # Audio settings
    sample_rate: int = 16000
    language_code: str = "en-US"
    
    # Queue settings
    queue_max_size: int = 10  # Maximum sentences in queue
    inter_sentence_gap_ms: int = int(os.getenv("LEIBNIZ_TTS_INTER_SENTENCE_GAP_MS", "100"))
    
    # Cache settings
    enable_cache: bool = True
    cache_dir: str = "/app/audio_cache"
    max_cache_size: int = 500
    cache_ttl_days: int = 30
    
    # Streaming settings
    fastrtc_chunk_duration_ms: int = int(os.getenv("LEIBNIZ_TTS_FASTRTC_CHUNK_MS", "40"))
    
    # Service settings
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Port
    port: int = 8025
    
    @staticmethod
    def from_env() -> "TTSStreamingConfig":
        """Load configuration from environment variables"""
        return TTSStreamingConfig(
            sarvam_api_key=os.getenv("SARVAM_API_KEY", "sk_2d8w6udi_gaPItmPcCEsf3CoON7RBzqPr"),
            sarvam_model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v2"),
            sarvam_speaker=os.getenv("SARVAM_TTS_SPEAKER", "anushka"),
            sarvam_language=os.getenv("SARVAM_TTS_LANGUAGE", "en-IN"),
            sarvam_pitch=float(os.getenv("SARVAM_TTS_PITCH", "0.0")),
            sarvam_pace=float(os.getenv("SARVAM_TTS_PACE", "1.0")),
            sarvam_loudness=float(os.getenv("SARVAM_TTS_LOUDNESS", "1.0")),
            sarvam_preprocessing=os.getenv("SARVAM_TTS_PREPROCESSING", "false").lower() == "true",
            sample_rate=int(os.getenv("LEIBNIZ_TTS_SAMPLE_RATE", "16000")),
            language_code=os.getenv("LEIBNIZ_TTS_LANGUAGE_CODE", "en-US"),
            queue_max_size=int(os.getenv("TTS_QUEUE_MAX_SIZE", "10")),
            enable_cache=os.getenv("LEIBNIZ_TTS_CACHE_ENABLED", "true").lower() == "true",
            cache_dir=os.getenv("LEIBNIZ_TTS_CACHE_DIR", "/app/audio_cache"),
            max_cache_size=int(os.getenv("LEIBNIZ_TTS_CACHE_MAX_SIZE", "500")),
            cache_ttl_days=int(os.getenv("LEIBNIZ_TTS_CACHE_TTL_DAYS", "30")),
            timeout=float(os.getenv("LEIBNIZ_TTS_TIMEOUT", "30.0")),
            retry_attempts=int(os.getenv("LEIBNIZ_TTS_RETRY_ATTEMPTS", "3")),
            retry_delay=float(os.getenv("LEIBNIZ_TTS_RETRY_DELAY", "1.0")),
            port=int(os.getenv("TTS_STREAMING_PORT", "8025")),
            inter_sentence_gap_ms=int(os.getenv("LEIBNIZ_TTS_INTER_SENTENCE_GAP_MS", "100")),
            fastrtc_chunk_duration_ms=int(os.getenv("LEIBNIZ_TTS_FASTRTC_CHUNK_MS", "40"))
        )
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.sarvam_api_key:
            logger.warning("SARVAM_API_KEY not set - service may not function properly")
        
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate}")
        
        if self.queue_max_size <= 0:
            raise ValueError(f"Invalid queue_max_size: {self.queue_max_size}")
        
        if self.inter_sentence_gap_ms < 0:
            raise ValueError(f"Invalid inter_sentence_gap_ms: {self.inter_sentence_gap_ms}")
        
        if self.fastrtc_chunk_duration_ms <= 0:
            raise ValueError(f"Invalid fastrtc_chunk_duration_ms: {self.fastrtc_chunk_duration_ms}")
        
        logger.info(
            f"TTS Streaming Config: model={self.sarvam_model}, speaker={self.sarvam_speaker}, "
            f"language={self.sarvam_language}, sample_rate={self.sample_rate}Hz, "
            f"queue_size={self.queue_max_size}, cache={self.enable_cache}, "
            f"gap={self.inter_sentence_gap_ms}ms, chunk={self.fastrtc_chunk_duration_ms}ms"
        )





