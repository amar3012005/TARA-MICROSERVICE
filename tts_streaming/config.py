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
    """Configuration for TTS Streaming microservice with hardcoded LemonFox settings"""
    
    # LemonFox API (hardcoded endpoint)
    lemonfox_api_key: str = os.getenv("LEMONFOX_API_KEY", "")
    lemonfox_voice: str = os.getenv("LEIBNIZ_LEMONFOX_VOICE", "sarah")
    lemonfox_language: str = os.getenv("LEIBNIZ_LEMONFOX_LANGUAGE", "en-us")
    lemonfox_api_endpoint: str = "https://api.lemonfox.ai/v1/audio/speech"  # Hardcoded
    
    # Audio settings
    sample_rate: int = 24000
    language_code: str = "en-US"
    
    # Queue settings
    queue_max_size: int = 10  # Maximum sentences in queue
    
    # Cache settings
    enable_cache: bool = True
    cache_dir: str = "/app/audio_cache"
    max_cache_size: int = 500
    cache_ttl_days: int = 30
    
    # Service settings
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Port
    port: int = 8005
    
    @staticmethod
    def from_env() -> "TTSStreamingConfig":
        """Load configuration from environment variables"""
        return TTSStreamingConfig(
            lemonfox_api_key=os.getenv("LEMONFOX_API_KEY", ""),
            lemonfox_voice=os.getenv("LEIBNIZ_LEMONFOX_VOICE", "sarah"),
            lemonfox_language=os.getenv("LEIBNIZ_LEMONFOX_LANGUAGE", "en-us"),
            sample_rate=int(os.getenv("LEIBNIZ_TTS_SAMPLE_RATE", "24000")),
            language_code=os.getenv("LEIBNIZ_TTS_LANGUAGE_CODE", "en-US"),
            queue_max_size=int(os.getenv("TTS_QUEUE_MAX_SIZE", "10")),
            enable_cache=os.getenv("LEIBNIZ_TTS_CACHE_ENABLED", "true").lower() == "true",
            cache_dir=os.getenv("LEIBNIZ_TTS_CACHE_DIR", "/app/audio_cache"),
            max_cache_size=int(os.getenv("LEIBNIZ_TTS_CACHE_MAX_SIZE", "500")),
            cache_ttl_days=int(os.getenv("LEIBNIZ_TTS_CACHE_TTL_DAYS", "30")),
            timeout=float(os.getenv("LEIBNIZ_TTS_TIMEOUT", "30.0")),
            retry_attempts=int(os.getenv("LEIBNIZ_TTS_RETRY_ATTEMPTS", "3")),
            retry_delay=float(os.getenv("LEIBNIZ_TTS_RETRY_DELAY", "1.0")),
            port=int(os.getenv("TTS_STREAMING_PORT", "8005"))
        )
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.lemonfox_api_key:
            logger.warning("LEMONFOX_API_KEY not set - service may not function properly")
        
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate}")
        
        if self.queue_max_size <= 0:
            raise ValueError(f"Invalid queue_max_size: {self.queue_max_size}")
        
        logger.info(
            f"TTS Streaming Config: voice={self.lemonfox_voice}, "
            f"language={self.lemonfox_language}, sample_rate={self.sample_rate}Hz, "
            f"queue_size={self.queue_max_size}, cache={self.enable_cache}"
        )


