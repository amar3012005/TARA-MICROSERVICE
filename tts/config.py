"""
TTS Configuration Module

Extracted from leibniz_tts.py (LeibnizTTSConfig, lines 203-252) for microservice deployment.

Configuration dataclass for TTS service with environment variable loading, validation,
and provider-specific settings.

Reference:
    leibniz_agent/leibniz_tts.py - Original configuration
    leibniz_agent/.env.leibniz - Environment variables (lines 172-268)
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """
    Configuration for TTS microservice.
    
    Supports 4 providers:
        - lemonfox: LemonFox.ai TTS (multilingual, natural voices)
        - google: Google Cloud TTS (stable, production-ready)
        - elevenlabs: ElevenLabs TTS (premium quality)
        - mock: Mock TTS (silent audio for testing)
    
    Emotion modulation:
        - lemonfox: Speed control (0.25-4.0)
        - google: Pitch/speaking_rate via SSML prosody tags
        - elevenlabs: Stability/similarity (no direct emotion)
    """
    
    # Provider selection
    provider: str = "lemonfox"  # Primary provider (lemonfox, google, elevenlabs, auto, mock)
    fallback_provider: str = "google"  # Fallback provider (google, elevenlabs)
    enable_fallback: bool = True  # Enable automatic fallback on primary failure
    
    # LemonFox TTS
    lemonfox_api_key: Optional[str] = None
    lemonfox_voice: str = "sarah"  # sarah, heart, bella, michael, etc.
    lemonfox_language: str = "en-us"  # en-us, de-de, es-es, fr-fr, it-it, pl-pl, pt-br, nl-nl
    
    # Google Cloud TTS
    google_voice: str = "en-US-Neural2-F"
    google_credentials_path: Optional[str] = None
    
    # ElevenLabs TTS (REMOVED - keeping only LemonFox)
    # elevenlabs_api_key: Optional[str] = None
    # elevenlabs_voice: str = "AnvlJBAqSLDzEevYr9Ap"
    # elevenlabs_model: str = " eleven_turbo_v2_5"
    # elevenlabs_stability: float = 0.5
    # elevenlabs_similarity_boost: float = 0.75
    
    # Audio settings
    language_code: str = "en-US"
    sample_rate: int = 24000
    speaking_rate: float = 1.0
    pitch: float = 0.0
    volume_gain_db: float = 0.0
    
    # Cache settings
    enable_cache: bool = True
    cache_dir: str = "/app/audio_cache"
    max_cache_size: int = 500
    cache_ttl_days: int = 30
    
    # Service settings
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    mock_mode: bool = False
    
    @staticmethod
    def from_env() -> "TTSConfig":
        """
        Load configuration from environment variables.
        
        Environment variables (15+ from .env.leibniz lines 172-268):
            - LEIBNIZ_TTS_PROVIDER: Primary provider (default: gemini)
            - LEIBNIZ_TTS_FALLBACK_PROVIDER: Fallback provider (default: google)
            - GEMINI_API_KEY: Gemini API key
            - GOOGLE_APPLICATION_CREDENTIALS: Path to Google service account JSON
            - GOOGLE_TTS_VOICE: Google voice name (default: en-US-Neural2-F)
            - ELEVENLABS_API_KEY: ElevenLabs API key
            - ELEVENLABS_VOICE_ID: ElevenLabs voice ID
            - ELEVENLABS_MODEL: ElevenLabs model (default:  eleven_turbo_v2_5)
            - LEIBNIZ_TTS_GEMINI_MODEL: Gemini model (default: gemini-2.5-flash-preview-tts)
            - LEIBNIZ_TTS_GEMINI_VOICE: Gemini voice character (default: Callirrhoe)
            - LEIBNIZ_TTS_GEMINI_EMOTION_SUPPORT: Enable emotion prompts (default: true)
            - LEIBNIZ_XTTS_SPEAKER_SAMPLE: Path to speaker sample WAV
            - LEIBNIZ_XTTS_LANGUAGE: XTTS language code (default: en)
            - LEIBNIZ_XTTS_DEVICE: XTTS device (cuda, cpu, auto)
            - LEIBNIZ_TTS_CACHE_DIR: Cache directory (default: /app/audio_cache)
            - LEIBNIZ_TTS_PROVIDER: Primary provider (default: lemonfox)
            - LEIBNIZ_TTS_FALLBACK_PROVIDER: Fallback provider (default: google)
            - LEMONFOX_API_KEY: LemonFox API key
            - GEMINI_API_KEY: Gemini API key (for STT/LLM, not TTS)
            - GOOGLE_APPLICATION_CREDENTIALS: Path to Google service account JSON
            - GOOGLE_TTS_VOICE: Google voice name (default: en-US-Neural2-F)
            - ELEVENLABS_API_KEY: ElevenLabs API key
            - ELEVENLABS_VOICE_ID: ElevenLabs voice ID
            - LEIBNIZ_TTS_CACHE_ENABLED: Enable caching (default: true)
            - LEIBNIZ_TTS_CACHE_MAX_SIZE: Max cache entries (default: 500)
            - LEIBNIZ_TTS_TIMEOUT: Synthesis timeout (default: 30.0s)
            - LEIBNIZ_TTS_SAMPLE_RATE: Audio sample rate (default: 24000)
            - MOCK_TTS: Enable mock provider for testing (default: false)
        
        Returns:
            TTSConfig instance with values from environment or defaults
        """
        return TTSConfig(
            # Provider selection
            provider=os.getenv("LEIBNIZ_TTS_PROVIDER", "lemonfox"),
            fallback_provider=os.getenv("LEIBNIZ_TTS_FALLBACK_PROVIDER", "google"),
            enable_fallback=os.getenv("LEIBNIZ_TTS_ENABLE_FALLBACK", "true").lower() == "true",
            
            # LemonFox TTS
            lemonfox_api_key=os.getenv("LEMONFOX_API_KEY"),
            lemonfox_voice=os.getenv("LEIBNIZ_LEMONFOX_VOICE", "sarah"),
            lemonfox_language=os.getenv("LEIBNIZ_LEMONFOX_LANGUAGE", "en-us"),
            
            # Google Cloud TTS
            google_voice=os.getenv("GOOGLE_TTS_VOICE", "en-US-Neural2-F"),
            google_credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            
            # ElevenLabs TTS (REMOVED)
            # elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
            # elevenlabs_voice=os.getenv("ELEVENLABS_VOICE_ID", "AnvlJBAqSLDzEevYr9Ap"),
            # elevenlabs_model=os.getenv("ELEVENLABS_MODEL", " eleven_turbo_v2_5"),
            # elevenlabs_stability=float(os.getenv("ELEVENLABS_STABILITY", "0.5")),
            # elevenlabs_similarity_boost=float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75")),
            
            # Audio settings
            language_code=os.getenv("LEIBNIZ_TTS_LANGUAGE_CODE", "en-US"),
            sample_rate=int(os.getenv("LEIBNIZ_TTS_SAMPLE_RATE", "24000")),
            speaking_rate=float(os.getenv("LEIBNIZ_TTS_SPEAKING_RATE", "1.0")),
            pitch=float(os.getenv("LEIBNIZ_TTS_PITCH", "0.0")),
            volume_gain_db=float(os.getenv("LEIBNIZ_TTS_VOLUME_GAIN_DB", "0.0")),
            
            # Cache settings
            enable_cache=os.getenv("LEIBNIZ_TTS_CACHE_ENABLED", "true").lower() == "true",
            cache_dir=os.getenv("LEIBNIZ_TTS_CACHE_DIR", "/app/audio_cache"),
            max_cache_size=int(os.getenv("LEIBNIZ_TTS_CACHE_MAX_SIZE", "500")),
            cache_ttl_days=int(os.getenv("LEIBNIZ_TTS_CACHE_TTL_DAYS", "30")),
            
            # Service settings
            timeout=float(os.getenv("LEIBNIZ_TTS_TIMEOUT", "30.0")),
            retry_attempts=int(os.getenv("LEIBNIZ_TTS_RETRY_ATTEMPTS", "3")),
            retry_delay=float(os.getenv("LEIBNIZ_TTS_RETRY_DELAY", "1.0")),
            mock_mode=os.getenv("MOCK_TTS", "false").lower() == "true",
        )
    
    def __post_init__(self):
        """
        Validate configuration after initialization.
        
        Validations:
            - Provider is valid (lemonfox, google, elevenlabs, auto, mock)
            - Fallback provider is valid (google, elevenlabs, lemonfox)
            - Sample rate is positive
            - Speaking rate is between 0.25 and 4.0
            - Pitch is between -20.0 and 20.0
            - ElevenLabs stability/similarity_boost are between 0.0 and 1.0
            - Timeout is positive
            - Retry attempts is positive
            - Max cache size is positive
        
        Raises:
            ValueError: If any validation fails
        """
        # Validate provider
        valid_providers = ["lemonfox", "google", "elevenlabs", "auto", "mock"]
        if self.provider not in valid_providers:
            raise ValueError(f"Invalid provider: {self.provider}. Must be one of: {valid_providers}")
        
        # Validate fallback provider (allow mock for minimal builds)
        valid_fallback = ["google", "elevenlabs", "lemonfox", "mock"]
        if self.fallback_provider not in valid_fallback:
            raise ValueError(f"Invalid fallback_provider: {self.fallback_provider}. Must be one of: {valid_fallback}")
        
        # Validate audio settings
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate}. Must be positive.")
        
        if not (0.25 <= self.speaking_rate <= 4.0):
            logger.warning(f"Speaking rate {self.speaking_rate} outside recommended range [0.25, 4.0]. Clamping.")
            self.speaking_rate = max(0.25, min(4.0, self.speaking_rate))
        
        if not (-20.0 <= self.pitch <= 20.0):
            logger.warning(f"Pitch {self.pitch} outside valid range [-20.0, 20.0]. Clamping.")
            self.pitch = max(-20.0, min(20.0, self.pitch))
        
        # Validate ElevenLabs settings (REMOVED)
        # if not (0.0 <= self.elevenlabs_stability <= 1.0):
        #     raise ValueError(f"Invalid elevenlabs_stability: {self.elevenlabs_stability}. Must be between 0.0 and 1.0.")
        
        # if not (0.0 <= self.elevenlabs_similarity_boost <= 1.0):
        #     raise ValueError(f"Invalid elevenlabs_similarity_boost: {self.elevenlabs_similarity_boost}. Must be between 0.0 and 1.0.")
        
        # Validate service settings
        if self.timeout <= 0:
            raise ValueError(f"Invalid timeout: {self.timeout}. Must be positive.")
        
        if self.retry_attempts <= 0:
            raise ValueError(f"Invalid retry_attempts: {self.retry_attempts}. Must be positive.")
        
        if self.max_cache_size <= 0:
            raise ValueError(f"Invalid max_cache_size: {self.max_cache_size}. Must be positive.")
        
        # Warn if no API keys set
        has_lemonfox = bool(self.lemonfox_api_key)
        has_google = bool(self.google_credentials_path)
        # has_elevenlabs = bool(self.elevenlabs_api_key)  # REMOVED
        
        if not (has_lemonfox or has_google or self.mock_mode):
            logger.warning(
                "️ No TTS provider API keys configured! "
                "Set at least one: LEMONFOX_API_KEY, GOOGLE_APPLICATION_CREDENTIALS, "
                "or enable MOCK_TTS=true"
            )
        
        # Log configuration
        logger.info(f"️ TTS Config: provider={self.provider}, fallback={self.fallback_provider}, "
                   f"cache={self.enable_cache}, sample_rate={self.sample_rate}Hz, mock={self.mock_mode}")
