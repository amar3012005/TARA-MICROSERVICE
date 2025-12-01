"""
Configuration for STT/VAD Microservice

Consolidates configuration from leibniz_vad.py (LeibnizVADConfig) and 
leibniz_stt.py (LeibnizSTTConfig) into unified service configuration.

Configuration is loaded from environment variables with LEIBNIZ_* prefix.
No load_dotenv() calls - follows microservice pattern.

Reference:
    leibniz_agent/leibniz_vad.py (lines 56-96)
    leibniz_agent/leibniz_stt.py (lines 213-243)
    leibniz_agent/services/shared/redis_client.py (RedisConfig pattern)
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# CRITICAL: Ensure Gemini API key is always available
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY", 
    "AIzaSyC6cvyEl4FNjIQCV_p5_2wJkOa1cUObFHU"  # Hardcoded for testing
)
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


@dataclass
class VADConfig:
    """
    Unified configuration for STT/VAD microservice.
    
    Merges VAD and STT configurations with environment variable loading.
    """
    
    # Audio settings
    sample_rate: int = 16000  # 16kHz for VAD (mono)
    stt_sample_rate: int = 48000  # 48kHz for STT if used separately
    channels: int = 1  # Mono audio
    
    # Gemini Live API settings
    model_name: str = "gemini-live-2.5-flash-preview"
    language_code: str = "en-US"  # English-only
    
    # Timeout settings (seconds)
    initial_timeout_s: float = 20.0      # First attempt timeout
    retry_timeout_s: float = 10.0        # Retry attempt timeout
    max_timeout_s: float = 30.0          # Maximum timeout
    start_timeout_s: float = 20.0        # Default timeout (dynamically adjusted)
    
    # Granular timeout configuration for conversation phases
    greeting_timeout_s: float = 25.0        # Initial greeting phase
    decision_timeout_s: float = 30.0        # Decision-making scenarios
    complex_query_timeout_s: float = 35.0   # Complex RAG queries
    post_service_timeout_s: float = 20.0    # Post-service continuation
    
    # VAD and silence detection
    silence_timeout: float = 2.5          # Silence detection timeout
    min_speech_ms: int = 250              # Minimum speech duration (ms)
    
    # Session management
    session_timeout: float = 600.0        # Session expiry (10 minutes)
    max_session_duration: float = 600.0   # Max session duration
    
    # VAD settings
    vad_sensitivity: str = "MEDIUM"       # VAD sensitivity level
    barge_in_threshold: float = 0.5       # Barge-in detection threshold
    warmup_trigger_delay: float = 2.0     # Warmup delay
    
    # Smart prompting
    smart_prompt_threshold_s: float = 6.0  # Smart prompt trigger delay
    smart_prompt_enabled: bool = True      # Enable smart prompting
    
    # Language detection
    enable_language_detection: bool = False  # DISABLED: Strict en-US lock prevents auto-detection
    strict_english_mode: bool = True         # Reject non-English input
    
    # Gemini VAD/AAD Configuration (Critical Fix #1 - Fix Early Cut-offs)
    # Automatic Activity Detection parameters for robust speech boundary detection
    vad_prefix_padding_ms: int = 300  # Captures speech onset (300ms recommended for ultra-low latency)
    vad_silence_duration_ms: int = 500  # Silence before end-of-speech (500ms for sentence completion)
    # Valid enum values from Gemini API spec
    vad_start_sensitivity: str = "START_SENSITIVITY_HIGH"  # Quick speech detection
    vad_end_sensitivity: str = "END_SENSITIVITY_LOW"  # Don't cut off mid-breath (patient)
    
    # Retry settings
    max_retries: int = 3                  # Maximum retry attempts
    retry_delay_base: float = 1.0         # Base delay for exponential backoff
    
    # Verbosity control (for debugging)
    verbose: bool = False                    # General debug output
    log_audio_callbacks: bool = False        # Audio callback logging
    log_timeout_checks: bool = False         # Timeout check logging
    log_state_transitions: bool = True       # State transition messages
    
    # Response settings
    response_modality: str = "TEXT"       # Transcription only
    
    @staticmethod
    def from_env() -> 'VADConfig':
        """
        Load configuration from environment variables.
        
        Environment variables use LEIBNIZ_VAD_* and LEIBNIZ_STT_* prefixes.
        
        Returns:
            VADConfig: Configuration instance loaded from environment
        """
        return VADConfig(
            # Audio settings
            sample_rate=int(os.getenv("LEIBNIZ_VAD_SAMPLE_RATE", "16000")),
            stt_sample_rate=int(os.getenv("LEIBNIZ_STT_SAMPLE_RATE", "48000")),
            channels=int(os.getenv("LEIBNIZ_VAD_CHANNELS", "1")),
            
            # Gemini settings
            model_name=os.getenv("LEIBNIZ_VAD_MODEL", "gemini-live-2.5-flash-preview"),
            language_code=os.getenv("LEIBNIZ_VAD_LANGUAGE", "en-US"),
            
            # Timeout settings
            initial_timeout_s=float(os.getenv("LEIBNIZ_VAD_TIMEOUT_INITIAL", "20.0")),
            retry_timeout_s=float(os.getenv("LEIBNIZ_VAD_TIMEOUT_RETRY", "10.0")),
            max_timeout_s=float(os.getenv("LEIBNIZ_VAD_TIMEOUT_MAX", "30.0")),
            start_timeout_s=float(os.getenv("LEIBNIZ_VAD_TIMEOUT_START", "20.0")),
            
            # Granular timeouts
            greeting_timeout_s=float(os.getenv("LEIBNIZ_VAD_TIMEOUT_GREETING", "25.0")),
            decision_timeout_s=float(os.getenv("LEIBNIZ_VAD_TIMEOUT_DECISION", "30.0")),
            complex_query_timeout_s=float(os.getenv("LEIBNIZ_VAD_TIMEOUT_COMPLEX", "35.0")),
            post_service_timeout_s=float(os.getenv("LEIBNIZ_VAD_TIMEOUT_POST_SERVICE", "20.0")),
            
            # VAD and silence
            silence_timeout=float(os.getenv("LEIBNIZ_VAD_SILENCE_TIMEOUT", "2.5")),
            min_speech_ms=int(os.getenv("LEIBNIZ_VAD_MIN_SPEECH_MS", "250")),
            
            # Session settings
            session_timeout=float(os.getenv("LEIBNIZ_VAD_SESSION_TIMEOUT", "600.0")),
            max_session_duration=float(os.getenv("LEIBNIZ_VAD_MAX_SESSION_DURATION", "600.0")),
            
            # VAD settings
            vad_sensitivity=os.getenv("LEIBNIZ_VAD_SENSITIVITY", "MEDIUM"),
            barge_in_threshold=float(os.getenv("LEIBNIZ_VAD_BARGE_IN_THRESHOLD", "0.5")),
            warmup_trigger_delay=float(os.getenv("LEIBNIZ_VAD_WARMUP_DELAY", "2.0")),
            
            # Smart prompting
            smart_prompt_threshold_s=float(os.getenv("LEIBNIZ_VAD_SMART_PROMPT_THRESHOLD", "6.0")),
            smart_prompt_enabled=os.getenv("LEIBNIZ_VAD_SMART_PROMPT_ENABLED", "true").lower() == "true",
            
            # Language detection
            enable_language_detection=os.getenv("LEIBNIZ_VAD_ENABLE_LANGUAGE_DETECTION", "false").lower() == "true",
            strict_english_mode=os.getenv("LEIBNIZ_VAD_STRICT_ENGLISH_MODE", "true").lower() == "true",
            
            # VAD/AAD Configuration
            vad_prefix_padding_ms=int(os.getenv("LEIBNIZ_VAD_PREFIX_PADDING_MS", "300")),
            vad_silence_duration_ms=int(os.getenv("LEIBNIZ_VAD_SILENCE_DURATION_MS", "500")),
            vad_start_sensitivity=os.getenv("LEIBNIZ_VAD_START_SENSITIVITY", "START_SENSITIVITY_HIGH"),
            vad_end_sensitivity=os.getenv("LEIBNIZ_VAD_END_SENSITIVITY", "END_SENSITIVITY_LOW"),
            
            # Retry settings
            max_retries=int(os.getenv("LEIBNIZ_VAD_MAX_RETRIES", "3")),
            retry_delay_base=float(os.getenv("LEIBNIZ_VAD_RETRY_DELAY_BASE", "1.0")),
            
            # Verbosity
            verbose=os.getenv("LEIBNIZ_VAD_VERBOSE", "false").lower() == "true",
            log_audio_callbacks=os.getenv("LEIBNIZ_VAD_LOG_AUDIO_CALLBACKS", "false").lower() == "true",
            log_timeout_checks=os.getenv("LEIBNIZ_VAD_LOG_TIMEOUT_CHECKS", "false").lower() == "true",
            log_state_transitions=os.getenv("LEIBNIZ_VAD_LOG_STATE_TRANSITIONS", "true").lower() == "true",
            
            # Response settings
            response_modality=os.getenv("LEIBNIZ_VAD_RESPONSE_MODALITY", "TEXT"),
        )
    
    def __post_init__(self):
        """
        Validate and normalize configuration.
        
        Ensures English-only language code and valid timeout values.
        Ported from LeibnizVADConfig.__post_init__() (lines 85-96).
        """
        # Normalize language code to en-US
        if self.language_code not in ["en-US", "en"]:
            logger.warning(
                f"️ STT/VAD service configured for English-only. "
                f"Got language_code='{self.language_code}', normalizing to 'en-US'. "
                f"For multilingual support, use TARA agent."
            )
            self.language_code = "en-US"
        
        if self.language_code == "en":
            self.language_code = "en-US"
        
        # Validate VAD parameters
        if not (0 <= self.vad_prefix_padding_ms <= 1000):
            logger.warning(f"Invalid vad_prefix_padding_ms={self.vad_prefix_padding_ms}, using default 300")
            self.vad_prefix_padding_ms = 300
        if not (0 <= self.vad_silence_duration_ms <= 5000):
            logger.warning(f"Invalid vad_silence_duration_ms={self.vad_silence_duration_ms}, using default 500")
            self.vad_silence_duration_ms = 500
        
        # Validate VAD sensitivity enums
        valid_start = ["START_SENSITIVITY_UNSPECIFIED", "START_SENSITIVITY_HIGH", "START_SENSITIVITY_LOW"]
        valid_end = ["END_SENSITIVITY_UNSPECIFIED", "END_SENSITIVITY_HIGH", "END_SENSITIVITY_LOW"]
        if self.vad_start_sensitivity not in valid_start:
            logger.warning(f"Invalid vad_start_sensitivity '{self.vad_start_sensitivity}', defaulting to START_SENSITIVITY_HIGH")
            self.vad_start_sensitivity = "START_SENSITIVITY_HIGH"
        if self.vad_end_sensitivity not in valid_end:
            logger.warning(f"Invalid vad_end_sensitivity '{self.vad_end_sensitivity}', defaulting to END_SENSITIVITY_LOW")
            self.vad_end_sensitivity = "END_SENSITIVITY_LOW"
        
        # Validate timeout values
        if self.initial_timeout_s <= 0:
            logger.warning(f"Invalid initial_timeout_s={self.initial_timeout_s}, using default 20.0")
            self.initial_timeout_s = 20.0
        
        if self.session_timeout <= 0:
            logger.warning(f"Invalid session_timeout={self.session_timeout}, using default 600.0")
            self.session_timeout = 600.0
        
        # Log configuration if verbose
        if self.verbose:
            logger.info(
                f" VADConfig loaded: model={self.model_name}, "
                f"sample_rate={self.sample_rate}Hz, timeout={self.initial_timeout_s}s, "
                f"language={self.language_code}"
            )
