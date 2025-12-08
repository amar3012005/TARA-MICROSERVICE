"""
Configuration for the Sarvam AI Saarika STT microservice.

Loads environment variables with LEIBNIZ_* and SARVAM_* prefixes to drive
audio buffering, silence detection, and API connectivity.

Reference: https://docs.sarvam.ai/api-reference-docs/getting-started/models/saarika
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

SARVAM_SUPPORTED_LANGUAGE_CODES = {
    "en-IN",
    "hi-IN",
    "bn-IN",
    "ta-IN",
    "te-IN",
    "gu-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "pa-IN",
    "od-IN",
    "unknown",
}


@dataclass
class VADConfig:
    """
    Unified configuration for STT/VAD microservice.
    
    Merges VAD and STT configurations with environment variable loading.
    """
    
    # Audio settings
    sample_rate: int = 16000  # 16kHz telephony-optimized audio
    stt_sample_rate: int = 16000  # Sarvam expects 16kHz mono
    channels: int = 1  # Mono audio
    
    # Sarvam AI Saarika STT settings
    model_name: str = "saarika:v2.5"
    language_code: str = "unknown"  # Default to automatic language detection
    enable_language_detection: bool = True
    sarvam_api_key: str = os.getenv("SARVAM_API_SUBSCRIPTION_KEY", "")
    sarvam_endpoint: Optional[str] = os.getenv("SARVAM_API_ENDPOINT")
    
    # Streaming Settings
    sarvam_high_vad_sensitivity: bool = True
    sarvam_vad_signals: bool = True
    
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
    
    # VAD and silence detection (Legacy - energy based parameters mostly unused now)
    silence_timeout: float = 1.5          # Silence detection timeout
    min_speech_ms: int = 350              # Minimum speech duration (ms)
    max_buffer_ms: int = 6000             # Force-flush buffer window
    partial_flush_ms: int = 3500          # Emit partial transcripts during long speech
    energy_activation: float = 450.0      # RMS threshold to start speech
    energy_release: float = 220.0         # RMS threshold to consider silence
    
    # Session management
    session_timeout: float = 600.0        # Session expiry (10 minutes)
    max_session_duration: float = 600.0   # Max session duration
    
    # VAD settings
    vad_sensitivity: str = "MEDIUM"       # VAD sensitivity level
    barge_in_threshold: float = 0.5       # Barge-in detection threshold
    warmup_trigger_delay: float = 2.0     # Warmup delay
    
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
            
            # Sarvam settings
            model_name=os.getenv("LEIBNIZ_VAD_MODEL", "saarika:v2.5"),
            language_code=os.getenv("LEIBNIZ_VAD_LANGUAGE", "unknown"),
            enable_language_detection=os.getenv("LEIBNIZ_VAD_ENABLE_LANGUAGE_DETECTION", "true").lower() == "true",
            sarvam_api_key=os.getenv("SARVAM_API_SUBSCRIPTION_KEY", ""),
            sarvam_endpoint=os.getenv("SARVAM_API_ENDPOINT"),
            
            # Streaming settings
            sarvam_high_vad_sensitivity=os.getenv("SARVAM_HIGH_VAD_SENSITIVITY", "true").lower() == "true",
            sarvam_vad_signals=os.getenv("SARVAM_VAD_SIGNALS", "true").lower() == "true",
            
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
            silence_timeout=float(os.getenv("LEIBNIZ_VAD_SILENCE_TIMEOUT", "1.5")),
            min_speech_ms=int(os.getenv("LEIBNIZ_VAD_MIN_SPEECH_MS", "350")),
            max_buffer_ms=int(os.getenv("LEIBNIZ_VAD_MAX_BUFFER_MS", "6000")),
            partial_flush_ms=int(os.getenv("LEIBNIZ_VAD_PARTIAL_FLUSH_MS", "3500")),
            energy_activation=float(os.getenv("LEIBNIZ_VAD_ENERGY_ACTIVATION", "450.0")),
            energy_release=float(os.getenv("LEIBNIZ_VAD_ENERGY_RELEASE", "220.0")),
            
            # Session settings
            session_timeout=float(os.getenv("LEIBNIZ_VAD_SESSION_TIMEOUT", "600.0")),
            max_session_duration=float(os.getenv("LEIBNIZ_VAD_MAX_SESSION_DURATION", "600.0")),
            
            # VAD settings
            vad_sensitivity=os.getenv("LEIBNIZ_VAD_SENSITIVITY", "MEDIUM"),
            barge_in_threshold=float(os.getenv("LEIBNIZ_VAD_BARGE_IN_THRESHOLD", "0.5")),
            warmup_trigger_delay=float(os.getenv("LEIBNIZ_VAD_WARMUP_DELAY", "2.0")),
            
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
        """
        # Normalize Sarvam language configuration
        normalized_code = (self.language_code or "unknown").strip()
        if self.enable_language_detection:
            normalized_code = "unknown"
        elif normalized_code not in SARVAM_SUPPORTED_LANGUAGE_CODES:
            logger.warning(
                "Language code '%s' is not supported by Saarika. "
                "Defaulting to 'en-IN'. Refer to Sarvam docs for valid codes.",
                normalized_code,
            )
            normalized_code = "en-IN"
        self.language_code = normalized_code

        # Validate detection thresholds
        if self.max_buffer_ms < 1000:
            logger.warning("max_buffer_ms too low (%s). Using 3000ms.", self.max_buffer_ms)
            self.max_buffer_ms = 3000
        
        # Log configuration if verbose
        if self.verbose:
            logger.info(
                f" VADConfig loaded: model={self.model_name}, "
                f"sample_rate={self.sample_rate}Hz, timeout={self.initial_timeout_s}s, "
                f"language={self.language_code}, streaming_vad={self.sarvam_high_vad_sensitivity}"
            )
