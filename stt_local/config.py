"""
Configuration for STT Local Microservice

Configuration for local STT service using Silero VAD and Faster Whisper.
Loaded from environment variables with LEIBNIZ_STT_LOCAL_* prefix.

Reference:
    services/stt_vad/config.py - VADConfig pattern
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class STTLocalConfig:
    """
    Configuration for STT Local microservice.
    
    Supports GPU acceleration with CUDA and configurable model sizes.
    """
    
    # Audio settings
    sample_rate: int = 16000  # 16kHz for VAD and Whisper (mono)
    channels: int = 1  # Mono audio
    
    # Faster Whisper settings
    whisper_model_size: str = "medium.en"  # medium.en (769M params, English-only, high accuracy)
    whisper_device: str = "cuda"  # cuda or cpu
    whisper_compute_type: str = "float16"  # float16, float32, int8
    whisper_language: str = "en"  # Language code (en, es, fr, etc.)
    whisper_beam_size: int = 5  # Beam search size
    whisper_vad_filter: bool = True  # Use VAD filtering (we use Silero, so False)
    
    # Silero VAD settings
    vad_threshold: float = 0.5  # Speech detection threshold (0.3-0.7)
    vad_min_speech_duration_ms: int = 200  # Minimum speech duration (lowered for responsiveness)
    vad_silence_timeout_ms: int = 1000  # Silence before ending speech (increased for natural pauses)
    vad_model_path: Optional[str] = None  # Optional custom model path
    
    # Streaming settings
    partial_update_interval_ms: int = 250  # Update partial transcripts every 250ms (faster updates)
    min_audio_length_for_stt: float = 0.3  # Minimum audio length for STT (lowered for faster response)
    
    # Timeout settings (seconds)
    initial_timeout_s: float = 20.0
    retry_timeout_s: float = 10.0
    max_timeout_s: float = 30.0
    start_timeout_s: float = 20.0
    
    # Session management
    session_timeout: float = 600.0  # Session expiry (10 minutes)
    max_session_duration: float = 600.0
    
    # Performance settings
    use_gpu: bool = True  # Enable GPU acceleration
    num_workers: int = 1  # Number of worker threads for Whisper
    
    # Verbosity control
    verbose: bool = False
    log_audio_callbacks: bool = False
    log_vad_events: bool = True
    log_state_transitions: bool = True
    
    @staticmethod
    def from_env() -> 'STTLocalConfig':
        """
        Load configuration from environment variables.
        
        Environment variables use LEIBNIZ_STT_LOCAL_* prefix.
        
        Returns:
            STTLocalConfig: Configuration instance loaded from environment
        """
        # Check for GPU availability
        use_gpu = os.getenv("LEIBNIZ_STT_LOCAL_USE_GPU", "true").lower() == "true"
        device = "cuda" if use_gpu else "cpu"
        
        return STTLocalConfig(
            # Audio settings
            sample_rate=int(os.getenv("LEIBNIZ_STT_LOCAL_SAMPLE_RATE", "16000")),
            channels=int(os.getenv("LEIBNIZ_STT_LOCAL_CHANNELS", "1")),
            
            # Whisper settings
            whisper_model_size=os.getenv("LEIBNIZ_STT_LOCAL_WHISPER_MODEL_SIZE", "medium.en"),
            whisper_device=os.getenv("LEIBNIZ_STT_LOCAL_WHISPER_DEVICE", device),
            whisper_compute_type=os.getenv("LEIBNIZ_STT_LOCAL_WHISPER_COMPUTE_TYPE", "float16" if use_gpu else "float32"),
            whisper_language=os.getenv("LEIBNIZ_STT_LOCAL_WHISPER_LANGUAGE", "en"),
            whisper_beam_size=int(os.getenv("LEIBNIZ_STT_LOCAL_WHISPER_BEAM_SIZE", "5")),
            whisper_vad_filter=os.getenv("LEIBNIZ_STT_LOCAL_WHISPER_VAD_FILTER", "false").lower() == "true",
            
            # Silero VAD settings
            vad_threshold=float(os.getenv("LEIBNIZ_STT_LOCAL_VAD_THRESHOLD", "0.5")),
            vad_min_speech_duration_ms=int(os.getenv("LEIBNIZ_STT_LOCAL_VAD_MIN_SPEECH_MS", "250")),
            vad_silence_timeout_ms=int(os.getenv("LEIBNIZ_STT_LOCAL_VAD_SILENCE_TIMEOUT_MS", "800")),
            vad_model_path=os.getenv("LEIBNIZ_STT_LOCAL_VAD_MODEL_PATH", None),
            
            # Streaming settings
            partial_update_interval_ms=int(os.getenv("LEIBNIZ_STT_LOCAL_PARTIAL_UPDATE_INTERVAL_MS", "500")),
            min_audio_length_for_stt=float(os.getenv("LEIBNIZ_STT_LOCAL_MIN_AUDIO_LENGTH", "0.5")),
            
            # Timeout settings
            initial_timeout_s=float(os.getenv("LEIBNIZ_STT_LOCAL_TIMEOUT_INITIAL", "20.0")),
            retry_timeout_s=float(os.getenv("LEIBNIZ_STT_LOCAL_TIMEOUT_RETRY", "10.0")),
            max_timeout_s=float(os.getenv("LEIBNIZ_STT_LOCAL_TIMEOUT_MAX", "30.0")),
            start_timeout_s=float(os.getenv("LEIBNIZ_STT_LOCAL_TIMEOUT_START", "20.0")),
            
            # Session settings
            session_timeout=float(os.getenv("LEIBNIZ_STT_LOCAL_SESSION_TIMEOUT", "600.0")),
            max_session_duration=float(os.getenv("LEIBNIZ_STT_LOCAL_MAX_SESSION_DURATION", "600.0")),
            
            # Performance settings
            use_gpu=use_gpu,
            num_workers=int(os.getenv("LEIBNIZ_STT_LOCAL_NUM_WORKERS", "1")),
            
            # Verbosity
            verbose=os.getenv("LEIBNIZ_STT_LOCAL_VERBOSE", "false").lower() == "true",
            log_audio_callbacks=os.getenv("LEIBNIZ_STT_LOCAL_LOG_AUDIO_CALLBACKS", "false").lower() == "true",
            log_vad_events=os.getenv("LEIBNIZ_STT_LOCAL_LOG_VAD_EVENTS", "true").lower() == "true",
            log_state_transitions=os.getenv("LEIBNIZ_STT_LOCAL_LOG_STATE_TRANSITIONS", "true").lower() == "true",
        )
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        # Validate Whisper model size (includes .en English-only variants)
        valid_sizes = [
            "tiny", "tiny.en", 
            "base", "base.en", 
            "small", "small.en", 
            "medium", "medium.en", 
            "large-v2", "large-v3"
        ]
        if self.whisper_model_size not in valid_sizes:
            logger.warning(f"Invalid whisper_model_size={self.whisper_model_size}, using default 'medium.en'")
            self.whisper_model_size = "medium.en"
        
        # Validate device
        if self.whisper_device not in ["cuda", "cpu"]:
            logger.warning(f"Invalid whisper_device={self.whisper_device}, using 'cpu'")
            self.whisper_device = "cpu"
        
        # Validate compute type
        valid_compute_types = ["float16", "float32", "int8"]
        if self.whisper_compute_type not in valid_compute_types:
            logger.warning(f"Invalid whisper_compute_type={self.whisper_compute_type}, using 'float32'")
            self.whisper_compute_type = "float32"
        
        # Validate VAD threshold
        if not (0.0 <= self.vad_threshold <= 1.0):
            logger.warning(f"Invalid vad_threshold={self.vad_threshold}, using default 0.5")
            self.vad_threshold = 0.5
        
        # Validate timeout values
        if self.initial_timeout_s <= 0:
            logger.warning(f"Invalid initial_timeout_s={self.initial_timeout_s}, using default 20.0")
            self.initial_timeout_s = 20.0
        
        # Log configuration if verbose
        if self.verbose:
            logger.info(
                f"STTLocalConfig loaded: model={self.whisper_model_size}, "
                f"device={self.whisper_device}, sample_rate={self.sample_rate}Hz, "
                f"timeout={self.initial_timeout_s}s"
            )


