"""
TTS_LABS Configuration

Configuration dataclass for ElevenLabs TTS streaming service.
Optimized for ultra-low latency with eleven_flash_v2_5 model.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TTSLabsConfig:
    """Configuration for TTS_LABS microservice with ElevenLabs settings"""
    
    # ElevenLabs API Configuration
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel voice
    elevenlabs_model_id: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
    
    # Latency Optimization (0-4, higher = lower latency but potentially lower quality)
    # 0: default, 1: normal, 2: strong, 3: max, 4: max + no text normalizer
    optimize_streaming_latency: int = int(os.getenv("ELEVENLABS_LATENCY_OPTIMIZATION", "4"))
    
    # Output format: pcm_24000 for low latency, mp3_44100_128 for quality
    # PCM is faster to decode but larger; MP3 is compressed but needs decoding
    output_format: str = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_24000")
    
    # Voice settings
    stability: float = float(os.getenv("ELEVENLABS_STABILITY", "0.5"))
    similarity_boost: float = float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75"))
    style: float = float(os.getenv("ELEVENLABS_STYLE", "0.0"))
    use_speaker_boost: bool = os.getenv("ELEVENLABS_SPEAKER_BOOST", "true").lower() == "true"
    
    # Audio settings (derived from output_format)
    @property
    def sample_rate(self) -> int:
        """Extract sample rate from output format"""
        if "pcm_" in self.output_format:
            return int(self.output_format.split("_")[1])
        elif "mp3_" in self.output_format:
            return int(self.output_format.split("_")[1])
        return 24000  # Default
    
    # Queue settings
    queue_max_size: int = 10
    inter_sentence_gap_ms: int = int(os.getenv("LEIBNIZ_TTS_INTER_SENTENCE_GAP_MS", "50"))
    
    # Cache settings
    enable_cache: bool = os.getenv("LEIBNIZ_TTS_CACHE_ENABLED", "true").lower() == "true"
    cache_dir: str = os.getenv("LEIBNIZ_TTS_CACHE_DIR", "/app/audio_cache")
    max_cache_size: int = int(os.getenv("LEIBNIZ_TTS_CACHE_MAX_SIZE", "500"))
    cache_ttl_days: int = int(os.getenv("LEIBNIZ_TTS_CACHE_TTL_DAYS", "30"))
    
    # Streaming settings
    fastrtc_chunk_duration_ms: int = int(os.getenv("LEIBNIZ_TTS_FASTRTC_CHUNK_MS", "40"))
    
    # WebSocket settings
    websocket_timeout: float = float(os.getenv("ELEVENLABS_WS_TIMEOUT", "30.0"))
    reconnect_attempts: int = int(os.getenv("ELEVENLABS_RECONNECT_ATTEMPTS", "3"))
    reconnect_delay: float = float(os.getenv("ELEVENLABS_RECONNECT_DELAY", "1.0"))
    
    # Flush settings for stream-input
    # try_trigger_generation: Flushes buffer after sending, reduces latency for short text
    try_trigger_generation: bool = os.getenv("ELEVENLABS_TRY_TRIGGER_GENERATION", "true").lower() == "true"
    
    # Service settings
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Port
    port: int = int(os.getenv("TTS_LABS_PORT", "8006"))
    
    @staticmethod
    def from_env() -> "TTSLabsConfig":
        """Load configuration from environment variables"""
        return TTSLabsConfig(
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            elevenlabs_model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"),
            optimize_streaming_latency=int(os.getenv("ELEVENLABS_LATENCY_OPTIMIZATION", "4")),
            output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_24000"),
            stability=float(os.getenv("ELEVENLABS_STABILITY", "0.5")),
            similarity_boost=float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75")),
            style=float(os.getenv("ELEVENLABS_STYLE", "0.0")),
            use_speaker_boost=os.getenv("ELEVENLABS_SPEAKER_BOOST", "true").lower() == "true",
            queue_max_size=int(os.getenv("TTS_QUEUE_MAX_SIZE", "10")),
            enable_cache=os.getenv("LEIBNIZ_TTS_CACHE_ENABLED", "true").lower() == "true",
            cache_dir=os.getenv("LEIBNIZ_TTS_CACHE_DIR", "/app/audio_cache"),
            max_cache_size=int(os.getenv("LEIBNIZ_TTS_CACHE_MAX_SIZE", "500")),
            cache_ttl_days=int(os.getenv("LEIBNIZ_TTS_CACHE_TTL_DAYS", "30")),
            timeout=float(os.getenv("LEIBNIZ_TTS_TIMEOUT", "30.0")),
            retry_attempts=int(os.getenv("LEIBNIZ_TTS_RETRY_ATTEMPTS", "3")),
            retry_delay=float(os.getenv("LEIBNIZ_TTS_RETRY_DELAY", "1.0")),
            port=int(os.getenv("TTS_LABS_PORT", "8006")),
            inter_sentence_gap_ms=int(os.getenv("LEIBNIZ_TTS_INTER_SENTENCE_GAP_MS", "50")),
            fastrtc_chunk_duration_ms=int(os.getenv("LEIBNIZ_TTS_FASTRTC_CHUNK_MS", "40")),
            websocket_timeout=float(os.getenv("ELEVENLABS_WS_TIMEOUT", "30.0")),
            reconnect_attempts=int(os.getenv("ELEVENLABS_RECONNECT_ATTEMPTS", "3")),
            reconnect_delay=float(os.getenv("ELEVENLABS_RECONNECT_DELAY", "1.0")),
            try_trigger_generation=os.getenv("ELEVENLABS_TRY_TRIGGER_GENERATION", "true").lower() == "true"
        )
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.elevenlabs_api_key:
            logger.warning("ELEVENLABS_API_KEY not set - service will not function properly")
        
        if self.optimize_streaming_latency < 0 or self.optimize_streaming_latency > 4:
            raise ValueError(f"Invalid optimize_streaming_latency: {self.optimize_streaming_latency} (must be 0-4)")
        
        if self.queue_max_size <= 0:
            raise ValueError(f"Invalid queue_max_size: {self.queue_max_size}")
        
        if self.inter_sentence_gap_ms < 0:
            raise ValueError(f"Invalid inter_sentence_gap_ms: {self.inter_sentence_gap_ms}")
        
        if self.fastrtc_chunk_duration_ms <= 0:
            raise ValueError(f"Invalid fastrtc_chunk_duration_ms: {self.fastrtc_chunk_duration_ms}")
        
        logger.info(
            f"TTS_LABS Config: model={self.elevenlabs_model_id}, voice={self.elevenlabs_voice_id}, "
            f"latency_opt={self.optimize_streaming_latency}, format={self.output_format}, "
            f"sample_rate={self.sample_rate}Hz, cache={self.enable_cache}, "
            f"gap={self.inter_sentence_gap_ms}ms, chunk={self.fastrtc_chunk_duration_ms}ms"
        )
    
    def get_websocket_url(self) -> str:
        """Get the ElevenLabs WebSocket URL for stream-input"""
        base_url = "wss://api.elevenlabs.io/v1/text-to-speech"
        params = f"model_id={self.elevenlabs_model_id}"
        if self.optimize_streaming_latency > 0:
            params += f"&optimize_streaming_latency={self.optimize_streaming_latency}"
        params += f"&output_format={self.output_format}"
        return f"{base_url}/{self.elevenlabs_voice_id}/stream-input?{params}"
    
    def get_voice_settings(self) -> dict:
        """Get voice settings for ElevenLabs API"""
        return {
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost
        }
