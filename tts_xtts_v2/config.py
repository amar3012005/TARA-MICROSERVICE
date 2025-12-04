"""Configuration for XTTS-v2 streaming microservice."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class XTTSStreamingConfig:
    """Environment-driven configuration for XTTS microservice."""

    xtts_model_dir: Path
    xtts_device: str
    default_language: str
    default_speaker_wav: Path
    default_voice_id: str
    sample_rate: int
    stream_chunk_tokens: int
    max_buffer_chunks: int
    fastrtc_chunk_duration_ms: int
    enable_cache: bool
    cache_dir: Path
    max_cache_size: int
    cache_provider_name: str = "xtts_v2"
    port: int = 8005

    @staticmethod
    def from_env() -> "XTTSStreamingConfig":
        model_dir = Path(
            os.getenv(
                "XTTS_MODEL_DIR",
                str(Path.home() / ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"),
            )
        )
        speaker_wav = Path(
            os.getenv(
                "LEIBNIZ_XTTS_SPEAKER_WAV",
                "/app/leibniz_agent/services/ElevenLabs_enigma.wav",
            )
        )

        return XTTSStreamingConfig(
            xtts_model_dir=model_dir,
            xtts_device=os.getenv("XTTS_DEVICE", "cuda"),
            default_language=os.getenv("LEIBNIZ_XTTS_LANGUAGE", "en"),
            default_speaker_wav=speaker_wav,
            default_voice_id=os.getenv("LEIBNIZ_XTTS_VOICE_ID", "xtts_voice_default"),
            sample_rate=int(os.getenv("LEIBNIZ_XTTS_SAMPLE_RATE", "24000")),
            stream_chunk_tokens=int(os.getenv("LEIBNIZ_XTTS_STREAM_CHUNK_TOKENS", "20")),
            max_buffer_chunks=int(os.getenv("LEIBNIZ_XTTS_MAX_BUFFER_CHUNKS", "8")),
            fastrtc_chunk_duration_ms=int(os.getenv("LEIBNIZ_TTS_FASTRTC_CHUNK_MS", "40")),
            enable_cache=os.getenv("LEIBNIZ_TTS_CACHE_ENABLED", "true").lower() == "true",
            cache_dir=Path(os.getenv("LEIBNIZ_TTS_CACHE_DIR", "/app/audio_cache")),
            max_cache_size=int(os.getenv("LEIBNIZ_TTS_CACHE_MAX_SIZE", "500")),
            port=int(os.getenv("TTS_STREAMING_PORT", "8005")),
        )

    def __post_init__(self):
        if not self.default_speaker_wav.exists():
            logger.warning(
                "Default speaker WAV %s does not exist. Set LEIBNIZ_XTTS_SPEAKER_WAV",
                self.default_speaker_wav,
            )

        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")

        if self.stream_chunk_tokens <= 0:
            raise ValueError("Stream chunk tokens must be positive")

        if self.max_buffer_chunks <= 0:
            raise ValueError("Max buffer chunks must be positive")

        logger.info(
            "XTTS Config | lang=%s | speaker=%s | chunk_tokens=%s | cache=%s",
            self.default_language,
            self.default_speaker_wav,
            self.stream_chunk_tokens,
            self.enable_cache,
        )





