"""
Mock TTS Provider

Extracted from leibniz_tts.py (MockTTSProvider, lines 951-1018) for microservice deployment.

Implements mock TTS provider for testing without real API calls:
    - Generates silent audio (zeros)
    - Configurable duration based on text length
    - WAV format output
    - No external dependencies

Reference:
    leibniz_agent/leibniz_tts.py - Original MockTTSProvider (lines 951-1018)
"""

import struct
import logging
from typing import AsyncIterator, Dict, Any, Optional, Tuple

# Import from parent directory (works in Docker)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TTSConfig

logger = logging.getLogger(__name__)


class MockTTSProvider:
    """
    Mock TTS provider for testing without real API calls.
    
    Features:
        - Generates silent audio (all zeros)
        - Duration based on text length (~100ms per character)
        - WAV format output (PCM 16-bit, mono)
        - No external dependencies
        - No API keys required
    
    Usage:
        - Unit tests without API costs
        - Development without API access
        - Load testing (predictable latency)
        - CI/CD pipelines
    
    Audio Characteristics:
        - Silent (amplitude = 0)
        - Duration: len(text) * 0.1 seconds (100ms per char)
        - Sample Rate: 24000 Hz (config.sample_rate)
        - Format: WAV PCM 16-bit mono
    """
    
    def __init__(self, config: TTSConfig):
        """
        Initialize Mock TTS provider.
        
        Args:
            config: TTSConfig (uses sample_rate only, no API keys needed)
        """
        self.config = config
        logger.info(" Mock TTS initialized (silent audio generation)")
    
    def _generate_silent_wav(self, duration_seconds: float) -> bytes:
        """
        Generate silent WAV audio of specified duration.
        
        Creates PCM 16-bit audio with all samples = 0 (silence).
        
        Args:
            duration_seconds: Duration in seconds
        
        Returns:
            WAV file bytes (silent audio)
        """
        sample_rate = self.config.sample_rate
        num_samples = int(duration_seconds * sample_rate)
        num_channels = 1  # Mono
        bytes_per_sample = 2  # 16-bit = 2 bytes
        
        # Generate silent PCM data (all zeros)
        pcm_data = b'\x00' * (num_samples * bytes_per_sample)
        
        # Calculate WAV header values
        byte_rate = sample_rate * num_channels * bytes_per_sample
        block_align = num_channels * bytes_per_sample
        data_size = len(pcm_data)
        file_size = 36 + data_size
        
        # Build WAV header (44 bytes)
        wav_header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',          # ChunkID
            file_size,        # ChunkSize
            b'WAVE',          # Format
            b'fmt ',          # Subchunk1ID
            16,               # Subchunk1Size (PCM)
            1,                # AudioFormat (PCM = 1)
            num_channels,     # NumChannels
            sample_rate,      # SampleRate
            byte_rate,        # ByteRate
            block_align,      # BlockAlign
            16,               # BitsPerSample
            b'data',          # Subchunk2ID
            data_size         # Subchunk2Size
        )
        
        return wav_header + pcm_data
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: str = "neutral",
        **kwargs
    ) -> bytes:
        """
        Generate mock audio (silent WAV).
        
        Duration is based on text length: ~100ms per character.
        Example: "Hello world" (11 chars) = 1.1 seconds of silence
        
        Args:
            text: Text to "synthesize" (only used for duration calculation)
            voice: Ignored (mock has no voices)
            language: Ignored (mock is language-agnostic)
            emotion: Ignored (mock has no emotion)
            **kwargs: Additional parameters (ignored)
        
        Returns:
            bytes: Silent WAV audio data
        """
        # Calculate duration: 100ms per character
        duration_seconds = len(text) * 0.1
        
        # Generate silent WAV
        wav_audio = self._generate_silent_wav(duration_seconds)
        
        logger.debug(f" Mock TTS generated {len(text)} chars → {duration_seconds:.1f}s silence ({len(wav_audio)} bytes)")
        return wav_audio
    
    async def stream_synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: str = "neutral",
        **kwargs
    ) -> AsyncIterator[bytes]:
        """
        Generate mock streaming audio (silent WAV chunks).
        
        Yields audio in 1-second chunks for streaming simulation.
        
        Args:
            text: Text to "synthesize"
            voice: Ignored
            language: Ignored
            emotion: Ignored
            **kwargs: Ignored
        
        Yields:
            bytes: Silent WAV chunks (1 second each)
        """
        # Calculate total duration
        duration_seconds = len(text) * 0.1
        
        # Yield 1-second chunks
        chunk_duration = 1.0
        chunks_count = int(duration_seconds / chunk_duration) + 1
        
        for i in range(chunks_count):
            # Last chunk may be shorter
            if i == chunks_count - 1:
                remaining = duration_seconds - (i * chunk_duration)
                chunk = self._generate_silent_wav(remaining)
            else:
                chunk = self._generate_silent_wav(chunk_duration)
            
            yield chunk
    
    def get_available_voices(self) -> Dict[str, Any]:
        """
        Get list of available mock voices.
        
        Returns:
            Dict with single mock voice
        
        Example:
            {
                "mock_voice": {
                    "name": "Mock Voice",
                    "language": "all",
                    "gender": "neutral",
                    "description": "Silent mock audio for testing"
                }
            }
        """
        return {
            "mock_voice": {
                "name": "Mock Voice",
                "language": "all",
                "gender": "neutral",
                "description": "Silent mock audio for testing"
            }
        }
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """
        Validate mock TTS configuration.
        
        Mock provider has no requirements, always valid.
        
        Returns:
            (True, None) - always valid
        """
        return True, None
