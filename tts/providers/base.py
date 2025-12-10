"""
TTS Provider Base Protocol

Extracted from leibniz_tts.py for microservice deployment.

Defines the unified interface all TTS providers must implement.

Reference:
    leibniz_agent/leibniz_tts.py - Original provider classes (lines 473-1372)
"""

from typing import Protocol, AsyncIterator, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TTSProvider(Protocol):
    """
    Unified interface for TTS providers.
    
    All providers (Google, ElevenLabs, Gemini, XTTS, Mock) implement this protocol
    to enable polymorphic usage and seamless fallback.
    
    Methods:
        synthesize: Generate complete audio file (blocking until complete)
        stream_synthesize: Generate streaming audio chunks (for real-time playback)
        get_available_voices: List available voice options for this provider
        validate_config: Check if provider is properly configured
    """
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: str = "neutral",
        **kwargs
    ) -> bytes:
        """
        Synthesize text to complete audio file.
        
        Args:
            text: Text to synthesize (may contain SSML for Google provider)
            voice: Voice name/ID (provider-specific, uses config default if None)
            language: Language code (e.g., "en-US", uses config default if None)
            emotion: Emotion descriptor (helpful, excited, calm, neutral)
            **kwargs: Provider-specific parameters (pitch, speaking_rate, stability, etc.)
        
        Returns:
            bytes: Complete WAV audio data (PCM 16-bit, sample_rate from config)
        
        Raises:
            Exception: On synthesis failure (network, API error, invalid config)
        """
        ...
    
    async def stream_synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: str = "neutral",
        **kwargs
    ) -> AsyncIterator[bytes]:
        """
        Synthesize text to streaming audio chunks.
        
        Yields audio chunks as they become available for real-time playback.
        Not all providers support streaming (e.g., Google returns complete audio).
        
        Args:
            text: Text to synthesize
            voice: Voice name/ID
            language: Language code
            emotion: Emotion descriptor
            **kwargs: Provider-specific parameters
        
        Yields:
            bytes: Audio chunks (WAV format, chunk size varies by provider)
        
        Raises:
            Exception: On synthesis failure or if streaming not supported
        """
        ...
    
    def get_available_voices(self) -> Dict[str, Any]:
        """
        Get list of available voices for this provider.
        
        Returns:
            Dict mapping voice IDs to metadata:
                {
                    "voice_id": {
                        "name": str,
                        "language": str,
                        "gender": str,
                        "description": str
                    },
                    ...
                }
        
        Examples:
            Google: {"en-US-Neural2-F": {"name": "Female Neural 2", ...}}
            ElevenLabs: {"AnvlJBAqSLDzEevYr9Ap": {"name": "Rachel", ...}}
            Gemini: {"Callirrhoe": {"name": "Callirrhoe (female)", ...}}
        """
        ...
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """
        Validate provider configuration (API keys, credentials, etc.).
        
        Returns:
            Tuple of (is_valid, error_message):
                - (True, None) if configuration is valid
                - (False, "error message") if configuration is invalid
        
        Examples:
            Google: Check GOOGLE_APPLICATION_CREDENTIALS exists
            ElevenLabs: Check ELEVENLABS_API_KEY is set
            Gemini: Check GEMINI_API_KEY is set
            XTTS: Check speaker_sample file exists
        """
        ...
