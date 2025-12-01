"""
LemonFox TTS Provider

Implements LemonFox.ai TTS API with async HTTP requests, WAV audio format, 
multilingual support, and speed control.

Extracted from leibniz_tts.py for microservice deployment.

Features:
    - Async HTTP API calls via aiohttp
    - WAV audio output (24kHz)
    - 8+ language support
    - Speed control (0.25-4.0)
    - Emotion-to-speed mapping
    - Multiple voice options

Reference:
    https://www.lemonfox.ai/apis/text-to-speech
"""

import io
import sys
import logging
import asyncio
from pathlib import Path
from typing import AsyncIterator, Dict, Any, Optional, Tuple

# Fix imports for Docker deployment (subdirectory as root)
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TTSConfig

logger = logging.getLogger(__name__)

# Lazy import aiohttp
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("️ aiohttp not available - LemonFox TTS will not work")


class LemonFoxTTSProvider:
    """
    LemonFox.ai TTS provider with multilingual support.
    
    Features:
        - Natural-sounding voices (sarah, heart, bella, michael, etc.)
        - 8+ languages (en-us, de-de, es-es, fr-fr, it-it, pl-pl, pt-br, nl-nl)
        - Speed control for emotion modulation
        - WAV output format (24kHz mono)
        - Async HTTP API
    
    API Documentation:
        https://www.lemonfox.ai/apis/text-to-speech
    """
    
    API_ENDPOINT = "https://api.lemonfox.ai/v1/audio/speech"
    
    def __init__(self, config: TTSConfig):
        """
        Initialize LemonFox TTS provider.
        
        Args:
            config: TTS configuration object
            
        Raises:
            ValueError: If API key is not set
        """
        if not config.lemonfox_api_key:
            raise ValueError(
                "LemonFox API key not set. Set LEMONFOX_API_KEY environment variable "
                "or configure lemonfox_api_key in TTSConfig."
            )
        
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp not installed. Install with: pip install aiohttp"
            )
        
        self.config = config
        self.api_key = config.lemonfox_api_key
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(
            f" LemonFox TTS initialized (voice: {config.lemonfox_voice}, "
            f"language: {config.lemonfox_language})"
        )
    
    async def _ensure_session(self):
        """Create aiohttp session if not exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to speech using LemonFox API.
        
        Args:
            text: Text to synthesize
            voice: Voice name (sarah, heart, bella, michael) - defaults to config
            language: Language code (en-us, de-de, etc.) - defaults to config
            emotion: Emotion for speed modulation (helpful, excited, calm, etc.)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            WAV audio bytes (24kHz mono)
            
        Raises:
            aiohttp.ClientError: On API request failure
            asyncio.TimeoutError: On request timeout
        """
        await self._ensure_session()
        
        # Map emotion to speed
        speed = self._get_speed_from_emotion(emotion) if emotion else 1.0
        
        # Build request payload
        payload = {
            "input": text,
            "voice": voice or self.config.lemonfox_voice,
            "response_format": "wav",  # Request WAV for consistency
            "speed": speed,
            "language": language or self.config.lemonfox_language
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.debug(
            f" LemonFox TTS: {len(text)} chars, voice={payload['voice']}, "
            f"language={payload['language']}, speed={speed:.2f}"
        )
        
        try:
            async with self.session.post(
                self.API_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientError(
                        f"LemonFox API error (status {response.status}): {error_text}"
                    )
                
                wav_audio = await response.read()
                
                logger.debug(
                    f" LemonFox TTS synthesized {len(text)} chars → "
                    f"{len(wav_audio)} bytes ({len(wav_audio)/1024:.1f} KB)"
                )
                
                return wav_audio
                
        except asyncio.TimeoutError:
            logger.error("⏱️ LemonFox API timeout (30s)")
            raise
        except aiohttp.ClientError as e:
            logger.error(f" LemonFox API error: {e}")
            raise
    
    async def stream_synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[bytes]:
        """
        Stream synthesize text (returns complete audio as single chunk).
        
        Note: LemonFox API doesn't support true streaming, so this calls
        synthesize() and yields the complete audio as a single chunk.
        
        Args:
            text: Text to synthesize
            voice: Voice name - defaults to config
            language: Language code - defaults to config
            emotion: Emotion for speed modulation
            **kwargs: Additional parameters
            
        Yields:
            Audio bytes (complete WAV file)
        """
        logger.debug(" LemonFox streaming (single chunk - no true streaming support)")
        
        # Call synthesize to get complete audio
        audio_bytes = await self.synthesize(
            text=text,
            voice=voice,
            language=language,
            emotion=emotion,
            **kwargs
        )
        
        # Yield as single chunk
        yield audio_bytes
    
    def get_available_voices(self) -> Dict[str, Any]:
        """
        Get available voices.
        
        Note: LemonFox doesn't provide a voice listing API endpoint,
        so this returns a hardcoded list of known voices.
        
        Returns:
            Dict mapping voice IDs to voice metadata
        """
        voices = {
            "sarah": {
                "name": "Sarah",
                "language": "multi",
                "gender": "female",
                "description": "Soft, natural female voice"
            },
            "heart": {
                "name": "Heart",
                "language": "multi",
                "gender": "female",
                "description": "Warm, friendly female voice"
            },
            "bella": {
                "name": "Bella",
                "language": "multi",
                "gender": "female",
                "description": "Elegant, refined female voice"
            },
            "michael": {
                "name": "Michael",
                "language": "multi",
                "gender": "male",
                "description": "Professional, clear male voice"
            },
            "alloy": {
                "name": "Alloy",
                "language": "multi",
                "gender": "neutral",
                "description": "Neutral, balanced voice"
            },
            "nova": {
                "name": "Nova",
                "language": "multi",
                "gender": "female",
                "description": "Energetic female voice"
            },
            "echo": {
                "name": "Echo",
                "language": "multi",
                "gender": "male",
                "description": "Deep, resonant male voice"
            },
            "onyx": {
                "name": "Onyx",
                "language": "multi",
                "gender": "male",
                "description": "Strong, authoritative male voice"
            }
        }
        
        logger.info(f" LemonFox TTS: {len(voices)} voices available")
        return voices
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """
        Validate provider configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.api_key or len(self.api_key) < 10:
            return (
                False,
                "lemonfox_api_key not set or invalid (set LEMONFOX_API_KEY environment variable)"
            )
        
        if not AIOHTTP_AVAILABLE:
            return (False, "aiohttp not installed (pip install aiohttp)")
        
        return (True, None)
    
    def _get_speed_from_emotion(self, emotion: str) -> float:
        """
        Map emotion to speech speed.
        
        Args:
            emotion: Emotion string (helpful, excited, calm, etc.)
            
        Returns:
            Speed value (0.25-4.0, 1.0 = normal)
        """
        emotion_speed_map = {
            "helpful": 1.0,
            "excited": 1.2,
            "calm": 0.95,
            "happy": 1.1,
            "neutral": 1.0,
            "professional": 1.0,
            "friendly": 1.05,
            "curious": 1.05,
            "empathetic": 0.95,
            "urgent": 1.25,
            "relaxed": 0.9
        }
        
        speed = emotion_speed_map.get(emotion.lower() if emotion else "neutral", 1.0)
        
        # Clamp to valid range
        return max(0.25, min(4.0, speed))
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug(" LemonFox session closed")
