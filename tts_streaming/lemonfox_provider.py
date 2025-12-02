"""
LemonFox TTS Provider

Hardcoded LemonFox API client with emotion-to-speed mapping.
Simplified from services/tts/providers/lemonfox.py and leibniz_tts.py.
"""

import asyncio
import logging
import unicodedata
from typing import Optional, Tuple

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class LemonFoxProvider:
    """
    LemonFox.ai TTS Provider with hardcoded API endpoint.
    
    Features:
    - Hardcoded API endpoint: https://api.lemonfox.ai/v1/audio/speech
    - Emotion-to-speed mapping
    - Async aiohttp client
    - WAV output format (24kHz mono)
    """
    
    API_ENDPOINT = "https://api.lemonfox.ai/v1/audio/speech"  # Hardcoded
    
    def __init__(self, api_key: str, voice: str = "sarah", language: str = "en-us"):
        """
        Initialize LemonFox TTS provider.
        
        Args:
            api_key: LemonFox API key
            voice: Voice name (sarah, heart, bella, michael, alloy, nova, echo, onyx)
            language: Language code (en-us, de-de, es-es, fr-fr, it-it, pl-pl, pt-br, nl-nl)
            
        Raises:
            ValueError: If API key is not set
            ImportError: If aiohttp is not installed
        """
        if not api_key:
            raise ValueError(
                "LemonFox API key not set. Set LEMONFOX_API_KEY environment variable."
            )
        
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp not installed. Install with: pip install aiohttp"
            )
        
        self.api_key = api_key
        self.voice = voice
        self.language = language
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(
            f"LemonFox TTS initialized (voice: {voice}, language: {language})"
        )
    
    async def _ensure_session(self):
        """Create aiohttp session if not exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: Optional[str] = None
    ) -> bytes:
        """
        Synthesize text to speech using LemonFox API.
        
        Args:
            text: Text to synthesize
            voice: Voice name (defaults to config voice)
            language: Language code (defaults to config language)
            emotion: Emotion for speed modulation
            
        Returns:
            WAV audio bytes (24kHz mono)
            
        Raises:
            aiohttp.ClientError: On API request failure
            asyncio.TimeoutError: On request timeout
        """
        text = self._sanitize_text(text)
        await self._ensure_session()
        
        # Map emotion to speed
        speed = self._get_speed_from_emotion(emotion) if emotion else 1.0
        
        # Build request payload
        payload = {
            "input": text,
            "voice": voice or self.voice,
            "response_format": "wav",
            "speed": speed,
            "language": language or self.language
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.debug(
            f"LemonFox TTS: {len(text)} chars, voice={payload['voice']}, "
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
                    f"LemonFox TTS synthesized {len(text)} chars → "
                    f"{len(wav_audio)} bytes ({len(wav_audio)/1024:.1f} KB)"
                )
                
                return wav_audio
                
        except asyncio.TimeoutError:
            logger.error("LemonFox API timeout (30s)")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"LemonFox API error: {e}")
            raise
    
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
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize input text for TTS synthesis.
        
        Args:
            text: Raw input text
            
        Returns:
            Sanitized text safe for TTS
        """
        # Normalize Unicode (NFKC)
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters (except whitespace)
        text = ''.join(
            char for char in text 
            if unicodedata.category(char)[0] != 'C' or char in '\t\n\r'
        )
        
        # Remove zero-width characters
        invisible_chars = {'\u200B', '\u200C', '\u200D', '\u200E', '\u200F', '\uFEFF'}
        text = ''.join(char for char in text if char not in invisible_chars)
        
        # Enforce length limits (LemonFox API limit ~5000 chars, conservative)
        max_length = 4000
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated to {max_length} characters")
        
        # Ensure minimum length
        text = text.strip()
        if not text:
            text = "Hello"  # Fallback for empty text
        
        return text
    
    async def close(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("LemonFox session closed")
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """
        Validate provider configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.api_key or len(self.api_key) < 10:
            return (
                False,
                "Invalid LemonFox API key (set LEMONFOX_API_KEY environment variable)"
            )
        
        return (True, None)


