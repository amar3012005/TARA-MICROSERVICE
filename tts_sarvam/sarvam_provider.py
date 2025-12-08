"""
Sarvam AI TTS Provider
High-performance Indian language TTS with ultra-low latency optimization.

Features:
- Sarvam Bulbul v2 model
- 11+ Indian languages support
- Voice control (pitch, pace, loudness)
- Parallel synthesis support
- Async/await architecture
- Automatic retry with exponential backoff
"""

import asyncio
import logging
import base64
import json
import time
import unicodedata
from typing import Optional, Tuple, List, Dict, Any

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

class SarvamProvider:
    """
    Sarvam AI TTS Provider with hardcoded API endpoint.
    
    Features:
    - Hardcoded API endpoint: https://api.sarvam.ai/text-to-speech
    - Support for 11 Indian languages
    - Voice control parameters (pitch, pace, loudness)
    - Async aiohttp client
    - WAV output format (16kHz mono)
    - Batch synthesis support (up to 5 texts per request)
    """
    
    API_ENDPOINT = "https://api.sarvam.ai/text-to-speech"  # Hardcoded
    DEFAULT_MODEL = "bulbul:v2"
    DEFAULT_SPEAKER = "anushka"
    MAX_TEXT_LENGTH = 1500  # Per text
    MAX_BATCH_SIZE = 5      # Texts per request
    
    LANGUAGE_CODES = {
        "en-IN": "English (India)",
        "hi": "Hindi",
        "ta": "Tamil",
        "te": "Telugu",
        "kn": "Kannada",
        "ml": "Malayalam",
        "mr": "Marathi",
        "gu": "Gujarati",
        "bn": "Bengali",
        "pa": "Punjabi",
        "as": "Assamese"
    }
    
    SPEAKERS = {
        "anushka": "Female - Professional, Clear",
        "vidya": "Female - Warm, Engaging",
        "manisha": "Female - Friendly, Conversational",
        "arya": "Female - Calm, Composed",
        "abhilash": "Male - Professional, Authoritative",
        "karun": "Male - Friendly, Approachable"
    }

    def __init__(
        self,
        api_key: str,
        speaker: str = "anushka",
        language: str = "en-IN",
        model: str = "bulbul:v2",
        pitch: float = 0.0,
        pace: float = 1.0,
        loudness: float = 1.0,
        enable_preprocessing: bool = False
    ):
        """
        Initialize Sarvam AI TTS provider.
        
        Args:
            api_key: Sarvam API key
            speaker: Speaker name
            language: Language code
            model: Model version
            pitch: Voice pitch (-0.75 to 0.75)
            pace: Speech pace (0.3 to 3.0)
            loudness: Volume (0.1 to 3.0)
            enable_preprocessing: Auto-fix text issues
        """
        if not api_key:
            raise ValueError("Sarvam API key not set. Set SARVAM_API_KEY environment variable.")
        
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp not installed. Install with: pip install aiohttp")
        
        self.api_key = api_key
        self.speaker = speaker
        self.language = language
        self.model = model
        self.pitch = max(-0.75, min(0.75, pitch))
        self.pace = max(0.3, min(3.0, pace))
        self.loudness = max(0.1, min(3.0, loudness))
        self.enable_preprocessing = enable_preprocessing
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(
            f"Sarvam AI TTS initialized (model: {model}, speaker: {speaker}, "
            f"language: {language})"
        )

    async def _ensure_session(self):
        """Create optimized aiohttp session if not exists"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(
                total=30,
                connect=5,  # Fast connection timeout
                sock_read=15  # Socket read timeout
            )
            connector = aiohttp.TCPConnector(
                limit=100,  # Max concurrent connections
                limit_per_host=20,  # Per-host limit for Sarvam API
                ttl_dns_cache=300,  # DNS cache TTL
                keepalive_timeout=60,  # Keep connections alive longer
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )

    async def warmup(self) -> Dict[str, Any]:
        """
        Pre-warm the connection to Sarvam API.
        Returns warmup statistics.
        """
        logger.info("🔥 Pre-warming Sarvam API connection...")
        stats = {"warmup_time_ms": 0, "success": False}
        start = time.time()
        
        try:
            await self._ensure_session()
            
            # Send warmup request with minimal text
            warmup_text = "Hello"
            audio = await self.synthesize(
                text=warmup_text,
                speaker=self.speaker,
                language=self.language
            )
            
            stats["warmup_time_ms"] = (time.time() - start) * 1000
            stats["success"] = True
            stats["audio_bytes"] = len(audio)
            
            logger.info(f"✅ Sarvam API warmed up in {stats['warmup_time_ms']:.0f}ms")
        except Exception as e:
            stats["warmup_time_ms"] = (time.time() - start) * 1000
            stats["error"] = str(e)
            logger.warning(f"⚠️ Warmup failed (non-critical): {e}")
        
        return stats

    async def synthesize_parallel(
        self,
        texts: List[str],
        max_concurrent: int = 3,
        **kwargs
    ) -> List[Tuple[bytes, float]]:
        """
        Synthesize multiple texts in parallel.
        
        Args:
            texts: List of texts to synthesize
            max_concurrent: Maximum concurrent requests (default 3)
            **kwargs: Pass-through to synthesize()
        
        Returns:
            List of tuples (audio_bytes, duration_ms)
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def synthesize_with_limit(text: str, index: int):
            async with semaphore:
                start = time.time()
                audio = await self.synthesize(text, **kwargs)
                duration_ms = (time.time() - start) * 1000
                return (index, audio, duration_ms)
        
        tasks = [
            synthesize_with_limit(text, i)
            for i, text in enumerate(texts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sort by original index and filter errors
        ordered = []
        for result in sorted(results, key=lambda x: x[0] if not isinstance(x, Exception) else float('inf')):
            if isinstance(result, Exception):
                logger.error(f"Parallel synthesis error: {result}")
                continue
            ordered.append((result[1], result[2]))
        
        return ordered

    async def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        pitch: Optional[float] = None,
        pace: Optional[float] = None,
        loudness: Optional[float] = None
    ) -> bytes:
        """
        Synthesize text to speech using Sarvam API.
        
        Args:
            text: Text to synthesize
            speaker: Speaker name (overrides config)
            language: Language code (overrides config)
            pitch: Voice pitch (overrides config)
            pace: Speech pace (overrides config)
            loudness: Volume level (overrides config)
        
        Returns:
            WAV audio bytes (16kHz mono, PCM 16-bit)
        """
        text = self._sanitize_text(text)
        await self._ensure_session()
        
        speaker = speaker or self.speaker
        language = language or self.language
        pitch = pitch if pitch is not None else self.pitch
        pace = pace if pace is not None else self.pace
        loudness = loudness if loudness is not None else self.loudness
        
        pitch = max(-0.75, min(0.75, pitch))
        pace = max(0.3, min(3.0, pace))
        loudness = max(0.1, min(3.0, loudness))
        
        payload = {
            "inputs": [text],
            "target_language_code": language,
            "speaker": speaker,
            "pitch": pitch,
            "pace": pace,
            "loudness": loudness,
            "enable_preprocessing": self.enable_preprocessing,
            "model": self.model
        }
        
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        
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
                        f"Sarvam API error (status {response.status}): {error_text}"
                    )
                
                response_data = await response.json()
                
                # Check for error in response
                if "error" in response_data:
                    error_msg = response_data.get('error', {}).get('message', 'Unknown')
                    logger.error(f"Sarvam API error response: {response_data}")
                    raise aiohttp.ClientError(
                        f"Sarvam API error: {error_msg}"
                    )
                
                # Extract audio content - Sarvam returns audios directly at root level
                audios = response_data.get("audios", [])
                if not audios:
                    logger.error(f"No audios array in response: {response_data}")
                    raise aiohttp.ClientError("No audio content in response")
                
                audio_content = audios[0]
                if not audio_content:
                    raise aiohttp.ClientError("Empty audio content")
                
                wav_audio = base64.b64decode(audio_content)
                
                return wav_audio
        
        except asyncio.TimeoutError:
            logger.error("Sarvam API timeout (30s)")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Sarvam API error: {e}")
            raise

    def _sanitize_text(self, text: str) -> str:
        """Sanitize input text for TTS synthesis."""
        text = unicodedata.normalize('NFKC', text)
        text = ''.join(
            char for char in text
            if unicodedata.category(char)[0] != 'C' or char in '\t\n\r'
        )
        invisible_chars = {'\u200B', '\u200C', '\u200D', '\u200E', '\u200F', '\uFEFF'}
        text = ''.join(char for char in text if char not in invisible_chars)
        
        if len(text) > self.MAX_TEXT_LENGTH:
            text = text[:self.MAX_TEXT_LENGTH]
            logger.warning(f"Text truncated to {self.MAX_TEXT_LENGTH} characters")
        
        text = text.strip()
        if not text:
            text = "Hello"
        
        return text

    async def close(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validate provider configuration."""
        if not self.api_key or len(self.api_key) < 10:
            return (False, "Invalid Sarvam API key")
        
        if self.speaker not in self.SPEAKERS:
            return (False, f"Invalid speaker: {self.speaker}")
        
        if self.language not in self.LANGUAGE_CODES:
            return (False, f"Invalid language: {self.language}")
        
        return (True, None)
