"""
TTS Synthesizer - Multi-Provider Text-to-Speech Service

Implements streaming TTS with LemonFox, Google Cloud, ElevenLabs, and Gemini providers.
Supports emotion modulation, caching, and fallback logic.

Reference:
    leibniz_agent/leibniz_tts.py - Monolith implementation
"""

import asyncio
import hashlib
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import TTSConfig
from .audio_cache import AudioCache
from .providers.base import TTSProvider

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Result of TTS synthesis"""
    audio_path: str
    duration_ms: float
    provider: str
    cached: bool
    metadata: Dict[str, Any]


class TTSSynthesizer:
    """
    Multi-provider TTS synthesizer with caching and streaming support.

    Ported from leibniz_tts.py LeibnizTTS class.
    Adapted for microservice context with relative imports.
    """

    def __init__(self, config: TTSConfig, cache: Optional[AudioCache] = None):
        self.config = config
        self.cache = cache or AudioCache()
        self.providers: Dict[str, TTSProvider] = {}
        
        # Set provider priority based on configuration
        if config.provider and config.provider != "auto":
            # Use only the specified provider (no fallbacks)
            self.provider_priority = [config.provider]
        else:
            # Use full priority list with fallbacks
            self.provider_priority = ["lemonfox", "google", "elevenlabs", "gemini", "mock"]
        
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "provider_usage": {},
            "errors": {}
        }

        # Initialize providers
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize TTS providers based on configuration"""
        try:
            # LemonFox provider
            if self.config.lemonfox_api_key:
                from .providers.lemonfox import LemonFoxTTSProvider
                self.providers["lemonfox"] = LemonFoxTTSProvider(
                    config=self.config
                )
                logger.info(" LemonFox provider initialized")
        except Exception as e:
            logger.warning(f"️ LemonFox provider initialization failed: {e}")

        try:
            # Google Cloud provider
            if self.config.google_credentials_path and os.path.exists(self.config.google_credentials_path):
                from .providers.google import GoogleTTSProvider
                self.providers["google"] = GoogleTTSProvider(
                    credentials_path=self.config.google_credentials_path,
                    voice_name=self.config.google_voice_name
                )
                logger.info(" Google Cloud provider initialized")
        except Exception as e:
            logger.warning(f"️ Google provider initialization failed: {e}")

        # ElevenLabs provider REMOVED - keeping only LemonFox
        # try:
        #     # ElevenLabs provider
        #     if self.config.elevenlabs_api_key:
        #         from .providers.elevenlabs import ElevenLabsProvider
        #         self.providers["elevenlabs"] = ElevenLabsProvider(
        #             api_key=self.config.elevenlabs_api_key,
        #             voice_id=self.config.elevenlabs_voice_id
        #         )
        #         logger.info(" ElevenLabs provider initialized")
        # except Exception as e:
        #     logger.warning(f"️ ElevenLabs provider initialization failed: {e}")

        try:
            # Gemini provider
            if self.config.gemini_api_key:
                from .providers.gemini import GeminiTTSProvider
                self.providers["gemini"] = GeminiTTSProvider(
                    api_key=self.config.gemini_api_key,
                    voice_name=self.config.gemini_voice_name
                )
                logger.info(" Gemini provider initialized")
        except Exception as e:
            logger.warning(f"️ Gemini provider initialization failed: {e}")

        # Mock provider (always available for testing)
        try:
            from .providers.mock import MockTTSProvider
            self.providers["mock"] = MockTTSProvider()
            logger.info(" Mock provider initialized")
        except Exception as e:
            logger.warning(f"️ Mock provider initialization failed: {e}")

        if not self.providers:
            logger.error(" No TTS providers available!")
            raise RuntimeError("No TTS providers could be initialized")

        logger.info(f" Initialized {len(self.providers)} TTS providers: {list(self.providers.keys())}")

    def _get_cache_key(self, text: str, voice: str, language: str, provider: str, emotion: str) -> str:
        """Generate cache key for audio content"""
        content = f"{text}|{voice}|{language}|{provider}|{emotion}"
        return hashlib.md5(content.encode()).hexdigest()

    def _apply_emotion_modulation(self, audio_data: np.ndarray, emotion: str) -> np.ndarray:
        """Apply emotion-based audio modulation"""
        if emotion == "excited":
            # Increase pitch and speed slightly
            # Note: This is a simplified implementation
            # Real emotion modulation would use more sophisticated DSP
            return audio_data * 1.1  # Slight volume boost
        elif emotion == "calm":
            # Reduce volume slightly for calmer effect
            return audio_data * 0.9
        elif emotion == "helpful":
            # Default modulation
            return audio_data
        else:
            return audio_data

    async def synthesize_to_file(
        self,
        text: str,
        emotion: str = "helpful",
        voice: str = None,
        language: str = "en-US",
        output_path: str = None
    ) -> SynthesisResult:
        """
        Synthesize text to audio file with caching and fallback.

        Args:
            text: Text to synthesize
            emotion: Emotion for voice modulation
            voice: Voice identifier (provider-specific)
            language: Language code
            output_path: Custom output path

        Returns:
            SynthesisResult with audio metadata
        """
        self.stats["total_requests"] += 1
        start_time = time.time()

        try:
            # Validate inputs
            if not text or not text.strip():
                raise ValueError("Empty text provided")

            # Get default voice if not specified
            if not voice:
                voice = self._get_default_voice()

            # Try providers in priority order
            for provider_name in self.provider_priority:
                if provider_name not in self.providers:
                    continue

                provider = self.providers[provider_name]

                try:
                    # Check cache first
                    cache_key = self._get_cache_key(text, voice, language, provider_name, emotion)
                    cached_path = self.cache.get(cache_key)

                    if cached_path and os.path.exists(cached_path):
                        self.stats["cache_hits"] += 1
                        # Load cached audio to get duration
                        audio_data, sample_rate = sf.read(cached_path)
                        duration_ms = len(audio_data) / sample_rate * 1000

                        return SynthesisResult(
                            audio_path=cached_path,
                            duration_ms=duration_ms,
                            provider=provider_name,
                            cached=True,
                            metadata={
                                "cache_key": cache_key,
                                "processing_time": time.time() - start_time
                            }
                        )

                    # Cache miss - synthesize
                    self.stats["cache_misses"] += 1

                    # Generate output path
                    if not output_path:
                        output_path = self.cache.get_cache_path(cache_key, ".wav")

                    # Synthesize audio
                    audio_path, metadata = await provider.synthesize(
                        text=text,
                        voice=voice,
                        language=language,
                        output_path=output_path
                    )

                    # Apply emotion modulation if supported
                    if emotion != "helpful":
                        audio_data, sample_rate = sf.read(audio_path)
                        modulated_audio = self._apply_emotion_modulation(audio_data, emotion)
                        sf.write(audio_path, modulated_audio, sample_rate)

                    # Load to get duration
                    audio_data, sample_rate = sf.read(audio_path)
                    duration_ms = len(audio_data) / sample_rate * 1000

                    # Cache the result
                    self.cache.put(cache_key, audio_path)

                    # Update stats
                    self.stats["provider_usage"][provider_name] = self.stats["provider_usage"].get(provider_name, 0) + 1

                    processing_time = time.time() - start_time
                    logger.info(
                        f" Synthesized {len(text)} chars with {provider_name} "
                        f"in {processing_time:.2f}s (cached: False)"
                    )

                    return SynthesisResult(
                        audio_path=audio_path,
                        duration_ms=duration_ms,
                        provider=provider_name,
                        cached=False,
                        metadata={
                            "cache_key": cache_key,
                            "processing_time": processing_time,
                            **metadata
                        }
                    )

                except Exception as e:
                    error_msg = f"{provider_name} synthesis failed: {e}"
                    logger.warning(f"️ {error_msg}")
                    self.stats["errors"][provider_name] = self.stats["errors"].get(provider_name, 0) + 1
                    continue

            # All providers failed
            raise RuntimeError("All TTS providers failed")

        except Exception as e:
            logger.error(f" TTS synthesis failed: {e}")
            raise

    async def synthesize_streaming(
        self,
        text: str,
        emotion: str = "helpful",
        voice: str = None,
        language: str = "en-US"
    ):
        """
        Synthesize text to audio stream with chunking.

        Yields audio chunks for streaming playback.

        Args:
            text: Text to synthesize
            emotion: Emotion for voice modulation
            voice: Voice identifier
            language: Language code

        Yields:
            tuple: (audio_chunk: bytes, is_final: bool)
        """
        # For streaming, we'll synthesize the full audio first
        # In a production implementation, this would synthesize in chunks
        result = await self.synthesize_to_file(text, emotion, voice, language)

        # Read audio file and yield chunks
        audio_data, sample_rate = sf.read(result.audio_path, dtype=np.int16)

        # Convert to bytes
        audio_bytes = audio_data.tobytes()

        # Yield in chunks (50ms chunks at 16kHz = 800 samples = 1600 bytes for int16)
        chunk_size = 1600
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            is_final = (i + chunk_size) >= len(audio_bytes)
            yield chunk, is_final

        # Clean up temp file if not cached
        if not result.cached:
            try:
                os.unlink(result.audio_path)
            except:
                pass

    def _get_default_voice(self) -> str:
        """Get default voice from highest priority available provider"""
        for provider_name in self.provider_priority:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                return provider.default_voice

        return "default"

    def get_provider_status(self) -> Dict[str, bool]:
        """Get availability status of all providers"""
        return {name: True for name in self.providers.keys()}

    def get_stats(self) -> Dict[str, Any]:
        """Get synthesizer statistics"""
        stats = self.stats.copy()
        stats["available_providers"] = list(self.providers.keys())
        stats["cache_stats"] = self.cache.get_stats() if self.cache else {}
        return stats

    async def cleanup(self):
        """Cleanup resources"""
        if self.cache:
            await self.cache.cleanup()

        # Close provider connections if needed
        for provider in self.providers.values():
            if hasattr(provider, 'close'):
                try:
                    await provider.close()
                except:
                    pass