"""
TTS (Text-to-Speech) Microservice for Leibniz Agent

This microservice provides multi-provider speech synthesis with emotion modulation
and file-based caching via HTTP REST API.

Features:
- Multi-provider support: Google Cloud TTS, ElevenLabs, Gemini Live, XTTS Local, Mock
- Emotion-based voice modulation (helpful, excited, calm, professional, etc.)
- File-based caching with MD5 keys and LRU cleanup
- Automatic provider fallback with retry logic (3 attempts)
- Streaming synthesis support for real-time playback
- HTTP REST API with FastAPI

Architecture:
    Client (HTTP) → FastAPI → TTSSynthesizer → AudioCache → Providers
    
Providers:
    - Google Cloud TTS: Stable, Neural2 voices, SSML support
    - ElevenLabs: Premium quality, natural voices, streaming
    - Gemini Live: Emotion-aware, voice characters, streaming
    - XTTS Local: Voice cloning, GPU acceleration (requires torch ~2GB)
    - Mock: Silent audio for testing

Reference:
    Cloud Transformation doc (lines 688-742) - Service architecture and specifications
    Main entry point: leibniz_agent/services/tts/app.py
    
Usage:
    from leibniz_agent.services.tts import TTSSynthesizer, TTSConfig, AudioCache, synthesize_speech
"""

from leibniz_agent.services.tts.config import TTSConfig
from leibniz_agent.services.tts.audio_cache import AudioCache
from leibniz_agent.services.tts.tts_synthesizer import TTSSynthesizer

# Convenience async function for direct synthesis
async def synthesize_speech(text: str, emotion: str = "helpful", **kwargs):
    """
    Convenience function for direct speech synthesis.
    
    Args:
        text: Text to synthesize
        emotion: Emotion type (helpful, excited, calm, professional, etc.)
        **kwargs: Additional parameters (provider, voice, cache_name, force_regenerate)
        
    Returns:
        Dict with synthesis result (success, filepath, provider, duration, cache_hit, elapsed)
    """
    config = TTSConfig.from_env()
    synthesizer = TTSSynthesizer(config)
    return await synthesizer.synthesize_to_file(text=text, emotion=emotion, **kwargs)

__all__ = [
    "TTSConfig",
    "AudioCache",
    "TTSSynthesizer",
    "synthesize_speech",
]
