"""
pytest Configuration and Fixtures

Provides reusable fixtures for TTS microservice testing:
    - tts_config: TTSConfig with mock provider
    - temp_cache_dir: Temporary directory for cache tests
    - audio_cache: AudioCache instance with temp directory
    - mock_providers: Dictionary of mocked provider instances
    - sample_audio: Sample WAV audio bytes for testing
"""

import os
import io
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from leibniz_agent.services.tts.config import TTSConfig
from leibniz_agent.services.tts.audio_cache import AudioCache


@pytest.fixture
def tts_config():
    """
    TTSConfig with mock provider and test settings.
    
    Uses mock provider to avoid requiring API keys during testing.
    """
    config = TTSConfig(
        provider="mock",
        fallback_provider="mock",
        enable_fallback=True,
        cache_enabled=True,
        cache_dir="./test_cache",
        cache_max_size=100,
        cache_ttl_days=7,
        sample_rate=24000,
        speaking_rate=1.0,
        pitch=0.0,
        timeout=30.0,
        retry_attempts=3,
        retry_delay=1.0,
        # Google Cloud settings (optional)
        google_credentials_path=None,
        google_voice="en-US-Neural2-F",
        # ElevenLabs settings (optional)
        elevenlabs_api_key=None,
        elevenlabs_voice="AnvlJBAqSLDzEevYr9Ap",
        elevenlabs_model=" eleven_turbo_v2_5",
        elevenlabs_stability=0.5,
        elevenlabs_similarity_boost=0.75,
        # Gemini settings (optional)
        gemini_api_key=None,
        gemini_model="gemini-2.0-flash-exp",
        gemini_voice="Callirrhoe",
        gemini_emotion_support=True,
        # XTTS settings (optional)
        xtts_speaker_sample=None,
        xtts_language="en",
        xtts_device="cpu",
    )
    return config


@pytest.fixture
def temp_cache_dir():
    """
    Temporary directory for cache testing.
    
    Automatically cleaned up after test completion.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def audio_cache(temp_cache_dir):
    """
    AudioCache instance with temporary directory.
    
    Provides isolated cache for testing without polluting file system.
    """
    cache = AudioCache(
        cache_dir=temp_cache_dir,
        max_size=100,
        ttl_days=7
    )
    return cache


@pytest.fixture
def sample_audio():
    """
    Generate sample WAV audio bytes for testing.
    
    Returns:
        bytes: Valid WAV file (1 second of 440Hz sine wave, 24kHz mono)
    """
    import soundfile as sf
    
    # Generate 1 second of 440Hz sine wave (A4 note)
    sample_rate = 24000
    duration = 1.0  # seconds
    frequency = 440.0  # Hz
    
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Normalize to int16 range
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Write to WAV in memory
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_int16, sample_rate, format='WAV', subtype='PCM_16')
    
    return wav_buffer.getvalue()


@pytest.fixture
def mock_google_provider():
    """
    Mock GoogleCloudTTSProvider for testing without API calls.
    """
    mock = MagicMock()
    mock.synthesize = AsyncMock(return_value=b'RIFF....WAV')  # Mock WAV bytes
    mock.stream_synthesize = AsyncMock()
    mock.get_available_voices = MagicMock(return_value={
        "en-US-Neural2-F": {
            "name": "en-US-Neural2-F",
            "language": "en-US",
            "gender": "FEMALE",
            "description": "en-US Female Neural2 F"
        }
    })
    mock.validate_config = MagicMock(return_value=(True, None))
    return mock


@pytest.fixture
def mock_elevenlabs_provider():
    """
    Mock ElevenLabsTTSProvider for testing without API calls.
    """
    mock = MagicMock()
    mock.synthesize = AsyncMock(return_value=b'RIFF....WAV')
    mock.stream_synthesize = AsyncMock()
    mock.get_available_voices = MagicMock(return_value={
        "AnvlJBAqSLDzEevYr9Ap": {
            "name": "Rachel",
            "language": "en",
            "gender": "female",
            "description": "Rachel - Calm, narration"
        }
    })
    mock.validate_config = MagicMock(return_value=(True, None))
    return mock


@pytest.fixture
def mock_gemini_provider():
    """
    Mock GeminiLiveTTSProvider for testing without API calls.
    """
    mock = MagicMock()
    mock.synthesize = AsyncMock(return_value=b'RIFF....WAV')
    mock.stream_synthesize = AsyncMock()
    mock.get_available_voices = MagicMock(return_value={
        "Callirrhoe": {
            "name": "Callirrhoe",
            "language": "en",
            "gender": "female",
            "description": "Callirrhoe (female, expressive)"
        }
    })
    mock.validate_config = MagicMock(return_value=(True, None))
    return mock


@pytest.fixture
def mock_providers(mock_google_provider, mock_elevenlabs_provider, mock_gemini_provider):
    """
    Dictionary of all mocked providers.
    
    Useful for testing synthesizer with multiple provider options.
    """
    return {
        "google": mock_google_provider,
        "elevenlabs": mock_elevenlabs_provider,
        "gemini": mock_gemini_provider,
    }


@pytest.fixture
def mock_tts_synthesizer(tts_config, mock_providers):
    """
    Mock TTSSynthesizer with mocked providers.
    
    Useful for testing service layer without actual synthesis.
    """
    from leibniz_agent.services.tts.tts_synthesizer import TTSSynthesizer
    
    with patch.multiple(
        'leibniz_agent.services.tts.tts_synthesizer',
        GoogleCloudTTSProvider=MagicMock(return_value=mock_providers["google"]),
        ElevenLabsTTSProvider=MagicMock(return_value=mock_providers["elevenlabs"]),
        GeminiLiveTTSProvider=MagicMock(return_value=mock_providers["gemini"]),
    ):
        synthesizer = TTSSynthesizer(tts_config)
        return synthesizer


# Async test support
@pytest.fixture
def event_loop():
    """
    Create event loop for async tests.
    
    Required for pytest-asyncio.
    """
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test markers
def pytest_configure(config):
    """
    Register custom test markers.
    """
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may require services)"
    )
    config.addinivalue_line(
        "markers", "requires_service: Requires running TTS service"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (>1s execution time)"
    )
