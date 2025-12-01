"""
Unit Tests for TTS Microservice

Tests core functionality without external dependencies:
    - TTSConfig validation and environment loading
    - AudioCache operations (key generation, hit/miss, LRU)
    - Cache statistics and cleanup
    - Emotion parameter handling

Target Coverage: 80%+

Run with:
    pytest leibniz_agent/services/tts/tests/test_unit.py -v
    pytest leibniz_agent/services/tts/tests/test_unit.py -v -m unit
"""

import os
import json
import time
import shutil
from pathlib import Path
import pytest

from leibniz_agent.services.tts.config import TTSConfig
from leibniz_agent.services.tts.audio_cache import AudioCache


@pytest.mark.unit
class TestTTSConfig:
    """Test TTSConfig validation and environment loading."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = TTSConfig()
        
        assert config.provider == "gemini"
        assert config.fallback_provider == "google"
        assert config.enable_fallback is True
        assert config.cache_enabled is True
        assert config.sample_rate == 24000
        assert config.speaking_rate == 1.0
        assert config.pitch == 0.0
        assert config.timeout == 30.0
        assert config.retry_attempts == 3
    
    def test_provider_validation(self):
        """Test provider value validation."""
        # Valid providers
        valid_providers = ["google", "elevenlabs", "gemini", "xtts_local", "mock"]
        for provider in valid_providers:
            config = TTSConfig(provider=provider)
            assert config.provider == provider
        
        # Invalid provider should raise ValueError
        with pytest.raises(ValueError, match="Invalid provider"):
            TTSConfig(provider="invalid_provider")
    
    def test_pitch_range_validation(self):
        """Test pitch parameter range validation."""
        # Valid pitch values
        assert TTSConfig(pitch=-20.0).pitch == -20.0
        assert TTSConfig(pitch=0.0).pitch == 0.0
        assert TTSConfig(pitch=20.0).pitch == 20.0
        
        # Invalid pitch values
        with pytest.raises(ValueError, match="pitch must be between -20.0 and 20.0"):
            TTSConfig(pitch=-25.0)
        
        with pytest.raises(ValueError, match="pitch must be between -20.0 and 20.0"):
            TTSConfig(pitch=25.0)
    
    def test_speaking_rate_range_validation(self):
        """Test speaking_rate parameter range validation."""
        # Valid speaking_rate values
        assert TTSConfig(speaking_rate=0.25).speaking_rate == 0.25
        assert TTSConfig(speaking_rate=1.0).speaking_rate == 1.0
        assert TTSConfig(speaking_rate=4.0).speaking_rate == 4.0
        
        # Invalid speaking_rate values
        with pytest.raises(ValueError, match="speaking_rate must be between 0.25 and 4.0"):
            TTSConfig(speaking_rate=0.1)
        
        with pytest.raises(ValueError, match="speaking_rate must be between 0.25 and 4.0"):
            TTSConfig(speaking_rate=5.0)
    
    def test_sample_rate_validation(self):
        """Test sample_rate must be positive."""
        assert TTSConfig(sample_rate=16000).sample_rate == 16000
        assert TTSConfig(sample_rate=24000).sample_rate == 24000
        assert TTSConfig(sample_rate=48000).sample_rate == 48000
        
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            TTSConfig(sample_rate=0)
        
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            TTSConfig(sample_rate=-100)
    
    def test_from_env_loads_defaults(self, monkeypatch):
        """Test from_env() loads default values when env vars not set."""
        # Clear relevant environment variables
        for key in os.environ.copy():
            if key.startswith("LEIBNIZ_TTS_") or key.startswith("GOOGLE_") or key.startswith("ELEVENLABS_"):
                monkeypatch.delenv(key, raising=False)
        
        config = TTSConfig.from_env()
        
        assert config.provider == "gemini"
        assert config.cache_enabled is True
        assert config.sample_rate == 24000
    
    def test_from_env_loads_custom_values(self, monkeypatch):
        """Test from_env() loads custom values from environment."""
        monkeypatch.setenv("LEIBNIZ_TTS_PROVIDER", "google")
        monkeypatch.setenv("LEIBNIZ_TTS_FALLBACK_PROVIDER", "elevenlabs")
        monkeypatch.setenv("LEIBNIZ_TTS_CACHE_ENABLED", "false")
        monkeypatch.setenv("LEIBNIZ_TTS_SAMPLE_RATE", "16000")
        monkeypatch.setenv("LEIBNIZ_TTS_SPEAKING_RATE", "1.2")
        monkeypatch.setenv("LEIBNIZ_TTS_PITCH", "5.0")
        monkeypatch.setenv("LEIBNIZ_TTS_TIMEOUT", "60.0")
        monkeypatch.setenv("LEIBNIZ_TTS_RETRY_ATTEMPTS", "5")
        
        config = TTSConfig.from_env()
        
        assert config.provider == "google"
        assert config.fallback_provider == "elevenlabs"
        assert config.cache_enabled is False
        assert config.sample_rate == 16000
        assert config.speaking_rate == 1.2
        assert config.pitch == 5.0
        assert config.timeout == 60.0
        assert config.retry_attempts == 5
    
    def test_cache_dir_default(self):
        """Test cache_dir defaults to ./audio_cache."""
        config = TTSConfig()
        assert config.cache_dir == "./audio_cache"
    
    def test_google_voice_default(self):
        """Test Google voice defaults to en-US-Neural2-F."""
        config = TTSConfig()
        assert config.google_voice == "en-US-Neural2-F"
    
    def test_elevenlabs_voice_default(self):
        """Test ElevenLabs voice defaults to Rachel."""
        config = TTSConfig()
        assert config.elevenlabs_voice == "21m00Tcm4TlvDq8ikWAM"


@pytest.mark.unit
class TestAudioCache:
    """Test AudioCache operations (key generation, hit/miss, LRU)."""
    
    def test_get_cache_key_consistency(self, audio_cache):
        """Test cache key generation is consistent for same inputs."""
        key1 = audio_cache.get_cache_key("Hello world", "en-US-Neural2-F", "en-US", "google", "neutral")
        key2 = audio_cache.get_cache_key("Hello world", "en-US-Neural2-F", "en-US", "google", "neutral")
        
        assert key1 == key2
        assert len(key1) == 32  # MD5 hash is 32 hex characters
    
    def test_get_cache_key_different_text(self, audio_cache):
        """Test cache key changes with different text."""
        key1 = audio_cache.get_cache_key("Hello world", "en-US-Neural2-F", "en-US", "google", "neutral")
        key2 = audio_cache.get_cache_key("Goodbye world", "en-US-Neural2-F", "en-US", "google", "neutral")
        
        assert key1 != key2
    
    def test_get_cache_key_different_voice(self, audio_cache):
        """Test cache key changes with different voice."""
        key1 = audio_cache.get_cache_key("Hello world", "en-US-Neural2-F", "en-US", "google", "neutral")
        key2 = audio_cache.get_cache_key("Hello world", "en-US-Neural2-C", "en-US", "google", "neutral")
        
        assert key1 != key2
    
    def test_get_cache_key_different_provider(self, audio_cache):
        """Test cache key changes with different provider."""
        key1 = audio_cache.get_cache_key("Hello world", "voice1", "en-US", "google", "neutral")
        key2 = audio_cache.get_cache_key("Hello world", "voice1", "en-US", "elevenlabs", "neutral")
        
        assert key1 != key2
    
    def test_get_cache_key_different_emotion(self, audio_cache):
        """Test cache key changes with different emotion."""
        key1 = audio_cache.get_cache_key("Hello world", "voice1", "en-US", "google", "neutral")
        key2 = audio_cache.get_cache_key("Hello world", "voice1", "en-US", "google", "excited")
        
        assert key1 != key2
    
    def test_cache_miss(self, audio_cache):
        """Test cache miss returns None."""
        result = audio_cache.get_cached_audio("nonexistent_key")
        assert result is None
    
    def test_cache_hit(self, audio_cache, sample_audio):
        """Test cache hit returns audio data."""
        key = audio_cache.get_cache_key("Test", "voice1", "en-US", "mock", "neutral")
        
        # Cache the audio
        audio_cache.cache_audio(key, sample_audio)
        
        # Retrieve from cache
        cached = audio_cache.get_cached_audio(key)
        assert cached == sample_audio
    
    def test_cache_stats_initial(self, audio_cache):
        """Test cache stats start at zero."""
        stats = audio_cache.get_stats()
        
        assert stats["total_entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
    
    def test_cache_stats_after_miss(self, audio_cache):
        """Test cache stats after cache miss."""
        audio_cache.get_cached_audio("nonexistent_key")
        
        stats = audio_cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0
        assert stats["hit_rate"] == 0.0
    
    def test_cache_stats_after_hit(self, audio_cache, sample_audio):
        """Test cache stats after cache hit."""
        key = audio_cache.get_cache_key("Test", "voice1", "en-US", "mock", "neutral")
        audio_cache.cache_audio(key, sample_audio)
        
        # Hit cache twice
        audio_cache.get_cached_audio(key)
        audio_cache.get_cached_audio(key)
        
        stats = audio_cache.get_stats()
        assert stats["hits"] == 2
        assert stats["total_entries"] == 1
    
    def test_cache_hit_rate_calculation(self, audio_cache, sample_audio):
        """Test cache hit rate calculation."""
        key1 = audio_cache.get_cache_key("Test1", "voice1", "en-US", "mock", "neutral")
        key2 = audio_cache.get_cache_key("Test2", "voice1", "en-US", "mock", "neutral")
        
        audio_cache.cache_audio(key1, sample_audio)
        
        # 2 hits, 1 miss
        audio_cache.get_cached_audio(key1)  # hit
        audio_cache.get_cached_audio(key1)  # hit
        audio_cache.get_cached_audio(key2)  # miss
        
        stats = audio_cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2.0 / 3.0)
    
    def test_lru_cleanup(self, temp_cache_dir, sample_audio):
        """Test LRU cleanup evicts oldest entries."""
        # Create cache with max_size=3
        cache = AudioCache(cache_dir=temp_cache_dir, max_size=3, ttl_days=30)
        
        # Cache 3 entries
        key1 = cache.get_cache_key("Test1", "v1", "en-US", "mock", "neutral")
        key2 = cache.get_cache_key("Test2", "v1", "en-US", "mock", "neutral")
        key3 = cache.get_cache_key("Test3", "v1", "en-US", "mock", "neutral")
        
        cache.cache_audio(key1, sample_audio)
        time.sleep(0.1)  # Ensure different timestamps
        cache.cache_audio(key2, sample_audio)
        time.sleep(0.1)
        cache.cache_audio(key3, sample_audio)
        
        # All 3 should be in cache
        assert cache.get_cached_audio(key1) is not None
        assert cache.get_cached_audio(key2) is not None
        assert cache.get_cached_audio(key3) is not None
        
        # Add 4th entry (should evict oldest: key1)
        key4 = cache.get_cache_key("Test4", "v1", "en-US", "mock", "neutral")
        cache.cache_audio(key4, sample_audio)
        
        # key1 should be evicted
        assert cache.get_cached_audio(key1) is None
        assert cache.get_cached_audio(key2) is not None
        assert cache.get_cached_audio(key3) is not None
        assert cache.get_cached_audio(key4) is not None
    
    def test_cache_audio_creates_directory(self, temp_cache_dir, sample_audio):
        """Test cache_audio creates cache directory if missing."""
        cache_dir = os.path.join(temp_cache_dir, "new_cache_dir")
        cache = AudioCache(cache_dir=cache_dir, max_size=100, ttl_days=30)
        
        key = cache.get_cache_key("Test", "v1", "en-US", "mock", "neutral")
        cache.cache_audio(key, sample_audio)
        
        assert os.path.exists(cache_dir)
        assert cache.get_cached_audio(key) is not None
    
    def test_cache_index_persistence(self, temp_cache_dir, sample_audio):
        """Test cache index is saved and loaded correctly."""
        cache1 = AudioCache(cache_dir=temp_cache_dir, max_size=100, ttl_days=30)
        
        key = cache1.get_cache_key("Test", "v1", "en-US", "mock", "neutral")
        cache1.cache_audio(key, sample_audio)
        
        # Create new cache instance (should load index)
        cache2 = AudioCache(cache_dir=temp_cache_dir, max_size=100, ttl_days=30)
        
        # Should find cached audio
        assert cache2.get_cached_audio(key) is not None
    
    def test_cache_ttl_expiration(self, temp_cache_dir, sample_audio):
        """Test expired entries are not returned."""
        # Create cache with 0-day TTL (everything expires)
        cache = AudioCache(cache_dir=temp_cache_dir, max_size=100, ttl_days=0)
        
        key = cache.get_cache_key("Test", "v1", "en-US", "mock", "neutral")
        cache.cache_audio(key, sample_audio)
        
        # Modify cache index to set old timestamp
        cache.cache_index[key]["last_accessed"] = time.time() - (2 * 86400)  # 2 days ago
        
        # Should return None (expired)
        result = cache.get_cached_audio(key)
        assert result is None


@pytest.mark.unit
def test_cache_key_with_special_characters(audio_cache):
    """Test cache key generation handles special characters."""
    text_with_special = "Hello, world! How are you? I'm fine. <tag>"
    key = audio_cache.get_cache_key(text_with_special, "voice1", "en-US", "google", "neutral")
    
    assert len(key) == 32
    assert key.isalnum()  # MD5 hash is alphanumeric


@pytest.mark.unit
def test_cache_key_with_unicode(audio_cache):
    """Test cache key generation handles Unicode text."""
    unicode_text = "Hello 世界 "
    key = audio_cache.get_cache_key(unicode_text, "voice1", "en-US", "google", "neutral")
    
    assert len(key) == 32
    assert key.isalnum()


@pytest.mark.unit
def test_cache_cleanup_on_error(temp_cache_dir):
    """Test cache handles corrupted index gracefully."""
    cache_dir = temp_cache_dir
    index_file = os.path.join(cache_dir, "cache_index.json")
    
    # Create corrupted index
    os.makedirs(cache_dir, exist_ok=True)
    with open(index_file, 'w') as f:
        f.write("corrupted json{{{")
    
    # Should handle error and start with empty index
    cache = AudioCache(cache_dir=cache_dir, max_size=100, ttl_days=30)
    stats = cache.get_stats()
    
    assert stats["total_entries"] == 0
