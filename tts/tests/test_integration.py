"""
Integration Tests for TTS Microservice

Tests synthesizer orchestration, caching, and multi-provider behavior:
    - Cache hit/miss behavior
    - Retry with exponential backoff
    - Provider fallback logic
    - WAV file validation
    - Statistics tracking

Target Coverage: 70%+

Run with:
    pytest leibniz_agent/services/tts/tests/test_integration.py -v
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from leibniz_agent.services.tts.config import TTSConfig
from leibniz_agent.services.tts.tts_synthesizer import TTSSynthesizer


@pytest.mark.integration
class TestSynthesizerCacheBehavior:
    """Test synthesizer cache integration."""
    
    @pytest.mark.asyncio
    async def test_cache_miss_triggers_synthesis(self, mock_tts_synthesizer, sample_audio):
        """Test cache miss calls provider synthesis."""
        synthesizer = mock_tts_synthesizer
        
        # Clear cache to force miss
        synthesizer.cache.clear()
        
        # Mock provider to return sample audio
        synthesizer.primary_provider = AsyncMock()
        synthesizer.primary_provider.synthesize = AsyncMock(return_value=sample_audio)
        
        result = await synthesizer.synthesize("Hello world")
        
        # Should have called provider
        synthesizer.primary_provider.synthesize.assert_called_once_with("Hello world")
        assert result == sample_audio
    
    @pytest.mark.asyncio
    async def test_cache_hit_skips_synthesis(self, mock_tts_synthesizer, sample_audio):
        """Test cache hit returns cached audio without provider call."""
        synthesizer = mock_tts_synthesizer
        
        # Pre-populate cache
        cache_key = synthesizer.cache.get_cache_key(
            text="Hello world",
            voice=synthesizer.config.google_voice,
            provider="mock",
            emotion="neutral"
        )
        synthesizer.cache.cache_audio(cache_key, sample_audio)
        
        # Mock provider to verify it's NOT called
        synthesizer.primary_provider = AsyncMock()
        synthesizer.primary_provider.synthesize = AsyncMock(return_value=b"should not see this")
        
        result = await synthesizer.synthesize("Hello world")
        
        # Should NOT have called provider
        synthesizer.primary_provider.synthesize.assert_not_called()
        assert result == sample_audio
    
    @pytest.mark.asyncio
    async def test_force_regenerate_bypasses_cache(self, mock_tts_synthesizer, sample_audio):
        """Test force_regenerate=True bypasses cache."""
        synthesizer = mock_tts_synthesizer
        
        # Pre-populate cache
        cache_key = synthesizer.cache.get_cache_key(
            text="Hello world",
            voice=synthesizer.config.google_voice,
            provider="mock",
            emotion="neutral"
        )
        old_audio = b"old audio"
        synthesizer.cache.cache_audio(cache_key, old_audio)
        
        # Mock provider to return new audio
        synthesizer.primary_provider = AsyncMock()
        synthesizer.primary_provider.synthesize = AsyncMock(return_value=sample_audio)
        
        result = await synthesizer.synthesize("Hello world", force_regenerate=True)
        
        # Should have called provider despite cache hit
        synthesizer.primary_provider.synthesize.assert_called_once()
        assert result == sample_audio
        assert result != old_audio


@pytest.mark.integration
class TestSynthesizerRetryLogic:
    """Test synthesizer retry and backoff."""
    
    @pytest.mark.asyncio
    async def test_retry_on_provider_failure(self, tts_config, sample_audio):
        """Test synthesizer retries on provider errors."""
        config = TTSConfig(
            provider="mock",
            retry_attempts=3,
            retry_delay=0.1  # Fast retry for testing
        )
        
        # Create synthesizer with mocked provider
        with patch('leibniz_agent.services.tts.tts_synthesizer.TTSSynthesizer._init_mock_provider') as mock_init:
            mock_provider = AsyncMock()
            
            # Fail twice, then succeed
            mock_provider.synthesize = AsyncMock(
                side_effect=[
                    Exception("Network error"),
                    Exception("Timeout"),
                    sample_audio
                ]
            )
            mock_init.return_value = mock_provider
            
            synthesizer = TTSSynthesizer(config)
            
            result = await synthesizer.synthesize("Test text")
            
            # Should have called provider 3 times
            assert mock_provider.synthesize.call_count == 3
            assert result == sample_audio
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self, tts_config):
        """Test exponential backoff delays (1s, 2s, 4s)."""
        config = TTSConfig(
            provider="mock",
            retry_attempts=3,
            retry_delay=0.1  # 100ms base for testing
        )
        
        with patch('leibniz_agent.services.tts.tts_synthesizer.TTSSynthesizer._init_mock_provider') as mock_init:
            mock_provider = AsyncMock()
            mock_provider.synthesize = AsyncMock(side_effect=Exception("Always fails"))
            mock_init.return_value = mock_provider
            
            synthesizer = TTSSynthesizer(config)
            
            import time
            start = time.time()
            
            with pytest.raises(Exception):
                await synthesizer.synthesize("Test text")
            
            elapsed = time.time() - start
            
            # Should have delays: 0.1s + 0.2s + 0.4s = 0.7s (approximately)
            assert elapsed >= 0.6  # Allow some tolerance


@pytest.mark.integration
class TestSynthesizerFallback:
    """Test provider fallback logic."""
    
    @pytest.mark.asyncio
    async def test_fallback_to_secondary_on_failure(self, sample_audio):
        """Test fallback activates when primary fails."""
        config = TTSConfig(
            provider="google",
            fallback_provider="mock",
            enable_fallback=True,
            retry_attempts=1  # Fast failure for testing
        )
        
        # Mock both providers
        with patch('leibniz_agent.services.tts.tts_synthesizer.TTSSynthesizer._init_google_provider') as mock_google, \
             patch('leibniz_agent.services.tts.tts_synthesizer.TTSSynthesizer._init_mock_provider') as mock_fallback:
            
            # Primary fails
            mock_google_provider = AsyncMock()
            mock_google_provider.synthesize = AsyncMock(side_effect=Exception("API error"))
            mock_google.return_value = mock_google_provider
            
            # Fallback succeeds
            mock_fallback_provider = AsyncMock()
            mock_fallback_provider.synthesize = AsyncMock(return_value=sample_audio)
            mock_fallback.return_value = mock_fallback_provider
            
            synthesizer = TTSSynthesizer(config)
            
            result = await synthesizer.synthesize("Test text")
            
            # Should have tried primary then fallback
            assert mock_google_provider.synthesize.called
            assert mock_fallback_provider.synthesize.called
            assert result == sample_audio
    
    @pytest.mark.asyncio
    async def test_fallback_disabled_raises_error(self):
        """Test error propagates when fallback disabled."""
        config = TTSConfig(
            provider="google",
            fallback_provider="mock",
            enable_fallback=False,  # Disabled
            retry_attempts=1
        )
        
        with patch('leibniz_agent.services.tts.tts_synthesizer.TTSSynthesizer._init_google_provider') as mock_google:
            mock_provider = AsyncMock()
            mock_provider.synthesize = AsyncMock(side_effect=Exception("API error"))
            mock_google.return_value = mock_provider
            
            synthesizer = TTSSynthesizer(config)
            
            with pytest.raises(Exception):
                await synthesizer.synthesize("Test text")


@pytest.mark.integration
class TestWAVValidation:
    """Test WAV file format validation."""
    
    @pytest.mark.asyncio
    async def test_all_providers_return_valid_wav(self, sample_audio):
        """Test all providers return valid WAV format."""
        # Test each provider type
        providers_to_test = ["mock"]  # Only test mock (always available)
        
        for provider_name in providers_to_test:
            config = TTSConfig(provider=provider_name)
            
            # Mock provider initialization to avoid dependencies
            with patch(f'leibniz_agent.services.tts.tts_synthesizer.TTSSynthesizer._init_{provider_name}_provider') as mock_init:
                mock_provider = AsyncMock()
                mock_provider.synthesize = AsyncMock(return_value=sample_audio)
                mock_init.return_value = mock_provider
                
                synthesizer = TTSSynthesizer(config)
                result = await synthesizer.synthesize("Test")
                
                # Validate WAV header
                assert result[:4] == b'RIFF', f"{provider_name} didn't return RIFF header"
                assert result[8:12] == b'WAVE', f"{provider_name} didn't return WAVE format"
    
    def test_wav_header_structure(self, sample_audio):
        """Test WAV file has correct header structure."""
        # RIFF header check
        assert sample_audio[:4] == b'RIFF'
        
        # File size check (chunk size field)
        import struct
        chunk_size = struct.unpack('<I', sample_audio[4:8])[0]
        assert chunk_size == len(sample_audio) - 8
        
        # WAVE format check
        assert sample_audio[8:12] == b'WAVE'
        
        # fmt chunk check
        assert sample_audio[12:16] == b'fmt '


@pytest.mark.integration
class TestStatisticsTracking:
    """Test synthesizer statistics collection."""
    
    @pytest.mark.asyncio
    async def test_stats_track_requests(self, mock_tts_synthesizer, sample_audio):
        """Test statistics track total requests."""
        synthesizer = mock_tts_synthesizer
        synthesizer.primary_provider = AsyncMock()
        synthesizer.primary_provider.synthesize = AsyncMock(return_value=sample_audio)
        
        # Make 3 requests
        for _ in range(3):
            await synthesizer.synthesize("Test")
        
        stats = synthesizer.get_stats()
        assert stats['total_requests'] >= 3
    
    @pytest.mark.asyncio
    async def test_stats_track_cache_hits(self, mock_tts_synthesizer, sample_audio):
        """Test statistics track cache hit rate."""
        synthesizer = mock_tts_synthesizer
        synthesizer.primary_provider = AsyncMock()
        synthesizer.primary_provider.synthesize = AsyncMock(return_value=sample_audio)
        
        # First call (miss)
        await synthesizer.synthesize("Hello")
        
        # Second call (hit)
        await synthesizer.synthesize("Hello")
        
        stats = synthesizer.get_stats()
        cache_stats = stats.get('cache_stats', {})
        
        # Should have at least 1 hit and 1 miss
        assert cache_stats.get('hits', 0) >= 1
        assert cache_stats.get('misses', 0) >= 1
    
    @pytest.mark.asyncio
    async def test_stats_track_provider_usage(self, mock_tts_synthesizer, sample_audio):
        """Test statistics track which provider was used."""
        synthesizer = mock_tts_synthesizer
        synthesizer.primary_provider = AsyncMock()
        synthesizer.primary_provider.synthesize = AsyncMock(return_value=sample_audio)
        
        await synthesizer.synthesize("Test")
        
        stats = synthesizer.get_stats()
        provider_stats = stats.get('provider_stats', {})
        
        # Should have tracked mock provider usage
        assert 'mock' in provider_stats or len(provider_stats) > 0
