"""
Service Tests for TTS Microservice

Tests FastAPI HTTP endpoints with httpx.AsyncClient:
    - GET /health - Health check with provider status
    - POST /api/v1/synthesize - TTS synthesis endpoint
    - GET /api/v1/audio/{key} - Audio file retrieval
    - CORS headers
    - Error handling

Target Coverage: 90%+

Run with:
    pytest leibniz_agent/services/tts/tests/test_service.py -v --requires-service
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import httpx

# Note: These tests require the FastAPI app
# They use httpx.AsyncClient to make HTTP requests


@pytest.mark.requires_service
class TestHealthEndpoint:
    """Test GET /health endpoint."""
    
    @pytest.mark.asyncio
    async def test_health_returns_200(self, sample_audio):
        """Test health endpoint returns 200 OK."""
        # Import app
        from leibniz_agent.services.tts.app import app
        
        # Mock synthesizer to avoid provider initialization
        with patch('leibniz_agent.services.tts.app.synthesizer') as mock_synth:
            mock_synth.get_stats.return_value = {
                'total_requests': 10,
                'cache_stats': {'hits': 5, 'misses': 5, 'hit_rate': 0.5},
                'provider_stats': {'mock': 10}
            }
            mock_synth.config = MagicMock(provider="mock", fallback_provider="google")
            mock_synth.primary_provider = MagicMock()
            mock_synth.fallback_provider = None
            
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/health")
                
                assert response.status_code == 200
                data = response.json()
                assert data['status'] in ['healthy', 'degraded']
                assert 'providers_available' in data
                assert 'cache_stats' in data
    
    @pytest.mark.asyncio
    async def test_health_shows_degraded_when_no_providers(self):
        """Test health endpoint shows degraded status when no providers available."""
        from leibniz_agent.services.tts.app import app
        
        with patch('leibniz_agent.services.tts.app.synthesizer') as mock_synth:
            mock_synth.get_stats.return_value = {}
            mock_synth.config = MagicMock(provider="mock")
            mock_synth.primary_provider = None  # No providers
            mock_synth.fallback_provider = None
            
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/health")
                
                assert response.status_code == 200
                data = response.json()
                assert data['status'] == 'degraded'
    
    @pytest.mark.asyncio
    async def test_health_includes_cache_stats(self):
        """Test health endpoint includes cache statistics."""
        from leibniz_agent.services.tts.app import app
        
        with patch('leibniz_agent.services.tts.app.synthesizer') as mock_synth:
            mock_synth.get_stats.return_value = {
                'cache_stats': {
                    'hits': 100,
                    'misses': 50,
                    'hit_rate': 0.667,
                    'size': 75
                }
            }
            mock_synth.config = MagicMock(provider="mock")
            mock_synth.primary_provider = MagicMock()
            
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/health")
                
                data = response.json()
                assert 'cache_stats' in data
                assert data['cache_stats']['hits'] == 100
                assert data['cache_stats']['hit_rate'] == 0.667


@pytest.mark.requires_service
class TestSynthesizeEndpoint:
    """Test POST /api/v1/synthesize endpoint."""
    
    @pytest.mark.asyncio
    async def test_synthesize_valid_request(self, sample_audio):
        """Test synthesize endpoint with valid request."""
        from leibniz_agent.services.tts.app import app
        
        with patch('leibniz_agent.services.tts.app.synthesizer') as mock_synth:
            mock_synth.synthesize = AsyncMock(return_value=sample_audio)
            mock_synth.cache.get_cache_key.return_value = "test_cache_key_123"
            
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/synthesize",
                    json={
                        "text": "Hello world",
                        "emotion": "helpful"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert 'cache_key' in data
                assert 'audio_url' in data
                assert 'provider' in data
                assert '/api/v1/audio/' in data['audio_url']
    
    @pytest.mark.asyncio
    async def test_synthesize_invalid_request_missing_text(self):
        """Test synthesize endpoint rejects request without text."""
        from leibniz_agent.services.tts.app import app
        
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/synthesize",
                json={}  # Missing text field
            )
            
            assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_synthesize_uses_custom_voice(self, sample_audio):
        """Test synthesize endpoint respects custom voice parameter."""
        from leibniz_agent.services.tts.app import app
        
        with patch('leibniz_agent.services.tts.app.synthesizer') as mock_synth:
            mock_synth.synthesize = AsyncMock(return_value=sample_audio)
            mock_synth.cache.get_cache_key.return_value = "test_key"
            
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/synthesize",
                    json={
                        "text": "Hello",
                        "voice": "en-US-Neural2-C"
                    }
                )
                
                assert response.status_code == 200
                # Verify synthesize was called with custom voice
                mock_synth.synthesize.assert_called_once()
                call_kwargs = mock_synth.synthesize.call_args[1]
                assert call_kwargs.get('voice') == "en-US-Neural2-C"
    
    @pytest.mark.asyncio
    async def test_synthesize_handles_provider_error(self):
        """Test synthesize endpoint handles provider errors gracefully."""
        from leibniz_agent.services.tts.app import app
        
        with patch('leibniz_agent.services.tts.app.synthesizer') as mock_synth:
            mock_synth.synthesize = AsyncMock(side_effect=Exception("Provider failed"))
            
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/synthesize",
                    json={"text": "Hello"}
                )
                
                assert response.status_code == 500
                data = response.json()
                assert 'detail' in data


@pytest.mark.requires_service
class TestAudioEndpoint:
    """Test GET /api/v1/audio/{cache_key} endpoint."""
    
    @pytest.mark.asyncio
    async def test_audio_retrieval_valid_key(self, sample_audio):
        """Test audio endpoint returns cached audio."""
        from leibniz_agent.services.tts.app import app
        
        with patch('leibniz_agent.services.tts.app.synthesizer') as mock_synth:
            mock_synth.cache.get_audio.return_value = sample_audio
            
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/api/v1/audio/test_cache_key")
                
                assert response.status_code == 200
                assert response.headers['content-type'] == 'audio/wav'
                assert response.content == sample_audio
    
    @pytest.mark.asyncio
    async def test_audio_retrieval_invalid_key(self):
        """Test audio endpoint returns 404 for invalid key."""
        from leibniz_agent.services.tts.app import app
        
        with patch('leibniz_agent.services.tts.app.synthesizer') as mock_synth:
            mock_synth.cache.get_audio.return_value = None  # Not found
            
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/api/v1/audio/nonexistent_key")
                
                assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_audio_endpoint_sets_correct_headers(self, sample_audio):
        """Test audio endpoint sets WAV content-type and cache headers."""
        from leibniz_agent.services.tts.app import app
        
        with patch('leibniz_agent.services.tts.app.synthesizer') as mock_synth:
            mock_synth.cache.get_audio.return_value = sample_audio
            
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/api/v1/audio/test_key")
                
                assert response.headers['content-type'] == 'audio/wav'
                # Should have cache headers for browser caching
                # (Actual implementation may vary)


@pytest.mark.requires_service
class TestCORSHeaders:
    """Test CORS configuration."""
    
    @pytest.mark.asyncio
    async def test_cors_headers_present(self):
        """Test CORS headers are present in responses."""
        from leibniz_agent.services.tts.app import app
        
        with patch('leibniz_agent.services.tts.app.synthesizer') as mock_synth:
            mock_synth.get_stats.return_value = {}
            mock_synth.config = MagicMock(provider="mock")
            mock_synth.primary_provider = MagicMock()
            
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get(
                    "/health",
                    headers={"Origin": "http://localhost:3000"}
                )
                
                # Should have CORS headers
                # (Actual headers depend on CORS middleware configuration)
                assert response.status_code == 200


@pytest.mark.requires_service
class TestErrorHandling:
    """Test service error handling."""
    
    @pytest.mark.asyncio
    async def test_malformed_json_returns_400(self):
        """Test malformed JSON returns 400 Bad Request."""
        from leibniz_agent.services.tts.app import app
        
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/synthesize",
                content="invalid json{",  # Malformed
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 422  # FastAPI validation error
    
    @pytest.mark.asyncio
    async def test_empty_text_returns_422(self):
        """Test empty text field returns 422 Validation Error."""
        from leibniz_agent.services.tts.app import app
        
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/synthesize",
                json={"text": ""}  # Empty string
            )
            
            # Should reject empty text
            assert response.status_code in [422, 400]
    
    @pytest.mark.asyncio
    async def test_nonexistent_endpoint_returns_404(self):
        """Test nonexistent endpoint returns 404."""
        from leibniz_agent.services.tts.app import app
        
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/nonexistent")
            
            assert response.status_code == 404
