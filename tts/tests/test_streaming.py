"""
Tests for TTS streaming endpoint functionality.

Tests Server-Sent Events streaming for real-time audio synthesis.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from leibniz_agent.services.tts.app import app
from leibniz_agent.services.tts.tts_synthesizer import TTSSynthesizer


class TestTTSStreaming:
    """Test cases for TTS streaming endpoint."""

    @pytest.fixture
    def mock_synthesizer(self):
        """Mock synthesizer for testing."""
        mock_synth = MagicMock(spec=TTSSynthesizer)

        # Mock streaming method as async generator
        async def mock_streaming(*args, **kwargs):
            # Simulate streaming chunks
            yield b"chunk1", False
            yield b"chunk2", False
            yield b"chunk3", True

        mock_synth.synthesize_streaming = mock_streaming
        mock_synth.get_provider_status.return_value = {"lemonfox": True, "google": False}
        mock_synth.get_stats.return_value = {
            "total_requests": 10,
            "cache": {"hits": 5, "misses": 5}
        }

        return mock_synth

    @pytest.fixture
    def client(self, mock_synthesizer):
        """Test client with mocked synthesizer."""
        with patch('leibniz_agent.services.tts.app.synthesizer', mock_synthesizer):
            yield TestClient(app)

    def test_streaming_endpoint_success(self, client, mock_synthesizer):
        """Test successful streaming synthesis."""
        response = client.get(
            "/api/v1/synthesize/stream",
            params={
                "text": "Hello world",
                "emotion": "helpful",
                "voice": "test_voice",
                "language": "en-US"
            }
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse Server-Sent Events
        content = response.text
        lines = content.strip().split('\n')

        # Should have metadata, audio chunks, and complete events
        events = []
        current_event = {}

        for line in lines:
            if line.startswith('event: '):
                if current_event:
                    events.append(current_event)
                current_event = {'event': line[7:]}
            elif line.startswith('data: '):
                current_event['data'] = line[6:]

        if current_event:
            events.append(current_event)

        # Verify events
        assert len(events) >= 5  # metadata + 3 audio + complete

        # Check metadata event
        metadata_event = next(e for e in events if e.get('event') == 'metadata')
        metadata = json.loads(metadata_event['data'])
        assert metadata['type'] == 'metadata'
        assert metadata['text_length'] == 11  # "Hello world"
        assert metadata['emotion'] == 'helpful'

        # Check audio events
        audio_events = [e for e in events if e.get('event') == 'audio']
        assert len(audio_events) == 3

        for i, event in enumerate(audio_events):
            data = json.loads(event['data'])
            assert data['type'] == 'audio'
            assert data['chunk_index'] == i
            assert 'data' in data  # base64 encoded chunk

        # Check final chunk is marked as final
        final_audio = json.loads(audio_events[-1]['data'])
        assert final_audio['is_final'] is True

        # Check complete event
        complete_event = next(e for e in events if e.get('event') == 'complete')
        complete_data = json.loads(complete_event['data'])
        assert complete_data['type'] == 'complete'
        assert complete_data['total_chunks'] == 3

    def test_streaming_endpoint_empty_text(self, client):
        """Test streaming with empty text."""
        response = client.get(
            "/api/v1/synthesize/stream",
            params={"text": ""}
        )

        assert response.status_code == 200
        content = response.text

        # Should have error event
        assert 'event: error' in content
        assert 'Empty text provided' in content

    def test_streaming_endpoint_synthesis_error(self, client, mock_synthesizer):
        """Test streaming with synthesis error."""
        # Mock synthesis to raise exception
        async def failing_streaming(*args, **kwargs):
            raise Exception("Synthesis failed")
            yield  # Make it an async generator
            return

        mock_synthesizer.synthesize_streaming = failing_streaming

        response = client.get(
            "/api/v1/synthesize/stream",
            params={"text": "Test text"}
        )

        assert response.status_code == 200
        content = response.text

        # Should have error event
        assert 'event: error' in content
        assert 'Synthesis failed' in content

    def test_streaming_endpoint_cors_headers(self, client):
        """Test CORS headers are set correctly."""
        response = client.get(
            "/api/v1/synthesize/stream",
            params={"text": "Test"}
        )

        assert response.headers.get("access-control-allow-origin") == "*"
        assert response.headers.get("cache-control") == "no-cache"
        assert response.headers.get("connection") == "keep-alive"

    def test_health_check_with_provider_status(self, client, mock_synthesizer):
        """Test health check includes provider status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "degraded"  # Only lemonfox available
        assert "lemonfox" in data["providers_available"]
        assert "google" not in data["providers_available"]
        assert data["cache_enabled"] is not None
        assert "total_requests" in data
        assert "cache_stats" in data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])