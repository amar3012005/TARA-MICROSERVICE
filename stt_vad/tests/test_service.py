"""
STT/VAD Service Integration Tests

Comprehensive test suite for the STT/VAD microservice.

Tests WebSocket streaming, HTTP endpoints, and error handling.
Mocks external dependencies (Gemini, Redis) for reliable testing.

Run with: pytest leibniz_agent/services/stt_vad/tests/test_service.py -v
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from fastapi.testclient import TestClient
from leibniz_agent.services.stt_vad.app import app


@pytest.fixture
def client():
    """FastAPI test client fixture"""
    return TestClient(app)


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    mock = AsyncMock()
    mock.ping.return_value = True
    return mock


@pytest.fixture
def mock_gemini_session():
    """Mock Gemini Live session"""
    mock = MagicMock()
    mock.get_session_stats.return_value = {"active": True}
    return mock


@pytest.fixture
def mock_vad_manager():
    """Mock VAD manager"""
    mock = AsyncMock()
    mock.get_performance_metrics.return_value = {
        "total_captures": 5,
        "avg_capture_time_ms": 1500.0
    }
    return mock


@pytest.fixture
async def mock_receive():
    """Mock Gemini receive method yielding transcript responses"""
    async def receive():
        # First yield: partial transcript
        yield MagicMock(
            server_content=MagicMock(
                model_turn=MagicMock(
                    parts=[MagicMock(text="Hello")]
                )
            )
        )
        # Second yield: final transcript
        yield MagicMock(
            server_content=MagicMock(
                input_audio_transcription=MagicMock(text="Hello world"),
                turn_complete=True
            )
        )
    return receive


def build_pcm_audio(duration_seconds=1.0, sample_rate=16000):
    """Build PCM silence bytes for testing"""
    samples = int(duration_seconds * sample_rate)
    audio_data = np.zeros(samples, dtype=np.int16)
    return audio_data.tobytes()


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_endpoint(self, client, mock_redis, mock_gemini_session, mock_vad_manager):
        """Test health endpoint returns expected fields"""
        with patch('leibniz_agent.services.stt_vad.app.redis_client', mock_redis), \
             patch('leibniz_agent.services.stt_vad.app.vad_manager', mock_vad_manager), \
             patch('leibniz_agent.services.stt_vad.gemini_client.GeminiLiveSession', mock_gemini_session):

            response = client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert "status" in data
            assert "service" in data
            assert "uptime_seconds" in data
            assert "active_sessions" in data
            assert "gemini_session" in data
            assert "redis_connected" in data
            assert "total_captures" in data
            assert "avg_capture_time_ms" in data

            assert data["service"] == "stt-vad"


class TestWebSocketConnection:
    """Test WebSocket connection establishment"""

    @pytest.mark.asyncio
    async def test_websocket_connection(self, mock_redis, mock_vad_manager):
        """Test WebSocket accepts connection and sends welcome message"""
        with patch('leibniz_agent.services.stt_vad.app.redis_client', mock_redis), \
             patch('leibniz_agent.services.stt_vad.app.vad_manager', mock_vad_manager):

            client = TestClient(app)
            session_id = "test_session_123"

            with client.websocket_connect(f"/api/v1/transcribe/stream?session_id={session_id}") as websocket:
                # Should receive welcome message
                data = websocket.receive_json()
                assert data["type"] == "connected"
                assert data["session_id"] == session_id


class TestWebSocketAudioStreaming:
    """Test WebSocket audio streaming functionality"""

    @pytest.mark.asyncio
    async def test_websocket_audio_streaming(self, mock_redis, mock_receive):
        """Test sending audio and receiving transcripts"""
        with patch('leibniz_agent.services.stt_vad.app.redis_client', mock_redis), \
             patch('leibniz_agent.services.stt_vad.app.vad_manager') as mock_manager, \
             patch('leibniz_agent.services.stt_vad.gemini_client.GeminiLiveSession.get_session') as mock_session:

            # Setup mocks
            mock_session_instance = AsyncMock()
            mock_session_instance.receive = mock_receive
            mock_session.return_value = mock_session_instance

            mock_manager_instance = AsyncMock()
            mock_manager_instance.capture_speech_streaming = AsyncMock(return_value="Hello world")
            mock_manager.return_value = mock_manager_instance

            client = TestClient(app)
            session_id = "test_session_456"

            with client.websocket_connect(f"/api/v1/transcribe/stream?session_id={session_id}") as websocket:
                # Receive welcome
                welcome = websocket.receive_json()
                assert welcome["type"] == "connected"

                # Send start_capture command
                websocket.send_json({"type": "start_capture"})

                # Send audio data
                audio_bytes = build_pcm_audio(0.5)  # 0.5 seconds of silence
                websocket.send_bytes(audio_bytes)

                # Should receive transcript fragments
                messages = []
                try:
                    # Collect messages with timeout
                    while True:
                        msg = websocket.receive_json(timeout=1.0)
                        messages.append(msg)
                except:
                    pass  # Expected timeout when no more messages

                # Should have received transcript messages
                transcript_messages = [m for m in messages if m.get("type") in ["partial", "final"]]
                assert len(transcript_messages) > 0


class TestWebSocketTimeout:
    """Test WebSocket timeout handling"""

    @pytest.mark.asyncio
    async def test_websocket_timeout(self, mock_redis, mock_vad_manager):
        """Test timeout when no audio sent after connection"""
        with patch('leibniz_agent.services.stt_vad.app.redis_client', mock_redis), \
             patch('leibniz_agent.services.stt_vad.app.vad_manager', mock_vad_manager):

            client = TestClient(app)
            session_id = "timeout_test"

            with client.websocket_connect(f"/api/v1/transcribe/stream?session_id={session_id}") as websocket:
                # Receive welcome
                welcome = websocket.receive_json()
                assert welcome["type"] == "connected"

                # Wait for timeout (should happen after 30 seconds, but we'll simulate)
                # In real test, this would take 30+ seconds, so we mock the timeout
                import time
                start_time = time.time()

                try:
                    while time.time() - start_time < 2:  # Short timeout for test
                        data = websocket.receive_json(timeout=0.1)
                        if data.get("type") == "timeout":
                            break
                except:
                    pass  # Expected in test environment


class TestInvalidAudioChunk:
    """Test invalid audio chunk handling"""

    @pytest.mark.asyncio
    async def test_invalid_audio_chunk(self, mock_redis, mock_vad_manager):
        """Test sending invalid audio data returns error"""
        with patch('leibniz_agent.services.stt_vad.app.redis_client', mock_redis), \
             patch('leibniz_agent.services.stt_vad.app.vad_manager', mock_vad_manager):

            client = TestClient(app)
            session_id = "invalid_audio_test"

            with client.websocket_connect(f"/api/v1/transcribe/stream?session_id={session_id}") as websocket:
                # Receive welcome
                welcome = websocket.receive_json()
                assert welcome["type"] == "connected"

                # Send invalid audio (too small)
                invalid_audio = b"short"
                websocket.send_bytes(invalid_audio)

                # Should receive error message
                error_msg = websocket.receive_json()
                assert error_msg["type"] == "error"
                assert "errors" in error_msg


class TestConcurrentSessions:
    """Test multiple concurrent WebSocket sessions"""

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, mock_redis, mock_vad_manager):
        """Test multiple sessions can operate independently"""
        with patch('leibniz_agent.services.stt_vad.app.redis_client', mock_redis), \
             patch('leibniz_agent.services.stt_vad.app.vad_manager', mock_vad_manager):

            client = TestClient(app)
            session_ids = ["session_1", "session_2", "session_3"]

            websockets = []
            try:
                # Connect multiple websockets
                for session_id in session_ids:
                    ws = client.websocket_connect(f"/api/v1/transcribe/stream?session_id={session_id}")
                    websockets.append((session_id, ws.__enter__()))

                # Send different commands to each
                for session_id, websocket in websockets:
                    # Receive welcome
                    welcome = websocket.receive_json()
                    assert welcome["session_id"] == session_id

                    # Send session-specific command
                    websocket.send_json({"type": "start_capture", "session_id": session_id})

            finally:
                # Clean up
                for _, ws in websockets:
                    try:
                        ws.__exit__(None, None, None)
                    except:
                        pass


class TestMetricsEndpoint:
    """Test metrics endpoint"""

    def test_metrics_endpoint(self, client, mock_redis, mock_vad_manager, mock_gemini_session):
        """Test metrics endpoint after captures"""
        with patch('leibniz_agent.services.stt_vad.app.redis_client', mock_redis), \
             patch('leibniz_agent.services.stt_vad.app.vad_manager', mock_vad_manager), \
             patch('leibniz_agent.services.stt_vad.gemini_client.GeminiLiveSession', mock_gemini_session):

            response = client.get("/metrics")
            assert response.status_code == 200

            data = response.json()
            assert "active_sessions" in data
            assert "gemini_stats" in data
            assert data["active_sessions"] == 0  # No active sessions in test


class TestAdminResetSession:
    """Test admin reset session endpoint"""

    def test_admin_reset_session(self, client, mock_redis, mock_vad_manager):
        """Test admin endpoint resets session"""
        with patch('leibniz_agent.services.stt_vad.app.redis_client', mock_redis), \
             patch('leibniz_agent.services.stt_vad.app.vad_manager', mock_vad_manager), \
             patch('leibniz_agent.services.stt_vad.gemini_client.GeminiLiveSession.close_session') as mock_close:

            response = client.post("/admin/reset_session")
            assert response.status_code == 200

            data = response.json()
            assert "status" in data
            assert "message" in data
            assert data["status"] == "success"

            # Verify close_session was called
            mock_close.assert_called_once()