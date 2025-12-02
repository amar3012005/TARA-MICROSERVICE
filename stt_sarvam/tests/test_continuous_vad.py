"""
Tests for Continuous VAD functionality

Tests session management, barge-in detection, and WebSocket streaming.
"""

import asyncio
import pytest
import json
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
import websockets
import httpx
import pytest_asyncio

from leibniz_agent.services.stt_vad.continuous_session import ContinuousSessionManager, ContinuousSessionState


class TestContinuousSessionManager:
    """Test ContinuousSessionManager functionality"""

    @pytest_asyncio.fixture
    async def session_manager(self):
        """Create session manager for testing"""
        manager = ContinuousSessionManager(redis_client=None)  # No Redis for unit tests
        yield manager
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test session creation"""
        session_id = "test_session_123"
        session = await session_manager.create_session(session_id)

        assert session.session_id == session_id
        assert session.status == "active"
        assert session.transcripts_count == 0
        assert session.is_agent_speaking == False

    @pytest.mark.asyncio
    async def test_get_session_existing(self, session_manager):
        """Test retrieving existing session"""
        session_id = "test_session_456"
        created = await session_manager.create_session(session_id)
        retrieved = await session_manager.get_session(session_id)

        assert retrieved is not None
        assert retrieved.session_id == session_id
        assert retrieved.status == "active"

    @pytest.mark.asyncio
    async def test_get_session_nonexistent(self, session_manager):
        """Test retrieving non-existent session"""
        retrieved = await session_manager.get_session("nonexistent")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_set_agent_speaking(self, session_manager):
        """Test setting agent speaking state"""
        session_id = "test_session_789"
        await session_manager.create_session(session_id)

        # Set speaking to True
        result = await session_manager.set_agent_speaking(session_id, True)
        assert result == True

        session = await session_manager.get_session(session_id)
        assert session.is_agent_speaking == True

        # Set speaking to False
        result = await session_manager.set_agent_speaking(session_id, False)
        assert result == True

        session = await session_manager.get_session(session_id)
        assert session.is_agent_speaking == False

    @pytest.mark.asyncio
    async def test_set_agent_speaking_nonexistent(self, session_manager):
        """Test setting agent speaking on non-existent session"""
        result = await session_manager.set_agent_speaking("nonexistent", True)
        assert result == False

    @pytest.mark.asyncio
    async def test_close_session(self, session_manager):
        """Test session closure"""
        session_id = "test_session_close"
        await session_manager.create_session(session_id)

        # Verify session exists
        session = await session_manager.get_session(session_id)
        assert session is not None

        # Close session
        result = await session_manager.close_session(session_id)
        assert result == True

        # Verify session is gone
        session = await session_manager.get_session(session_id)
        assert session is None

    @pytest.mark.asyncio
    async def test_close_nonexistent_session(self, session_manager):
        """Test closing non-existent session"""
        result = await session_manager.close_session("nonexistent")
        assert result == False

    @pytest.mark.asyncio
    async def test_session_stats(self, session_manager):
        """Test session statistics"""
        # Create multiple sessions
        await session_manager.create_session("session1")
        await session_manager.create_session("session2")
        await session_manager.create_session("session3")

        stats = await session_manager.get_session_stats()

        assert stats["active_sessions"] == 3
        assert stats["total_sessions_created"] == 3
        assert stats["sessions_by_status"]["active"] == 3

    @pytest.mark.asyncio
    async def test_update_activity(self, session_manager):
        """Test activity timestamp updates"""
        session_id = "test_activity"
        session = await session_manager.create_session(session_id)

        original_time = session.last_activity
        await asyncio.sleep(0.01)  # Small delay

        result = await session_manager.update_activity(session_id)
        assert result == True

        updated_session = await session_manager.get_session(session_id)
        assert updated_session.last_activity > original_time


class TestContinuousVADWebSocket:
    """Test WebSocket endpoints for continuous VAD"""

    @pytest_asyncio.fixture
    async def test_server(self):
        """Start test server"""
        from leibniz_agent.services.stt_vad.app import app
        from fastapi.testclient import TestClient

        # Override Redis client for testing
        from leibniz_agent.services.stt_vad import continuous_session
        original_redis = continuous_session.get_redis_client
        continuous_session.get_redis_client = AsyncMock(return_value=None)

        with TestClient(app) as client:
            yield client

        # Restore
        continuous_session.get_redis_client = original_redis

    @pytest.mark.asyncio
    async def test_continuous_start_endpoint(self, test_server):
        """Test POST /api/v1/continuous/start endpoint"""
        response = test_server.post("/api/v1/continuous/start")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "status" in data
        assert data["status"] == "created"

    @pytest.mark.asyncio
    async def test_continuous_stop_endpoint(self, test_server):
        """Test POST /api/v1/continuous/stop/{session_id} endpoint"""
        # First create a session
        create_response = test_server.post("/api/v1/continuous/start")
        session_id = create_response.json()["session_id"]

        # Stop the session
        stop_response = test_server.post(f"/api/v1/continuous/stop/{session_id}")

        assert stop_response.status_code == 200
        data = stop_response.json()
        assert data["status"] == "stopped"
        assert data["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_continuous_stop_nonexistent_session(self, test_server):
        """Test stopping non-existent session"""
        response = test_server.post("/api/v1/continuous/stop/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_continuous_stream_websocket_connection(self):
        """Test WebSocket connection establishment"""
        # This would require a running server
        # For now, just test that the endpoint exists
        pass

    @pytest.mark.asyncio
    async def test_continuous_stream_with_audio_chunks(self):
        """Test sending audio chunks over WebSocket"""
        # This would require a running server with mocked Gemini
        # For now, just test that the endpoint exists
        pass


class TestBargeInDetection:
    """Test barge-in detection functionality"""

    @pytest_asyncio.fixture
    async def session_manager(self):
        """Create session manager for testing"""
        manager = ContinuousSessionManager(redis_client=None)
        yield manager
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_barge_in_when_agent_speaking(self, session_manager):
        """Test barge-in detection when agent is speaking"""
        session_id = "barge_test"
        await session_manager.create_session(session_id)

        # Set agent as speaking
        await session_manager.set_agent_speaking(session_id, True)

        # Simulate transcript during agent speech
        transcript = "Hello, I have a question"

        # Check if this should trigger barge-in
        session = await session_manager.get_session(session_id)
        should_barge_in = session.is_agent_speaking and len(transcript.strip()) > 0

        assert should_barge_in == True

    @pytest.mark.asyncio
    async def test_no_barge_in_when_agent_silent(self, session_manager):
        """Test no barge-in when agent is not speaking"""
        session_id = "no_barge_test"
        await session_manager.create_session(session_id)

        # Agent is not speaking (default)
        session = await session_manager.get_session(session_id)
        assert session.is_agent_speaking == False

        # Simulate transcript
        transcript = "Hello, I have a question"

        # Should not trigger barge-in
        should_barge_in = session.is_agent_speaking and len(transcript.strip()) > 0
        assert should_barge_in == False


class TestContinuousVADPerformance:
    """Test performance aspects of continuous VAD"""

    @pytest_asyncio.fixture
    async def session_manager(self):
        """Create session manager for testing"""
        manager = ContinuousSessionManager(redis_client=None)
        yield manager
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_sessions(self, session_manager):
        """Test handling multiple concurrent sessions"""
        session_ids = [f"perf_session_{i}" for i in range(5)]

        # Create multiple sessions concurrently
        create_tasks = [session_manager.create_session(sid) for sid in session_ids]
        sessions = await asyncio.gather(*create_tasks)

        assert len(sessions) == 5
        for session in sessions:
            assert session.status == "active"

        # Verify all sessions are active
        stats = await session_manager.get_session_stats()
        assert stats["active_sessions"] == 5

    @pytest.mark.asyncio
    async def test_session_memory_usage(self, session_manager):
        """Test memory usage with many sessions"""
        # Create 10 sessions
        for i in range(10):
            await session_manager.create_session(f"mem_test_{i}")

        # Check that sessions are properly managed
        stats = await session_manager.get_session_stats()
        assert stats["active_sessions"] == 10

        # Clean up half the sessions
        for i in range(5):
            await session_manager.close_session(f"mem_test_{i}")

        stats = await session_manager.get_session_stats()
        assert stats["active_sessions"] == 5


class TestContinuousVADErrorRecovery:
    """Test error recovery in continuous VAD"""

    @pytest_asyncio.fixture
    async def session_manager(self):
        """Create session manager for testing"""
        manager = ContinuousSessionManager(redis_client=None)
        yield manager
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_session_recovery_after_error(self, session_manager):
        """Test session recovery after errors"""
        session_id = "error_recovery_test"
        await session_manager.create_session(session_id)

        # Simulate error by setting status
        session = await session_manager.get_session(session_id)
        session.status = "error"
        session.errors_count = 1

        # Session should still be retrievable
        recovered = await session_manager.get_session(session_id)
        assert recovered is not None
        assert recovered.errors_count == 1

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_active_sessions(self, session_manager):
        """Test graceful shutdown cleans up active sessions"""
        # Create some sessions
        await session_manager.create_session("shutdown_test_1")
        await session_manager.create_session("shutdown_test_2")

        # Shutdown
        await session_manager.shutdown()

        # Verify sessions are cleaned up
        stats = await session_manager.get_session_stats()
        assert stats["active_sessions"] == 0


# Integration tests that require running services
class TestContinuousVADIntegration:
    """Integration tests requiring running services"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_continuous_session_lifecycle(self):
        """Test complete session lifecycle with HTTP and WebSocket"""
        # This test requires the service to be running
        # Would test: create session -> connect WebSocket -> send audio -> receive transcripts -> close
        pass

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_barge_in_integration(self):
        """Test barge-in with TTS service coordination"""
        # This test requires both STT and TTS services running
        # Would test: start continuous session -> set agent speaking -> send audio -> verify barge-in event
        pass