"""
Pytest fixtures for shared module tests.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    client = AsyncMock()
    
    # Mock common methods
    client.xadd = AsyncMock(return_value="1234567890-0")
    client.xread = AsyncMock(return_value=[])
    client.xreadgroup = AsyncMock(return_value=[])
    client.xack = AsyncMock(return_value=1)
    client.xgroup_create = AsyncMock(return_value=True)
    client.xautoclaim = AsyncMock(return_value=(b"0-0", []))
    client.xinfo_groups = AsyncMock(return_value=[])
    client.ping = AsyncMock(return_value=True)
    
    return client


@pytest.fixture
def sample_voice_event():
    """Create a sample VoiceEvent for testing."""
    from shared.events import VoiceEvent
    
    return VoiceEvent(
        event_type="voice.stt.final",
        session_id="test_session_123",
        source="test_service",
        payload={"text": "Hello world", "confidence": 0.95},
        metadata={"language": "en"}
    )


@pytest.fixture
def sample_event_dict():
    """Create a sample event as a Redis-compatible dict."""
    return {
        "event_type": "voice.stt.final",
        "session_id": "test_session_123",
        "source": "test_service",
        "timestamp": "1234567890.123",
        "correlation_id": "abc-123-def",
        "payload": '{"text": "Hello world", "confidence": 0.95}',
        "metadata": '{"language": "en"}'
    }


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


