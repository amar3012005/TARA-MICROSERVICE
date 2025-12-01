"""
Test configuration and fixtures for appointment service tests.

Mocks Redis client and configuration for isolated testing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from leibniz_agent.services.appointment.app import app
from leibniz_agent.services.appointment.config import AppointmentConfig


@pytest.fixture
def test_config():
    """Test configuration fixture"""
    return AppointmentConfig(
        redis_url="redis://localhost:6379",
        session_ttl=1800,
        max_retries=3,
        max_confirmation_attempts=2
    )


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    mock_client = AsyncMock()
    # Store for session data persistence across calls
    stored_data = {}
    
    async def mock_get(key):
        return stored_data.get(key)
    
    async def mock_setex(key, ttl, value):
        stored_data[key] = value
        return True
    
    async def mock_delete(key):
        if key in stored_data:
            del stored_data[key]
            return 1
        return 0
    
    mock_client.get.side_effect = mock_get
    mock_client.setex.side_effect = mock_setex
    mock_client.delete.side_effect = mock_delete
    mock_client.incr.return_value = 1
    mock_client.ping.return_value = True
    mock_client.ttl.return_value = 1800
    mock_client.scan.return_value = (0, [])
    return mock_client


@pytest.fixture
def client(mock_redis_client, test_config):
    """FastAPI test client with mocked dependencies"""
    with patch('leibniz_agent.services.appointment.app.redis_client', mock_redis_client), \
         patch('leibniz_agent.services.appointment.app.config', test_config):
        yield TestClient(app)


@pytest.fixture
def async_client(mock_redis_client, test_config):
    """HTTP client for testing with mocked dependencies"""
    with patch('leibniz_agent.services.appointment.app.redis_client', mock_redis_client), \
         patch('leibniz_agent.services.appointment.app.config', test_config):
        yield TestClient(app)