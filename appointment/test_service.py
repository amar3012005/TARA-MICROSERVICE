#!/usr/bin/env python3
"""
Simple test script for appointment service functionality.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import modules properly
from config import AppointmentConfig
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

# Import app module using importlib
import importlib.util
spec = importlib.util.spec_from_file_location("app", os.path.join(os.path.dirname(__file__), "app.py"))
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)
app = app_module.app

def test_basic_functionality():
    """Test basic service functionality"""
    print(" Testing appointment service functionality...")

    # Create test config
    config = AppointmentConfig()

    # Create mock Redis with proper async methods
    mock_redis = AsyncMock()
    stored_sessions = {}
    
    async def mock_setex(key, ttl, value):
        stored_sessions[key] = value
        return True
    
    async def mock_get(key):
        return stored_sessions.get(key)
    
    async def mock_ping():
        return True
    
    async def mock_scan(cursor, match=None, count=None):
        # Return empty results for simplicity
        return 0, []
    
    async def mock_ttl(key):
        return 1800  # Return TTL in seconds
    
    async def mock_incr(key):
        return 1
    
    mock_redis.setex = mock_setex
    mock_redis.get = mock_get
    mock_redis.ping = mock_ping
    mock_redis.scan = mock_scan
    mock_redis.ttl = mock_ttl
    mock_redis.incr = mock_incr
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True
    mock_redis.incr.return_value = 1
    mock_redis.ping.return_value = True

    # Patch dependencies and config
    with patch.object(app_module, 'redis_client', mock_redis), \
         patch.object(app_module, 'config', config):
        # Manually set the config in the app module
        app_module.config = config
        client = TestClient(app)

        # Test health endpoint
        response = client.get('/health')
        if response.status_code != 200:
            print(f" Health check failed: {response.status_code}")
            return False
        print(" Health check passed")

        # Test root endpoint
        response = client.get('/')
        if response.status_code != 200:
            print(f" Root endpoint failed: {response.status_code}")
            return False
        print(" Root endpoint passed")

        # Test session creation
        response = client.post('/api/v1/session/create')
        if response.status_code != 200:
            print(f" Session creation failed: {response.status_code}")
            return False

        data = response.json()
        if 'session_id' not in data:
            print(" Invalid session creation response")
            return False
        print(" Session creation passed")

        session_id = data['session_id']

        # Test session processing
        response = client.post(f'/api/v1/session/{session_id}/process',
                             json={'user_input': 'John Doe'})
        if response.status_code != 200:
            print(f" Session processing failed: {response.status_code}")
            return False
        print(" Session processing passed")

        # Test session status
        response = client.get(f'/api/v1/session/{session_id}/status')
        if response.status_code != 200:
            print(f" Session status failed: {response.status_code}")
            return False
        print(" Session status passed")

        # Test metrics
        response = client.get('/metrics')
        if response.status_code != 200:
            print(f" Metrics endpoint failed: {response.status_code}")
            return False
        print(" Metrics endpoint passed")

        print(" All tests passed!")
        return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)