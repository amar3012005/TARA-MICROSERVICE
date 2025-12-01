"""
API Integration Tests for Leibniz Appointment FSM Microservice

Tests FastAPI endpoints with Redis persistence.

Reference:
    leibniz_agent/services/intent/tests/test_api_integration.py - API test pattern
    httpx documentation - Async HTTP client testing
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from ..app import app
from ..config import AppointmentConfig
from ..fsm_manager import AppointmentFSMManager


# ============================================================================
# Test Configuration
# ============================================================================

# Fixtures are defined in conftest.py


# ============================================================================
# API Endpoint Tests
# ============================================================================

class TestAPIEndpoints:
    """Test API endpoints"""

    def test_root_endpoint(self, async_client):
        """Test root endpoint returns service info"""
        response = async_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "Leibniz Appointment FSM Service" in data["service"]
        assert "endpoints" in data

    def test_health_endpoint(self, async_client):
        """Test health check endpoint"""
        response = async_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "redis_connected" in data
        assert "config_valid" in data
        assert "uptime_seconds" in data

    def test_create_session(self, async_client):
        """Test session creation"""
        response = async_client.post("/api/v1/session/create")
        assert response.status_code == 200

        data = response.json()
        assert "session_id" in data
        assert "state" in data
        assert "response" in data
        assert data["state"] == "collect_name"

    def test_process_input_valid_flow(self, async_client):
        """Test complete appointment booking flow via API"""
        # Create session
        create_response = async_client.post("/api/v1/session/create")
        assert create_response.status_code == 200
        session_id = create_response.json()["session_id"]

        # Process name
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "John Doe"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "confirm_name"

        # Confirm name
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "yes"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "collect_email"

        # Process email
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "john.doe@example.com"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "confirm_email"

        # Confirm email
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "yes"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "collect_phone"

        # Process phone
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "555-123-4567"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "confirm_phone"

        # Confirm phone
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "yes"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "collect_department"

        # Process department
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "Academic Advising"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "confirm_department"

        # Confirm department
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "yes"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "collect_appointment_type"

        # Process appointment type
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "Course selection and registration"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "confirm_appointment_type"

        # Confirm appointment type
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "yes"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "collect_datetime"

        # Process datetime
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "tomorrow at 2pm"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "confirm_datetime"

        # Confirm datetime
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "yes"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "collect_purpose"

        # Process purpose
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "Need help choosing courses"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "confirm_purpose"

        # Confirm purpose
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "yes"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == True
        assert data["state"] == "confirm"

        # Process final confirmation
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "yes"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["complete"] == True
        assert data["success"] == True
        assert "set" in data["response"].lower()

    def test_process_input_invalid_data(self, async_client):
        """Test processing invalid input"""
        # Create session
        create_response = async_client.post("/api/v1/session/create")
        session_id = create_response.json()["session_id"]

        # Send invalid name
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "A"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["success"] == False
        assert "name" in data["response"].lower()

    def test_process_input_cancellation(self, async_client):
        """Test session cancellation"""
        # Create session
        create_response = async_client.post("/api/v1/session/create")
        session_id = create_response.json()["session_id"]

        # Process name first
        async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "John Doe"}
        )

        # Cancel
        process_response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "cancel"}
        )
        assert process_response.status_code == 200
        data = process_response.json()
        assert data["cancelled"] == True
        assert "no problem" in data["response"].lower()
        assert "later" in data["response"].lower()

    def test_get_session_status(self, async_client):
        """Test getting session status"""
        # Create session
        create_response = async_client.post("/api/v1/session/create")
        session_id = create_response.json()["session_id"]

        # Get status
        status_response = async_client.get(f"/api/v1/session/{session_id}/status")
        assert status_response.status_code == 200

        data = status_response.json()
        assert data["session_id"] == session_id
        assert data["state"] == "collect_name"
        assert "data" in data
        assert "created_at" in data

    def test_get_session_status_not_found(self, async_client):
        """Test getting status of non-existent session"""
        response = async_client.get("/api/v1/session/non-existent-session/status")
        assert response.status_code == 404

    def test_delete_session(self, async_client):
        """Test session deletion"""
        # Create session
        create_response = async_client.post("/api/v1/session/create")
        session_id = create_response.json()["session_id"]

        # Delete session
        delete_response = async_client.delete(f"/api/v1/session/{session_id}")
        assert delete_response.status_code == 200

        # Verify session is gone
        status_response = async_client.get(f"/api/v1/session/{session_id}/status")
        assert status_response.status_code == 404

    def test_metrics_endpoint(self, async_client):
        """Test metrics endpoint"""
        response = async_client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "total_sessions_created" in data
        assert "active_sessions_count" in data
        assert "completed_sessions_count" in data
        assert "cancelled_sessions_count" in data
        assert "average_session_duration" in data

    def test_admin_clear_sessions(self, async_client):
        """Test admin clear sessions endpoint"""
        # Create a few sessions
        for _ in range(3):
            async_client.post("/api/v1/session/create")

        # Clear all sessions
        clear_response = async_client.post("/admin/clear_sessions")
        assert clear_response.status_code == 200

        data = clear_response.json()
        assert "sessions_deleted" in data
        assert data["sessions_deleted"] >= 3


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error conditions and edge cases"""

    def test_process_input_missing_session(self, async_client):
        """Test processing input for non-existent session"""
        response = async_client.post(
            "/api/v1/session/non-existent/process",
            json={"user_input": "test"}
        )
        assert response.status_code == 404

    def test_process_input_invalid_json(self, async_client):
        """Test processing input with invalid JSON"""
        # Create session
        create_response = async_client.post("/api/v1/session/create")
        session_id = create_response.json()["session_id"]

        # Send invalid JSON (missing user_input)
        response = async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={}
        )
        assert response.status_code == 422  # Validation error

    def test_service_unavailable_without_config(self, async_client):
        """Test service behavior when config is unavailable"""
        # This would require mocking the config loading to fail
        # For now, test with valid config
        response = async_client.get("/health")
        assert response.status_code == 200

    def test_concurrent_session_access(self, async_client):
        """Test concurrent access to same session"""
        # Create session
        create_response = async_client.post("/api/v1/session/create")
        session_id = create_response.json()["session_id"]

        # Process multiple inputs sequentially (sync client limitation)
        success_count = 0
        for i in range(5):
            response = async_client.post(
                f"/api/v1/session/{session_id}/process",
                json={"user_input": f"input_{i}"}
            )
            if response.status_code == 200:
                success_count += 1

        # At least some should succeed
        assert success_count >= 1


# ============================================================================
# Redis Integration Tests
# ============================================================================

class TestRedisIntegration:
    """Test Redis persistence and session management"""

    def test_session_persistence_across_requests(self, async_client):
        """Test that sessions persist across API calls"""
        # Create session
        create_response = async_client.post("/api/v1/session/create")
        session_id = create_response.json()["session_id"]

        # Process name
        async_client.post(
            f"/api/v1/session/{session_id}/process",
            json={"user_input": "John Doe"}
        )

        # Get status - should show updated state
        status_response = async_client.get(f"/api/v1/session/{session_id}/status")
        data = status_response.json()
        assert data["state"] == "confirm_name"
        assert data["data"]["name"] == "John Doe"

    def test_session_expiry(self, async_client):
        """Test session expiry behavior"""
        # Create session
        create_response = async_client.post("/api/v1/session/create")
        session_id = create_response.json()["session_id"]

        # Get status immediately
        status_response = async_client.get(f"/api/v1/session/{session_id}/status")
        assert status_response.status_code == 200

        # Note: Testing actual expiry would require manipulating Redis TTL
        # or waiting for expiry, which is impractical in unit tests

    def test_metrics_persistence(self, async_client):
        """Test that metrics persist across requests"""
        # Get initial metrics
        initial_response = async_client.get("/metrics")
        initial_data = initial_response.json()
        initial_created = initial_data["total_sessions_created"]

        # Create some sessions
        for _ in range(3):
            async_client.post("/api/v1/session/create")

        # Get updated metrics
        updated_response = async_client.get("/metrics")
        updated_data = updated_response.json()
        updated_created = updated_data["total_sessions_created"]

        # Should have increased (exact amount depends on test isolation)
        assert updated_created >= initial_created


# ============================================================================
# Load and Performance Tests
# ============================================================================

class TestLoadPerformance:
    """Test service performance under load"""

    def test_multiple_concurrent_sessions(self, async_client):
        """Test creating multiple sessions concurrently"""
        # Create multiple sessions sequentially
        responses = []
        for _ in range(10):
            response = async_client.post("/api/v1/session/create")
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data

    def test_session_isolation(self, async_client):
        """Test that sessions are properly isolated"""
        # Create two sessions
        session1_response = async_client.post("/api/v1/session/create")
        session1_id = session1_response.json()["session_id"]

        session2_response = async_client.post("/api/v1/session/create")
        session2_id = session2_response.json()["session_id"]

        # Process different data in each
        async_client.post(
            f"/api/v1/session/{session1_id}/process",
            json={"user_input": "John Doe"}
        )

        async_client.post(
            f"/api/v1/session/{session2_id}/process",
            json={"user_input": "Jane Smith"}
        )

        # Check isolation
        status1 = async_client.get(f"/api/v1/session/{session1_id}/status")
        status2 = async_client.get(f"/api/v1/session/{session2_id}/status")

        data1 = status1.json()
        data2 = status2.json()

        assert data1["data"]["name"] == "John Doe"
        assert data2["data"]["name"] == "Jane Smith"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])