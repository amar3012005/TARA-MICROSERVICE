"""
Test Suite for Leibniz Appointment FSM Microservice

Tests FSM logic, validation, and API endpoints.

Reference:
    leibniz_agent/services/intent/tests/test_intent_flow.py - Test structure pattern
    leibniz_appointment_fsm.py - Original FSM logic for validation
"""

import pytest
from datetime import datetime, timedelta

from ..fsm_manager import AppointmentFSMManager
from ..models import AppointmentState, AppointmentData
from ..config import AppointmentConfig
from ..validation import validate_name, validate_email, validate_phone, validate_datetime, parse_yes_no


# ============================================================================
# Test Configuration
# ============================================================================

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
def fsm_manager(test_config):
    """FSM manager fixture"""
    return AppointmentFSMManager(test_config)


# ============================================================================
# FSM Logic Tests
# ============================================================================

class TestFSMLogic:
    """Test FSM state transitions and logic"""

    def test_initialization(self, fsm_manager):
        """Test FSM initialization"""
        assert fsm_manager.state == AppointmentState.INIT
        assert fsm_manager.data == AppointmentData()
        assert fsm_manager.retry_counts == {}
        assert fsm_manager.confirmation_attempts == {}

    def test_init_handler(self, fsm_manager):
        """Test initial state handler"""
        response = fsm_manager._handle_init()
        assert "appointment" in response.lower()
        assert "name" in response.lower()
        assert fsm_manager.state == AppointmentState.COLLECT_NAME

    def test_name_collection(self, fsm_manager):
        """Test name input collection"""
        fsm_manager.state = AppointmentState.COLLECT_NAME

        # Valid name
        result = fsm_manager.process_input("John Doe")
        assert not result["complete"]
        assert not result["cancelled"]
        assert fsm_manager.data.name == "John Doe"
        assert fsm_manager.state == AppointmentState.CONFIRM_NAME

    def test_email_collection(self, fsm_manager):
        """Test email input collection"""
        fsm_manager.state = AppointmentState.COLLECT_EMAIL
        fsm_manager.data.name = "John Doe"

        # Valid email
        result = fsm_manager.process_input("john.doe@example.com")
        assert not result["complete"]
        assert not result["cancelled"]
        assert fsm_manager.data.email == "john.doe@example.com"
        assert fsm_manager.state == AppointmentState.CONFIRM_EMAIL

    def test_phone_collection(self, fsm_manager):
        """Test phone input collection"""
        fsm_manager.state = AppointmentState.COLLECT_PHONE
        fsm_manager.data.name = "John Doe"
        fsm_manager.data.email = "john.doe@example.com"

        # Valid phone
        result = fsm_manager.process_input("555-123-4567")
        assert not result["complete"]
        assert not result["cancelled"]
        assert fsm_manager.data.phone == "+495551234567"
        assert fsm_manager.state == AppointmentState.CONFIRM_PHONE

    def test_department_collection(self, fsm_manager):
        """Test department selection"""
        fsm_manager.state = AppointmentState.COLLECT_DEPARTMENT
        fsm_manager.data.name = "John Doe"
        fsm_manager.data.email = "john.doe@example.com"
        fsm_manager.data.phone = "555-123-4567"

        # Valid department
        result = fsm_manager.process_input("Academic Advising")
        assert not result["complete"]
        assert not result["cancelled"]
        assert fsm_manager.data.department == "academic_advising"
        assert fsm_manager.state == AppointmentState.CONFIRM_DEPARTMENT

    def test_appointment_type_collection(self, fsm_manager):
        """Test appointment type selection"""
        fsm_manager.state = AppointmentState.COLLECT_APPOINTMENT_TYPE
        fsm_manager.data.name = "John Doe"
        fsm_manager.data.email = "john.doe@example.com"
        fsm_manager.data.phone = "555-123-4567"
        fsm_manager.data.department = "academic_advising"

        # Valid appointment type
        result = fsm_manager.process_input("Course selection and registration")
        assert not result["complete"]
        assert not result["cancelled"]
        assert fsm_manager.data.appointment_type == "Course selection and registration"
        assert fsm_manager.state == AppointmentState.CONFIRM_APPOINTMENT_TYPE

    def test_datetime_collection(self, fsm_manager):
        """Test datetime input collection"""
        fsm_manager.state = AppointmentState.COLLECT_DATETIME
        fsm_manager.data.name = "John Doe"
        fsm_manager.data.email = "john.doe@example.com"
        fsm_manager.data.phone = "555-123-4567"
        fsm_manager.data.department = "academic_advising"
        fsm_manager.data.appointment_type = "Course selection and registration"

        # Valid datetime
        result = fsm_manager.process_input("tomorrow at 2pm")
        assert not result["complete"]
        assert not result["cancelled"]
        assert fsm_manager.data.preferred_datetime is not None
        assert fsm_manager.state == AppointmentState.CONFIRM_DATETIME

    def test_final_confirmation(self, fsm_manager):
        """Test final confirmation"""
        from datetime import datetime

        # Set up complete data
        fsm_manager.state = AppointmentState.CONFIRM
        fsm_manager.data = AppointmentData(
            name="John Doe",
            email="john.doe@example.com",
            phone="555-123-4567",
            department="academic_advising",
            appointment_type="Course selection and registration",
            preferred_datetime="Monday, January 15, 2024 at 02:00 PM"
        )

        # Confirm yes
        result = fsm_manager.process_input("yes")
        assert result["complete"]
        assert not result["cancelled"]
        assert "set" in result["response"].lower() or "confirmed" in result["response"].lower()

    def test_cancellation(self, fsm_manager):
        """Test cancellation"""
        fsm_manager.state = AppointmentState.COLLECT_NAME
        result = fsm_manager.process_input("cancel")
        assert not result["complete"]
        assert result["cancelled"]
        assert "problem" in result["response"].lower()

    def test_serialization(self, fsm_manager):
        """Test FSM serialization/deserialization"""
        # Set up some data
        fsm_manager.state = AppointmentState.COLLECT_EMAIL
        fsm_manager.data.name = "John Doe"
        fsm_manager.retry_counts["name"] = 1

        # Serialize
        data = fsm_manager.to_dict()
        assert data["state"] == "collect_email"
        assert data["data"]["name"] == "John Doe"
        assert data["retry_counts"]["name"] == 1

        # Deserialize
        new_fsm = AppointmentFSMManager.from_dict(data, fsm_manager.config)
        assert new_fsm.state == AppointmentState.COLLECT_EMAIL
        assert new_fsm.data.name == "John Doe"
        assert new_fsm.retry_counts["name"] == 1


# ============================================================================
# Validation Function Tests
# ============================================================================

class TestValidationFunctions:
    """Test individual validation functions"""

    def test_validate_name(self):
        """Test name validation"""
        valid, _ = validate_name("John Doe")
        assert valid == True

        valid, _ = validate_name("Jane Smith")
        assert valid == True

        valid, _ = validate_name("A")
        assert valid == False  # Too short

        valid, _ = validate_name("")
        assert valid == False  # Empty

    def test_validate_email(self):
        """Test email validation"""
        valid, _ = validate_email("john.doe@example.com")
        assert valid == True

        valid, _ = validate_email("jane@example.co.uk")
        assert valid == True

        valid, _ = validate_email("invalid-email")
        assert valid == False

    def test_validate_phone(self):
        """Test phone validation"""
        valid, _ = validate_phone("555-123-4567")
        assert valid == True

        valid, _ = validate_phone("(555) 123-4567")
        assert valid == True

        valid, _ = validate_phone("123")
        assert valid == False  # Too short

    def test_validate_datetime(self):
        """Test datetime validation"""
        # Valid inputs
        result = validate_datetime("Monday, December 15, 2025 at 02:00 PM")
        assert result == (True, "")

        # Invalid inputs (past date)
        past_date = "Monday, January 1, 2020 at 02:00 PM"
        valid, msg = validate_datetime(past_date)
        assert valid == False
        assert "passed" in msg

    def test_parse_yes_no(self):
        """Test yes/no parsing"""
        assert parse_yes_no("yes") == True
        assert parse_yes_no("Yes") == True
        assert parse_yes_no("no") == False
        assert parse_yes_no("maybe") is None


if __name__ == "__main__":
    pytest.main([__file__])