"""
Data models for Leibniz Appointment FSM Microservice

Contains enums, dataclasses, and Pydantic models for appointment booking.

Reference:
    leibniz_agent/leibniz_appointment_fsm.py (lines 87-174) - Original models
"""

import re
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class AppointmentState(Enum):
    """FSM states for appointment booking process"""
    INIT = "init"
    COLLECT_NAME = "collect_name"
    CONFIRM_NAME = "confirm_name"
    COLLECT_EMAIL = "collect_email"
    CONFIRM_EMAIL = "confirm_email"
    COLLECT_PHONE = "collect_phone"
    CONFIRM_PHONE = "confirm_phone"
    COLLECT_DEPARTMENT = "collect_department"
    CONFIRM_DEPARTMENT = "confirm_department"
    COLLECT_APPOINTMENT_TYPE = "collect_appointment_type"
    CONFIRM_APPOINTMENT_TYPE = "confirm_appointment_type"
    COLLECT_DATETIME = "collect_datetime"
    CONFIRM_DATETIME = "confirm_datetime"
    COLLECT_PURPOSE = "collect_purpose"
    CONFIRM_PURPOSE = "confirm_purpose"
    CONFIRM = "confirm"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


# ============================================================================
# Constants
# ============================================================================

# Department options
DEPARTMENTS = {
    "academic_advising": "Academic Advising",
    "faculty": "Faculty Consultation",
    "examination_office": "Examination Office",
    "international_office": "International Office",
    "career_services": "Career Services",
    "counseling": "Psychological Counseling",
    "financial_aid": "Financial Aid",
    "it_services": "IT Services",
    "registration": "Student Registration",
    "admissions": "Admissions Office"
}

# Appointment types by department
APPOINTMENT_TYPES = {
    "academic_advising": [
        "Course selection and registration",
        "Academic planning and degree requirements",
        "Transfer credit evaluation",
        "Academic probation support",
        "Study abroad advising"
    ],
    "faculty": [
        "Research supervision meeting",
        "Thesis/dissertation consultation",
        "Course content questions",
        "Grade appeal discussion",
        "Recommendation letter request"
    ],
    "examination_office": [
        "Exam registration issues",
        "Grade correction request",
        "Exam schedule conflicts",
        "Missing exam results",
        "Exam deferral request"
    ],
    "international_office": [
        "Visa and residence permit questions",
        "International student orientation",
        "Degree recognition",
        "Language course registration",
        "International exchange programs"
    ],
    "career_services": [
        "Resume and cover letter review",
        "Job search strategies",
        "Interview preparation",
        "Internship opportunities",
        "Career counseling"
    ],
    "counseling": [
        "Initial consultation",
        "Academic stress support",
        "Personal counseling",
        "Crisis intervention",
        "Group therapy information"
    ],
    "financial_aid": [
        "Scholarship applications",
        "Student loan questions",
        "Financial aid eligibility",
        "Payment plan setup",
        "Tuition waiver requests"
    ],
    "it_services": [
        "Computer account issues",
        "WiFi and network problems",
        "Software installation help",
        "Email and calendar issues",
        "Password reset"
    ],
    "registration": [
        "Course registration problems",
        "Student ID issues",
        "Enrollment verification",
        "Transcript requests",
        "Student record updates"
    ],
    "admissions": [
        "Application status check",
        "Admission requirements",
        "Document submission",
        "Application fee questions",
        "Program information"
    ]
}

# Validation patterns
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
PHONE_PATTERN_INTL = re.compile(r'^\+[1-9]\d{1,14}$')
PHONE_PATTERN_GERMAN = re.compile(r'^0\d{2,4}[\s/-]?\d{3,}[\s/-]?\d{0,}$')
NAME_PATTERN = re.compile(r"^[a-zA-Z\s\-']{2,50}$")

# Date/time patterns
DATE_RELATIVE = re.compile(r'\b(today|tomorrow|next week|next month)\b', re.IGNORECASE)
DATE_WEEKDAY = re.compile(r'\bnext (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', re.IGNORECASE)
DATE_FORMATTED = re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b')
TIME_PATTERN = re.compile(r'\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)\b', re.IGNORECASE)
TIME_SIMPLE = re.compile(r'\b\d{1,2}\s*(am|pm|AM|PM)\b', re.IGNORECASE)
TIME_RELATIVE = re.compile(r'\b(morning|afternoon|evening)\b', re.IGNORECASE)

# Cancellation keywords
CANCEL_KEYWORDS = [
    "cancel", "stop", "quit", "exit", "never mind", "forget it",
    "cancel that", "stop this", "quit this", "exit this",
    "i changed my mind", "don't want to", "no thanks"
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AppointmentData:
    """Data structure for collected appointment information"""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None  # Key from DEPARTMENTS dict
    appointment_type: Optional[str] = None
    preferred_datetime: Optional[str] = None  # Natural language or formatted
    purpose: Optional[str] = None
    student_id: Optional[str] = None  # Optional field
    preferred_language: str = "English"  # Default to English
    booking_timestamp: Optional[str] = None  # When booking was made

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for submission"""
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "department": DEPARTMENTS.get(self.department, self.department),
            "appointment_type": self.appointment_type,
            "preferred_datetime": self.preferred_datetime,
            "purpose": self.purpose,
            "student_id": self.student_id,
            "preferred_language": self.preferred_language,
            "booking_timestamp": self.booking_timestamp
        }


# ============================================================================
# Pydantic Models for API
# ============================================================================

class SessionCreateRequest(BaseModel):
    """Request model for creating a new appointment session"""
    initial_data: Optional[Dict[str, Any]] = Field(None, description="Optional initial appointment data")


class SessionCreateResponse(BaseModel):
    """Response model for session creation"""
    session_id: str = Field(..., description="Unique session identifier")
    state: str = Field(..., description="Current FSM state")
    response: str = Field(..., description="Initial system response")


class ProcessInputRequest(BaseModel):
    """Request model for processing user input"""
    user_input: str = Field(..., min_length=1, description="User's spoken/typed input")


class ProcessInputResponse(BaseModel):
    """Response model for input processing"""
    response: str = Field(..., description="System response to user")
    state: str = Field(..., description="Current FSM state")
    previous_state: str = Field(..., description="Previous FSM state")
    complete: bool = Field(..., description="Whether booking is complete")
    cancelled: bool = Field(..., description="Whether booking was cancelled")
    data: Optional[Dict[str, Any]] = Field(None, description="Collected appointment data if complete")
    error: Optional[str] = Field(None, description="Error message if any")
    success: bool = Field(..., description="Whether the input was processed successfully")


class SessionStatusResponse(BaseModel):
    """Response model for session status query"""
    session_id: str = Field(..., description="Session identifier")
    state: str = Field(..., description="Current FSM state")
    data: Dict[str, Any] = Field(..., description="Collected appointment data")
    created_at: str = Field(..., description="Session creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    expires_at: str = Field(..., description="Session expiry timestamp")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status: healthy/degraded/unhealthy")
    redis_connected: bool = Field(..., description="Redis connectivity status")
    config_valid: bool = Field(..., description="Configuration validation status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint"""
    total_sessions_created: int = Field(..., description="Total sessions created")
    active_sessions_count: int = Field(..., description="Currently active sessions")
    completed_sessions_count: int = Field(..., description="Successfully completed bookings")
    cancelled_sessions_count: int = Field(..., description="Cancelled bookings")
    average_session_duration: float = Field(..., description="Average session duration in seconds")


class AdminClearSessionsResponse(BaseModel):
    """Response model for clearing sessions"""
    sessions_deleted: int = Field(..., description="Number of sessions deleted")
    message: str = Field(..., description="Operation result message")


class CurrentPromptResponse(BaseModel):
    """Response model for current prompt endpoint"""
    session_id: str = Field(..., description="Session identifier")
    state: str = Field(..., description="Current FSM state")
    prompt: str = Field(..., description="Current prompt to speak")