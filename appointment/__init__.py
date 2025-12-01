"""
Leibniz Appointment FSM Microservice - Stateful appointment booking with Redis persistence

This service provides a complete appointment booking system using a finite state machine
pattern. It manages multi-step conversations for collecting user information (name, email,
phone, department, appointment type, datetime, purpose) with robust validation, retry logic,
and confirmation flows.

Key Features:
- 17-state FSM for structured data collection
- Natural language datetime parsing (today, tomorrow, next week, weekdays)
- Redis-based session persistence (30min TTL)
- Validation with retry logic (max 3 attempts per field)
- Confirmation flow with correction ability
- Cancellation available at any time
- Optional MongoDB storage for completed bookings

Architecture:
- FastAPI web framework for REST endpoints
- Redis for session state management
- FSM Manager for business logic
- Validation utilities for data checking
- Pydantic models for API contracts
"""

__version__ = "1.0.0"