"""
Configuration for Leibniz Appointment FSM Microservice

Extracted from leibniz_appointment_fsm.py for microservice deployment.

Reference:
    leibniz_agent/leibniz_appointment_fsm.py (lines 180-200) - Original configuration
    leibniz_agent/services/intent/config.py - Configuration pattern
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AppointmentConfig:
    """
    Configuration for Appointment FSM service.

    Attributes:
        redis_url: Redis connection URL (default: redis://localhost:6379)
        session_ttl: Session TTL in seconds (default: 1800 = 30min)
        max_retries: Max validation retries per field (default: 3)
        max_confirmation_attempts: Max empty confirmation attempts (default: 2)
        enable_mongodb: Enable MongoDB storage for completed bookings (default: False)
        mongodb_uri: MongoDB connection URI (optional)
        log_state_transitions: Log FSM state transitions (default: True)
    """

    redis_url: str = "redis://localhost:6379"
    session_ttl: int = 1800
    max_retries: int = 3
    max_confirmation_attempts: int = 2
    enable_mongodb: bool = False
    mongodb_uri: Optional[str] = None
    log_state_transitions: bool = True

    def __post_init__(self):
        """Validate configuration after initialization"""
        # Validate session TTL
        if self.session_ttl <= 0:
            raise ValueError(
                f"session_ttl must be positive, got {self.session_ttl}"
            )

        # Validate max retries
        if self.max_retries < 1:
            raise ValueError(
                f"max_retries must be at least 1, got {self.max_retries}"
            )

        # Validate max confirmation attempts
        if self.max_confirmation_attempts < 1:
            raise ValueError(
                f"max_confirmation_attempts must be at least 1, got {self.max_confirmation_attempts}"
            )

        # Validate Redis URL format
        if not self.redis_url.startswith(("redis://", "rediss://", "unix://")):
            raise ValueError(
                f"redis_url must start with redis://, rediss://, or unix://, got {self.redis_url}"
            )

        # Warn if MongoDB enabled but no URI provided
        if self.enable_mongodb and not self.mongodb_uri:
            logger.warning(
                "️ MONGODB_URI not set but enable_mongodb=True. "
                "Completed bookings will not be persisted to MongoDB."
            )

        # Log configuration if enabled
        if self.log_state_transitions:
            logger.info(
                f" AppointmentConfig loaded: redis_url={self.redis_url}, "
                f"session_ttl={self.session_ttl}s, max_retries={self.max_retries}, "
                f"mongodb_enabled={self.enable_mongodb}"
            )

    @staticmethod
    def from_env() -> "AppointmentConfig":
        """
        Load configuration from environment variables.

        Environment Variables:
            LEIBNIZ_REDIS_HOST: Redis host (default: localhost)
            LEIBNIZ_REDIS_PORT: Redis port (default: 6379)
            LEIBNIZ_REDIS_DB: Redis database (default: 0)
            LEIBNIZ_APPOINTMENT_SESSION_TTL: Session TTL in seconds (default: 1800)
            LEIBNIZ_APPOINTMENT_MAX_RETRIES: Max validation retries (default: 3)
            LEIBNIZ_APPOINTMENT_MAX_CONFIRMATION_ATTEMPTS: Max confirmation attempts (default: 2)
            LEIBNIZ_APPOINTMENT_ENABLE_MONGODB: Enable MongoDB (default: false)
            LEIBNIZ_APPOINTMENT_MONGODB_URI: MongoDB URI (optional)
            LEIBNIZ_APPOINTMENT_LOG_STATE_TRANSITIONS: Log state transitions (default: true)

        Returns:
            AppointmentConfig instance loaded from environment
        """
        # Build Redis URL from components
        redis_host = os.getenv("LEIBNIZ_REDIS_HOST", "localhost")
        redis_port = int(os.getenv("LEIBNIZ_REDIS_PORT", "6379"))
        redis_db = int(os.getenv("LEIBNIZ_REDIS_DB", "0"))
        redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

        # Load session TTL with validation
        try:
            session_ttl = int(
                os.getenv("LEIBNIZ_APPOINTMENT_SESSION_TTL", "1800")
            )
        except ValueError:
            logger.warning(
                "Invalid LEIBNIZ_APPOINTMENT_SESSION_TTL, using default 1800"
            )
            session_ttl = 1800

        # Load max retries with validation
        try:
            max_retries = int(
                os.getenv("LEIBNIZ_APPOINTMENT_MAX_RETRIES", "3")
            )
        except ValueError:
            logger.warning(
                "Invalid LEIBNIZ_APPOINTMENT_MAX_RETRIES, using default 3"
            )
            max_retries = 3

        # Load max confirmation attempts with validation
        try:
            max_confirmation_attempts = int(
                os.getenv("LEIBNIZ_APPOINTMENT_MAX_CONFIRMATION_ATTEMPTS", "2")
            )
        except ValueError:
            logger.warning(
                "Invalid LEIBNIZ_APPOINTMENT_MAX_CONFIRMATION_ATTEMPTS, using default 2"
            )
            max_confirmation_attempts = 2

        # Load boolean flags
        enable_mongodb = os.getenv(
            "LEIBNIZ_APPOINTMENT_ENABLE_MONGODB", "false"
        ).lower() in ("true", "1", "yes")

        # Load MongoDB URI
        mongodb_uri = os.getenv("LEIBNIZ_APPOINTMENT_MONGODB_URI")

        # Load logging flag
        log_state_transitions = os.getenv(
            "LEIBNIZ_APPOINTMENT_LOG_STATE_TRANSITIONS", "true"
        ).lower() in ("true", "1", "yes")

        return AppointmentConfig(
            redis_url=redis_url,
            session_ttl=session_ttl,
            max_retries=max_retries,
            max_confirmation_attempts=max_confirmation_attempts,
            enable_mongodb=enable_mongodb,
            mongodb_uri=mongodb_uri,
            log_state_transitions=log_state_transitions,
        )