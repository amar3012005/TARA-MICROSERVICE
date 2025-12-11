import json
import logging
import time
from typing import Any, Dict, Optional


class StructuredLogger:
    """
    Lightweight structured logger that emits JSON log lines.

    This wraps a standard `logging.Logger` instance so existing logging
    configuration (handlers, formatters, sinks) continues to work, while
    providing a consistent machine-parseable structure.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def _log(
        self,
        level: str,
        event_type: str,
        message: str,
        session_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "event_type": event_type,
            "message": message,
        }

        if session_id is not None:
            entry["session_id"] = session_id

        if data:
            # Avoid mutating caller data
            entry["data"] = dict(data)

        log_method = getattr(self.logger, level.lower(), self.logger.info)
        try:
            log_method(json.dumps(entry, default=str))
        except Exception:
            # Fallback to plain logging if JSON serialization fails
            log_method(f"[STRUCTURED_LOG_FALLBACK] {entry}")

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #

    def event(
        self,
        session_id: Optional[str],
        event_type: str,
        message: str,
        level: str = "INFO",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Generic structured event."""
        self._log(level=level, event_type=event_type, message=message, session_id=session_id, data=data)

    def state_transition(
        self,
        session_id: str,
        old_state: str,
        new_state: str,
        trigger: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Structured log for state transitions."""
        payload = {"old_state": old_state, "new_state": new_state, "trigger": trigger}
        if data:
            payload["data"] = data
        self._log(
            level="INFO",
            event_type="state_transition",
            message=f"{old_state} -> {new_state} ({trigger})",
            session_id=session_id,
            data=payload,
        )

    def event_received(
        self,
        session_id: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Structured log when an event is received/consumed."""
        self._log(
            level="INFO",
            event_type="event_received",
            message=f"Received {event_type}",
            session_id=session_id,
            data={"event_type": event_type, "payload": payload or {}},
        )

    def latency_recorded(
        self,
        session_id: str,
        operation: str,
        duration_ms: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Structured log for latency measurements."""
        data = {"operation": operation, "duration_ms": duration_ms}
        if extra:
            data.update(extra)
        self._log(
            level="INFO",
            event_type="latency",
            message=f"{operation} took {duration_ms:.0f}ms",
            session_id=session_id,
            data=data,
        )


