"""
Unified Event Schema for Leibniz TARA Microservices.

Defines the standard event structure for the event-driven architecture using Redis Streams.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import time
import json
import uuid

@dataclass
class VoiceEvent:
    """
    Standard event model for all voice pipeline events.
    """
    event_type: str
    session_id: str
    payload: Dict[str, Any]
    source: str
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_redis_dict(self) -> Dict[str, str]:
        """
        Convert to Redis-compatible dictionary (all values must be strings/bytes).
        The payload and metadata are JSON serialized.
        """
        return {
            "event_type": self.event_type,
            "session_id": self.session_id,
            "source": self.source,
            "timestamp": str(self.timestamp),
            "correlation_id": self.correlation_id,
            "payload": json.dumps(self.payload),
            "metadata": json.dumps(self.metadata)
        }

    @classmethod
    def from_redis_dict(cls, data: Dict[str, Any]) -> 'VoiceEvent':
        """Create VoiceEvent from Redis stream data."""
        # Handle bytes if returned from Redis
        def decode(val):
            return val.decode('utf-8') if isinstance(val, bytes) else val

        return cls(
            event_type=decode(data.get("event_type") or data.get(b"event_type")),
            session_id=decode(data.get("session_id") or data.get(b"session_id")),
            source=decode(data.get("source") or data.get(b"source")),
            timestamp=float(decode(data.get("timestamp") or data.get(b"timestamp") or 0.0)),
            correlation_id=decode(data.get("correlation_id") or data.get(b"correlation_id")),
            payload=json.loads(decode(data.get("payload") or data.get(b"payload") or "{}")),
            metadata=json.loads(decode(data.get("metadata") or data.get(b"metadata") or "{}"))
        )

    def validate_payload(self) -> None:
        """Validate payload schema for critical event types."""
        required = {
            EventTypes.STT_FINAL: ["text", "is_final"],
            EventTypes.BARGE_IN: ["reason"],
            EventTypes.PLAYBACK_DONE: ["duration_ms"],
            EventTypes.TTS_CHUNK_READY: ["audio_base64"],
        }
        
        fields = required.get(self.event_type)
        if fields:
            missing = [f for f in fields if f not in self.payload]
            if missing:
                raise ValueError(
                    f"Event {self.event_type} payload missing required fields: {missing}. "
                    f"Payload: {self.payload}"
                )

# Event Type Constants
class EventTypes:
    # STT Events
    STT_PARTIAL = "voice.stt.partial"
    STT_FINAL = "voice.stt.final"
    
    # Intent Events
    INTENT_DETECTED = "voice.intent.detected"
    
    # RAG Events
    RAG_ANSWER_READY = "voice.rag.answer_ready"  # Streaming chunk or full
    
    # TTS Events
    TTS_REQUEST = "voice.tts.request"      # Orchestrator -> TTS
    TTS_CHUNK_READY = "voice.tts.chunk_ready"  # TTS -> Orchestrator/Client
    TTS_COMPLETE = "voice.tts.complete"
    TTS_CANCEL = "voice.tts.cancel"        # Barge-in cancellation
    
    # WebRTC/Client Events
    PLAYBACK_STARTED = "voice.webrtc.playback_started"
    PLAYBACK_DONE = "voice.webrtc.playback_done"
    BARGE_IN = "voice.webrtc.barge_in"
    
    # Orchestrator Events
    ORCHESTRATOR_STATE = "voice.orchestrator.state"
    ORCHESTRATOR_ERROR = "voice.orchestrator.error"
