"""
Tests for VoiceEvent and EventTypes.
"""

import pytest
import json
import time
from shared.events import VoiceEvent, EventTypes


class TestVoiceEvent:
    """Tests for VoiceEvent dataclass."""
    
    def test_create_event_with_required_fields(self):
        """Test creating an event with required fields only."""
        event = VoiceEvent(
            event_type=EventTypes.STT_FINAL,
            session_id="session_123",
            source="stt_service",
            payload={"text": "Hello"}
        )
        
        assert event.event_type == EventTypes.STT_FINAL
        assert event.session_id == "session_123"
        assert event.source == "stt_service"
        assert event.payload == {"text": "Hello"}
        assert event.timestamp > 0
        assert event.correlation_id is not None
        assert event.metadata == {}
    
    def test_create_event_with_all_fields(self, sample_voice_event):
        """Test creating an event with all fields."""
        assert sample_voice_event.event_type == "voice.stt.final"
        assert sample_voice_event.session_id == "test_session_123"
        assert sample_voice_event.payload["text"] == "Hello world"
        assert sample_voice_event.metadata["language"] == "en"
    
    def test_to_dict(self, sample_voice_event):
        """Test converting event to dictionary."""
        result = sample_voice_event.to_dict()
        
        assert isinstance(result, dict)
        assert result["event_type"] == "voice.stt.final"
        assert result["session_id"] == "test_session_123"
        assert isinstance(result["payload"], dict)
        assert isinstance(result["metadata"], dict)
    
    def test_to_redis_dict(self, sample_voice_event):
        """Test converting event to Redis-compatible dict."""
        result = sample_voice_event.to_redis_dict()
        
        # All values should be strings
        for key, value in result.items():
            assert isinstance(value, str), f"{key} should be string"
        
        # payload and metadata should be JSON strings
        payload = json.loads(result["payload"])
        assert payload["text"] == "Hello world"
        
        metadata = json.loads(result["metadata"])
        assert metadata["language"] == "en"
    
    def test_from_redis_dict(self, sample_event_dict):
        """Test creating event from Redis dict."""
        event = VoiceEvent.from_redis_dict(sample_event_dict)
        
        assert event.event_type == "voice.stt.final"
        assert event.session_id == "test_session_123"
        assert event.source == "test_service"
        assert event.payload["text"] == "Hello world"
        assert event.metadata["language"] == "en"
    
    def test_from_redis_dict_with_bytes(self):
        """Test creating event from Redis dict with byte values."""
        byte_dict = {
            b"event_type": b"voice.stt.final",
            b"session_id": b"test_session_123",
            b"source": b"test_service",
            b"timestamp": b"1234567890.123",
            b"correlation_id": b"abc-123-def",
            b"payload": b'{"text": "Hello world"}',
            b"metadata": b'{}'
        }
        
        event = VoiceEvent.from_redis_dict(byte_dict)
        
        assert event.event_type == "voice.stt.final"
        assert event.payload["text"] == "Hello world"
    
    def test_roundtrip_conversion(self, sample_voice_event):
        """Test converting to Redis dict and back."""
        redis_dict = sample_voice_event.to_redis_dict()
        restored_event = VoiceEvent.from_redis_dict(redis_dict)
        
        assert restored_event.event_type == sample_voice_event.event_type
        assert restored_event.session_id == sample_voice_event.session_id
        assert restored_event.source == sample_voice_event.source
        assert restored_event.payload == sample_voice_event.payload
        assert restored_event.metadata == sample_voice_event.metadata


class TestEventTypes:
    """Tests for EventTypes constants."""
    
    def test_stt_event_types(self):
        """Test STT event type constants."""
        assert EventTypes.STT_PARTIAL == "voice.stt.partial"
        assert EventTypes.STT_FINAL == "voice.stt.final"
    
    def test_intent_event_types(self):
        """Test Intent event type constants."""
        assert EventTypes.INTENT_DETECTED == "voice.intent.detected"
    
    def test_rag_event_types(self):
        """Test RAG event type constants."""
        assert EventTypes.RAG_ANSWER_READY == "voice.rag.answer_ready"
    
    def test_tts_event_types(self):
        """Test TTS event type constants."""
        assert EventTypes.TTS_REQUEST == "voice.tts.request"
        assert EventTypes.TTS_CHUNK_READY == "voice.tts.chunk_ready"
        assert EventTypes.TTS_COMPLETE == "voice.tts.complete"
        assert EventTypes.TTS_CANCEL == "voice.tts.cancel"
    
    def test_webrtc_event_types(self):
        """Test WebRTC event type constants."""
        assert EventTypes.PLAYBACK_STARTED == "voice.webrtc.playback_started"
        assert EventTypes.PLAYBACK_DONE == "voice.webrtc.playback_done"
        assert EventTypes.BARGE_IN == "voice.webrtc.barge_in"
    
    def test_orchestrator_event_types(self):
        """Test Orchestrator event type constants."""
        assert EventTypes.ORCHESTRATOR_STATE == "voice.orchestrator.state"
        assert EventTypes.ORCHESTRATOR_ERROR == "voice.orchestrator.error"


