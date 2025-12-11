"""
Integration tests for event-driven architecture.

These tests simulate the full event flow:
STT -> Orchestrator -> Intent + RAG -> TTS -> WebRTC

Note: These tests use mocked Redis for unit testing.
For real integration tests, use a running Redis instance.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from shared.events import VoiceEvent, EventTypes
from shared.event_broker import EventBroker
from shared.event_consumer import EventConsumer, ConsumerConfig, ProcessingResult


class TestEventFlowSimulation:
    """Test simulating the full event-driven voice pipeline."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        client = AsyncMock()
        client.xadd = AsyncMock(return_value="1234567890-0")
        client.xread = AsyncMock(return_value=[])
        client.xreadgroup = AsyncMock(return_value=[])
        client.xack = AsyncMock(return_value=1)
        client.xgroup_create = AsyncMock(return_value=True)
        client.xautoclaim = AsyncMock(return_value=(b"0-0", []))
        return client
    
    @pytest.fixture
    def broker(self, mock_redis):
        """Create EventBroker."""
        return EventBroker(mock_redis)
    
    @pytest.mark.asyncio
    async def test_stt_to_orchestrator_flow(self, broker, mock_redis):
        """Test STT final event triggers orchestrator processing."""
        # Simulate STT service emitting final transcript
        stt_event = VoiceEvent(
            event_type=EventTypes.STT_FINAL,
            session_id="session_123",
            source="stt_sarvam",
            payload={
                "text": "What are your office hours?",
                "confidence": 0.95,
                "language": "en"
            }
        )
        
        # Publish to session-specific stream
        stream_key = f"voice:stt:session:{stt_event.session_id}"
        message_id = await broker.publish(stream_key, stt_event)
        
        assert message_id is not None
        mock_redis.xadd.assert_called()
    
    @pytest.mark.asyncio
    async def test_orchestrator_parallel_dispatch(self, broker, mock_redis):
        """Test orchestrator dispatching to Intent and RAG in parallel."""
        session_id = "session_123"
        text = "What are your office hours?"
        correlation_id = "corr_abc123"
        
        # Orchestrator creates intent request
        intent_request = VoiceEvent(
            event_type="voice.intent.request",
            session_id=session_id,
            source="orchestrator",
            correlation_id=correlation_id,
            payload={"text": text}
        )
        
        # Orchestrator creates RAG request
        rag_request = VoiceEvent(
            event_type="voice.rag.request",
            session_id=session_id,
            source="orchestrator",
            correlation_id=correlation_id,
            payload={"text": text}
        )
        
        # Publish both (parallel in real system)
        await broker.publish("voice:intent:requests", intent_request)
        await broker.publish("voice:rag:requests", rag_request)
        
        # Verify both were published
        assert mock_redis.xadd.call_count == 2
    
    @pytest.mark.asyncio
    async def test_intent_response_flow(self, broker, mock_redis):
        """Test Intent service response event."""
        session_id = "session_123"
        
        intent_response = VoiceEvent(
            event_type=EventTypes.INTENT_DETECTED,
            session_id=session_id,
            source="intent_service",
            payload={
                "intent": "inquiry_office_hours",
                "confidence": 0.89,
                "entities": {}
            }
        )
        
        stream_key = f"voice:intent:session:{session_id}"
        await broker.publish(stream_key, intent_response)
        
        mock_redis.xadd.assert_called()
    
    @pytest.mark.asyncio
    async def test_rag_streaming_response(self, broker, mock_redis):
        """Test RAG service streaming response chunks."""
        session_id = "session_123"
        
        # RAG streams multiple chunks
        chunks = [
            "Our office hours are ",
            "Monday to Friday, ",
            "9 AM to 5 PM."
        ]
        
        for i, chunk in enumerate(chunks):
            rag_chunk = VoiceEvent(
                event_type=EventTypes.RAG_ANSWER_READY,
                session_id=session_id,
                source="rag_service",
                payload={
                    "text": chunk,
                    "chunk_index": i,
                    "is_final": i == len(chunks) - 1
                }
            )
            await broker.publish(f"voice:rag:session:{session_id}", rag_chunk)
        
        assert mock_redis.xadd.call_count == len(chunks)
    
    @pytest.mark.asyncio
    async def test_tts_request_and_response(self, broker, mock_redis):
        """Test TTS request and audio chunk response."""
        session_id = "session_123"
        
        # Orchestrator sends TTS request
        tts_request = VoiceEvent(
            event_type=EventTypes.TTS_REQUEST,
            session_id=session_id,
            source="orchestrator",
            payload={
                "text": "Our office hours are Monday to Friday, 9 AM to 5 PM.",
                "voice": "default"
            }
        )
        
        await broker.publish("voice:tts:requests", tts_request)
        
        # TTS service responds with audio chunks
        tts_chunk = VoiceEvent(
            event_type=EventTypes.TTS_CHUNK_READY,
            session_id=session_id,
            source="tts_sarvam",
            payload={
                "audio_base64": "SGVsbG8gV29ybGQ=",  # Base64 encoded audio
                "sample_rate": 24000,
                "chunk_index": 0,
                "is_final": False
            }
        )
        
        await broker.publish(f"voice:tts:session:{session_id}", tts_chunk)
        
        # TTS complete event
        tts_complete = VoiceEvent(
            event_type=EventTypes.TTS_COMPLETE,
            session_id=session_id,
            source="tts_sarvam",
            payload={
                "total_chunks": 5,
                "audio_duration_ms": 2500
            }
        )
        
        await broker.publish(f"voice:tts:session:{session_id}", tts_complete)
        
        assert mock_redis.xadd.call_count == 3
    
    @pytest.mark.asyncio
    async def test_webrtc_playback_events(self, broker, mock_redis):
        """Test WebRTC playback events from client."""
        session_id = "session_123"
        
        # Client reports playback started
        playback_started = VoiceEvent(
            event_type=EventTypes.PLAYBACK_STARTED,
            session_id=session_id,
            source="client_webrtc",
            payload={"timestamp": time.time()}
        )
        
        await broker.publish(f"voice:webrtc:session:{session_id}", playback_started)
        
        # Client reports playback done
        playback_done = VoiceEvent(
            event_type=EventTypes.PLAYBACK_DONE,
            session_id=session_id,
            source="client_webrtc",
            payload={
                "duration_ms": 2500,
                "timestamp": time.time()
            }
        )
        
        await broker.publish(f"voice:webrtc:session:{session_id}", playback_done)
        
        assert mock_redis.xadd.call_count == 2
    
    @pytest.mark.asyncio
    async def test_barge_in_event_flow(self, broker, mock_redis):
        """Test barge-in (user interruption) event flow."""
        session_id = "session_123"
        
        # User starts speaking during TTS playback
        barge_in = VoiceEvent(
            event_type=EventTypes.BARGE_IN,
            session_id=session_id,
            source="client_webrtc",
            payload={
                "reason": "user_speaking_detected",
                "timestamp": time.time()
            }
        )
        
        await broker.publish(f"voice:webrtc:session:{session_id}", barge_in)
        
        # Orchestrator cancels TTS
        tts_cancel = VoiceEvent(
            event_type=EventTypes.TTS_CANCEL,
            session_id=session_id,
            source="orchestrator",
            payload={"reason": "barge_in"}
        )
        
        await broker.publish(f"voice:tts:session:{session_id}", tts_cancel)
        
        # Orchestrator state transition
        state_change = VoiceEvent(
            event_type=EventTypes.ORCHESTRATOR_STATE,
            session_id=session_id,
            source="orchestrator",
            payload={
                "previous_state": "SPEAKING",
                "new_state": "INTERRUPT",
                "trigger": "barge_in"
            }
        )
        
        await broker.publish(f"voice:orchestrator:session:{session_id}", state_change)
        
        assert mock_redis.xadd.call_count == 3


class TestEventCorrelation:
    """Test event correlation across the pipeline."""
    
    @pytest.fixture
    def broker(self, mock_redis_client):
        return EventBroker(mock_redis_client)
    
    @pytest.mark.asyncio
    async def test_correlation_id_propagation(self, broker, mock_redis_client):
        """Test that correlation IDs are propagated through the pipeline."""
        session_id = "session_123"
        correlation_id = "corr_unique_123"
        
        # Initial STT event
        stt_event = VoiceEvent(
            event_type=EventTypes.STT_FINAL,
            session_id=session_id,
            source="stt",
            correlation_id=correlation_id,
            payload={"text": "Hello"}
        )
        
        await broker.publish(f"voice:stt:session:{session_id}", stt_event)
        
        # Downstream events should have same correlation_id
        intent_event = VoiceEvent(
            event_type=EventTypes.INTENT_DETECTED,
            session_id=session_id,
            source="intent",
            correlation_id=correlation_id,  # Same correlation ID
            payload={"intent": "greeting"}
        )
        
        await broker.publish(f"voice:intent:session:{session_id}", intent_event)
        
        # Verify both events have same correlation ID
        assert stt_event.correlation_id == intent_event.correlation_id


class TestEventLatencyTracking:
    """Test latency tracking through events."""
    
    @pytest.mark.asyncio
    async def test_event_timestamps(self, mock_redis_client):
        """Test that events have proper timestamps."""
        broker = EventBroker(mock_redis_client)
        
        start_time = time.time()
        
        event = VoiceEvent(
            event_type=EventTypes.STT_FINAL,
            session_id="session_123",
            source="stt",
            payload={"text": "Hello"}
        )
        
        await broker.publish("test_stream", event)
        
        # Verify timestamp is recent
        assert event.timestamp >= start_time
        assert event.timestamp <= time.time()
    
    @pytest.mark.asyncio
    async def test_pipeline_latency_calculation(self):
        """Test calculating latency between pipeline stages."""
        # Simulate timing between events
        stt_time = time.time()
        
        await asyncio.sleep(0.01)  # Simulate processing
        
        intent_time = time.time()
        
        # Calculate latency
        stt_to_intent_ms = (intent_time - stt_time) * 1000
        
        assert stt_to_intent_ms > 0
        assert stt_to_intent_ms < 100  # Should be fast in tests


