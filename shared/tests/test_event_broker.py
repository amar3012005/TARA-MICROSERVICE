"""
Tests for EventBroker Redis Streams wrapper.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from shared.event_broker import EventBroker
from shared.events import VoiceEvent, EventTypes


class TestEventBroker:
    """Tests for EventBroker class."""
    
    @pytest.fixture
    def broker(self, mock_redis_client):
        """Create an EventBroker instance with mock Redis."""
        return EventBroker(mock_redis_client)
    
    @pytest.mark.asyncio
    async def test_publish_event(self, broker, mock_redis_client, sample_voice_event):
        """Test publishing an event to a stream."""
        stream_key = "voice:stt:session:test"
        
        result = await broker.publish(stream_key, sample_voice_event)
        
        assert result == "1234567890-0"
        mock_redis_client.xadd.assert_called_once()
        
        # Verify the call arguments
        call_args = mock_redis_client.xadd.call_args
        assert call_args[0][0] == stream_key
        assert "event_type" in call_args[0][1]
    
    @pytest.mark.asyncio
    async def test_publish_with_max_len(self, broker, mock_redis_client, sample_voice_event):
        """Test publishing with max stream length."""
        stream_key = "voice:stt:session:test"
        max_len = 5000
        
        await broker.publish(stream_key, sample_voice_event, max_len=max_len)
        
        call_kwargs = mock_redis_client.xadd.call_args[1]
        assert call_kwargs.get("maxlen") == max_len
    
    @pytest.mark.asyncio
    async def test_create_group_success(self, broker, mock_redis_client):
        """Test creating a consumer group."""
        stream_key = "voice:intent:requests"
        group_name = "intent_processors"
        
        result = await broker.create_group(stream_key, group_name)
        
        assert result is True
        mock_redis_client.xgroup_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_group_already_exists(self, broker, mock_redis_client):
        """Test creating a group that already exists."""
        mock_redis_client.xgroup_create.side_effect = Exception("BUSYGROUP Consumer Group name already exists")
        
        stream_key = "voice:intent:requests"
        group_name = "intent_processors"
        
        result = await broker.create_group(stream_key, group_name)
        
        # Should return True even if group exists
        assert result is True
    
    @pytest.mark.asyncio
    async def test_consume_from_stream(self, broker, mock_redis_client):
        """Test consuming events from a stream."""
        mock_redis_client.xread.return_value = [
            (b"voice:stt:session:test", [
                (b"1234567890-0", {
                    b"event_type": b"voice.stt.final",
                    b"session_id": b"test_123",
                    b"source": b"stt",
                    b"timestamp": b"1234567890.0",
                    b"correlation_id": b"abc",
                    b"payload": b'{"text": "hello"}',
                    b"metadata": b'{}'
                })
            ])
        ]
        
        streams = {"voice:stt:session:test": "0"}
        result = await broker.consume(streams, count=10, block=100)
        
        assert len(result) == 1
        mock_redis_client.xread.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_consume_group(self, broker, mock_redis_client):
        """Test consuming from a consumer group."""
        mock_redis_client.xreadgroup.return_value = [
            (b"voice:intent:requests", [
                (b"1234567890-0", {
                    b"event_type": b"voice.intent.detected",
                    b"session_id": b"test_123",
                    b"source": b"intent",
                    b"timestamp": b"1234567890.0",
                    b"correlation_id": b"abc",
                    b"payload": b'{"intent": "greeting"}',
                    b"metadata": b'{}'
                })
            ])
        ]
        
        result = await broker.consume_group(
            group_name="intent_processors",
            consumer_name="worker_1",
            streams={"voice:intent:requests": ">"}
        )
        
        assert len(result) == 1
        mock_redis_client.xreadgroup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_acknowledge_message(self, broker, mock_redis_client):
        """Test acknowledging a processed message."""
        stream_key = "voice:tts:requests"
        group_name = "tts_processors"
        message_id = "1234567890-0"
        
        await broker.ack(stream_key, group_name, message_id)
        
        mock_redis_client.xack.assert_called_once_with(
            stream_key, group_name, message_id
        )
    
    @pytest.mark.asyncio
    async def test_acknowledge_multiple_messages(self, broker, mock_redis_client):
        """Test acknowledging multiple messages."""
        stream_key = "voice:tts:requests"
        group_name = "tts_processors"
        message_ids = ["1234567890-0", "1234567890-1", "1234567890-2"]
        
        await broker.ack(stream_key, group_name, message_ids)
        
        mock_redis_client.xack.assert_called_once_with(
            stream_key, group_name, *message_ids
        )
    
    @pytest.mark.asyncio
    async def test_claim_stuck_messages(self, broker, mock_redis_client):
        """Test claiming stuck/idle messages."""
        mock_redis_client.xautoclaim.return_value = (
            b"0-0",
            [(b"1234567890-0", {b"event_type": b"test"})]
        )
        
        result = await broker.claim_stuck_messages(
            stream_key="voice:tts:requests",
            group_name="tts_processors",
            consumer_name="worker_1",
            min_idle_ms=60000
        )
        
        assert len(result) == 1
        mock_redis_client.xautoclaim.assert_called_once()


class TestEventBrokerEdgeCases:
    """Edge case tests for EventBroker."""
    
    @pytest.fixture
    def broker(self, mock_redis_client):
        return EventBroker(mock_redis_client)
    
    @pytest.mark.asyncio
    async def test_publish_empty_payload(self, broker, mock_redis_client):
        """Test publishing event with empty payload."""
        event = VoiceEvent(
            event_type=EventTypes.TTS_COMPLETE,
            session_id="test",
            source="tts",
            payload={}
        )
        
        await broker.publish("test_stream", event)
        
        mock_redis_client.xadd.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_consume_empty_result(self, broker, mock_redis_client):
        """Test consuming when no messages available."""
        mock_redis_client.xread.return_value = []
        
        result = await broker.consume({"test_stream": "0"})
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_consume_with_timeout(self, broker, mock_redis_client):
        """Test consuming with block timeout."""
        mock_redis_client.xread.return_value = None
        
        result = await broker.consume({"test_stream": "0"}, block=100)
        
        assert result is None or result == []


