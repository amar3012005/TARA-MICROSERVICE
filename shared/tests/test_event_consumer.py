"""
Tests for EventConsumer with retry logic and DLQ.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from shared.event_consumer import (
    EventConsumer, 
    MultiStreamConsumer,
    ConsumerConfig, 
    ConsumerMetrics,
    ProcessingResult,
    create_consumer
)
from shared.event_broker import EventBroker
from shared.events import VoiceEvent, EventTypes


class TestConsumerConfig:
    """Tests for ConsumerConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ConsumerConfig(
            stream_key="test_stream",
            group_name="test_group",
            consumer_name="test_consumer"
        )
        
        assert config.stream_key == "test_stream"
        assert config.group_name == "test_group"
        assert config.consumer_name == "test_consumer"
        assert config.max_retries == 3
        assert config.initial_backoff_ms == 100
        assert config.max_backoff_ms == 5000
        assert config.batch_size == 10
        assert config.block_ms == 1000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConsumerConfig(
            stream_key="test_stream",
            group_name="test_group",
            consumer_name="test_consumer",
            max_retries=5,
            initial_backoff_ms=200,
            dlq_stream_key="test_stream:dlq"
        )
        
        assert config.max_retries == 5
        assert config.initial_backoff_ms == 200
        assert config.dlq_stream_key == "test_stream:dlq"


class TestConsumerMetrics:
    """Tests for ConsumerMetrics dataclass."""
    
    def test_initial_metrics(self):
        """Test initial metric values."""
        metrics = ConsumerMetrics()
        
        assert metrics.events_processed == 0
        assert metrics.events_succeeded == 0
        assert metrics.events_retried == 0
        assert metrics.events_failed == 0
        assert metrics.events_sent_to_dlq == 0
    
    def test_avg_processing_time_empty(self):
        """Test average processing time with no events."""
        metrics = ConsumerMetrics()
        
        assert metrics.avg_processing_time_ms() == 0.0
    
    def test_avg_processing_time(self):
        """Test average processing time calculation."""
        metrics = ConsumerMetrics()
        metrics.events_processed = 10
        metrics.total_processing_time_ms = 500.0
        
        assert metrics.avg_processing_time_ms() == 50.0
    
    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ConsumerMetrics()
        metrics.events_processed = 100
        metrics.events_succeeded = 95
        metrics.events_failed = 5
        
        result = metrics.to_dict()
        
        assert result["events_processed"] == 100
        assert result["events_succeeded"] == 95
        assert result["events_failed"] == 5
        assert "uptime_seconds" in result


class TestEventConsumer:
    """Tests for EventConsumer base class."""
    
    @pytest.fixture
    def consumer_config(self):
        """Create test consumer config."""
        return ConsumerConfig(
            stream_key="voice:test:requests",
            group_name="test_processors",
            consumer_name="test_worker_1",
            max_retries=3,
            initial_backoff_ms=10,  # Short for tests
            dlq_stream_key="voice:test:requests:dlq"
        )
    
    @pytest.fixture
    def mock_broker(self, mock_redis_client):
        """Create mock EventBroker."""
        broker = EventBroker(mock_redis_client)
        broker.publish = AsyncMock(return_value="1234567890-0")
        broker.create_group = AsyncMock(return_value=True)
        broker.consume_group = AsyncMock(return_value=[])
        broker.ack = AsyncMock()
        broker.claim_stuck_messages = AsyncMock(return_value=[])
        return broker
    
    @pytest.fixture
    def simple_consumer(self, mock_redis_client, mock_broker, consumer_config):
        """Create a simple test consumer."""
        class TestConsumer(EventConsumer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.processed_events = []
            
            async def process_event(self, event: VoiceEvent) -> ProcessingResult:
                self.processed_events.append(event)
                return ProcessingResult.SUCCESS
        
        return TestConsumer(mock_redis_client, mock_broker, consumer_config)
    
    @pytest.mark.asyncio
    async def test_consumer_start_creates_group(self, simple_consumer, mock_broker):
        """Test that starting consumer creates the consumer group."""
        await simple_consumer.start()
        
        mock_broker.create_group.assert_called_once_with(
            "voice:test:requests",
            "test_processors",
            start_id="0"
        )
        
        await simple_consumer.stop()
    
    @pytest.mark.asyncio
    async def test_consumer_stop_gracefully(self, simple_consumer):
        """Test graceful consumer shutdown."""
        await simple_consumer.start()
        await simple_consumer.stop()
        
        assert simple_consumer._running is False
    
    @pytest.mark.asyncio
    async def test_process_success_increments_metrics(self, simple_consumer, mock_broker):
        """Test that successful processing increments metrics."""
        event = VoiceEvent(
            event_type=EventTypes.TTS_REQUEST,
            session_id="test",
            source="test",
            payload={"text": "hello"}
        )
        
        await simple_consumer._process_message("1234567890-0", event.to_redis_dict())
        
        assert simple_consumer.metrics.events_processed == 1
        assert simple_consumer.metrics.events_succeeded == 1
        mock_broker.ack.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_retry_increments_metrics(self, mock_redis_client, mock_broker, consumer_config):
        """Test that retry increments retry metrics."""
        class RetryConsumer(EventConsumer):
            async def process_event(self, event: VoiceEvent) -> ProcessingResult:
                return ProcessingResult.RETRY
        
        consumer = RetryConsumer(mock_redis_client, mock_broker, consumer_config)
        event = VoiceEvent(
            event_type=EventTypes.TTS_REQUEST,
            session_id="test",
            source="test",
            payload={"text": "hello"}
        )
        
        await consumer._process_message("1234567890-0", event.to_redis_dict())
        
        assert consumer.metrics.events_retried == 1
    
    @pytest.mark.asyncio
    async def test_process_fail_sends_to_dlq(self, mock_redis_client, mock_broker, consumer_config):
        """Test that failed events are sent to DLQ."""
        class FailConsumer(EventConsumer):
            async def process_event(self, event: VoiceEvent) -> ProcessingResult:
                return ProcessingResult.FAIL
        
        consumer = FailConsumer(mock_redis_client, mock_broker, consumer_config)
        event = VoiceEvent(
            event_type=EventTypes.TTS_REQUEST,
            session_id="test",
            source="test",
            payload={"text": "hello"}
        )
        
        await consumer._process_message("1234567890-0", event.to_redis_dict())
        
        assert consumer.metrics.events_failed == 1
        assert consumer.metrics.events_sent_to_dlq == 1
        
        # Verify DLQ publish was called
        dlq_calls = [call for call in mock_broker.publish.call_args_list 
                     if "dlq" in str(call)]
        assert len(dlq_calls) == 1
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded_sends_to_dlq(self, mock_redis_client, mock_broker, consumer_config):
        """Test that exceeding max retries sends to DLQ."""
        class RetryConsumer(EventConsumer):
            async def process_event(self, event: VoiceEvent) -> ProcessingResult:
                return ProcessingResult.RETRY
        
        consumer = RetryConsumer(mock_redis_client, mock_broker, consumer_config)
        consumer._retry_counts["1234567890-0"] = consumer_config.max_retries  # Already at max
        
        event = VoiceEvent(
            event_type=EventTypes.TTS_REQUEST,
            session_id="test",
            source="test",
            payload={"text": "hello"}
        )
        
        await consumer._process_message("1234567890-0", event.to_redis_dict())
        
        assert consumer.metrics.events_sent_to_dlq == 1
    
    @pytest.mark.asyncio
    async def test_filter_event_skips_filtered(self, mock_redis_client, mock_broker, consumer_config):
        """Test that filtered events are skipped but acknowledged."""
        class FilteringConsumer(EventConsumer):
            async def process_event(self, event: VoiceEvent) -> ProcessingResult:
                return ProcessingResult.SUCCESS
            
            def filter_event(self, event: VoiceEvent) -> bool:
                return event.payload.get("should_process", False)
        
        consumer = FilteringConsumer(mock_redis_client, mock_broker, consumer_config)
        
        # Event that should be filtered out
        event = VoiceEvent(
            event_type=EventTypes.TTS_REQUEST,
            session_id="test",
            source="test",
            payload={"text": "hello", "should_process": False}
        )
        
        await consumer._process_message("1234567890-0", event.to_redis_dict())
        
        # Should be acknowledged but not counted as processed
        mock_broker.ack.assert_called_once()
        assert consumer.metrics.events_processed == 0
    
    def test_get_metrics(self, simple_consumer):
        """Test getting current metrics."""
        simple_consumer.metrics.events_processed = 50
        simple_consumer.metrics.events_succeeded = 45
        
        metrics = simple_consumer.get_metrics()
        
        assert metrics["events_processed"] == 50
        assert metrics["events_succeeded"] == 45


class TestCreateConsumer:
    """Tests for create_consumer factory function."""
    
    @pytest.mark.asyncio
    async def test_create_simple_consumer(self, mock_redis_client):
        """Test creating consumer with handler function."""
        broker = EventBroker(mock_redis_client)
        processed = []
        
        async def handler(event: VoiceEvent) -> ProcessingResult:
            processed.append(event)
            return ProcessingResult.SUCCESS
        
        consumer = create_consumer(
            redis_client=mock_redis_client,
            broker=broker,
            stream_key="test_stream",
            group_name="test_group",
            consumer_name="test_consumer",
            handler=handler
        )
        
        assert consumer is not None
        assert consumer.config.stream_key == "test_stream"


class TestProcessingResult:
    """Tests for ProcessingResult enum."""
    
    def test_result_values(self):
        """Test result enum values."""
        assert ProcessingResult.SUCCESS.value == "success"
        assert ProcessingResult.RETRY.value == "retry"
        assert ProcessingResult.FAIL.value == "fail"


