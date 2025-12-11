"""
Event Consumer Base Class with Retry Logic and Dead Letter Queue (DLQ)

Provides a resilient event consumer framework for Redis Streams with:
- Automatic retry with exponential backoff
- Dead Letter Queue for failed events
- Consumer group support for horizontal scaling
- Graceful shutdown handling
- Metrics collection

Reference:
    docs/EVENT_DRIVEN_TRANSFORMATION_GUIDE.md - Architecture guide
"""

import asyncio
import logging
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Awaitable
from enum import Enum

from redis.asyncio import Redis

from .events import VoiceEvent
from .event_broker import EventBroker

logger = logging.getLogger(__name__)


class ProcessingResult(Enum):
    """Result of processing an event."""
    SUCCESS = "success"
    RETRY = "retry"
    FAIL = "fail"  # Send to DLQ


@dataclass
class ConsumerConfig:
    """Configuration for an event consumer."""
    stream_key: str
    group_name: str
    consumer_name: str
    max_retries: int = 3
    initial_backoff_ms: int = 100
    max_backoff_ms: int = 5000
    backoff_multiplier: float = 2.0
    batch_size: int = 10
    block_ms: int = 1000
    dlq_stream_key: Optional[str] = None  # Default: {stream_key}:dlq
    claim_idle_ms: int = 60000  # Claim messages stuck for 60s
    claim_batch_size: int = 100
    

@dataclass
class ConsumerMetrics:
    """Metrics for an event consumer."""
    events_processed: int = 0
    events_succeeded: int = 0
    events_retried: int = 0
    events_failed: int = 0
    events_sent_to_dlq: int = 0
    events_claimed: int = 0
    processing_errors: int = 0
    last_event_time: float = 0.0
    total_processing_time_ms: float = 0.0
    started_at: float = field(default_factory=time.time)
    
    def avg_processing_time_ms(self) -> float:
        if self.events_processed == 0:
            return 0.0
        return self.total_processing_time_ms / self.events_processed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "events_processed": self.events_processed,
            "events_succeeded": self.events_succeeded,
            "events_retried": self.events_retried,
            "events_failed": self.events_failed,
            "events_sent_to_dlq": self.events_sent_to_dlq,
            "events_claimed": self.events_claimed,
            "processing_errors": self.processing_errors,
            "avg_processing_time_ms": self.avg_processing_time_ms(),
            "uptime_seconds": time.time() - self.started_at
        }


class EventConsumer(ABC):
    """
    Abstract base class for resilient event consumers.
    
    Subclasses must implement:
        - process_event(event: VoiceEvent) -> ProcessingResult
        - filter_event(event: VoiceEvent) -> bool (optional)
    
    Features:
        - Consumer group support for horizontal scaling
        - Automatic retries with exponential backoff
        - Dead Letter Queue for failed events
        - Stuck message claiming
        - Graceful shutdown
        - Metrics collection
    
    Usage:
        class MyConsumer(EventConsumer):
            async def process_event(self, event: VoiceEvent) -> ProcessingResult:
                # Process the event
                return ProcessingResult.SUCCESS
        
        consumer = MyConsumer(redis_client, broker, config)
        await consumer.start()
    """
    
    def __init__(
        self,
        redis_client: Redis,
        broker: EventBroker,
        config: ConsumerConfig
    ):
        self.redis = redis_client
        self.broker = broker
        self.config = config
        self.metrics = ConsumerMetrics()
        
        # Set default DLQ stream if not specified
        if not config.dlq_stream_key:
            config.dlq_stream_key = f"{config.stream_key}:dlq"
        
        # Internal state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._claim_task: Optional[asyncio.Task] = None
        self._retry_counts: Dict[str, int] = {}  # message_id -> retry count
        
        logger.info(f"EventConsumer initialized: {config.consumer_name}")
        logger.info(f"  Stream: {config.stream_key}")
        logger.info(f"  Group: {config.group_name}")
        logger.info(f"  DLQ: {config.dlq_stream_key}")
    
    @abstractmethod
    async def process_event(self, event: VoiceEvent) -> ProcessingResult:
        """
        Process an event. Must be implemented by subclasses.
        
        Args:
            event: The event to process
            
        Returns:
            ProcessingResult indicating success, retry, or fail
        """
        pass
    
    def filter_event(self, event: VoiceEvent) -> bool:
        """
        Optional filter to skip events. Override in subclass if needed.
        
        Args:
            event: The event to filter
            
        Returns:
            True to process the event, False to skip
        """
        return True
    
    async def start(self):
        """Start the consumer."""
        if self._running:
            logger.warning(f"{self.config.consumer_name} already running")
            return
        
        self._running = True
        
        # Ensure consumer group exists
        await self.broker.create_group(
            self.config.stream_key,
            self.config.group_name,
            start_id="0"  # Start from beginning to not miss any messages
        )
        
        # Start main consumer loop
        self._task = asyncio.create_task(
            self._consume_loop(),
            name=f"consumer_{self.config.consumer_name}"
        )
        
        # Start stuck message claimer
        self._claim_task = asyncio.create_task(
            self._claim_loop(),
            name=f"claimer_{self.config.consumer_name}"
        )
        
        logger.info(f"✅ {self.config.consumer_name} started")
    
    async def stop(self):
        """Stop the consumer gracefully."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel tasks
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        if self._claim_task:
            self._claim_task.cancel()
            try:
                await self._claim_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"🛑 {self.config.consumer_name} stopped")
        logger.info(f"   Final metrics: {self.metrics.to_dict()}")
    
    async def _consume_loop(self):
        """Main consumer loop."""
        logger.info(f"🎧 {self.config.consumer_name} consuming from {self.config.stream_key}")
        
        while self._running:
            try:
                # Read from consumer group
                result = await self.broker.consume_group(
                    group_name=self.config.group_name,
                    consumer_name=self.config.consumer_name,
                    streams={self.config.stream_key: ">"},  # Only new messages
                    count=self.config.batch_size,
                    block=self.config.block_ms
                )
                
                if not result:
                    continue
                
                # Process messages
                for stream_name, messages in result:
                    for message_id, message_data in messages:
                        await self._process_message(message_id, message_data)
                        
            except asyncio.CancelledError:
                logger.info(f"{self.config.consumer_name} consume loop cancelled")
                break
            except Exception as e:
                logger.error(f"{self.config.consumer_name} consume error: {e}")
                self.metrics.processing_errors += 1
                await asyncio.sleep(1.0)  # Back off on error
        
        logger.info(f"{self.config.consumer_name} consume loop ended")
    
    async def _claim_loop(self):
        """Background loop to claim stuck messages."""
        while self._running:
            try:
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
                # Claim stuck messages
                claimed = await self.broker.claim_stuck_messages(
                    stream_key=self.config.stream_key,
                    group_name=self.config.group_name,
                    consumer_name=self.config.consumer_name,
                    min_idle_ms=self.config.claim_idle_ms,
                    count=self.config.claim_batch_size
                )
                
                if claimed:
                    self.metrics.events_claimed += len(claimed)
                    logger.info(f"📥 {self.config.consumer_name} claimed {len(claimed)} stuck messages")
                    
                    # Process claimed messages
                    for message_id, message_data in claimed:
                        await self._process_message(message_id, message_data)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self.config.consumer_name} claim loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_message(self, message_id: str, message_data: Dict[str, Any]):
        """Process a single message with retry logic."""
        start_time = time.time()
        
        try:
            # Parse event
            event = VoiceEvent.from_redis_dict(message_data)
            
            # Apply filter
            if not self.filter_event(event):
                # Skip but acknowledge
                await self.broker.ack(
                    self.config.stream_key,
                    self.config.group_name,
                    message_id
                )
                return
            
            self.metrics.events_processed += 1
            self.metrics.last_event_time = time.time()
            
            # Process event
            result = await self.process_event(event)
            
            processing_time = (time.time() - start_time) * 1000
            self.metrics.total_processing_time_ms += processing_time
            
            if result == ProcessingResult.SUCCESS:
                self.metrics.events_succeeded += 1
                # Acknowledge message
                await self.broker.ack(
                    self.config.stream_key,
                    self.config.group_name,
                    message_id
                )
                # Clear retry count
                if message_id in self._retry_counts:
                    del self._retry_counts[message_id]
                    
            elif result == ProcessingResult.RETRY:
                await self._handle_retry(message_id, event)
                
            elif result == ProcessingResult.FAIL:
                await self._send_to_dlq(message_id, event, "processing_failed")
                
        except Exception as e:
            logger.error(f"Error processing message {message_id}: {e}")
            self.metrics.processing_errors += 1
            
            # Try to parse event for DLQ
            try:
                event = VoiceEvent.from_redis_dict(message_data)
                await self._handle_retry(message_id, event)
            except:
                # Can't even parse, send raw data to DLQ
                await self._send_raw_to_dlq(message_id, message_data, str(e))
    
    async def _handle_retry(self, message_id: str, event: VoiceEvent):
        """Handle retry with exponential backoff."""
        retry_count = self._retry_counts.get(message_id, 0) + 1
        self._retry_counts[message_id] = retry_count
        
        if retry_count > self.config.max_retries:
            logger.warning(f"Message {message_id} exceeded max retries ({self.config.max_retries})")
            await self._send_to_dlq(message_id, event, "max_retries_exceeded")
            return
        
        self.metrics.events_retried += 1
        
        # Calculate backoff
        backoff_ms = min(
            self.config.initial_backoff_ms * (self.config.backoff_multiplier ** (retry_count - 1)),
            self.config.max_backoff_ms
        )
        
        logger.info(f"Retrying message {message_id} (attempt {retry_count}/{self.config.max_retries}) in {backoff_ms}ms")
        
        # Wait and then let it be reprocessed
        await asyncio.sleep(backoff_ms / 1000.0)
        
        # The message will be reclaimed in the next claim loop
        # Or we can republish it (safer approach)
        try:
            await self.broker.publish(self.config.stream_key, event)
            # Acknowledge the old message
            await self.broker.ack(
                self.config.stream_key,
                self.config.group_name,
                message_id
            )
        except Exception as e:
            logger.error(f"Failed to republish message for retry: {e}")
    
    async def _send_to_dlq(self, message_id: str, event: VoiceEvent, reason: str):
        """Send a failed event to the Dead Letter Queue."""
        self.metrics.events_failed += 1
        self.metrics.events_sent_to_dlq += 1
        
        try:
            # Add failure metadata
            event.metadata["dlq_reason"] = reason
            event.metadata["dlq_timestamp"] = time.time()
            event.metadata["original_message_id"] = message_id
            event.metadata["retry_count"] = self._retry_counts.get(message_id, 0)
            event.metadata["consumer"] = self.config.consumer_name
            
            # Publish to DLQ
            await self.broker.publish(self.config.dlq_stream_key, event)
            
            # Acknowledge original message
            await self.broker.ack(
                self.config.stream_key,
                self.config.group_name,
                message_id
            )
            
            # Clear retry count
            if message_id in self._retry_counts:
                del self._retry_counts[message_id]
            
            logger.warning(f"Message {message_id} sent to DLQ: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to send message {message_id} to DLQ: {e}")
    
    async def _send_raw_to_dlq(self, message_id: str, raw_data: Dict[str, Any], error: str):
        """Send unparseable raw data to DLQ."""
        self.metrics.events_failed += 1
        self.metrics.events_sent_to_dlq += 1
        
        try:
            # Create a wrapper event for raw data
            dlq_event = VoiceEvent(
                event_type="dlq.unparseable",
                session_id="unknown",
                source=self.config.consumer_name,
                payload={
                    "raw_data": {k: str(v)[:1000] for k, v in raw_data.items()},  # Truncate large values
                    "error": error
                },
                metadata={
                    "dlq_reason": "unparseable_event",
                    "dlq_timestamp": time.time(),
                    "original_message_id": message_id
                }
            )
            
            await self.broker.publish(self.config.dlq_stream_key, dlq_event)
            
            # Acknowledge original message
            await self.broker.ack(
                self.config.stream_key,
                self.config.group_name,
                message_id
            )
            
            logger.warning(f"Unparseable message {message_id} sent to DLQ")
            
        except Exception as e:
            logger.error(f"Failed to send raw message {message_id} to DLQ: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.to_dict()


class MultiStreamConsumer(EventConsumer):
    """
    Event consumer that can consume from multiple streams simultaneously.
    
    Useful for orchestrator-style consumers that need to react to events
    from STT, Intent, RAG, TTS, and WebRTC streams.
    """
    
    def __init__(
        self,
        redis_client: Redis,
        broker: EventBroker,
        config: ConsumerConfig,
        additional_streams: List[str] = None
    ):
        super().__init__(redis_client, broker, config)
        self.additional_streams = additional_streams or []
        
    async def _consume_loop(self):
        """Multi-stream consumer loop."""
        # Build streams dict
        streams = {self.config.stream_key: ">"}
        for stream in self.additional_streams:
            streams[stream] = ">"
        
        logger.info(f"🎧 {self.config.consumer_name} consuming from {list(streams.keys())}")
        
        while self._running:
            try:
                result = await self.broker.consume_group(
                    group_name=self.config.group_name,
                    consumer_name=self.config.consumer_name,
                    streams=streams,
                    count=self.config.batch_size,
                    block=self.config.block_ms
                )
                
                if not result:
                    continue
                
                for stream_name, messages in result:
                    for message_id, message_data in messages:
                        # Add stream info to processing
                        message_data["__source_stream"] = stream_name
                        await self._process_message(message_id, message_data)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self.config.consumer_name} consume error: {e}")
                self.metrics.processing_errors += 1
                await asyncio.sleep(1.0)


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def create_consumer(
    redis_client: Redis,
    broker: EventBroker,
    stream_key: str,
    group_name: str,
    consumer_name: str,
    handler: Callable[[VoiceEvent], Awaitable[ProcessingResult]],
    **kwargs
) -> EventConsumer:
    """
    Factory function to create a simple event consumer with a handler function.
    
    Args:
        redis_client: Redis client
        broker: EventBroker instance
        stream_key: Stream to consume from
        group_name: Consumer group name
        consumer_name: Unique consumer name
        handler: Async function to process events
        **kwargs: Additional config options
        
    Returns:
        EventConsumer instance
    """
    config = ConsumerConfig(
        stream_key=stream_key,
        group_name=group_name,
        consumer_name=consumer_name,
        **kwargs
    )
    
    class SimpleConsumer(EventConsumer):
        async def process_event(self, event: VoiceEvent) -> ProcessingResult:
            return await handler(event)
    
    return SimpleConsumer(redis_client, broker, config)


