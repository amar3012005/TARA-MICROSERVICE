"""
Redis Streams Event Broker Wrapper.

Provides a simplified async interface for publishing and consuming events via Redis Streams.
"""

import logging
import asyncio
import json
from typing import Dict, List, Tuple, Optional, Union
import redis.asyncio as redis
from .events import VoiceEvent

logger = logging.getLogger(__name__)

class EventBroker:
    """
    Wrapper around Redis Streams for voice agent events.
    """
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def publish(self, stream_key: str, event: VoiceEvent, max_len: int = 10000) -> str:
        """
        Publish an event to a Redis Stream.
        
        Args:
            stream_key: The Redis key for the stream (e.g., "voice:stt:session:123")
            event: VoiceEvent object
            max_len: Maximum stream length (older entries are trimmed)
            
        Returns:
            The message ID of the published event.
        """
        try:
            # Convert event to Redis-compatible dict
            data = event.to_redis_dict()
            
            # XADD: Add to stream with approx maxlen
            message_id = await self.redis.xadd(stream_key, data, maxlen=max_len, approximate=True)
            return message_id
        except Exception as e:
            logger.error(f"Failed to publish event to {stream_key}: {e}")
            raise

    async def create_group(self, stream_key: str, group_name: str, start_id: str = "$") -> bool:
        """
        Create a consumer group if it doesn't exist.
        
        Args:
            stream_key: Stream key
            group_name: Consumer group name
            start_id: Start reading from ('$' = new messages, '0' = all messages)
            
        Returns:
            True if created, False if already exists
        """
        try:
            # MKSTREAM=True creates the stream if it doesn't exist
            await self.redis.xgroup_create(stream_key, group_name, id=start_id, mkstream=True)
            return True
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                return False  # Group already exists
            raise e

    async def consume(
        self, 
        streams: Dict[str, str], 
        count: int = 10, 
        block: int = 100
    ) -> List[Tuple[str, List[Tuple[str, Dict]]]]:
        """
        Consume events from one or more streams (XREAD).
        
        Args:
            streams: Dict mapping stream_key -> last_id (e.g. {"voice:stt": "$"})
            count: Max messages per stream
            block: Block time in ms (0 = infinite)
            
        Returns:
            List of [stream_key, [(msg_id, data), ...]]
        """
        return await self.redis.xread(streams, count=count, block=block)

    async def consume_group(
        self,
        group_name: str,
        consumer_name: str,
        streams: Dict[str, str],
        count: int = 10,
        block: int = 100
    ) -> List[Tuple[str, List[Tuple[str, Dict]]]]:
        """
        Consume events as part of a consumer group (XREADGROUP).
        
        Args:
            group_name: Consumer group name
            consumer_name: Unique consumer name (e.g. "pod-1")
            streams: Dict mapping stream_key -> id (">" for new messages)
            count: Max messages
            block: Block time in ms
            
        Returns:
            List of [stream_key, [(msg_id, data), ...]]
        """
        return await self.redis.xreadgroup(
            groupname=group_name,
            consumername=consumer_name,
            streams=streams,
            count=count,
            block=block
        )

    async def ack(self, stream_key: str, group_name: str, message_ids: Union[str, List[str]]):
        """Acknowledge processed messages."""
        if isinstance(message_ids, str):
            message_ids = [message_ids]
        if message_ids:
            await self.redis.xack(stream_key, group_name, *message_ids)

    async def claim_stuck_messages(
        self,
        stream_key: str,
        group_name: str,
        consumer_name: str,
        min_idle_time: int = 60000,
        count: int = 10
    ) -> List[Tuple[str, Dict]]:
        """
        Claim pending messages that have been idle for too long (XAUTOCLAIM).
        Useful for recovering from crashed consumers.
        
        Args:
            min_idle_time: Min time in ms that a message must be pending
            
        Returns:
            List of (msg_id, data) tuples
        """
        # XAUTOCLAIM returns: start_id, messages_list
        _, messages = await self.redis.xautoclaim(
            stream_key, group_name, consumer_name, min_idle_time, count=count
        )
        return messages
