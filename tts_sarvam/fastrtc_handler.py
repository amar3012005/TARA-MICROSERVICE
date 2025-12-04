"""
FastRTC Handler for TTS Streaming Service

Provides audio output streaming via FastRTC for browser playback.
"""

import asyncio
import json
import logging
import time
from typing import Tuple, Set, Optional

import numpy as np
from fastrtc import AsyncStreamHandler

logger = logging.getLogger(__name__)


class FastRTCTTSHandler(AsyncStreamHandler):
    """
    FastRTC AsyncStreamHandler that streams TTS audio to browser.
    
    Receives synthesized audio from TTS queue and streams it via FastRTC emit().
    """
    
    # Registry of active handler instances
    active_instances: Set['FastRTCTTSHandler'] = set()
    default_chunk_duration_ms: int = 40
    default_min_buffer_chunks: int = 1  # Start playing immediately when audio arrives
    
    def __init__(
        self,
        tts_queue=None,
        redis_client=None,
        chunk_duration_ms: Optional[int] = None,
        min_buffer_chunks: Optional[int] = None
    ):
        """
        Initialize FastRTC TTS Handler.
        
        Args:
            tts_queue: TTSStreamingQueue instance (injected at runtime)
        """
        super().__init__()
        self.tts_queue = tts_queue
        self.redis_client = redis_client
        self.session_id = f"fastrtc_tts_{int(time.time())}"
        self._started = False
        self._sample_rate = 24000
        self._chunk_duration_ms = None
        self._chunk_size_samples = None
        self._min_buffer_chunks = None
        self._buffer_warmed = False
        self._configure_stream_parameters(
            chunk_duration_ms=chunk_duration_ms,
            min_buffer_chunks=min_buffer_chunks
        )
        self._audio_output_queue = asyncio.Queue()
        self._current_audio_chunk = None
        self._chunk_position = 0
        self._playback_end_time = 0.0  # Track when current audio finishes
        
        logger.info("🔊 FastRTC TTS Handler initialized")
        logger.info(f"   Handler instance: {id(self)}")
        logger.info(f"   Session ID: {self.session_id}")
    
    def _configure_stream_parameters(
        self,
        chunk_duration_ms: Optional[int] = None,
        min_buffer_chunks: Optional[int] = None
    ):
        """Configure streaming parameters such as chunk duration and buffering."""
        resolved_chunk_ms = (
            chunk_duration_ms if chunk_duration_ms is not None
            else FastRTCTTSHandler.default_chunk_duration_ms
        )
        resolved_buffer_chunks = (
            min_buffer_chunks if min_buffer_chunks is not None
            else FastRTCTTSHandler.default_min_buffer_chunks
        )
        
        self._chunk_duration_ms = max(5, int(resolved_chunk_ms))
        self._min_buffer_chunks = max(1, int(resolved_buffer_chunks))
        self._chunk_size_samples = max(
            1,
            int(self._sample_rate * (self._chunk_duration_ms / 1000.0))
        )
    
    async def start_up(self):
        """Called when WebRTC stream starts"""
        self._started = True
        self._audio_output_queue = asyncio.Queue()
        self._current_audio_chunk = None
        self._chunk_position = 0
        self._buffer_warmed = False
        
        # Register this instance
        FastRTCTTSHandler.active_instances.add(self)
        
        logger.info("=" * 70)
        logger.info("🚀 FastRTC TTS stream started")
        logger.info(f"   Handler instance: {id(self)} | Session: {self.session_id}")
        logger.info(f"   Active instances: {len(FastRTCTTSHandler.active_instances)}")
        logger.info(
            f"   Chunk duration: {self._chunk_duration_ms}ms | "
            f"Min buffer chunks: {self._min_buffer_chunks}"
        )
        logger.info("=" * 70)
        
        if not self.tts_queue:
            # It's okay if queue is missing, we might be receiving via broadcast
            logger.info("ℹ️ TTS queue not directly injected - waiting for broadcast audio")
        else:
            logger.info("✅ TTS queue ready | Ready for audio streaming")
            logger.info("📊 Flow: Text → TTS Queue → FastRTC → Browser Speakers")
        
        await self._publish_connection_event()

        logger.info("=" * 70)
    
    async def receive(self, audio: tuple) -> None:
        """
        Receive method (not used for TTS output, but required by AsyncStreamHandler).
        
        This handler is for OUTPUT only (TTS), not input (STT).
        """
        # TTS service doesn't receive audio - this is output-only
        pass
    
    async def emit(self) -> Tuple[int, np.ndarray]:
        """
        Emit audio chunks to browser for playback.
        
        Returns:
            Tuple of (sample_rate, audio_array) for FastRTC streaming
        """
        silence_chunk = np.zeros(self._chunk_size_samples, dtype=np.int16)
        sleep_interval = self._chunk_duration_ms / 1000.0
        
        if self._audio_output_queue.empty() and self._current_audio_chunk is None:
            await asyncio.sleep(sleep_interval)
            return (self._sample_rate, silence_chunk)
        
        if not self._buffer_warmed:
            buffered_chunks = self._audio_output_queue.qsize()
            if self._current_audio_chunk is not None:
                buffered_chunks += 1
            
            if buffered_chunks < self._min_buffer_chunks:
                logger.debug(
                    "⏳ FastRTC buffer warming: queue_size=%s/%s",
                    buffered_chunks,
                    self._min_buffer_chunks
                )
                await asyncio.sleep(sleep_interval)
                return (self._sample_rate, silence_chunk)
            
            self._buffer_warmed = True
            logger.debug(
                "✅ FastRTC buffer warmed: queue_size=%s (min=%s)",
                buffered_chunks,
                self._min_buffer_chunks
            )
        
        # Process current chunk or get next one
        if self._current_audio_chunk is None:
            try:
                self._current_audio_chunk = self._audio_output_queue.get_nowait()
                self._chunk_position = 0
            except asyncio.QueueEmpty:
                await asyncio.sleep(sleep_interval)
                return (self._sample_rate, silence_chunk)
        
        if self._current_audio_chunk is not None:
            audio_data, sample_rate = self._current_audio_chunk
            
            # Ensure audio_data is numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.int16)
            
            # Get next chunk from current audio
            remaining = len(audio_data) - self._chunk_position
            if remaining > 0:
                take_samples = min(self._chunk_size_samples, remaining)
                chunk = audio_data[self._chunk_position:self._chunk_position + take_samples]
                self._chunk_position += take_samples
                
                # Ensure 1D array (FastRTC handles 1D as mono)
                if chunk.ndim > 1:
                    chunk = chunk.flatten()
                
                # Ensure int16
                if chunk.dtype != np.int16:
                    chunk = chunk.astype(np.int16)

                return (sample_rate, chunk)
            else:
                # Current chunk finished, get next
                self._current_audio_chunk = None
                self._chunk_position = 0
        
        await asyncio.sleep(sleep_interval)
        return (self._sample_rate, silence_chunk)
    
    async def add_audio_chunk(self, audio_bytes: bytes, sample_rate: int):
        """
        Add audio chunk to output queue for streaming.
        
        Args:
            audio_bytes: Raw audio bytes (int16 PCM)
            sample_rate: Sample rate of audio
        """
        try:
            # Convert bytes to numpy array (int16)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Track when this audio chunk will finish playing
            duration_s = len(audio_bytes) / (sample_rate * 2)  # int16 = 2 bytes/sample
            self._playback_end_time = time.time() + duration_s
            
            # Add to queue as int16 (do not convert to float32)
            await self._audio_output_queue.put((audio_int16, sample_rate))
            
            queue_size = self._audio_output_queue.qsize()
            logger.debug(
                "Added audio chunk to FastRTC queue: %s samples @ %sHz (int16), "
                "playback ends at %.2f, queue_size=%s",
                len(audio_int16),
                sample_rate,
                self._playback_end_time,
                queue_size
            )
        except Exception as e:
            logger.error(f"Error adding audio chunk: {e}")
    
    @classmethod
    async def broadcast_audio(cls, audio_bytes: bytes, sample_rate: int):
        """
        Broadcast audio to all active handler instances.
        
        Args:
            audio_bytes: Raw audio bytes
            sample_rate: Sample rate
        """
        if not cls.active_instances:
            logger.warning("No active FastRTC instances to broadcast to")
            return
            
        logger.info(f"Broadcasting audio to {len(cls.active_instances)} active clients")
        for instance in cls.active_instances:
            try:
                await instance.add_audio_chunk(audio_bytes, sample_rate)
            except Exception as e:
                logger.error(f"Failed to broadcast to instance {id(instance)}: {e}")

    def update_stream_settings(
        self,
        *,
        chunk_duration_ms: Optional[int] = None,
        min_buffer_chunks: Optional[int] = None
    ):
        """Update runtime streaming parameters."""
        if chunk_duration_ms is None and min_buffer_chunks is None:
            return
        
        self._configure_stream_parameters(
            chunk_duration_ms=chunk_duration_ms,
            min_buffer_chunks=min_buffer_chunks
        )
        self._buffer_warmed = False
        logger.info(
            "Updated FastRTC stream settings: chunk=%sms, min_buffer_chunks=%s",
            self._chunk_duration_ms,
            self._min_buffer_chunks
        )

    def copy(self) -> 'FastRTCTTSHandler':
        """Create a copy of this handler for FastRTC"""
        return FastRTCTTSHandler(
            tts_queue=self.tts_queue,
            redis_client=self.redis_client,
            chunk_duration_ms=self._chunk_duration_ms,
            min_buffer_chunks=self._min_buffer_chunks
        )
    
    async def shutdown(self) -> None:
        """Cleanup resources when stream closes"""
        logger.info("=" * 70)
        logger.info("🛑 FastRTC TTS stream shutting down...")
        logger.info(f"   Handler instance: {id(self)} | Started: {self._started}")
        
        # Deregister this instance
        if self in FastRTCTTSHandler.active_instances:
            FastRTCTTSHandler.active_instances.remove(self)
            
        logger.info(f"   Remaining instances: {len(FastRTCTTSHandler.active_instances)}")
        logger.info("=" * 70)
        
        self._started = False
        self._current_audio_chunk = None
        self._chunk_position = 0
        self._buffer_warmed = False
        
        # Clear queue
        while not self._audio_output_queue.empty():
            try:
                self._audio_output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info("✅ FastRTC TTS stream closed")

    async def _publish_connection_event(self):
        """Publish FastRTC connection event to Redis for orchestrator coordination."""
        if not self.redis_client:
            logger.warning("⚠️ Redis client unavailable - cannot publish TTS connection event")
            return

        payload = json.dumps({
            "session_id": self.session_id,
            "timestamp": time.time(),
            "event": "tts_connected",
            "source": "tts_streaming_fastrtc"
        })
        channel = "leibniz:events:tts:connected"

        try:
            await self.redis_client.publish(channel, payload)
            logger.info(f"📡 Published TTS connection event → {channel}")
        except Exception as exc:
            logger.warning(f"⚠️ Failed to publish TTS connection event: {exc}")

