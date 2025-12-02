"""
FastRTC Handler for TTS Streaming Service

Provides audio output streaming via FastRTC for browser playback.
"""

import asyncio
import logging
import time
from typing import Optional, Tuple, Set
from collections import deque

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
    
    def __init__(self, tts_queue=None):
        """
        Initialize FastRTC TTS Handler.
        
        Args:
            tts_queue: TTSStreamingQueue instance (injected at runtime)
        """
        super().__init__()
        self.tts_queue = tts_queue
        self.session_id = f"fastrtc_tts_{int(time.time())}"
        self._started = False
        self._sample_rate = 24000
        
        # Audio output buffer (FIFO queue for audio chunks)
        self._audio_output_queue = asyncio.Queue()
        self._current_audio_chunk = None
        self._chunk_position = 0
        
        logger.info("🔊 FastRTC TTS Handler initialized")
        logger.info(f"   Handler instance: {id(self)}")
        logger.info(f"   Session ID: {self.session_id}")
    
    async def start_up(self):
        """Called when WebRTC stream starts"""
        self._started = True
        self._audio_output_queue = asyncio.Queue()
        self._current_audio_chunk = None
        self._chunk_position = 0
        
        # Register this instance
        FastRTCTTSHandler.active_instances.add(self)
        
        logger.info("=" * 70)
        logger.info("🚀 FastRTC TTS stream started")
        logger.info(f"   Handler instance: {id(self)} | Session: {self.session_id}")
        logger.info(f"   Active instances: {len(FastRTCTTSHandler.active_instances)}")
        logger.info("=" * 70)
        
        if not self.tts_queue:
            # It's okay if queue is missing, we might be receiving via broadcast
            logger.info("ℹ️ TTS queue not directly injected - waiting for broadcast audio")
        else:
            logger.info("✅ TTS queue ready | Ready for audio streaming")
            logger.info("📊 Flow: Text → TTS Queue → FastRTC → Browser Speakers")
        
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
        # Return silence if no audio available
        if self._audio_output_queue.empty() and self._current_audio_chunk is None:
            await asyncio.sleep(0.02)  # Prevent busy loop
            # Return int16 silence (1D)
            return (self._sample_rate, np.zeros(int(self._sample_rate * 0.02), dtype=np.int16))
        
        # Process current chunk or get next one
        if self._current_audio_chunk is None:
            try:
                self._current_audio_chunk = self._audio_output_queue.get_nowait()
                self._chunk_position = 0
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.02)
                return (self._sample_rate, np.zeros(int(self._sample_rate * 0.02), dtype=np.int16))
        
        # Emit audio in small chunks (20ms = ~480 samples at 24kHz)
        chunk_size = int(self._sample_rate * 0.02)  # 20ms chunks
        
        if self._current_audio_chunk is not None:
            audio_data, sample_rate = self._current_audio_chunk
            
            # Ensure audio_data is numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.int16)
            
            # Get next chunk from current audio
            remaining = len(audio_data) - self._chunk_position
            if remaining > 0:
                take_samples = min(chunk_size, remaining)
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
        
        await asyncio.sleep(0.02)
        return (self._sample_rate, np.zeros(chunk_size, dtype=np.int16))
    
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
            
            # Add to queue as int16 (do not convert to float32)
            await self._audio_output_queue.put((audio_int16, sample_rate))
            
            logger.debug(f"Added audio chunk to FastRTC queue: {len(audio_int16)} samples @ {sample_rate}Hz (int16)")
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

    def copy(self) -> 'FastRTCTTSHandler':
        """Create a copy of this handler for FastRTC"""
        return FastRTCTTSHandler(tts_queue=self.tts_queue)
    
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
        
        # Clear queue
        while not self._audio_output_queue.empty():
            try:
                self._audio_output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info("✅ FastRTC TTS stream closed")
