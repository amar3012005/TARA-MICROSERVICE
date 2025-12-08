"""
FastRTC Handler for Direct STT/VAD Integration
==============================================

Direct integration of FastRTC with VADManager - no WebSocket overhead.
Provides ultra-low latency by calling VADManager directly within the same process.

This handler receives audio from FastRTC and forwards it directly to VADManager,
which processes it with Gemini Live API for real-time transcription.
"""

import asyncio
import json
import logging
import time
from typing import Optional

import numpy as np
from fastrtc import AsyncStreamHandler

logger = logging.getLogger(__name__)


class FastRTCSTTHandler(AsyncStreamHandler):
    """
    FastRTC AsyncStreamHandler that directly integrates with VADManager.
    
    Provides direct audio streaming without network overhead for ultra-low latency.
    """
    
    def __init__(self, vad_manager=None, redis_client=None):
        """
        Initialize FastRTC STT Handler.
        
        Args:
            vad_manager: VADManager instance (injected at runtime)
            redis_client: Redis client for publishing connection events
        """
        super().__init__()
        self.vad_manager = vad_manager
        self.redis_client = redis_client
        self.session_id = f"fastrtc_{int(time.time())}"
        self._chunk_count = 0
        self._started = False
        
        # Audio buffering for 100ms chunks (16kHz * 2 bytes * 0.1s = 3200 bytes)
        self._audio_buffer = bytearray()
        self._buffer_limit = 3200
        
        logger.info("🎙️ FastRTC STT Handler initialized")
        logger.info(f"   Handler instance: {id(self)}")
        logger.info(f"   Session ID: {self.session_id}")
    
    async def start_up(self):
        """Called when WebRTC stream starts - initialize connection."""
        self._started = True
        self._chunk_count = 0
        self._audio_buffer.clear()
        
        logger.info("=" * 70)
        logger.info("🚀 FastRTC stream started | Initializing STT/VAD pipeline...")
        logger.info(f"   Handler instance: {id(self)} | Session: {self.session_id}")
        logger.info("=" * 70)
        
        if not self.vad_manager:
            logger.warning("⚠️ VADManager not injected - transcription will not work")
            logger.warning("   Ensure vad_manager is set before starting stream")
        else:
            logger.info("✅ VADManager ready | Pipeline initialized")
            logger.info("📊 Flow: Browser → FastRTC → VADManager → Gemini Live → STT")
            logger.info("� Start speaking - audio will trigger automatic pipeline processing")
        
        # Publish connection event to Redis for orchestrator
        await self._publish_connection_event()
        
        logger.info("=" * 70)
    
    async def _publish_connection_event(self):
        """Publish FastRTC connection event to Redis for orchestrator coordination."""
        if not self.redis_client:
            logger.warning("⚠️ Redis client unavailable - cannot publish STT connection event")
            return

        payload = json.dumps({
            "session_id": self.session_id,
            "timestamp": time.time(),
            "event": "stt_connected",
            "source": "stt_vad_fastrtc"
        })
        channel = "leibniz:events:stt:connected"

        try:
            await self.redis_client.publish(channel, payload)
            logger.info(f"📡 Published STT connection event → {channel}")
        except Exception as exc:
            logger.warning(f"⚠️ Failed to publish STT connection event: {exc}")
    
    async def receive(self, audio: tuple) -> None:
        """
        Receive audio from FastRTC and forward to VADManager.
        
        Buffers audio to 100ms chunks before sending to reduce API overhead.
        
        Args:
            audio: Tuple of (sample_rate: int, audio_array: np.ndarray) from FastRTC
        """
        if not self.vad_manager:
            # Only log warning once
            if self._chunk_count == 0:
                logger.warning("⚠️ VADManager not available - dropping audio chunk")
            return
        
        try:
            # Parse FastRTC audio format
            if not isinstance(audio, tuple) or len(audio) != 2:
                logger.warning(f"⚠️ Unexpected audio format: {type(audio)}")
                return
            
            sample_rate, audio_array = audio
            
            # Validate and normalize audio
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array, dtype=np.float32)
            
            # Handle multi-dimensional arrays
            if audio_array.ndim == 2:
                audio_array = audio_array.squeeze()
            elif audio_array.ndim > 2:
                audio_array = audio_array.flatten()
            
            # Log audio stats on first chunk for debugging
            if self._chunk_count == 0:
                max_val = np.max(np.abs(audio_array)) if audio_array.size > 0 else 0
                logger.info(f"🔍 Audio Input Stats:")
                logger.info(f"   - Sample Rate: {sample_rate}Hz")
                logger.info(f"   - DType: {audio_array.dtype}")
                logger.info(f"   - Shape: {audio_array.shape}")
                logger.info(f"   - Max Value: {max_val}")
            
            # CRITICAL: Resample to 16kHz for Gemini Live API
            # Gemini Live API requires 16-bit PCM at 16kHz mono
            # Reference: https://ai.google.dev/gemini-api/docs/live
            target_rate = 16000
            if sample_rate != target_rate:
                if sample_rate > target_rate and sample_rate % target_rate == 0:
                    # Efficient integer downsampling (e.g. 48000 -> 16000 take every 3rd sample)
                    step = int(sample_rate / target_rate)
                    audio_array = audio_array[::step]
                    sample_rate = target_rate
                else:
                    # Fallback: Naive resampling for non-integer ratios
                    new_length = int(len(audio_array) * target_rate / sample_rate)
                    indices = np.linspace(0, len(audio_array) - 1, new_length).astype(int)
                    audio_array = audio_array[indices]
                    sample_rate = target_rate

            # Convert to int16 PCM format for Gemini Live API
            # No amplification - pass through audio as-is for accurate transcription
            if audio_array.dtype == np.int16:
                audio_int16 = audio_array
            elif audio_array.dtype in (np.float32, np.float64):
                max_val = np.max(np.abs(audio_array)) if audio_array.size > 0 else 0
                if max_val > 1.0:
                    # Already in int16 scale, just clip and cast
                    audio_int16 = np.clip(audio_array, -32768, 32767).astype(np.int16)
                else:
                    # Float [-1, 1] scale, convert to int16
                    audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
            else:
                # Convert other dtypes to float32 first, then to int16
                audio_array = audio_array.astype(np.float32)
                max_val = np.max(np.abs(audio_array)) if audio_array.size > 0 else 0
                if max_val > 1.0:
                    audio_int16 = np.clip(audio_array, -32768, 32767).astype(np.int16)
                else:
                    audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
            
            audio_bytes = audio_int16.tobytes()
            
            # Buffer chunks
            self._audio_buffer.extend(audio_bytes)
            
            # Send when buffer limit reached (100ms)
            if len(self._audio_buffer) >= self._buffer_limit:
                chunk_to_send = bytes(self._audio_buffer) # Create copy
                self._audio_buffer.clear() # Reset buffer
                
                # Send directly to VADManager (non-blocking)
                await self.vad_manager.process_audio_chunk_streaming(
                    session_id=self.session_id,
                    audio_chunk=chunk_to_send,
                    streaming_callback=self._transcript_callback
                )
                
                self._chunk_count += 1
                
                if self._chunk_count == 1:
                    logger.info(f"✅ First buffered chunk sent | Size: {len(chunk_to_send)} bytes | 16000Hz")
                elif self._chunk_count % 10 == 0: # Log less frequently (every 1s)
                    logger.info(f"📤 Audio chunk #{self._chunk_count} | {len(chunk_to_send)} bytes")
                
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _transcript_callback(self, text: str, is_final: bool):
        """
        Callback for receiving transcripts from VADManager.
        
        This is called by VADManager's background receive loop when
        transcripts are available. We log them for visibility.
        
        Args:
            text: Transcript text
            is_final: Whether this is a final transcript
        """
        status = "✅ FINAL" if is_final else "🔄 PARTIAL"
        logger.info(f"📝 [{status}] Transcript: '{text[:150]}'")
    
    async def emit(self):
        """
        Emit method required by AsyncStreamHandler.
        Returns silence since we're only receiving audio (no TTS output in this service).
        """
        # Return silence - this service only does STT, not TTS
        await asyncio.sleep(0.02)  # Prevent busy loop
        return (16000, np.zeros((1, 1600), dtype=np.int16))
    
    def copy(self) -> 'FastRTCSTTHandler':
        """Create a copy of this handler for FastRTC."""
        return FastRTCSTTHandler(vad_manager=self.vad_manager, redis_client=self.redis_client)
    
    async def shutdown(self) -> None:
        """Cleanup resources when stream closes."""
        logger.info("=" * 70)
        logger.info("🛑 FastRTC stream shutting down...")
        logger.info(f"   Handler instance: {id(self)} | Started: {self._started}")
        logger.info(f"   Total chunks processed: {self._chunk_count}")
        logger.info("=" * 70)
        
        self._started = False
        self._audio_buffer.clear()
        logger.info("✅ FastRTC stream closed")
