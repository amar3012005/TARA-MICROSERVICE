"""
STT Manager for STT Local Microservice

Orchestrates Silero VAD and Faster Whisper for streaming transcription.
Manages audio buffering, VAD detection, and streaming partial updates.

Reference:
    services/stt_vad/vad_manager.py - VADManager pattern
"""

import asyncio
import time
import logging
from typing import Optional, Callable, Dict, Any
import numpy as np

from config import STTLocalConfig
from vad_utils import SileroVAD, VADStateMachine
from whisper_service import WhisperService
from utils import normalize_english_transcript, TranscriptBuffer

logger = logging.getLogger(__name__)


class STTManager:
    """
    Manages speech-to-text pipeline with VAD and Whisper.
    
    Provides streaming transcription with partial updates every 500ms
    and final transcripts on speech end.
    """
    
    def __init__(self, config: STTLocalConfig, redis_client: Optional[Any] = None):
        """
        Initialize STT manager.
        
        Args:
            config: STTLocalConfig instance
            redis_client: Optional Redis client for state persistence
        """
        self.config = config
        self.redis_client = redis_client
        
        # Initialize VAD and Whisper
        logger.info("🎙️ Initializing STT Manager components...")
        self.vad = SileroVAD(
            threshold=config.vad_threshold,
            device=config.whisper_device
        )
        self.vad_state = VADStateMachine(
            self.vad,
            min_speech_duration_ms=config.vad_min_speech_duration_ms,
            silence_timeout_ms=config.vad_silence_timeout_ms
        )
        self.whisper = WhisperService(config)
        
        # Per-session state
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Metrics
        self.capture_count = 0
        self.total_capture_time = 0.0
        self.consecutive_timeouts = 0
        
        logger.info("✅ STT Manager initialized")
    
    async def process_audio_chunk_streaming(
        self,
        session_id: str,
        audio_chunk: bytes,
        streaming_callback: Optional[Callable[[str, bool], None]] = None
    ) -> None:
        """
        Process audio chunk for continuous streaming transcription.
        
        OPTIMIZED FOR ULTRA-LOW LATENCY:
        - Processes audio in 512-sample chunks (32ms at 16kHz)
        - Runs VAD on each chunk
        - Buffers speech segments
        - Emits partial transcripts every 500ms during speech
        - Emits final transcript on speech end
        
        Args:
            session_id: Session identifier
            audio_chunk: Raw PCM audio bytes (16-bit, 16kHz, mono)
            streaming_callback: Callback for streaming fragments (text, is_final)
        """
        # Get or create session state
        if session_id not in self._active_sessions:
            self._active_sessions[session_id] = {
                'vad_state': VADStateMachine(
                    self.vad,
                    min_speech_duration_ms=self.config.vad_min_speech_duration_ms,
                    silence_timeout_ms=self.config.vad_silence_timeout_ms
                ),
                'transcript_buffer': TranscriptBuffer(),
                'speech_buffer': [],
                'last_partial_time': 0.0,
                'is_speaking': False,
                'chunks_processed': 0
            }
        
        session_data = self._active_sessions[session_id]
        vad_state = session_data['vad_state']
        transcript_buffer = session_data['transcript_buffer']
        speech_buffer = session_data['speech_buffer']
        
        try:
            # Convert bytes to numpy array (int16 -> float32)
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Process in 512-sample chunks (Silero VAD requirement)
            chunk_size = 512
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i+chunk_size]
                
                # Pad last chunk if needed
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                
                # Process chunk through VAD state machine
                vad_result = vad_state.process_chunk(chunk, sample_rate=self.config.sample_rate)
                
                session_data['chunks_processed'] += 1
                current_time = time.time()
                
                # Handle VAD events
                if vad_result['event'] == 'speech_start':
                    # Speech started
                    session_data['is_speaking'] = True
                    session_data['last_partial_time'] = current_time
                    speech_buffer.clear()
                    speech_buffer.append(chunk)
                    
                    if self.config.log_vad_events:
                        logger.info(f"🎤 [{session_id}] Speech started")
                
                elif vad_result['event'] == 'speaking':
                    # Continue speaking - add to buffer
                    speech_buffer.append(chunk)
                    
                    # Check if time for partial update
                    time_since_partial = current_time - session_data['last_partial_time']
                    if time_since_partial >= (self.config.partial_update_interval_ms / 1000.0):
                        # Generate partial transcript
                        if len(speech_buffer) > 0:
                            speech_audio = np.concatenate(speech_buffer)
                            partial_text, confidence = self.whisper.transcribe_streaming(speech_audio)
                            
                            if partial_text:
                                # Add to transcript buffer and emit
                                complete_text = transcript_buffer.add_fragment(partial_text)
                                
                                if complete_text and streaming_callback:
                                    try:
                                        streaming_callback(complete_text, False)
                                        
                                        if self.config.log_vad_events:
                                            logger.info(f"📝 [{session_id}] Partial: '{complete_text[:100]}'")
                                    except Exception as e:
                                        logger.error(f"Streaming callback error: {e}")
                                
                                session_data['last_partial_time'] = current_time
                
                elif vad_result['event'] == 'speech_end':
                    # Speech ended - generate final transcript
                    session_data['is_speaking'] = False
                    
                    if 'audio_segment' in vad_result:
                        speech_audio = vad_result['audio_segment']
                    elif len(speech_buffer) > 0:
                        speech_audio = np.concatenate(speech_buffer)
                    else:
                        speech_audio = np.array([], dtype=np.float32)
                    
                    if len(speech_audio) > 0:
                        # Transcribe final segment
                        final_text, confidence = self.whisper.transcribe_final(speech_audio)
                        
                        if final_text:
                            # Add to transcript buffer
                            complete_text = transcript_buffer.add_fragment(final_text)
                            
                            # Get final transcript
                            final_transcript = transcript_buffer.get_final_transcript()
                            
                            if final_transcript:
                                # Normalize and emit
                                normalized = normalize_english_transcript(final_transcript)
                                
                                if streaming_callback:
                                    try:
                                        streaming_callback(normalized, True)
                                        
                                        logger.info("=" * 70)
                                        logger.info(f"📝 [✅ FINAL] STT Fragment")
                                        logger.info(f"   Text: '{normalized[:150]}'")
                                        logger.info(f"   Session: {session_id}")
                                        logger.info("=" * 70)
                                    except Exception as e:
                                        logger.error(f"Final callback error: {e}")
                    
                    # Reset state
                    speech_buffer.clear()
                    transcript_buffer.reset()
                    vad_state.reset()
                    
                    # Update metrics
                    self.capture_count += 1
                    self.consecutive_timeouts = 0
                
                elif vad_result['event'] == 'speech_too_short':
                    # Speech too short - ignore
                    speech_buffer.clear()
                    vad_state.reset()
        
        except Exception as e:
            logger.error(f"Error processing audio chunk for {session_id}: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        return {
            "total_captures": self.capture_count,
            "avg_capture_time_ms": (self.total_capture_time / self.capture_count * 1000) if self.capture_count > 0 else 0.0,
            "consecutive_timeouts": self.consecutive_timeouts,
            "active_sessions": len(self._active_sessions)
        }
    
    def cleanup_session(self, session_id: str):
        """Clean up session state."""
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            logger.debug(f"Cleaned up session: {session_id}")

