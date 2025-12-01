"""
VAD Manager for STT/VAD Microservice

Manages speech capture with bidirectional conversation state tracking.
Adapted from LeibnizBidirectionalVAD for microservice architecture.

Key Differences from Monolith:
- Audio input via WebSocket (not sounddevice callback)
- Session state persisted in Redis
- No direct prewarm integration (optional hook)

Reference:
    leibniz_agent/leibniz_vad.py (lines 278-706) - LeibnizBidirectionalVAD
    leibniz_agent/leibniz_stt.py - STT patterns
"""

import asyncio
import time
import logging
from typing import Optional, Callable, Dict, Any

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

try:
    import websockets.exceptions
except ImportError:
    websockets = None

from google.genai import types

from config import VADConfig
from gemini_client import GeminiLiveSession, GeminiQuotaExceededError
from utils import normalize_english_transcript, TranscriptBuffer
from collections import deque

logger = logging.getLogger(__name__)


class VADManager:
    """
    Manages speech capture with bidirectional conversation state.
    
    Attributes:
        config: VAD configuration
        gemini_session: Gemini Live session manager
        redis_client: Optional Redis client for state persistence
        conversation_state: Current state ("idle", "listening", "agent_speaking", "processing")
        is_agent_speaking: Agent speaking flag for barge-in detection
        is_listening: Active listening flag
        barge_in_detected: Barge-in detection flag
        consecutive_timeouts: Timeout counter
        capture_count: Total captures for metrics
        total_capture_time: Cumulative capture time
    """
    
    def __init__(self, config: VADConfig, redis_client: Optional[Any] = None):
        """
        Initialize VAD manager.
        
        Args:
            config: VAD configuration instance
            redis_client: Optional Redis client for state persistence
        """
        self.config = config
        self.gemini_session = GeminiLiveSession
        self.redis_client = redis_client
        
        # Conversation state
        self.conversation_state = "idle"
        self.is_agent_speaking = False
        self.is_listening = False
        self.barge_in_detected = False
        
        # Metrics
        self.consecutive_timeouts = 0
        self.capture_count = 0
        self.total_capture_time = 0.0
        
        # Concurrency control
        self._async_lock = asyncio.Lock()
        self._active = False
        
        # Transcript buffering (no rolling window to avoid duplication)
        self._transcript_buffers: Dict[str, TranscriptBuffer] = {}  # Per-session buffers
        
        logger.info(f" VADManager initialized (model={config.model_name}, timeout={config.initial_timeout_s}s)")
    
    async def capture_speech_streaming(
        self,
        session_id: str,
        audio_queue: asyncio.Queue,
        streaming_callback: Optional[Callable[[str, bool], None]] = None
    ) -> Optional[str]:
        """
        Capture speech with streaming transcript fragments.
        
        Adapted from LeibnizBidirectionalVAD.capture_speech_bidirectional() (lines 412-706).
        
        Args:
            session_id: Session identifier for tracking
            audio_queue: Queue containing PCM audio chunks from WebSocket
            streaming_callback: Callback for streaming fragments (text, is_final)
            
        Returns:
            Final transcript or None on timeout/error
        """
        # Check if already active (but don't block - each session is independent)
        if self._active:
            logger.debug(f" Concurrent capture for session {session_id}")
        
        capture_start = time.time()
        final_transcript = None
        self._active = True  # Set active flag
        
        try:
            # Set listening state
            self.conversation_state = "listening"
            self.is_listening = True
            self.barge_in_detected = False
            
            logger.info("=" * 70)
            logger.info(f"🎙️ VAD Manager: Listening for speech | Session: {session_id}")
            logger.info(f"⏱️ Timeout: {self.config.start_timeout_s}s | Model: {self.config.model_name}")
            logger.info(f"🔌 Connecting to Gemini Live API...")
            logger.info("=" * 70)
            
            # Check for multiple consecutive timeouts - reset session
            if self.consecutive_timeouts >= 3:
                logger.warning(f"️ [{session_id}] Multiple timeouts, resetting session")
                await self.gemini_session.close_session()
                self.consecutive_timeouts = 0
            
            # Get Gemini Live session
            try:
                session = await self.gemini_session.get_session(self.config)
                logger.info("=" * 70)
                logger.info(f"✅ Gemini Live connected | Session: {session_id}")
                logger.info("📊 Pipeline: ACTIVE | Ready for audio streaming")
                logger.info("=" * 70)
            except GeminiQuotaExceededError as e:
                logger.warning(f"⚠️ [{session_id}] Gemini API quota exceeded, using mock response for testing")
                # Provide mock response for testing when API is unavailable
                mock_transcript = "Hello, this is a mock transcription for testing purposes."
                if streaming_callback:
                    try:
                        streaming_callback(mock_transcript, True)
                    except Exception as callback_error:
                        logger.error(f"Mock callback error: {callback_error}")
                return mock_transcript
            
            if not session:
                logger.error(f"❌ [{session_id}] Failed to get Gemini session")
                return None
            
            logger.info(f"✅ [{session_id}] Ready for audio input | Pipeline active")
            
            # Initialize capture variables
            start_time = time.time()
            last_fragment_time = start_time  # Track time of last fragment (NOT start_time!)
            fragments = []
            speech_detected = False
            final_callback_emitted = False
            
            # Background task to send audio to Gemini
            async def send_audio_task():
                """Send audio chunks from queue to Gemini Live"""
                chunk_count = 0
                total_bytes_sent = 0
                
                try:
                    while self.is_listening:
                        try:
                            # Get audio chunk with timeout
                            audio_chunk = await asyncio.wait_for(
                                audio_queue.get(),
                                timeout=0.5
                            )
                            
                            # Observability tracking
                            chunk_count += 1
                            total_bytes_sent += len(audio_chunk)
                            
                            # Send to Gemini Live
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=audio_chunk,
                                    mime_type=f"audio/pcm;rate={self.config.sample_rate}"
                                )
                            )
                            # Update activity time for watchdog
                            self.gemini_session.update_activity_time(is_receive=False)
                            
                            # Log every 50 chunks to avoid spam
                            if chunk_count % 50 == 0:
                                logger.debug(
                                    f" [{session_id}] Audio egress: "
                                    f"chunk #{chunk_count}, "
                                    f"{len(audio_chunk)} bytes, "
                                    f"total: {total_bytes_sent} bytes"
                                )
                            
                        except asyncio.TimeoutError:
                            # No audio in queue, check overall timeout
                            elapsed = time.time() - start_time
                            if elapsed > self.config.start_timeout_s:
                                logger.warning(f"⏱️ [{session_id}] Timeout after {elapsed:.1f}s")
                                break
                            continue
                            
                except Exception as e:
                    logger.error(f" [{session_id}] Audio send error: {e}")
                finally:
                    logger.debug(
                        f" [{session_id}] Audio send complete: "
                        f"{chunk_count} chunks, {total_bytes_sent} bytes total"
                    )
                    
                    # NOTE: Gemini Live API doesn't support explicit end_of_turn signal
                    # Session will auto-finalize when audio stops flowing
            
            # Start audio sending task
            audio_task = asyncio.create_task(send_audio_task())
            
            try:
                # Process Gemini responses (handle ConnectionClosedOK as normal completion)
                try:
                    async for response in session.receive():
                        elapsed = time.time() - start_time
                        
                        # Check timeout
                        if elapsed > self.config.start_timeout_s:
                            logger.warning(f"⏱️ [{session_id}] Timeout after {elapsed:.1f}s")
                            break
                        
                        # Update receive activity time for watchdog
                        self.gemini_session.update_activity_time(is_receive=True)
                        
                        # Handle server content (partials)
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.text:
                                    partial_text = part.text
                                    
                                    # Invoke streaming callback for partial
                                    if streaming_callback:
                                        try:
                                            streaming_callback(partial_text, False)
                                        except Exception as e:
                                            logger.error(f"Callback error: {e}")
                                    
                                    logger.debug(f" [{session_id}] Partial: {partial_text[:50]}")
                        
                        # Handle input audio transcription (finals)
                        if (response.server_content and 
                            response.server_content.input_transcription):
                            
                            final_text = response.server_content.input_transcription.text
                            if final_text:
                                speech_detected = True
                                fragments.append(final_text)
                                last_fragment_time = time.time()  # Update timing for early completion
                                
                                # Invoke streaming callback for final fragment
                                if streaming_callback and not final_callback_emitted:
                                    try:
                                        streaming_callback(final_text, False)
                                    except Exception as e:
                                        logger.error(f"Callback error: {e}")
                                
                        logger.info("=" * 70)
                        logger.info(f"🎤 SPEECH DETECTED | Session: {session_id}")
                        logger.info(f"📝 STT Fragment: '{final_text[:150]}'")
                        logger.info("=" * 70)
                        
                        # Handle turn completion
                        if response.server_content and response.server_content.turn_complete:
                            logger.info(f" [{session_id}] Turn complete")
                            break
                        
                        # Early completion detection (300ms silence AFTER LAST FRAGMENT, not start)
                        now = time.time()
                        silence_since_fragment = now - last_fragment_time
                        if speech_detected and silence_since_fragment > 0.3:
                            logger.info(
                                f" [{session_id}] Early completion detected "
                                f"({silence_since_fragment:.2f}s silence after last fragment)"
                            )
                            break
                
                except (websockets.exceptions.ConnectionClosedOK if websockets else Exception) as e:
                    # Gemini closes with 1000 (OK) when done - this is NORMAL, not an error
                    if websockets and isinstance(e, websockets.exceptions.ConnectionClosedOK):
                        logger.debug(f" [{session_id}] Gemini closed connection normally (1000 OK)")
                    else:
                        # Re-raise if not ConnectionClosedOK
                        raise
                
            finally:
                # Stop audio task
                self.is_listening = False
                audio_task.cancel()
                try:
                    await audio_task
                except asyncio.CancelledError:
                    pass
            
            # Build final transcript
            if fragments:
                raw_transcript = " ".join(fragments)
                
                # Apply normalization
                final_transcript = normalize_english_transcript(raw_transcript)
                
                # Invoke final callback
                if streaming_callback and not final_callback_emitted:
                    try:
                        streaming_callback(final_transcript, True)
                        final_callback_emitted = True
                    except Exception as e:
                        logger.error(f"Final callback error: {e}")
                
                logger.info("=" * 70)
                logger.info(f"✅ FINAL TRANSCRIPT | Session: {session_id}")
                logger.info(f"📝 Text: '{final_transcript}'")
                logger.info("=" * 70)
                
                # Reset timeout counter on success
                self.consecutive_timeouts = 0
            else:
                # Timeout - no speech detected
                logger.warning("=" * 70)
                logger.warning(f"⏱️ TIMEOUT | Session: {session_id} | No speech detected")
                logger.warning("=" * 70)
                self.consecutive_timeouts += 1
            
            # Update metrics
            capture_time = time.time() - capture_start
            self.capture_count += 1
            self.total_capture_time += capture_time
            
            # Update Redis state if available
            if self.redis_client:
                try:
                    await self.redis_client.hset(
                        f"session:{session_id}",
                        mapping={
                            "last_activity": time.time(),
                            "capture_count": self.capture_count,
                            "consecutive_timeouts": self.consecutive_timeouts
                        }
                    )
                    await self.redis_client.expire(f"session:{session_id}", 3600)
                except Exception as e:
                    logger.warning(f"Redis update failed: {e}")
            
            return final_transcript
            
        except Exception as e:
            logger.error(f" [{session_id}] Capture error: {e}", exc_info=True)
            self.consecutive_timeouts += 1
            return None
            
        finally:
            self._active = False
            self.conversation_state = "idle"
            self.is_listening = False
    
    async def set_agent_speaking_state(self, is_speaking: bool, context: str = ""):
        """
        Update agent speaking state.
        
        Ported from LeibnizBidirectionalVAD.set_agent_speaking_state() (lines 333-346).
        
        Args:
            is_speaking: Whether agent is currently speaking
            context: Context information for logging
        """
        self.is_agent_speaking = is_speaking
        
        if is_speaking:
            self.conversation_state = "agent_speaking"
            # Cancel any ongoing capture (barge-in)
            if self._active:
                self.barge_in_detected = True
                logger.info(f" Barge-in detected during: {context}")
        else:
            self.conversation_state = "idle"
        
        if self.config.log_state_transitions:
            logger.debug(f"Agent speaking: {is_speaking} ({context})")
    
    def set_dynamic_timeout(self, attempt_count: int = 0, conversation_context: str = "initial"):
        """
        Set timeout based on conversation context.
        
        Ported from LeibnizBidirectionalVAD.set_dynamic_timeout() (lines 354-383).
        
        Args:
            attempt_count: Retry attempt number
            conversation_context: Context type (greeting, decision, complex_query, post_service)
        """
        # Select timeout based on context
        if conversation_context == "greeting":
            self.config.start_timeout_s = self.config.greeting_timeout_s
        elif conversation_context == "decision":
            self.config.start_timeout_s = self.config.decision_timeout_s
        elif conversation_context == "complex_query":
            self.config.start_timeout_s = self.config.complex_query_timeout_s
        elif conversation_context == "post_service":
            self.config.start_timeout_s = self.config.post_service_timeout_s
        else:
            self.config.start_timeout_s = self.config.initial_timeout_s
        
        # Reduce timeout on retries
        if attempt_count > 0:
            self.config.start_timeout_s = self.config.retry_timeout_s
        
        if self.config.log_timeout_checks:
            logger.debug(
                f"⏱️ Timeout set to {self.config.start_timeout_s}s "
                f"(context={conversation_context}, attempt={attempt_count})"
            )
    
    async def process_audio_chunk_streaming(
        self,
        session_id: str,
        audio_chunk: bytes,
        streaming_callback: Optional[Callable[[str, bool], None]] = None
    ) -> None:
        """
        Process individual audio chunk for continuous streaming transcription.
        
        OPTIMIZED FOR ULTRA-LOW LATENCY:
        - Decoupled send/receive: Only sends audio (non-blocking)
        - Background receive loop handles transcripts asynchronously
        - No waiting for responses - immediate audio forwarding

        Args:
            session_id: Session identifier for tracking
            audio_chunk: Raw PCM audio bytes (16-bit, 16kHz, mono)
            streaming_callback: Callback for streaming fragments (text, is_final)
        """
        # Get or create Gemini session for this session_id
        session_key = f"session_{session_id}"
        if not hasattr(self, '_active_sessions'):
            self._active_sessions = {}
        if not hasattr(self, '_receive_tasks'):
            self._receive_tasks = {}

        if session_key not in self._active_sessions:
            # Create new session for continuous streaming
            try:
                logger.info(f"🔌 Creating Gemini Live session for continuous streaming | Session: {session_id}")
                session = await self.gemini_session.get_session(self.config)
                if not session:
                    logger.error(f"❌ Failed to create Gemini session for {session_id}")
                    return

                self._active_sessions[session_key] = {
                    'session': session,
                    'transcript_buffer': [],
                    'last_activity': time.time(),
                    'chunks_sent': 0,
                    'callback': streaming_callback
                }
                logger.info(f"✅ Continuous session created | Session: {session_id} | Ready for streaming")
                
                # Start background receive loop for this session
                receive_task = asyncio.create_task(
                    self._receive_loop(session_id, session, streaming_callback)
                )
                self._receive_tasks[session_key] = receive_task

            except GeminiQuotaExceededError as e:
                logger.warning(f"Gemini API quota exceeded for {session_id}, using mock response")
                # Mock response for testing
                mock_text = "Hello, this is a test transcription."
                if streaming_callback:
                    try:
                        streaming_callback(mock_text, True)
                    except Exception as callback_error:
                        logger.error(f"Mock callback error: {callback_error}")
                return

        session_data = self._active_sessions[session_key]
        session = session_data['session']

        try:
            # CRITICAL: Only send audio (non-blocking) - no waiting for response
            await session.send_realtime_input(
                audio=types.Blob(
                    data=audio_chunk,
                    mime_type=f"audio/pcm;rate={self.config.sample_rate}"
                )
            )
            # Update activity time for watchdog
            self.gemini_session.update_activity_time(is_receive=False)

            session_data['chunks_sent'] += 1
            session_data['last_activity'] = time.time()

        except Exception as e:
            logger.error(f"Error sending audio chunk for {session_id}: {e}")
            # Clean up failed session
            if session_key in self._active_sessions:
                del self._active_sessions[session_key]
            if session_key in self._receive_tasks:
                task = self._receive_tasks[session_key]
                task.cancel()
                del self._receive_tasks[session_key]
            # Clean up buffers
            if session_id in self._transcript_buffers:
                del self._transcript_buffers[session_id]

    async def _receive_loop(
        self,
        session_id: str,
        session: Any,
        streaming_callback: Optional[Callable[[str, bool], None]]
    ):
        """
        Background receive loop for processing Gemini responses asynchronously.
        """
        logger.info(f"👂 [{session_id}] Starting background receive loop")
        
        # Initialize transcript buffer for this session
        if session_id not in self._transcript_buffers:
            self._transcript_buffers[session_id] = TranscriptBuffer()

        transcript_buffer = self._transcript_buffers[session_id]
        session_key = f"session_{session_id}"

        try:
            async for response in session.receive():
                # DEBUG: Log raw response to see what is coming back
                # logger.debug(f"📨 [{session_id}] Received response: {response}")
                
                # Update receive activity time for watchdog
                self.gemini_session.update_activity_time(is_receive=True)
                
                has_content = False
                
                # Process server content
                if response.server_content:
                    has_content = True
                    
                    # 1. Handle Input Transcription (User Speech)
                    if response.server_content.input_transcription:
                        final_text = response.server_content.input_transcription.text
                        logger.debug(f"🔍 [{session_id}] Input Transcription: '{final_text}'")
                        if final_text:
                            # Use TranscriptBuffer to handle word boundaries
                            complete_text = transcript_buffer.add_fragment(final_text.strip())
                            
                            if complete_text:
                                # Clean and normalize transcript
                                normalized_text = normalize_english_transcript(complete_text.strip())

                                # Send to streaming callback immediately
                                if streaming_callback:
                                    try:
                                        streaming_callback(normalized_text, True)  # Mark as final
                                    except Exception as e:
                                        logger.error(f"Streaming callback error: {e}")

                                logger.info("=" * 70)
                                logger.info(f"📝 [✅ FINAL] STT Fragment")
                                logger.info(f"   Text: '{normalized_text[:150]}'")
                                logger.info(f"   Session: {session_id}")
                                logger.info("=" * 70)

                    # 2. Handle Model Turn (Gemini Response) - IGNORE or LOG
                    if response.server_content.model_turn:
                        # Handle partial responses (Model Turn)
                        # CRITICAL: Do NOT emit model turn as transcript if we only want STT.
                        for part in response.server_content.model_turn.parts:
                            if part.text:
                                logger.debug(f"🤖 [IGNORED] Model Reply: {part.text[:100]}")

                    # 3. Handle Turn Completion
                    if response.server_content.turn_complete:
                        final_transcript = transcript_buffer.get_final_transcript()
                        
                        if final_transcript:
                            normalized_transcript = normalize_english_transcript(final_transcript)
                            
                            # Send final transcript
                            if streaming_callback:
                                try:
                                    streaming_callback(normalized_transcript, True)
                                except Exception as e:
                                    logger.error(f"Final callback error: {e}")
                            
                            logger.info("=" * 70)
                            logger.info(f"📝 [✅ FINAL] Complete Transcript")
                            logger.info(f"   Text: '{normalized_transcript[:150]}'")
                            logger.info(f"   Session: {session_id}")
                            logger.info("=" * 70)

        except StopAsyncIteration:
            # Session ended normally
            logger.debug(f"Gemini session ended for {session_id}")
        except asyncio.CancelledError:
            # Task was cancelled (normal shutdown)
            logger.debug(f"Receive loop cancelled for {session_id}")
            raise
        except Exception as e:
            logger.error(f"Error in receive loop for {session_id}: {e}")
        finally:
            # Clean up
            if session_key in self._active_sessions:
                del self._active_sessions[session_key]
            if session_id in self._transcript_buffers:
                del self._transcript_buffers[session_id]
