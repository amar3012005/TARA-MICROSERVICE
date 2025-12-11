"""
VAD Manager for Sarvam AI Streaming STT.

Manages streaming sessions, routing audio to SarvamStreamingClient,
and handling VAD events and transcripts.

Features:
- Session lifecycle management (register, stream, unregister)
- Real-time partial and final transcript handling
- Redis publishing for RAG pre-LLM incremental retrieval
- VAD signal processing (speech_start, speech_end)

Reference: https://docs.sarvam.ai/api-reference-docs/speech-to-text/apis/streaming
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, Optional

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

from config import VADConfig
from sarvam_streaming_client import SarvamStreamingClient
from utils import normalize_english_transcript

logger = logging.getLogger(__name__)

# Type alias for streaming callback
# Signature: (text: str, is_final: bool) -> None
StreamingCallback = Optional[Callable[[str, bool], None]]

# Redis channel for STT events
REDIS_STT_CHANNEL = "leibniz:events:stt"

class VADManager:
    """
    Orchestrator for Sarvam Streaming STT sessions.
    
    Uses Sarvam's server-side VAD for speech boundary detection.
    Publishes both partial and final transcripts to Redis for 
    RAG pre-LLM incremental retrieval.
    """

    def __init__(
        self,
        config: VADConfig,
        redis_client: Optional[Any] = None,
        *,
        sarvam_client: Optional[Any] = None,  # Deprecated, kept for signature compatibility
    ):
        self.config = config
        self.redis_client = redis_client
        
        # Session storage: session_id -> {client, callback, transcript_buffer, ...}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Metrics
        self.capture_count = 0
        self.partial_count = 0
        self.consecutive_timeouts = 0
        self._total_bytes_processed = 0
        self._total_chunks_processed = 0
        
        logger.info(f"🎙️ VADManager initialized | Model: {config.model_name} | Language: {config.language_code}")

    async def register_session(self, session_id: str, callback: StreamingCallback):
        """
        Register a new streaming session and connect to Sarvam.
        
        Args:
            session_id: Unique session identifier
            callback: Function to call with transcripts (text, is_final)
        """
        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already registered - reusing existing")
            # Update callback if provided
            if callback:
                self.sessions[session_id]["callback"] = callback
            return

        logger.info("=" * 70)
        logger.info(f"📝 Registering new streaming session: {session_id}")
        logger.info("=" * 70)
        
        # Create on_message handler for this session
        async def on_message(message: Any):
            await self._handle_sarvam_message(session_id, message, callback)

        # Create and connect client
        client = SarvamStreamingClient(self.config, on_message)
        
        try:
            success = await client.connect()
            if not success:
                logger.error(f"❌ Failed to connect client for session {session_id}")
                raise RuntimeError("Failed to connect to Sarvam API")
            
            self.sessions[session_id] = {
                "client": client,
                "callback": callback,
                "start_time": time.time(),
                "transcript_buffer": [],  # Buffer for accumulating partial transcripts
                "last_partial": "",  # Track last partial to avoid duplicates
                "speech_active": False,  # Track if speech is currently detected
                "chunks_sent": 0,
                "bytes_sent": 0,
                "last_flush_time": 0.0,  # Track last flush for periodic flushing
                "current_request_id": None,  # Track current request_id for accumulation
                "accumulated_transcript": "",  # Accumulate transcripts for same request_id
                "pending_transcripts": [],  # Queue of transcripts waiting for finalization
            }
            
            logger.info(f"✅ Session {session_id} registered | Ready for streaming")
            
        except Exception as e:
            logger.error(f"❌ Failed to register session {session_id}: {e}")
            raise

    async def unregister_session(self, session_id: str):
        """
        Unregister a session and disconnect from Sarvam.
        
        Args:
            session_id: Session identifier to unregister
        """
        if session_id not in self.sessions:
            logger.debug(f"Session {session_id} not found - already unregistered")
            return
        
        session_data = self.sessions[session_id]
        session_duration = time.time() - session_data.get("start_time", time.time())
        
        logger.info("=" * 70)
        logger.info(f"🛑 Unregistering session: {session_id}")
        logger.info(f"   Duration: {session_duration:.1f}s")
        logger.info(f"   Chunks: {session_data.get('chunks_sent', 0)}")
        logger.info(f"   Bytes: {session_data.get('bytes_sent', 0)}")
        logger.info("=" * 70)
        
        try:
            await session_data["client"].disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting client for {session_id}: {e}")
        finally:
            del self.sessions[session_id]

    async def stream_audio(self, session_id: str, audio_chunk: bytes) -> bool:
        """
        Stream audio chunk to the registered session.
        
        Args:
            session_id: Session identifier
            audio_chunk: Raw PCM audio bytes (16kHz, 16-bit mono)
            
        Returns:
            bool: True if audio was sent successfully
        """
        if session_id not in self.sessions:
            logger.warning(f"⚠️ Streaming audio for unknown session {session_id}")
            return False

        session_data = self.sessions[session_id]
        success = await session_data["client"].send_audio(audio_chunk)
        
        if success:
            # Update session metrics
            session_data["chunks_sent"] = session_data.get("chunks_sent", 0) + 1
            session_data["bytes_sent"] = session_data.get("bytes_sent", 0) + len(audio_chunk)
            self._total_chunks_processed += 1
            self._total_bytes_processed += len(audio_chunk)
            
            # Periodic flush for real-time processing (every 500ms during speech)
            # This forces Sarvam to process and return transcripts more frequently
            now = time.time()
            last_flush = session_data.get("last_flush_time", 0.0)
            if session_data.get("speech_active", False) and (now - last_flush) > 0.5:
                try:
                    await session_data["client"].flush()
                    session_data["last_flush_time"] = now
                except Exception as e:
                    logger.debug(f"Flush error (non-critical): {e}")
        
        return success

    async def process_audio_chunk_streaming(
        self,
        session_id: str,
        audio_chunk: bytes,
        streaming_callback: StreamingCallback = None,
    ) -> Optional[str]:
        """
        Process audio chunk for streaming transcription.
        
        Auto-registers session if needed. This is the main entry point
        for FastRTC integration.
        
        Args:
            session_id: Session identifier
            audio_chunk: Raw PCM audio bytes (16kHz, 16-bit mono)
            streaming_callback: Callback for transcript delivery (text, is_final)
            
        Returns:
            None - transcripts are delivered via callback, not return value
        """
        # Auto-register session if not exists
        if session_id not in self.sessions:
            if streaming_callback:
                try:
                    await self.register_session(session_id, streaming_callback)
                except Exception as e:
                    logger.error(f"❌ Failed to auto-register session {session_id}: {e}")
                    return None
            else:
                logger.warning(f"⚠️ Cannot auto-register session {session_id} without callback")
                return None
        
        # Stream the audio
        await self.stream_audio(session_id, audio_chunk)
        
        # Streaming mode - transcripts delivered via callback
        return None

    async def _handle_sarvam_message(self, session_id: str, message: Any, callback: StreamingCallback):
        """
        Handle incoming messages from Sarvam API.
        
        Message types:
        - speech_start: Voice activity detected
        - speech_end: Voice activity ended  
        - transcript: Transcription result (may be partial or final)
        
        Supports both dict and Pydantic model objects from the SDK.
        
        Args:
            session_id: Session identifier
            message: Message from Sarvam API (dict or Pydantic model)
            callback: Callback for transcript delivery
        """
        # Parse message to dict format
        msg_dict = self._parse_message_to_dict(message)
        msg_type = msg_dict.get("type")
        
        # Debug: Log message structure
        logger.debug(f"📋 Parsed message for {session_id}: type={msg_type}, keys={list(msg_dict.keys())}")
        
        # Get session data for state tracking
        session_data = self.sessions.get(session_id, {})
        
        # Handle Sarvam message types
        # Sarvam sends: type="events" with signal_type, or type="data" with transcript
        
        if msg_type == "events":
            # Handle VAD events: START_SPEECH, END_SPEECH
            data = msg_dict.get("data", {})
            signal_type = data.get("signal_type") if isinstance(data, dict) else getattr(data, 'signal_type', None)
            
            if signal_type == "START_SPEECH":
                logger.info(f"🎤 Speech detected [{session_id}]")
                if session_data:
                    session_data["speech_active"] = True
                    session_data["transcript_buffer"] = []  # Reset buffer on new speech
                    session_data["current_request_id"] = None  # Reset request tracking
                    session_data["accumulated_transcript"] = ""  # Reset accumulation
                    session_data["pending_transcripts"] = []  # Clear pending
                return
            elif signal_type == "END_SPEECH":
                logger.info(f"🔇 Speech ended [{session_id}]")
                if session_data:
                    session_data["speech_active"] = False
                    # Finalize any accumulated transcript after speech ends
                    if session_data.get("accumulated_transcript"):
                        await self._finalize_transcript(
                            session_id,
                            session_data["accumulated_transcript"],
                            callback,
                            session_data
                        )
                        session_data["accumulated_transcript"] = ""
                        session_data["current_request_id"] = None
                return
            else:
                logger.debug(f"📋 Unknown event signal_type: {signal_type}")
                return
        
        elif msg_type == "data":
            # Handle transcript data - extract from nested data structure
            await self._process_transcript(session_id, message, msg_dict, callback, session_data)
        
        elif msg_type == "transcript" or self._has_transcript_content(message, msg_dict):
            # Fallback for other transcript formats
            await self._process_transcript(session_id, message, msg_dict, callback, session_data)
        
        else:
            # Unknown message type - log for debugging
            logger.debug(f"📋 Unknown message type: {msg_type} | Message: {msg_dict} | Type: {type(message).__name__}")
    
    def _parse_message_to_dict(self, message: Any) -> dict:
        """
        Parse Sarvam API message to dict format.
        
        Handles:
        - Pydantic models (with model_dump or dict method)
        - Objects with __dict__
        - Plain dicts
        - Objects with direct attributes
        """
        # Try Pydantic v2 model_dump first
        if hasattr(message, 'model_dump'):
            try:
                return message.model_dump()
            except Exception as e:
                logger.debug(f"model_dump failed: {e}")
        
        # Then Pydantic v1 dict
        if hasattr(message, 'dict') and callable(message.dict):
            try:
                return message.dict()
            except Exception as e:
                logger.debug(f"dict() failed: {e}")
        
        # Plain dict
        if isinstance(message, dict):
            return message
        
        # Object with __dict__ (but not a class)
        if hasattr(message, '__dict__') and not isinstance(message, type):
            try:
                return dict(message.__dict__)
            except Exception as e:
                logger.debug(f"__dict__ access failed: {e}")
        
        # Try to extract known attributes
        result = {}
        for attr in ['type', 'text', 'transcript', 'is_final', 'confidence', 'data', 'message']:
            if hasattr(message, attr):
                try:
                    value = getattr(message, attr)
                    result[attr] = value
                except Exception as e:
                    logger.debug(f"Failed to get attribute {attr}: {e}")
        
        # If still empty, try to get string representation
        if not result:
            logger.warning(f"⚠️ Could not parse message: {type(message).__name__} | {repr(message)[:200]}")
        
        return result
    
    def _has_transcript_content(self, message: Any, msg_dict: dict) -> bool:
        """Check if message contains transcript content."""
        # Check dict fields
        if msg_dict.get("text") or msg_dict.get("transcript"):
            return True
        # Check object attributes
        if hasattr(message, 'text') and getattr(message, 'text', None):
            return True
        if hasattr(message, 'transcript') and getattr(message, 'transcript', None):
            return True
        return False
    
    async def _process_transcript(
        self,
        session_id: str,
        message: Any,
        msg_dict: dict,
        callback: StreamingCallback,
        session_data: dict
    ):
        """
        Process transcript message and publish to Redis.
        
        Handles both partial (interim) and final transcripts.
        Sarvam sends transcripts in type="data" with nested data.transcript
        """
        # Extract text from Sarvam's nested structure: type="data", data.transcript
        data = msg_dict.get("data", {})
        
        # Handle nested data structure (Sarvam format)
        request_id = None
        if isinstance(data, dict):
            text = data.get("transcript") or data.get("text")
            language_code = data.get("language_code")
            request_id = data.get("request_id")
        else:
            # Try direct attributes if data is an object
            text = getattr(data, 'transcript', None) or getattr(data, 'text', None)
            language_code = getattr(data, 'language_code', None)
            request_id = getattr(data, 'request_id', None)
        
        # Fallback to top-level fields
        if not text:
            text = (
                msg_dict.get("text") or
                msg_dict.get("transcript") or
                getattr(message, 'text', None) or
                getattr(message, 'transcript', None)
            )
        
        if not text:
            logger.debug(f"📋 Empty transcript message: {type(message).__name__}")
            return
        
        if not isinstance(text, str):
            text = str(text)
        
        text = text.strip()
        if not text:
            return
        
        # Accumulate transcripts by request_id for accuracy
        # Sarvam sends incremental updates with same request_id that need to be merged
        msg_type = msg_dict.get("type")
        
        if session_data:
            current_request_id = session_data.get("current_request_id")
            accumulated = session_data.get("accumulated_transcript", "")
            
            # If this is a new request_id, start fresh accumulation
            if request_id and request_id != current_request_id:
                session_data["current_request_id"] = request_id
                session_data["accumulated_transcript"] = text
                accumulated = text
                logger.debug(f"🆕 New request_id: {request_id[:30]}... | Starting accumulation")
            elif request_id == current_request_id:
                # Same request_id - accumulate/merge transcripts intelligently
                # Sarvam sends incremental updates, use the longest/most complete one
                if len(text) > len(accumulated):
                    # New text is longer, likely more complete
                    session_data["accumulated_transcript"] = text
                    accumulated = text
                    logger.debug(f"📈 Updated transcript (longer): '{text[:50]}...'")
                elif text != accumulated and text not in accumulated:
                    # Different text, might be continuation - merge intelligently
                    # Check if new text starts where old ends
                    if accumulated.endswith(text[:10]) or text.startswith(accumulated[-10:]):
                        # Overlapping, use the longer one
                        session_data["accumulated_transcript"] = text if len(text) > len(accumulated) else accumulated
                    else:
                        # Different parts, merge
                        session_data["accumulated_transcript"] = accumulated + " " + text
                    accumulated = session_data["accumulated_transcript"]
                    logger.debug(f"🔗 Merged transcript: '{accumulated[:50]}...'")
            else:
                # No request_id tracking, use text as-is
                accumulated = text
            
            # Use accumulated transcript
            text = accumulated.strip()
        
        # Determine if this is final (only after END_SPEECH)
        # Don't mark as final if speech is still active
        is_final = False
        if session_data:
            # Only finalize if speech has ended (END_SPEECH was received)
            if not session_data.get("speech_active", False) and request_id:
                # Speech ended, this is the final transcript for this request
                is_final = True
            else:
                # Speech still active, this is a partial/interim update
                is_final = False
        
        # Normalize text
        normalized_text = text
        if self.config.language_code.startswith("en"):
            normalized_text = normalize_english_transcript(text)
        
        # Check for duplicate (avoid spam)
        if session_data:
            last_partial = session_data.get("last_partial", "")
            if normalized_text == last_partial and not is_final:
                return  # Skip duplicate partial
            if not is_final:
                session_data["last_partial"] = normalized_text
        
        # Log transcript
        status = "✅ FINAL" if is_final else "🔄 PARTIAL"
        lang_info = f" | Language: {language_code}" if language_code else ""
        req_info = f" | RequestID: {request_id[:20]}..." if request_id else ""
        logger.info("=" * 70)
        logger.info(f"📝 [{status}] Transcript | Session: {session_id}{lang_info}{req_info}")
        logger.info(f"   Text: '{normalized_text[:150]}'")
        logger.info("=" * 70)
        
        # Invoke callback for real-time display
        if callback:
            try:
                callback(normalized_text, is_final)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        # Publish to Redis (both partials and finals for RAG pre-retrieval)
        if self.redis_client:
            await self._publish_to_redis(session_id, normalized_text, is_final)
        
        # Update metrics
        if is_final:
            self.capture_count += 1
            if session_data:
                session_data["last_partial"] = ""  # Reset on final
                session_data["accumulated_transcript"] = ""  # Clear accumulation
                session_data["current_request_id"] = None  # Reset request tracking
        else:
            self.partial_count += 1
    
    async def _finalize_transcript(
        self,
        session_id: str,
        text: str,
        callback: StreamingCallback,
        session_data: dict
    ):
        """
        Finalize and emit accumulated transcript after speech ends.
        
        This ensures we only send the complete, accurate transcript after
        all incremental updates have been accumulated.
        """
        if not text or not text.strip():
            return
        
        # Normalize text
        normalized_text = text.strip()
        if self.config.language_code.startswith("en"):
            normalized_text = normalize_english_transcript(text)
        
        logger.info("=" * 70)
        logger.info(f"📝 [✅ FINAL] Complete Accumulated Transcript | Session: {session_id}")
        logger.info(f"   Text: '{normalized_text[:150]}'")
        logger.info("=" * 70)
        
        # Invoke callback
        if callback:
            try:
                callback(normalized_text, True)
            except Exception as e:
                logger.error(f"Final callback error: {e}")
        
        # Publish to Redis
        if self.redis_client:
            await self._publish_to_redis(session_id, normalized_text, True)
        
        # Update metrics
        self.capture_count += 1

    async def _publish_to_redis(self, session_id: str, text: str, is_final: bool = True):
        """
        Publish transcript to Redis for downstream processing.
        
        Publishes both partial and final transcripts with is_final flag
        to enable RAG pre-LLM incremental retrieval.
        
        Args:
            session_id: Session identifier
            text: Transcript text
            is_final: True for final transcript, False for partial
        """
        try:
            event_data = {
                "text": text,
                "session_id": session_id,
                "timestamp": time.time(),
                "source": "stt-sarvam",
                "is_final": is_final,
            }
            
            await self.redis_client.publish(REDIS_STT_CHANNEL, json.dumps(event_data))
            
            status = "FINAL" if is_final else "PARTIAL"
            logger.info(f"📢 Published {status} transcript to Redis: {REDIS_STT_CHANNEL}")
            
        except Exception as e:
            logger.error(f"Redis publish error: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for monitoring.
        
        Returns:
            dict: Metrics including session counts, transcript counts, data processed
        """
        # Aggregate session metrics
        session_metrics = []
        for session_id, session_data in self.sessions.items():
            session_metrics.append({
                "session_id": session_id,
                "duration": time.time() - session_data.get("start_time", time.time()),
                "chunks_sent": session_data.get("chunks_sent", 0),
                "bytes_sent": session_data.get("bytes_sent", 0),
                "speech_active": session_data.get("speech_active", False),
            })
        
        return {
            "active_sessions": len(self.sessions),
            "total_final_transcripts": self.capture_count,
            "total_partial_transcripts": self.partial_count,
            "total_chunks_processed": self._total_chunks_processed,
            "total_bytes_processed": self._total_bytes_processed,
            "consecutive_timeouts": self.consecutive_timeouts,
            "sessions": session_metrics,
        }

    def reset_streaming_state(self):
        """
        Reset all sessions (sync method - async cleanup may be missed).
        
        Warning: This is a sync method that clears sessions without
        proper async disconnect. Use for emergency reset only.
        """
        logger.warning("=" * 70)
        logger.warning("⚠️ Resetting streaming state - clearing all sessions")
        logger.warning(f"   Sessions being cleared: {list(self.sessions.keys())}")
        logger.warning("=" * 70)
        
        self.sessions.clear()
        self.consecutive_timeouts = 0
        
    async def reset_streaming_state_async(self):
        """
        Reset all sessions with proper async cleanup.
        """
        logger.warning("=" * 70)
        logger.warning("⚠️ Resetting streaming state (async)")
        logger.warning(f"   Sessions being cleared: {list(self.sessions.keys())}")
        logger.warning("=" * 70)
        
        # Disconnect all sessions
        for session_id in list(self.sessions.keys()):
            try:
                await self.unregister_session(session_id)
            except Exception as e:
                logger.error(f"Error unregistering session {session_id}: {e}")
        
        self.consecutive_timeouts = 0
        logger.info("✅ Streaming state reset complete")
