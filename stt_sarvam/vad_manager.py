"""
VAD Manager for Sarvam AI Streaming STT.

Manages streaming sessions, routing audio to SarvamStreamingClient,
and handling VAD events and transcripts.
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

StreamingCallback = Optional[Callable[[str, bool], None]]

class VADManager:
    """
    Orchestrator for Sarvam Streaming STT sessions.
    Replaces energy-based VAD with Sarvam's server-side VAD.
    """

    def __init__(
        self,
        config: VADConfig,
        redis_client: Optional[Any] = None,
        *,
        sarvam_client: Optional[Any] = None, # Deprecated, kept for signature compatibility
    ):
        self.config = config
        self.redis_client = redis_client
        
        # Session storage: session_id -> {client, callback, ...}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Metrics
        self.capture_count = 0
        self.consecutive_timeouts = 0

    async def register_session(self, session_id: str, callback: StreamingCallback):
        """
        Register a new streaming session and connect to Sarvam.
        """
        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already registered")
            return

        logger.info(f"📝 Registering new streaming session: {session_id}")
        
        # Create on_message handler for this session
        async def on_message(message: dict):
            await self._handle_sarvam_message(session_id, message, callback)

        # Create and connect client
        client = SarvamStreamingClient(self.config, on_message)
        
        try:
            await client.connect()
            self.sessions[session_id] = {
                "client": client,
                "callback": callback,
                "start_time": time.time()
            }
        except Exception as e:
            logger.error(f"❌ Failed to register session {session_id}: {e}")
            raise

    async def unregister_session(self, session_id: str):
        """
        Unregister a session and disconnect from Sarvam.
        """
        if session_id in self.sessions:
            logger.info(f"🛑 Unregistering session: {session_id}")
            try:
                await self.sessions[session_id]["client"].disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting client for {session_id}: {e}")
            finally:
                del self.sessions[session_id]

    async def stream_audio(self, session_id: str, audio_chunk: bytes):
        """
        Stream audio chunk to the registered session.
        """
        if session_id not in self.sessions:
            logger.warning(f"⚠️ Streaming audio for unknown session {session_id}")
            return

        await self.sessions[session_id]["client"].send_audio(audio_chunk)

    async def process_audio_chunk_streaming(
        self,
        session_id: str,
        audio_chunk: bytes,
        streaming_callback: StreamingCallback = None,
    ) -> Optional[str]:
        """
        Compatibility method for app.py integration.
        Auto-registers session if needed using the provided callback.
        """
        if session_id not in self.sessions:
            if streaming_callback:
                await self.register_session(session_id, streaming_callback)
            else:
                logger.warning(f"⚠️ Cannot auto-register session {session_id} without callback")
                return None
        
        await self.stream_audio(session_id, audio_chunk)
        return None # Streaming doesn't return immediate results

    async def _handle_sarvam_message(self, session_id: str, message: Any, callback: StreamingCallback):
        """
        Handle incoming messages from Sarvam API.
        Supports both dict and Pydantic model objects.
        """
        # Handle Pydantic models (convert to dict) or dict directly
        if hasattr(message, 'dict'):
            # Pydantic model
            msg_dict = message.dict()
        elif hasattr(message, '__dict__'):
            # Object with __dict__
            msg_dict = message.__dict__
        elif isinstance(message, dict):
            msg_dict = message
        else:
            # Try to access attributes directly
            msg_dict = {}
            for attr in ['type', 'text', 'transcript']:
                if hasattr(message, attr):
                    msg_dict[attr] = getattr(message, attr)
        
        msg_type = msg_dict.get("type") or getattr(message, 'type', None)
        
        if msg_type == "speech_start":
            logger.info(f"🎤 Speech detected [{session_id}]")
        
        elif msg_type == "speech_end":
            logger.info(f"🔇 Speech ended [{session_id}]")
        
        elif msg_type == "transcript" or "transcript" in str(msg_dict) or hasattr(message, 'text'):
            # Handle transcript - check multiple possible field names
            text = (msg_dict.get("text") or 
                   msg_dict.get("transcript") or 
                   getattr(message, 'text', None) or
                   getattr(message, 'transcript', None))
            
            if not text:
                # Debug: log message structure if no text found
                logger.debug(f"📋 Message structure: {type(message)} | {msg_dict}")
                return
            
            if not isinstance(text, str):
                text = str(text)
            
            if not text.strip():
                return

            # In Sarvam streaming, 'transcript' type usually implies final result
            # Partial transcripts might come as updates, final ones as 'transcript' type
            is_final = (msg_type == "transcript")
            
            # Normalization
            normalized_text = text.strip()
            if is_final and self.config.language_code.startswith("en"):
                normalized_text = normalize_english_transcript(text)
            
            # Invoke callback for real-time transcript display
            if callback:
                try:
                    callback(normalized_text, is_final)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            # Publish to Redis (Final only)
            if is_final and self.redis_client:
                await self._publish_to_redis(session_id, normalized_text)
                self.capture_count += 1

    async def _publish_to_redis(self, session_id: str, text: str):
        """Publish final transcript to Redis."""
        try:
            event_data = {
                "text": text,
                "session_id": session_id,
                "timestamp": time.time(),
                "source": "stt-sarvam",
            }
            await self.redis_client.publish("leibniz:events:stt", json.dumps(event_data))
            logger.info(f"📢 Published to Redis: leibniz:events:stt")
        except Exception as e:
            logger.error(f"Redis publish error: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "active_sessions": len(self.sessions),
            "total_captures": self.capture_count
        }

    def reset_streaming_state(self):
        """Reset all sessions (Warning: asynchronous cleanup might be missed)."""
        logger.warning("Resetting streaming state - clearing sessions")
        # Ideally we should disconnect all, but this is sync method.
        # We rely on GC or timeouts.
        self.sessions.clear()
