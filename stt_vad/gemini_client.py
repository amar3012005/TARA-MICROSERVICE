"""
Gemini Live API Session Management for STT/VAD Microservice

Manages Gemini Live API session lifecycle with connection pooling and warmup.
Adapted from LeibnizPersistentSession and OptimizedGeminiConnection.

Reference:
    leibniz_agent/leibniz_vad.py (lines 98-256) - LeibnizPersistentSession
    leibniz_agent/leibniz_stt.py (lines 245-298) - OptimizedGeminiConnection
"""

import asyncio
import os
import time
import logging
from typing import Optional, Any, Dict

from google import genai
from google.genai import types

from config import VADConfig

logger = logging.getLogger(__name__)


class GeminiQuotaExceededError(Exception):
    """Raised when Gemini API quota is exceeded"""
    pass


class GeminiLiveSession:
    """
    Manages Gemini Live API session lifecycle with singleton pattern.
    
    Provides persistent session management with automatic expiry, warmup,
    event loop handling, and watchdog for auto-recovery in microservice environment.
    """
    
    # Class-level singleton state
    _client: Optional[genai.Client] = None
    _session: Optional[Any] = None
    _session_context: Optional[Any] = None
    _session_lock: Optional[asyncio.Lock] = None
    _session_loop: Optional[asyncio.AbstractEventLoop] = None
    
    # Session metadata
    _creation_time: float = 0.0
    _last_activity: float = 0.0
    _last_receive_time: float = 0.0  # Track receive-specific activity
    _total_uses: int = 0
    
    # Configuration
    _config: Optional[VADConfig] = None
    
    # Watchdog state
    _watchdog_task: Optional[asyncio.Task] = None
    _consecutive_timeouts: int = 0
    _is_watchdog_running: bool = False
    
    @classmethod
    async def get_session(cls, config: VADConfig, force_new: bool = False) -> Any:
        """
        Create a NEW Gemini Live session for each capture.
        
        CRITICAL: Each WebSocket connection needs its own session because
        session.receive() can only have one active iterator at a time.
        
        Args:
            config: VAD configuration instance
            force_new: Ignored (always creates new session)
            
        Returns:
            NEW Gemini Live session ready for audio streaming
            
        Raises:
            ValueError: If GEMINI_API_KEY not set
        """
        # Ensure client exists
        if cls._client is None:
            # Get API key from environment
            api_key = os.getenv("GEMINI_API_KEY", "AIzaSyAPNe7vAaTeRtjCU13JEf-wbbabPhsk5Gw")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable not set. "
                    "Set it in .env.leibniz or docker-compose environment."
                )
            cls._client = genai.Client(api_key=api_key)
            logger.info("=" * 70)
            logger.info("✅ Gemini client initialized")
            logger.info(f"🔑 API Key: {'*' * 20} (hardcoded)")
            logger.info("=" * 70)
        
        # ALWAYS create new session for each capture (no pooling)
        # This is required because session.receive() is single-use
        logger.info("🔌 Creating new Gemini Live session...")
        logger.info(f"📋 Model: {config.model_name} | Language: {config.language_code}")
        
        start_time = time.time()
        
        try:
            # Session configuration for English transcription with strict language lock
            session_config = {
                "response_modalities": ["TEXT"],  # TEXT only for transcription
                "input_audio_transcription": {},   # Enable user speech transcription
                "system_instruction": "You are a speech-to-text transcription assistant. Transcribe all audio input in English only. Do not translate or transliterate to other languages. Provide accurate, verbatim transcription of the user's speech."
            }
            
            # CRITICAL: Strict language lock to prevent hallucinations
            # This prevents auto-detection which causes Hindi/Hebrew hallucinations
            session_config["speech_config"] = {
                "language_code": config.language_code  # en-US - STRICT, no auto-detection
            }
            
            # CRITICAL: VAD/AAD configuration for better speech boundaries
            # Prevents incomplete transcripts (cutting off start/end)
            session_config["realtime_input_config"] = {
                "automatic_activity_detection": {
                    "prefix_padding_ms": config.vad_prefix_padding_ms,
                    "silence_duration_ms": config.vad_silence_duration_ms,
                    "start_of_speech_sensitivity": config.vad_start_sensitivity,
                    "end_of_speech_sensitivity": config.vad_end_sensitivity
                }
            }
            
            # Connect to Gemini Live (returns context manager) - STORE CONTEXT!
            logger.info("🌐 Connecting to Gemini Live API...")
            cls._session_context = cls._client.aio.live.connect(
                model=config.model_name,
                config=session_config
            )
            cls._session = await cls._session_context.__aenter__()
            cls._session_loop = asyncio.get_running_loop()
            
            # Track stats
            cls._creation_time = time.time()
            cls._last_activity = time.time()
            cls._last_receive_time = time.time()
            cls._total_uses += 1
            cls._config = config
            
            # Start watchdog if not already running
            if not cls._is_watchdog_running:
                cls._watchdog_task = asyncio.create_task(cls._watchdog_loop())
                cls._is_watchdog_running = True
                logger.info("🔍 Watchdog started for session recovery")
            
            connection_time = time.time() - start_time
            logger.info("=" * 70)
            logger.info(f"✅ Gemini Live session connected | Time: {connection_time:.3f}s")
            logger.info(f"📊 Session ready for audio streaming")
            logger.info("=" * 70)
            
            return cls._session
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for quota exceeded errors
            if ("quota" in error_msg or "1011" in error_msg or 
                "billing" in error_msg or "exceeded" in error_msg or
                "current quota" in error_msg):
                logger.error("=" * 70)
                logger.error(f"❌ Gemini API quota exceeded: {e}")
                logger.error("=" * 70)
                raise GeminiQuotaExceededError(f"Gemini API quota exceeded: {e}")
            else:
                logger.error("=" * 70)
                logger.error(f"❌ Gemini session creation failed: {e}")
                logger.error("=" * 70)
                raise
    
    @classmethod
    async def close_session(cls):
        """
        Gracefully close current session and cleanup resources.
        
        Note: With per-capture sessions, this is mainly for cleanup on shutdown.
        """
        # Cleanup singleton state
        if cls._session_context:
            try:
                await cls._session_context.__aexit__(None, None, None)
                logger.info(" Gemini session closed gracefully")
            except Exception as e:
                logger.error(f"Error closing session: {e}")
            finally:
                cls._session = None
                cls._session_context = None
                cls._session_loop = None
                cls._creation_time = 0.0
                cls._last_activity = 0.0
                cls._last_receive_time = 0.0
        
        # Stop watchdog
        if cls._watchdog_task and not cls._watchdog_task.done():
            cls._watchdog_task.cancel()
            try:
                await cls._watchdog_task
            except asyncio.CancelledError:
                pass
        cls._watchdog_task = None
        cls._is_watchdog_running = False
    
    @classmethod
    def get_session_stats(cls) -> Dict[str, Any]:
        """
        Get session statistics for monitoring.
        
        Returns:
            dict: {
                "session_exists": bool,
                "session_age": float,
                "last_used_ago": float,
                "total_uses": int,
                "creation_time": float
            }
        """
        now = time.time()
        
        return {
            "session_exists": cls._session is not None,
            "session_age": now - cls._creation_time if cls._creation_time > 0 else 0.0,
            "last_used_ago": now - cls._last_activity if cls._last_activity > 0 else 0.0,
            "total_uses": cls._total_uses,
            "creation_time": cls._creation_time,
            "event_loop_bound": cls._session_loop is not None
        }
    
    @classmethod
    async def warmup_session(cls, config: VADConfig):
        """
        Pre-warm session for faster first request.
        
        Creates session in background without blocking.
        """
        try:
            logger.info(" Triggering session warmup")
            await cls.get_session(config)
            logger.info(" Session warmup complete")
        except Exception as e:
            logger.error(f" Session warmup failed: {e}")
    
    @classmethod
    async def _watchdog_loop(cls):
        """
        Background watchdog task to detect stalls and restart session.
        
        Monitors time since last activity and restarts after 2 consecutive timeouts.
        Checks both send and receive activity independently.
        """
        try:
            while True:
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
                if not cls._session:
                    continue
                
                now = time.time()
                time_since_activity = now - cls._last_activity
                time_since_receive = now - cls._last_receive_time
                
                # If no activity for 60 seconds, consider stalled
                # Check both send and receive activity
                if time_since_activity > 60.0 and time_since_receive > 60.0:
                    cls._consecutive_timeouts += 1
                    
                    if cls._consecutive_timeouts >= 2:
                        logger.warning(
                            f"🔍 Watchdog: No activity for {time_since_activity:.1f}s (send) and "
                            f"{time_since_receive:.1f}s (receive) - restarting session"
                        )
                        await cls._restart_session()
                        cls._consecutive_timeouts = 0
                else:
                    # Reset consecutive timeouts on activity
                    if cls._consecutive_timeouts > 0:
                        cls._consecutive_timeouts = 0
        
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"🔍 Watchdog error: {e}")
    
    @classmethod
    async def _restart_session(cls):
        """
        Restart session after watchdog detects stall.
        
        Closes existing session and creates a new one.
        """
        logger.info("🔄 Restarting Gemini session (watchdog recovery)...")
        
        # Close existing session
        await cls.close_session()
        
        # Wait a bit before recreating
        await asyncio.sleep(1.0)
        
        # Recreate session if config exists
        if cls._config:
            try:
                await cls.get_session(cls._config, force_new=True)
                logger.info("✅ Session restarted successfully")
            except Exception as e:
                logger.error(f"❌ Failed to restart session: {e}")
    
    @classmethod
    def update_activity_time(cls, is_receive: bool = False):
        """
        Update activity timestamp for watchdog monitoring.
        
        Args:
            is_receive: If True, also update receive-specific timestamp
        """
        cls._last_activity = time.time()
        if is_receive:
            cls._last_receive_time = time.time()
