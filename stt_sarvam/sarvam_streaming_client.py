"""
Sarvam AI Streaming STT Client
==============================

Manages persistent WebSocket connection to Sarvam AI's Speech-to-Text Streaming API.
Handles real-time audio streaming, VAD events, and transcript delivery.

Features:
- Persistent connection management with auto-reconnect
- Real-time audio streaming (base64 encoded PCM)
- VAD signal handling (speech_start, speech_end)
- Partial and final transcript handling
- Ultra-low latency with high VAD sensitivity
- Flush signal support for immediate processing

Reference: https://docs.sarvam.ai/api-reference-docs/speech-to-text/apis/streaming
"""

import asyncio
import base64
import logging
import time
from typing import Optional, Callable, Awaitable, Any

from sarvamai import AsyncSarvamAI
from config import VADConfig

logger = logging.getLogger(__name__)


# Connection states for tracking
class ConnectionState:
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

class SarvamStreamingClient:
    """
    Client for Sarvam AI's WebSocket Streaming STT API.
    
    Provides ultra-low latency streaming transcription with:
    - Automatic VAD (Voice Activity Detection)
    - Partial and final transcript streaming
    - Auto-reconnect with exponential backoff
    - Rate limit handling
    """
    
    # Maximum reconnect attempts before giving up
    MAX_RECONNECT_ATTEMPTS = 5
    # Base delay for exponential backoff (seconds)
    BASE_RECONNECT_DELAY = 1.0
    # Maximum reconnect delay (seconds)
    MAX_RECONNECT_DELAY = 30.0

    def __init__(self, config: VADConfig, on_message: Callable[[Any], Awaitable[None]]):
        """
        Initialize the streaming client.

        Args:
            config: VADConfig instance containing API key and settings.
            on_message: Async callback function to handle incoming messages from Sarvam.
                        Signature: async def on_message(message: Any) -> None
                        Message types: speech_start, speech_end, transcript (partial/final)
        """
        self.config = config
        self.on_message = on_message
        
        # Validate API key
        if not config.sarvam_api_key:
            logger.warning("⚠️ SARVAM_API_SUBSCRIPTION_KEY not set - client will operate in mock mode")
            self._mock_mode = True
        else:
            self._mock_mode = False
            self.client = AsyncSarvamAI(api_subscription_key=config.sarvam_api_key)
        
        # Connection state
        self.ws_connection = None
        self.is_connected = False
        self.state = ConnectionState.DISCONNECTED
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._receive_task: Optional[asyncio.Task] = None
        self._connection_ctx = None
        
        # Rate limiting
        self._rate_limited = False
        self._rate_limit_until = 0.0
        
        # Reconnection
        self._reconnect_delay = self.BASE_RECONNECT_DELAY
        self._reconnect_attempts = 0
        
        # Metrics
        self._chunks_sent = 0
        self._bytes_sent = 0
        self._connection_time: Optional[float] = None
        self._last_activity: float = time.time()
        
        logger.info(f"🎙️ SarvamStreamingClient initialized | Model: {config.model_name} | Sample Rate: {config.sample_rate}Hz")

    async def connect(self) -> bool:
        """
        Establish WebSocket connection to Sarvam Streaming API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        async with self._connect_lock:
            if self.is_connected:
                logger.debug("Already connected to Sarvam Streaming API")
                return True
            
            # Check mock mode
            if self._mock_mode:
                logger.warning("🔌 Mock mode active - no actual connection will be made")
                self.is_connected = True
                self.state = ConnectionState.CONNECTED
                return True

            self.state = ConnectionState.CONNECTING
            start_time = time.time()

            try:
                logger.info("=" * 70)
                logger.info("🔌 Connecting to Sarvam Streaming STT API...")
                logger.info(f"   Model: {self.config.model_name}")
                logger.info(f"   Language: {self.config.language_code}")
                logger.info(f"   Sample Rate: {self.config.sample_rate}Hz")
                logger.info(f"   High VAD Sensitivity: {self.config.sarvam_high_vad_sensitivity}")
                logger.info(f"   VAD Signals: {self.config.sarvam_vad_signals}")
                logger.info("=" * 70)
                
                # Determine language code
                language_code = self.config.language_code
                
                # Create connection with optimized parameters
                # Reference: https://docs.sarvam.ai/api-reference-docs/speech-to-text/apis/streaming
                self._connection_ctx = self.client.speech_to_text_streaming.connect(
                    language_code=language_code,
                    model=self.config.model_name,
                    sample_rate=self.config.sample_rate,  # 16000 (recommended) or 8000
                    high_vad_sensitivity=self.config.sarvam_high_vad_sensitivity,  # Better speech detection
                    vad_signals=self.config.sarvam_vad_signals,  # Get speech_start/speech_end events
                    input_audio_codec="pcm",  # Raw PCM audio (16-bit signed)
                    flush_signal=True  # Enable manual flush for immediate processing
                )
                
                # Enter the async context manager
                self.ws_connection = await self._connection_ctx.__aenter__()
                self.is_connected = True
                self.state = ConnectionState.CONNECTED
                self._connection_time = time.time()
                self._reconnect_attempts = 0
                self._reconnect_delay = self.BASE_RECONNECT_DELAY
                
                connection_duration = time.time() - start_time
                logger.info("=" * 70)
                logger.info(f"✅ Connected to Sarvam Streaming API in {connection_duration:.3f}s")
                logger.info(f"   Ready for audio streaming")
                logger.info("=" * 70)

                # Start background receiver for transcripts
                self._receive_task = asyncio.create_task(self._receive_loop())
                
                return True

            except Exception as e:
                self.state = ConnectionState.ERROR
                self.is_connected = False
                self.ws_connection = None
                
                # Clean up context if it was created
                if self._connection_ctx:
                    try:
                        await self._connection_ctx.__aexit__(None, None, None)
                    except Exception:
                        pass
                    self._connection_ctx = None
                
                # Check for specific error types
                error_str = str(e).lower()
                if "api" in error_str and ("key" in error_str or "auth" in error_str):
                    logger.error("❌ Sarvam API authentication failed - check SARVAM_API_SUBSCRIPTION_KEY")
                elif "rate" in error_str or "limit" in error_str:
                    logger.error("❌ Sarvam API rate limit exceeded")
                    self._rate_limited = True
                    self._rate_limit_until = time.time() + 60.0
                else:
                    logger.error(f"❌ Failed to connect to Sarvam Streaming API: {e}")
                
                return False

    async def disconnect(self):
        """Close the WebSocket connection gracefully."""
        async with self._connect_lock:
            if not self.is_connected and self.state == ConnectionState.DISCONNECTED:
                return

            logger.info("🔌 Disconnecting from Sarvam Streaming API...")
            
            # Cancel receive task
            if self._receive_task and not self._receive_task.done():
                self._receive_task.cancel()
                try:
                    await asyncio.wait_for(self._receive_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                self._receive_task = None

            # Close WebSocket connection
            if self._connection_ctx:
                try:
                    await self._connection_ctx.__aexit__(None, None, None)
                except Exception as e:
                    logger.debug(f"Error closing connection context: {e}")
                self._connection_ctx = None

            self.ws_connection = None
            self.is_connected = False
            self.state = ConnectionState.DISCONNECTED
            
            # Log session metrics
            if self._connection_time:
                session_duration = time.time() - self._connection_time
                logger.info(f"✅ Disconnected | Session: {session_duration:.1f}s | Chunks: {self._chunks_sent} | Bytes: {self._bytes_sent}")
            else:
                logger.info("✅ Disconnected")

    async def _receive_loop(self):
        """
        Background task to receive messages from the WebSocket.
        
        Handles message types:
        - speech_start: Voice activity detected
        - speech_end: Voice activity ended
        - transcript: Transcription result (partial or final)
        """
        logger.info("👂 Starting Sarvam receive loop...")
        message_count = 0
        
        try:
            async for message in self.ws_connection:
                message_count += 1
                self._last_activity = time.time()
                
                # Pass message to handler (VADManager will parse it)
                try:
                    await self.on_message(message)
                except Exception as callback_error:
                    logger.error(f"❌ Error in message callback: {callback_error}")
                    
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled (normal shutdown)")
            raise
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limit errors
            if "rate limit" in error_str or "1003" in error_str or "429" in error_str:
                self._rate_limited = True
                self._rate_limit_until = time.time() + 60.0
                logger.warning("=" * 70)
                logger.warning("⚠️ Sarvam API rate limit exceeded")
                logger.warning("   Pausing audio transmission for 60 seconds")
                logger.warning("   Visit https://www.sarvam.ai/ to review your API subscription")
                logger.warning("=" * 70)
            elif "1000" in error_str or "closed" in error_str:
                # Normal WebSocket close
                logger.debug(f"WebSocket closed normally: {e}")
            else:
                logger.error(f"❌ Error in receive loop: {e}")
        finally:
            logger.info(f"🛑 Receive loop ended | Messages received: {message_count}")
            self.is_connected = False
            self.state = ConnectionState.DISCONNECTED

    async def send_audio(self, audio_bytes: bytes) -> bool:
        """
        Send audio data to the streaming API.
        
        Args:
            audio_bytes: Raw PCM audio bytes (16kHz, 16-bit mono expected)
            
        Returns:
            bool: True if audio was sent successfully
        """
        # Check mock mode
        if self._mock_mode:
            self._chunks_sent += 1
            self._bytes_sent += len(audio_bytes)
            return True
        
        # Check if rate limited
        if self._rate_limited:
            if time.time() < self._rate_limit_until:
                return False
            else:
                self._rate_limited = False
                self._reconnect_delay = self.BASE_RECONNECT_DELAY
                logger.info("✅ Rate limit window expired. Resuming audio transmission.")
        
        # Auto-reconnect if disconnected
        if not self.is_connected or not self.ws_connection:
            if self._reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
                logger.error(f"❌ Max reconnection attempts ({self.MAX_RECONNECT_ATTEMPTS}) exceeded")
                return False
            
            self.state = ConnectionState.RECONNECTING
            self._reconnect_attempts += 1
            
            try:
                logger.info(f"🔄 Reconnecting to Sarvam API (attempt {self._reconnect_attempts}/{self.MAX_RECONNECT_ATTEMPTS})...")
                await asyncio.sleep(self._reconnect_delay)
                
                if await self.connect():
                    self._reconnect_delay = self.BASE_RECONNECT_DELAY
                else:
                    self._reconnect_delay = min(self._reconnect_delay * 2, self.MAX_RECONNECT_DELAY)
                    return False
            except Exception as e:
                self._reconnect_delay = min(self._reconnect_delay * 2, self.MAX_RECONNECT_DELAY)
                logger.debug(f"Reconnection failed, will retry in {self._reconnect_delay}s: {e}")
                return False

        try:
            # Convert raw PCM to base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            async with self._send_lock:
                # Use SDK transcribe method
                # Note: SDK validation may require encoding="audio/wav" even when sending PCM
                # The input_audio_codec="pcm" in connect() tells server the actual format
                await self.ws_connection.transcribe(
                    audio=audio_b64,
                    encoding="audio/wav",  # SDK validation literal
                    sample_rate=self.config.sample_rate
                )
            
            # Update metrics
            self._chunks_sent += 1
            self._bytes_sent += len(audio_bytes)
            self._last_activity = time.time()
            self._reconnect_delay = self.BASE_RECONNECT_DELAY
            
            return True
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "rate limit" in error_str or "1003" in error_str or "429" in error_str:
                self._rate_limited = True
                self._rate_limit_until = time.time() + 60.0
                logger.warning("⚠️ Rate limit exceeded during send. Pausing for 60 seconds.")
            elif "closed" in error_str or "connection" in error_str:
                logger.warning("⚠️ Connection lost - will reconnect on next send")
                self.is_connected = False
                self.state = ConnectionState.DISCONNECTED
            else:
                logger.error(f"❌ Error sending audio: {e}")
                self.is_connected = False
            
            return False

    async def flush(self) -> bool:
        """
        Force immediate processing of buffered audio.
        
        Useful for getting transcripts without waiting for silence detection.
        
        Returns:
            bool: True if flush was sent successfully
        """
        if self._mock_mode:
            return True
            
        if not self.is_connected or not self.ws_connection:
            return False
        
        try:
            async with self._send_lock:
                await self.ws_connection.flush()
            logger.debug("⚡ Flush signal sent - forcing immediate processing")
            return True
        except Exception as e:
            logger.error(f"❌ Error sending flush: {e}")
            return False

    def get_stats(self) -> dict:
        """Get client statistics for monitoring."""
        return {
            "is_connected": self.is_connected,
            "state": self.state,
            "chunks_sent": self._chunks_sent,
            "bytes_sent": self._bytes_sent,
            "rate_limited": self._rate_limited,
            "reconnect_attempts": self._reconnect_attempts,
            "last_activity": self._last_activity,
            "connection_time": self._connection_time,
        }
