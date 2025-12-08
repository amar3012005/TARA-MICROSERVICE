"""
Sarvam AI Streaming STT Client
==============================

Manages persistent WebSocket connection to Sarvam AI's Speech-to-Text Streaming API.
Handles real-time audio streaming, VAD events, and transcript delivery.

Features:
- Persistent connection management with auto-reconnect
- Real-time audio streaming (base64 encoded)
- VAD signal handling (speech_start, speech_end)
- Partial and final transcript handling
"""

import asyncio
import base64
import logging
import json
import time
from typing import Optional, Callable, Awaitable

from sarvamai import AsyncSarvamAI
from config import VADConfig

logger = logging.getLogger(__name__)

class SarvamStreamingClient:
    """
    Client for Sarvam AI's WebSocket Streaming API.
    """

    def __init__(self, config: VADConfig, on_message: Callable[[dict], Awaitable[None]]):
        """
        Initialize the streaming client.

        Args:
            config: VADConfig instance containing API key and settings.
            on_message: Async callback function to handle incoming messages from Sarvam.
                        Signature: async def on_message(message: dict) -> None
        """
        self.config = config
        self.on_message = on_message
        self.client = AsyncSarvamAI(api_subscription_key=config.sarvam_api_key)
        self.ws_connection = None
        self.is_connected = False
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._receive_task: Optional[asyncio.Task] = None
        self._connection_ctx = None  # To hold the async context manager
        self._rate_limited = False
        self._rate_limit_until = 0.0  # Timestamp when rate limit expires
        self._reconnect_delay = 1.0  # Exponential backoff delay

    async def connect(self):
        """Establish WebSocket connection to Sarvam Streaming API."""
        async with self._connect_lock:
            if self.is_connected:
                return

            try:
                logger.info("🔌 Connecting to Sarvam Streaming API...")
                
                # Determine language code (default to unknown/auto-detect if configured)
                language_code = self.config.language_code
                if self.config.enable_language_detection:
                    # Sarvam streaming API might require specific language code or support 'unknown'
                    # Documentation suggests 'unknown' for auto-detect or omitted if supported
                    # Checking config normalization: it sets 'unknown' if enable_language_detection is True
                    pass

                # Create connection context
                # Note: We need to keep the context open, so we enter it manually
                # or run a loop inside it. 
                # The SDK usage `async with client... as ws` suggests the connection is valid within the block.
                # To maintain persistence, we should probably run the receive loop and allow sending from outside.
                # However, `async with` limits scope.
                # Let's try to enter the context and keep it open.
                
                self._connection_ctx = self.client.speech_to_text_streaming.connect(
                    language_code=language_code,
                    model=self.config.model_name,
                    sample_rate=self.config.sample_rate,  # 16000 or 8000
                    high_vad_sensitivity=True, # Always True as per plan for better VAD
                    vad_signals=True,          # Always True to get speech start/end
                    # input_audio_codec="pcm" if raw else "wav" - we will send wav header or use pcm?
                    # The plan says base64 audio. SDK transcribe takes audio data.
                    # If we send raw PCM, we must specify input_audio_codec="pcm" and matching sample_rate.
                    input_audio_codec="pcm" 
                )
                
                self.ws_connection = await self._connection_ctx.__aenter__()
                self.is_connected = True
                logger.info(f"✅ Connected to Sarvam Streaming API | Model: {self.config.model_name} | Language: {language_code}")

                # Start background receiver
                self._receive_task = asyncio.create_task(self._receive_loop())

            except Exception as e:
                logger.error(f"❌ Failed to connect to Sarvam Streaming API: {e}")
                self.is_connected = False
                self.ws_connection = None
                if self._connection_ctx:
                    await self._connection_ctx.__aexit__(None, None, None)
                    self._connection_ctx = None
                raise

    async def disconnect(self):
        """Close the WebSocket connection."""
        async with self._connect_lock:
            if not self.is_connected:
                return

            logger.info("🔌 Disconnecting from Sarvam Streaming API...")
            
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
                self._receive_task = None

            if self._connection_ctx:
                await self._connection_ctx.__aexit__(None, None, None)
                self._connection_ctx = None

            self.ws_connection = None
            self.is_connected = False
            logger.info("✅ Disconnected")

    async def _receive_loop(self):
        """Background task to receive messages from the WebSocket."""
        try:
            # The SDK might provide an iterator or recv method.
            # Based on docs: `async for message in ws:` or `msg = await ws.recv()`
            # Let's iterate.
            async for message in self.ws_connection:
                await self.on_message(message)
        except Exception as e:
            error_str = str(e)
            # Check for rate limit errors
            if "rate limit" in error_str.lower() or "1003" in error_str:
                self._rate_limited = True
                # Set rate limit expiry to 60 seconds from now (typical rate limit window)
                self._rate_limit_until = time.time() + 60.0
                logger.warning("⚠️ Rate limit exceeded. Pausing audio sending for 60 seconds.")
                logger.warning("   Visit https://www.sarvam.ai/ to review your API subscription.")
            else:
                logger.error(f"❌ Error in receive loop: {e}")
        finally:
            logger.info("🛑 Receive loop ended")
            # If loop ends unexpectedly, we might want to trigger reconnect logic externally
            # For now, mark as disconnected
            self.is_connected = False

    async def send_audio(self, audio_bytes: bytes):
        """
        Send audio data to the streaming API.
        
        Args:
            audio_bytes: Raw PCM audio bytes (16kHz, 16-bit mono expected)
        """
        # Check if rate limited
        if self._rate_limited:
            if time.time() < self._rate_limit_until:
                # Still rate limited, skip sending
                return
            else:
                # Rate limit expired, reset
                self._rate_limited = False
                self._reconnect_delay = 1.0
                logger.info("✅ Rate limit window expired. Resuming audio transmission.")
        
        if not self.is_connected or not self.ws_connection:
            # Try to reconnect with exponential backoff
            try:
                await asyncio.sleep(self._reconnect_delay)
                await self.connect()
                # Reset delay on successful connection
                self._reconnect_delay = 1.0
            except Exception as e:
                # Exponential backoff: double the delay, max 60 seconds
                self._reconnect_delay = min(self._reconnect_delay * 2, 60.0)
                logger.debug(f"Reconnection failed, will retry in {self._reconnect_delay}s: {e}")
                return

        try:
            # Convert raw PCM to base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            async with self._send_lock:
                 # Use SDK transcribe method
                 # Note: SDK validation requires encoding="audio/wav" even when sending PCM data
                 # The input_audio_codec="pcm" in connect() handles the actual format
                await self.ws_connection.transcribe(
                    audio=audio_b64,
                    encoding="audio/wav",  # SDK validation requires this literal value
                    sample_rate=self.config.sample_rate
                )
                # Reset reconnect delay on successful send
                self._reconnect_delay = 1.0
        except Exception as e:
            error_str = str(e)
            # Check for rate limit errors
            if "rate limit" in error_str.lower() or "1003" in error_str:
                self._rate_limited = True
                self._rate_limit_until = time.time() + 60.0
                logger.warning("⚠️ Rate limit exceeded during send. Pausing for 60 seconds.")
            else:
                logger.error(f"❌ Error sending audio: {e}")
            # If send fails, mark connection as dead
            self.is_connected = False


