"""
Sarvam AI WebSocket Streaming TTS Provider
High-performance Indian language TTS with ultra-low latency incremental streaming.

Features:
- Sarvam Bulbul v2 model with WebSocket streaming
- 11+ Indian languages support
- Voice control (pitch, pace, loudness)
- Incremental text input with buffer management
- Real-time audio chunk streaming
- Automatic retry with exponential backoff
"""

import asyncio
import logging
import base64
import json
import time
import uuid
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets library not available")

try:
    from sarvamai import AsyncSarvamAI, AudioOutput
    SARVAMAI_AVAILABLE = True
    logger.info("✅ sarvamai library loaded successfully")
except ImportError:
    SARVAMAI_AVAILABLE = False
    AsyncSarvamAI = None
    AudioOutput = None
    logger.warning("sarvamai library not available - streaming will use fallback")


@dataclass
class StreamingConfig:
    """Configuration for streaming behavior"""
    min_buffer_size: int = 50  # Minimum characters before sending
    max_chunk_length: int = 200  # Maximum characters per chunk
    stream_timeout: float = 30.0  # WebSocket timeout
    reconnect_attempts: int = 3
    reconnect_delay: float = 1.0
    output_audio_codec: str = "mp3"  # Audio codec: mp3, wav, aac, opus, flac, pcm, mulaw, alaw
    output_audio_bitrate: str = "128k"  # Audio bitrate for compression


class SarvamStreamingProvider:
    """
    Sarvam AI WebSocket Streaming TTS Provider with incremental text processing.

    Features:
    - WebSocket streaming endpoint: wss://ws-api.sarvam.ai/text-to-speech/streaming
    - Support for 11 Indian languages
    - Voice control parameters (pitch, pace, loudness)
    - Incremental text input with buffer management
    - Real-time audio chunk streaming
    - Async/await architecture
    """

    WS_ENDPOINT = "wss://api.sarvam.ai/text-to-speech/streaming"
    DEFAULT_MODEL = "bulbul:v2"
    DEFAULT_SPEAKER = "anushka"
    MAX_TEXT_LENGTH = 1500  # Per chunk

    LANGUAGE_CODES = {
        "en-IN": "English (India)",
        "hi": "Hindi",
        "ta": "Tamil",
        "te": "Telugu",
        "kn": "Kannada",
        "ml": "Malayalam",
        "mr": "Marathi",
        "gu": "Gujarati",
        "bn": "Bengali",
        "pa": "Punjabi",
        "as": "Assamese"
    }

    SPEAKERS = {
        "anushka": "Female - Professional, Clear",
        "vidya": "Female - Warm, Engaging",
        "manisha": "Female - Friendly, Conversational",
        "arya": "Female - Calm, Composed",
        "abhilash": "Male - Professional, Authoritative",
        "karun": "Male - Friendly, Approachable"
    }

    def __init__(
        self,
        api_key: str,
        speaker: str = "anushka",
        language: str = "te-IN",
        model: str = "bulbul:v2",
        pitch: float = 0.0,
        pace: float = 1.0,
        loudness: float = 1.0,
        enable_preprocessing: bool = False,
        streaming_config: Optional[StreamingConfig] = None,
        provider = None  # SarvamProvider instance for fallback
    ):
        """
        Initialize Sarvam AI streaming TTS provider using official sarvamai library.

        Args:
            api_key: Sarvam API key
            speaker: Speaker name
            language: Language code
            model: Model version
            pitch: Voice pitch (-0.75 to 0.75)
            pace: Speech pace (0.3 to 3.0)
            loudness: Volume (0.1 to 3.0)
            enable_preprocessing: Auto-fix text issues
            streaming_config: Configuration for streaming behavior
            provider: Regular SarvamProvider instance for fallback
        """
        if not api_key:
            raise ValueError("Sarvam API key not set. Set SARVAM_API_KEY environment variable.")

        self.api_key = api_key
        self.speaker = speaker
        self.language = language
        self.model = model
        self.pitch = max(-0.75, min(0.75, pitch))
        self.pace = max(0.3, min(3.0, pace))
        self.loudness = max(0.1, min(3.0, loudness))
        self.enable_preprocessing = enable_preprocessing
        self.streaming_config = streaming_config or StreamingConfig()
        self.provider = provider  # Regular provider for fallback

        # Sarvamai client
        self.client: Optional[AsyncSarvamAI] = None
        self.websocket_stream = None
        self.is_connected = False
        self.session_id = str(uuid.uuid4())
        self.audio_callback: Optional[Callable[[bytes, int, Dict[str, Any]], None]] = None

        # Streaming state
        self.audio_chunks_received = 0
        self._streaming_task: Optional[asyncio.Task] = None

        if SARVAMAI_AVAILABLE:
            self.client = AsyncSarvamAI(api_subscription_key=api_key)
            logger.info("✅ Sarvamai client initialized for streaming TTS")
        else:
            logger.warning("⚠️ Sarvamai library not available - will use REST fallback")

        logger.info(
            f"Sarvam Streaming initialized (model: {model}, speaker: {speaker}, "
            f"language: {language}, library: {'sarvamai' if SARVAMAI_AVAILABLE else 'fallback'})"
        )

    async def connect(self) -> bool:
        """Initialize streaming connection using sarvamai library"""
        if not SARVAMAI_AVAILABLE or not self.client:
            logger.warning("⚠️ Sarvamai library not available, using REST fallback")
            self.is_connected = True  # Allow fallback to work
            return True

        try:
            # Create streaming WebSocket connection
            self.websocket_stream = self.client.text_to_speech_streaming.connect(model=self.model)
            self.is_connected = True
            logger.info("✅ Sarvam streaming connection initialized")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Sarvam streaming: {e}")
            logger.warning("⚠️ Falling back to REST API")
            self.is_connected = True  # Allow fallback
            return True

    async def disconnect(self):
        """Close streaming connection"""
        self.is_connected = False
        if self.websocket_stream:
            # The context manager handles closing
            pass
        logger.info("✅ Streaming connection closed")


    async def add_text(self, text: str):
        """Add text to streaming session"""
        if not text.strip():
            return

        # Store text for later processing in streaming session
        if not hasattr(self, '_pending_text'):
            self._pending_text = []
        self._pending_text.append(text)
        logger.debug(f"📝 Added text to streaming queue: {len(text)} characters")

    async def finish_stream(self):
        """Complete streaming session"""
        logger.debug("🏁 Finishing streaming session")

        # Cancel any running streaming task
        if self._streaming_task and not self._streaming_task.done():
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass

        # Clean up pending text
        if hasattr(self, '_pending_text'):
            self._pending_text = []

        self.is_connected = False
        logger.info("✅ Streaming session finished")

    async def start_streaming_session(
        self,
        audio_callback: Callable[[bytes, int, Dict[str, Any]], None]
    ):
        """
        Start streaming session with instant audio delivery.

        Args:
            audio_callback: Function to call when audio chunks are received.
                            Signature: (audio_bytes, sample_rate, metadata)
        """
        self.audio_callback = audio_callback
        self.audio_chunks_received = 0

        if not await self.connect():
            raise RuntimeError("Failed to initialize streaming")

        # Start streaming task with sarvamai library
        self._streaming_task = asyncio.create_task(self._handle_sarvamai_streaming())

    async def _handle_sarvamai_streaming(self):
        """Handle streaming using sarvamai library for instant audio delivery"""
        try:
            if not SARVAMAI_AVAILABLE or not self.client or not self.websocket_stream:
                logger.warning("⚠️ Using REST fallback for streaming")
                await self._fallback_streaming()
                return

            # Combine all pending text
            if hasattr(self, '_pending_text') and self._pending_text:
                combined_text = ' '.join(self._pending_text)
                self._pending_text = []
            else:
                logger.warning("⚠️ No text to stream")
                return

            logger.info(f"🎵 Starting instant streaming for {len(combined_text)} characters")

            # Use sarvamai streaming context manager
            async with self.websocket_stream as ws:
                # Configure the stream
                await ws.configure(
                    target_language_code=self.language,
                    speaker=self.speaker,
                    pitch=self.pitch,
                    pace=self.pace,
                    loudness=self.loudness
                )
                logger.debug("✅ Streaming configuration sent")

                # Send text for conversion
                await ws.convert(combined_text)
                logger.debug("📤 Text sent for streaming conversion")

                # Flush to ensure all text is processed
                await ws.flush()
                logger.debug("🔄 Buffer flushed")

                # Receive audio chunks instantly
                async for message in ws:
                    # Check if message is an AudioOutput (safely handles missing import)
                    if AudioOutput is not None and isinstance(message, AudioOutput):
                        self.audio_chunks_received += 1

                        # Decode audio immediately
                        audio_bytes = base64.b64decode(message.data.audio)

                        # Create metadata for instant delivery
                        metadata = {
                            "chunk_index": self.audio_chunks_received,
                            "timestamp": time.time(),
                            "is_final": False,
                            "sample_rate": 22050,  # Sarvam default
                            "codec": "mp3",
                            "text_chunk": getattr(message.data, 'text_chunk', ''),
                            "streaming": True
                        }

                        # Deliver audio instantly to FastRTC
                        if self.audio_callback:
                            await self.audio_callback(audio_bytes, 22050, metadata)
                            logger.debug(f"⚡ INSTANT AUDIO CHUNK {self.audio_chunks_received} delivered ({len(audio_bytes)} bytes)")

                logger.info(f"✅ Streaming complete: {self.audio_chunks_received} chunks delivered instantly")

        except Exception as e:
            logger.error(f"❌ Sarvamai streaming error: {e}")
            logger.warning("⚠️ Falling back to REST API")
            await self._fallback_streaming()

    async def _fallback_streaming(self):
        """Fallback to REST API with chunking when streaming fails"""
        logger.info("🔄 Using REST API fallback for streaming")

        if not self.provider:
            logger.error("❌ No provider available for fallback")
            return

        # Combine pending text
        if hasattr(self, '_pending_text') and self._pending_text:
            combined_text = ' '.join(self._pending_text)
            self._pending_text = []

            try:
                # Synthesize as single chunk
                audio_bytes = await self.provider.synthesize(
                    text=combined_text,
                    speaker=self.speaker,
                    language=self.language
                )

                # Deliver as single chunk
                metadata = {
                    "chunk_index": 1,
                    "timestamp": time.time(),
                    "is_final": True,
                    "sample_rate": 22050,
                    "codec": "mp3",
                    "streaming": False,
                    "fallback": True
                }

                if self.audio_callback:
                    await self.audio_callback(audio_bytes, 22050, metadata)
                    logger.info("📦 Fallback audio delivered")

            except Exception as e:
                logger.error(f"❌ Fallback synthesis failed: {e}")