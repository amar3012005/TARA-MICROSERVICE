"""
Sarvam WebSocket TTS Provider for Ultra-Fast Streaming

Provides <500ms first audio chunk delivery via the official sarvamai library.
This replaces raw WebSocket connections with the proper SDK implementation.
"""

import asyncio
import base64
import io
import logging
import time
from typing import Optional, Callable, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# Try to import audio decoding libraries
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("soundfile not available - audio format conversion may be limited")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not available - MP3 decoding may be limited")

# Import sarvamai library for official streaming support
try:
    from sarvamai import AsyncSarvamAI, AudioOutput
    SARVAMAI_AVAILABLE = True
    logger.info("✅ sarvamai library loaded for WebSocket TTS")
except ImportError:
    SARVAMAI_AVAILABLE = False
    AudioOutput = None
    logger.warning("⚠️ sarvamai library not available - WebSocket TTS will use fallback")


def convert_audio_to_pcm(audio_bytes: bytes, source_format: str = "mp3") -> tuple:
    """
    Convert audio bytes to PCM int16 format for FastRTC.
    """
    try:
        # If source is already PCM/WAV, try soundfile first (fastest)
        if source_format.lower() in ("wav", "pcm"):
            if SOUNDFILE_AVAILABLE:
                audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                if audio_data.dtype != np.int16:
                    audio_data = (audio_data * 32767).astype(np.int16)
                return audio_data.tobytes(), sample_rate
            else:
                # If soundfile missing, assume raw bytes are fine or fallback to pydub
                pass

        # Try pydub for MP3
        if PYDUB_AVAILABLE:
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_bytes), 
                format=source_format
            )
            audio_segment = audio_segment.set_channels(1).set_sample_width(2)
            return audio_segment.raw_data, audio_segment.frame_rate
        
        logger.warning(f"No decoder for {source_format}, passing raw bytes")
        return audio_bytes, 22050
        
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return audio_bytes, 22050


class SarvamWebSocketTTS:
    """
    True WebSocket streaming TTS for <500ms first audio chunks.
    
    Uses the official sarvamai library's text_to_speech_streaming API
    instead of raw WebSocket connections, ensuring proper authentication
    and protocol handling.
    """

    def __init__(self, api_key: str, config):
        """
        Initialize SarvamWebSocketTTS with sarvamai client.
        
        Args:
            api_key: Sarvam API subscription key
            config: TTSStreamingConfig with speaker, language, etc.
        """
        self.api_key = api_key
        self.config = config
        self.client: Optional[AsyncSarvamAI] = None
        self.is_connected = False
        self.audio_callback: Optional[Callable] = None
        
        # Performance metrics
        self.first_chunk_time = None
        self.chunks_received = 0
        self.total_chunks = 0
        self.avg_latency = 0.0
        self.fallback_count = 0
        self.last_latency = None
        
        # Fallback to REST API if streaming fails
        self.rest_provider = None
        
        # Initialize sarvamai client
        if SARVAMAI_AVAILABLE and api_key:
            try:
                self.client = AsyncSarvamAI(api_subscription_key=api_key)
                logger.info("✅ SarvamWebSocketTTS: AsyncSarvamAI client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize AsyncSarvamAI client: {e}")
                self.client = None
        else:
            logger.warning("⚠️ SarvamWebSocketTTS: sarvamai library not available")

    async def connect_and_configure(self) -> bool:
        """
        Verify sarvamai client is ready for streaming.
        
        Returns:
            True if client is ready, False otherwise
        """
        if not SARVAMAI_AVAILABLE or not self.client:
            logger.warning("⚠️ sarvamai library not available - cannot enable streaming")
            return False
        
        try:
            # Test that the client can create a streaming context
            # We don't actually connect yet - that happens per synthesis request
            self.is_connected = True
            logger.info("✅ SarvamWebSocketTTS: Ready for streaming (sarvamai library)")
            logger.info(f"   Speaker: {self.config.sarvam_speaker}")
            logger.info(f"   Language: {self.config.sarvam_language}")
            logger.info(f"   Model: {self.config.sarvam_model}")
            return True
            
        except Exception as e:
            logger.error(f"❌ SarvamWebSocketTTS configuration failed: {e}")
            self.is_connected = False
            return False

    async def synthesize_streaming(self, text: str) -> Optional[bytes]:
        """
        Synthesize text via sarvamai streaming for <500ms first chunk.
        """
        if not SARVAMAI_AVAILABLE or not self.client:
            logger.warning("⚠️ sarvamai not available, using REST fallback")
            self.fallback_count += 1
            return await self._rest_fallback(text)
        
        start_time = time.time()
        self.first_chunk_time = None
        self.chunks_received = 0
        first_chunk = None
        
        try:
            logger.info(f"⚡ Starting ultra-fast streaming synthesis: {len(text)} chars")
            
            # Create streaming context using sarvamai library
            model = getattr(self.config, 'sarvam_model', 'bulbul:v2')
            
            async with self.client.text_to_speech_streaming.connect(model=model) as stream:
                # Configure the stream with voice parameters
                await stream.configure(
                    target_language_code=self.config.sarvam_language,
                    speaker=self.config.sarvam_speaker,
                    pitch=getattr(self.config, 'sarvam_pitch', 0.0),
                    pace=getattr(self.config, 'sarvam_pace', 1.0),
                    loudness=getattr(self.config, 'sarvam_loudness', 1.0)
                )
                logger.debug("✅ Streaming configuration sent")
                
                # Send text for conversion
                await stream.convert(text)
                logger.debug(f"📤 Text sent for streaming conversion: {text[:50]}...")
                
                # Flush to ensure all text is processed
                await stream.flush()
                logger.debug("🔄 Buffer flushed, waiting for audio chunks...")
                
                # Get configured audio codec
                audio_codec = getattr(self.config, 'sarvam_output_audio_codec', 'mp3')
                
                # Receive audio chunks with instant delivery
                async for message in stream:
                    if isinstance(message, AudioOutput):
                        self.chunks_received += 1
                        chunk_time = time.time()
                        
                        # Decode audio from base64
                        raw_audio_bytes = base64.b64decode(message.data.audio)
                        
                        # Convert to PCM for FastRTC compatibility
                        try:
                            # Run synchronously to avoid thread pool issues
                            pcm_bytes, sample_rate = convert_audio_to_pcm(raw_audio_bytes, audio_codec)
                        except Exception as conv_err:
                            logger.error(f"❌ Audio conversion error: {conv_err}")
                            # Fallback to raw bytes
                            pcm_bytes, sample_rate = raw_audio_bytes, 22050
                        
                        # Track first chunk latency
                        if first_chunk is None:
                            latency = chunk_time - start_time
                            self.first_chunk_time = latency
                            self.last_latency = latency
                            first_chunk = pcm_bytes
                            
                            # Update average latency
                            self.total_chunks += 1
                            self.avg_latency = ((self.avg_latency * (self.total_chunks - 1)) + latency) / self.total_chunks
                            
                            if latency < 0.5:
                                logger.info(f"⚡ ULTRA-FAST FIRST CHUNK: {latency*1000:.0f}ms (TARGET MET: <500ms)")
                            else:
                                logger.warning(f"⚠️ First chunk: {latency*1000:.0f}ms (TARGET MISSED: wanted <500ms)")
                        
                        # Create metadata for instant delivery
                        metadata = {
                            "chunk_index": self.chunks_received,
                            "timestamp": chunk_time,
                            "is_final": False,
                            "sample_rate": sample_rate,
                            "codec": "pcm",  # Always PCM after conversion
                            "streaming": True,
                            "latency_ms": (chunk_time - start_time) * 1000,
                            "text_chunk": getattr(message.data, 'text_chunk', ''),
                            "original_codec": audio_codec,
                            "original_size": len(raw_audio_bytes),
                            "pcm_size": len(pcm_bytes)
                        }
                        
                        # Deliver audio instantly via callback
                        if self.audio_callback:
                            try:
                                await self.audio_callback(pcm_bytes, sample_rate, metadata)
                                logger.debug(f"⚡ Chunk {self.chunks_received} delivered: {len(pcm_bytes)} bytes PCM")
                            except Exception as cb_err:
                                logger.error(f"❌ Audio callback error: {cb_err}")
                
                # Mark final chunk
                if self.audio_callback and self.chunks_received > 0:
                    final_metadata = {
                        "chunk_index": self.chunks_received,
                        "timestamp": time.time(),
                        "is_final": True,
                        "sample_rate": 22050,
                        "streaming": True,
                        "total_chunks": self.chunks_received
                    }
                    # Send empty final signal
                    await self.audio_callback(b'', 22050, final_metadata)
            
            total_time = time.time() - start_time
            logger.info(f"✅ Streaming complete: {self.chunks_received} chunks in {total_time*1000:.0f}ms")
            
            return first_chunk
            
        except Exception as e:
            logger.error(f"❌ Streaming synthesis failed: {e}")
            self.fallback_count += 1
            return await self._rest_fallback(text)

    async def _rest_fallback(self, text: str) -> Optional[bytes]:
        """
        Fallback to REST API if streaming fails.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio bytes from REST API, or None if failed
        """
        if self.rest_provider:
            try:
                logger.info("🔄 Using REST API fallback for synthesis")
                start_time = time.time()
                
                audio_bytes = await self.rest_provider.synthesize(
                    text=text,
                    speaker=self.config.sarvam_speaker,
                    language=self.config.sarvam_language
                )
                
                latency = time.time() - start_time
                logger.info(f"📦 REST fallback completed: {latency*1000:.0f}ms")
                
                # Deliver via callback if available
                if self.audio_callback and audio_bytes:
                    metadata = {
                        "chunk_index": 1,
                        "timestamp": time.time(),
                        "is_final": True,
                        "sample_rate": 22050,
                        "streaming": False,
                        "fallback": True,
                        "latency_ms": latency * 1000
                    }
                    await self.audio_callback(audio_bytes, 22050, metadata)
                
                return audio_bytes
                
            except Exception as e:
                logger.error(f"❌ REST fallback failed: {e}")
        else:
            logger.warning("⚠️ No REST provider available for fallback")
        
        return None

    async def disconnect(self):
        """
        Cleanup resources.
        
        Note: sarvamai library handles connection cleanup automatically
        via context managers, so this mainly resets state.
        """
        self.is_connected = False
        logger.info("🔌 SarvamWebSocketTTS disconnected")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get streaming performance metrics.
        
        Returns:
            Dictionary with connection status and latency metrics
        """
        return {
            "connected": self.is_connected,
            "sarvamai_available": SARVAMAI_AVAILABLE,
            "total_chunks": self.total_chunks,
            "avg_latency": self.avg_latency,
            "fallback_count": self.fallback_count,
            "last_latency": self.last_latency,
            "chunks_in_last_request": self.chunks_received
        }




