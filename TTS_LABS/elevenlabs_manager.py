"""
ElevenLabs WebSocket Manager for Ultra-Low Latency Streaming

Implements bidirectional WebSocket streaming using the ElevenLabs stream-input API.
Optimized for <150ms first chunk latency with eleven_flash_v2_5 model.

Reference: https://elevenlabs.io/docs/api-reference/text-to-speech/stream-input
"""

import asyncio
import base64
import json
import logging
import time
from typing import Optional, Callable, Dict, Any, AsyncGenerator

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .config import TTSLabsConfig

logger = logging.getLogger(__name__)


class ElevenLabsStreamManager:
    """
    Manages WebSocket connection to ElevenLabs stream-input API.
    
    Features:
    - Bidirectional streaming: send text chunks, receive audio chunks
    - Ultra-low latency with try_trigger_generation
    - Automatic reconnection on failure
    - PCM audio output for minimal decoding overhead
    
    Protocol:
    1. Connect to wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input
    2. Send BOS (Beginning of Stream) message with voice settings
    3. Send text chunks as they arrive from RAG
    4. Optionally send try_trigger_generation to flush buffer early
    5. Send EOS (End of Stream) to signal completion
    6. Receive audio chunks as they're generated
    """
    
    def __init__(self, config: TTSLabsConfig):
        """
        Initialize ElevenLabs stream manager.
        
        Args:
            config: TTSLabsConfig with API key and settings
        """
        self.config = config
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.is_streaming = False
        
        # Audio callback for streaming output
        self.audio_callback: Optional[Callable[[bytes, int, Dict[str, Any]], Any]] = None
        
        # Performance metrics
        self.first_chunk_time: Optional[float] = None
        self.chunks_received = 0
        self.total_audio_bytes = 0
        self.stream_start_time: Optional[float] = None
        self.last_latency_ms: Optional[float] = None
        
        # Reconnection state
        self.reconnect_count = 0
        
        logger.info(f"ElevenLabs StreamManager initialized")
        logger.info(f"  Model: {config.elevenlabs_model_id}")
        logger.info(f"  Voice: {config.elevenlabs_voice_id}")
        logger.info(f"  Latency optimization: {config.optimize_streaming_latency}")
        logger.info(f"  Output format: {config.output_format}")
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection to ElevenLabs.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.config.elevenlabs_api_key:
            logger.error("ELEVENLABS_API_KEY not configured")
            return False
        
        try:
            ws_url = self.config.get_websocket_url()
            
            # Add API key to headers
            headers = {
                "xi-api-key": self.config.elevenlabs_api_key
            }
            
            logger.info(f"Connecting to ElevenLabs WebSocket...")
            logger.debug(f"URL: {ws_url}")
            
            self.ws = await asyncio.wait_for(
                websockets.connect(
                    ws_url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ),
                timeout=self.config.websocket_timeout
            )
            
            self.is_connected = True
            self.reconnect_count = 0
            
            logger.info("ElevenLabs WebSocket connected")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"WebSocket connection timeout ({self.config.websocket_timeout}s)")
            return False
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Close WebSocket connection."""
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self.ws = None
                self.is_connected = False
                self.is_streaming = False
        
        logger.info("ElevenLabs WebSocket disconnected")
    
    async def send_bos(self, text: str = "") -> bool:
        """
        Send Beginning of Stream (BOS) message.
        
        This initializes the stream with voice settings.
        Optionally includes initial text for faster first chunk.
        
        Args:
            text: Optional initial text to include with BOS
            
        Returns:
            True if sent successfully
        """
        if not self.ws or not self.is_connected:
            logger.error("Cannot send BOS: not connected")
            return False
        
        try:
            # More aggressive chunk_length_schedule to favor earlier first chunk.
            # According to ElevenLabs docs, the first value controls how many
            # characters must be buffered before audio generation begins.
            # We lower this from the default [120, 160, 250, 290] to start
            # generating after ~50 characters while keeping later chunks similar.
            generation_config = {
                "chunk_length_schedule": [50, 120, 250, 290]
            }

            bos_message = {
                "text": text if text else " ",  # Space is required if no text
                "voice_settings": self.config.get_voice_settings(),
                "xi_api_key": self.config.elevenlabs_api_key,
                "generation_config": generation_config,
            }
            
            # Add try_trigger_generation for faster first chunk when we have
            # real text (not just the prewarm space). This hints ElevenLabs
            # to flush the current buffer immediately rather than waiting
            # strictly on chunk_length_schedule.
            if self.config.try_trigger_generation and text:
                bos_message["try_trigger_generation"] = True
            
            await self.ws.send(json.dumps(bos_message))
            
            self.is_streaming = True
            self.stream_start_time = time.time()
            self.first_chunk_time = None
            self.chunks_received = 0
            self.total_audio_bytes = 0
            
            logger.info(f"BOS sent with {len(text)} chars initial text")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send BOS: {e}")
            return False
    
    async def send_text_chunk(self, text: str, flush: bool = False) -> bool:
        """
        Send a text chunk for synthesis.
        
        Args:
            text: Text chunk to synthesize
            flush: If True, trigger generation immediately (lower latency)
            
        Returns:
            True if sent successfully
        """
        if not self.ws or not self.is_connected:
            logger.error("Cannot send text: not connected")
            return False
        
        if not self.is_streaming:
            logger.warning("Stream not started, sending BOS first")
            await self.send_bos(text)
            return True
        
        try:
            message = {
                "text": text
            }
            
            # try_trigger_generation flushes the buffer for lower latency
            if flush or self.config.try_trigger_generation:
                message["try_trigger_generation"] = True
            
            await self.ws.send(json.dumps(message))
            logger.debug(f"Sent text chunk: {len(text)} chars (flush={flush})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send text chunk: {e}")
            return False
    
    async def send_eos(self) -> bool:
        """
        Send End of Stream (EOS) message.
        
        This signals the server to finalize and send any remaining audio.
        
        Returns:
            True if sent successfully
        """
        if not self.ws or not self.is_connected:
            return False
        
        try:
            eos_message = {
                "text": ""  # Empty text signals EOS
            }
            
            await self.ws.send(json.dumps(eos_message))
            logger.info("EOS sent")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send EOS: {e}")
            return False
    
    async def receive_audio(self) -> AsyncGenerator[tuple[bytes, Dict[str, Any]], None]:
        """
        Receive audio chunks from ElevenLabs.
        
        Yields:
            Tuple of (audio_bytes, metadata_dict)
        """
        if not self.ws or not self.is_connected:
            logger.error("Cannot receive audio: not connected")
            return
        
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    
                    # Check for audio data
                    if "audio" in data:
                        audio_b64 = data["audio"]
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            self.chunks_received += 1
                            self.total_audio_bytes += len(audio_bytes)
                            
                            # Track first chunk latency
                            if self.first_chunk_time is None and self.stream_start_time:
                                self.first_chunk_time = time.time()
                                latency_ms = (self.first_chunk_time - self.stream_start_time) * 1000
                                self.last_latency_ms = latency_ms
                                
                                if latency_ms < 150:
                                    logger.info(f"ULTRA-FAST FIRST CHUNK: {latency_ms:.0f}ms (TARGET MET)")
                                elif latency_ms < 300:
                                    logger.info(f"Fast first chunk: {latency_ms:.0f}ms")
                                else:
                                    logger.warning(f"Slow first chunk: {latency_ms:.0f}ms (target: <150ms)")
                            
                            metadata = {
                                "chunk_index": self.chunks_received,
                                "is_final": data.get("isFinal", False),
                                "alignment": data.get("alignment"),
                                "normalizedAlignment": data.get("normalizedAlignment"),
                                "sample_rate": self.config.sample_rate,
                                "format": self.config.output_format
                            }
                            
                            yield audio_bytes, metadata
                    
                    # Check for final message
                    if data.get("isFinal"):
                        logger.info(f"Stream complete: {self.chunks_received} chunks, {self.total_audio_bytes} bytes")
                        break
                    
                    # Check for error
                    if "error" in data:
                        logger.error(f"ElevenLabs error: {data['error']}")
                        break
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from ElevenLabs: {e}")
                    continue
                    
        except ConnectionClosed as e:
            logger.warning(f"WebSocket closed: {e}")
        except WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
        except Exception as e:
            logger.error(f"Error receiving audio: {e}")
        finally:
            self.is_streaming = False
    
    async def synthesize_stream(
        self,
        text_generator: AsyncGenerator[str, None],
        audio_callback: Optional[Callable[[bytes, int, Dict[str, Any]], Any]] = None
    ) -> Dict[str, Any]:
        """
        Stream text from generator and receive audio chunks.
        
        This is the main entry point for continuous streaming synthesis.
        Text chunks from the generator are sent to ElevenLabs as they arrive,
        and audio chunks are delivered via the callback as they're received.
        
        Args:
            text_generator: Async generator yielding text chunks
            audio_callback: Callback for audio chunks (bytes, sample_rate, metadata)
            
        Returns:
            Stats dict with latency and chunk info
        """
        self.audio_callback = audio_callback
        
        # Connect if not already connected
        if not self.is_connected:
            if not await self.connect():
                return {"error": "Connection failed"}
        
        stats = {
            "chunks_received": 0,
            "total_audio_bytes": 0,
            "first_chunk_latency_ms": None,
            "total_time_ms": 0
        }
        
        start_time = time.time()
        first_text = True
        
        try:
            # Start receiver task
            receive_task = asyncio.create_task(self._receive_and_callback())
            
            # Send text chunks as they arrive
            async for text_chunk in text_generator:
                if not text_chunk:
                    continue
                
                if first_text:
                    # Send BOS with first text chunk for fastest first audio
                    await self.send_bos(text_chunk)
                    first_text = False
                else:
                    # Send subsequent chunks with flush for low latency
                    await self.send_text_chunk(text_chunk, flush=True)
            
            # Send EOS to finalize
            await self.send_eos()
            
            # Wait for receiver to complete
            await receive_task
            
        except Exception as e:
            logger.error(f"Synthesis stream error: {e}")
            stats["error"] = str(e)
        finally:
            stats["chunks_received"] = self.chunks_received
            stats["total_audio_bytes"] = self.total_audio_bytes
            stats["first_chunk_latency_ms"] = self.last_latency_ms
            stats["total_time_ms"] = (time.time() - start_time) * 1000
        
        return stats
    
    async def _receive_and_callback(self):
        """Internal method to receive audio and invoke callback."""
        async for audio_bytes, metadata in self.receive_audio():
            if self.audio_callback:
                try:
                    result = self.audio_callback(audio_bytes, self.config.sample_rate, metadata)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Audio callback error: {e}")
    
    async def synthesize_text(
        self,
        text: str,
        audio_callback: Optional[Callable[[bytes, int, Dict[str, Any]], Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize a single text string.
        
        Convenience method for non-streaming use cases.
        
        Args:
            text: Full text to synthesize
            audio_callback: Callback for audio chunks
            
        Returns:
            Stats dict
        """
        async def text_gen():
            yield text
        
        return await self.synthesize_stream(text_gen(), audio_callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            "is_connected": self.is_connected,
            "is_streaming": self.is_streaming,
            "chunks_received": self.chunks_received,
            "total_audio_bytes": self.total_audio_bytes,
            "first_chunk_latency_ms": self.last_latency_ms,
            "reconnect_count": self.reconnect_count
        }


class ElevenLabsProvider:
    """
    High-level provider interface for ElevenLabs TTS.
    
    Manages connection lifecycle and provides simple synthesize() method.
    Compatible with the orchestrator's TTS service interface.
    """
    
    def __init__(self, config: TTSLabsConfig):
        """Initialize provider."""
        self.config = config
        self.manager: Optional[ElevenLabsStreamManager] = None
        self._lock = asyncio.Lock()
    
    async def warmup(self):
        """Pre-warm the connection."""
        logger.info("Warming up ElevenLabs connection...")
        
        async with self._lock:
            self.manager = ElevenLabsStreamManager(self.config)
            if await self.manager.connect():
                logger.info("ElevenLabs connection pre-warmed")
            else:
                logger.warning("ElevenLabs warmup failed")
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None
    ) -> bytes:
        """
        Synthesize text to audio bytes.
        
        Args:
            text: Text to synthesize
            voice: Voice ID (optional, uses config default)
            language: Language code (optional, ignored for ElevenLabs)
            
        Returns:
            Audio bytes (PCM or MP3 depending on config)
        """
        audio_chunks = []
        
        async def collect_audio(audio_bytes: bytes, sample_rate: int, metadata: Dict[str, Any]):
            audio_chunks.append(audio_bytes)
        
        async with self._lock:
            if not self.manager:
                self.manager = ElevenLabsStreamManager(self.config)
            
            # Override voice if provided
            if voice and voice != self.config.elevenlabs_voice_id:
                # Create new manager with different voice
                temp_config = TTSLabsConfig.from_env()
                temp_config.elevenlabs_voice_id = voice
                temp_manager = ElevenLabsStreamManager(temp_config)
                await temp_manager.synthesize_text(text, collect_audio)
                await temp_manager.disconnect()
            else:
                await self.manager.synthesize_text(text, collect_audio)
        
        return b''.join(audio_chunks)
    
    async def close(self):
        """Close provider and cleanup resources."""
        if self.manager:
            await self.manager.disconnect()
            self.manager = None
