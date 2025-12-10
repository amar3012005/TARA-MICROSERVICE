"""
ElevenLabs WebSocket Manager for Ultra-Low Latency Streaming

Implements bidirectional WebSocket streaming using the ElevenLabs stream-input API.
Optimized for <150ms first chunk latency with eleven_turbo_v2_5 model.

Reference: https://elevenlabs.io/docs/api-reference/text-to-speech/stream-input

This module provides:
1. stream_text_to_audio() - Stateless function for streaming text -> audio
2. ElevenLabsStreamManager - Class wrapper for backward compatibility
3. ElevenLabsProvider - High-level provider for HTTP synthesis
"""

import asyncio
import base64
import json
import logging
import time
from typing import Optional, Callable, Dict, Any, AsyncGenerator, Union

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .config import TTSLabsConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Stateless Streaming Function (Recommended)
# =============================================================================

async def stream_text_to_audio(
    config: TTSLabsConfig,
    text_iterator: Union[AsyncGenerator[str, None], list[str], str],
    audio_callback: Callable[[bytes, int, Dict[str, Any]], Any],
) -> Dict[str, Any]:
    """
    Stream text to audio using ElevenLabs WebSocket API.
    
    This is the main entry point for streaming synthesis. It creates a fresh
    WebSocket connection, sends text chunks as they arrive, and invokes the
    audio callback for each received audio chunk. The connection is guaranteed
    to be cleaned up after completion.
    
    Protocol (per ElevenLabs docs):
    1. Connect to wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input
    2. Send BOS message with voice settings and optional initial text
    3. Send text chunks as {"text": "...", "try_trigger_generation": true}
    4. Send EOS as {"text": ""}
    5. Receive audio chunks until isFinal=true
    
    Args:
        config: TTSLabsConfig with API key and settings
        text_iterator: Async generator, list, or string of text to synthesize
        audio_callback: Callback invoked for each audio chunk:
                       callback(audio_bytes, sample_rate, metadata)
    
    Returns:
        Stats dict with:
        - chunks_received: Number of audio chunks
        - total_audio_bytes: Total bytes of audio received
        - first_chunk_latency_ms: Time to first audio chunk
        - total_time_ms: Total synthesis time
        - error: Error message if any
    """
    if not config.elevenlabs_api_key:
        logger.error("ELEVENLABS_API_KEY not configured")
        return {"error": "API key not configured"}
    
    # Normalize text_iterator to async generator
    if isinstance(text_iterator, str):
        async def _str_gen():
            yield text_iterator
        text_iter = _str_gen()
    elif isinstance(text_iterator, list):
        async def _list_gen():
            for item in text_iterator:
                yield item
        text_iter = _list_gen()
    else:
        text_iter = text_iterator
    
    # Build WebSocket URL
    ws_url = config.get_websocket_url()
    headers = {"xi-api-key": config.elevenlabs_api_key}
    
    # Stats tracking
    stats = {
        "chunks_received": 0,
        "total_audio_bytes": 0,
        "first_chunk_latency_ms": None,
        "total_time_ms": 0,
    }
    start_time = time.time()
    first_chunk_time: Optional[float] = None
    stream_start_time: Optional[float] = None
    
    # Shared state for concurrent tasks
    send_complete = asyncio.Event()
    receive_error: Optional[str] = None
    
    async def send_text(ws):
        """Send text chunks to ElevenLabs."""
        nonlocal stream_start_time
        first_text = True
        
        try:
            async for text_chunk in text_iter:
                if not text_chunk:
                    continue
                
                if first_text:
                    # BOS message with voice settings and initial text
                    bos_message = {
                        "text": text_chunk,
                        "voice_settings": config.get_voice_settings(),
                        "xi_api_key": config.elevenlabs_api_key,
                        "generation_config": {
                            # Lower first value = faster first chunk (default is 120)
                            "chunk_length_schedule": [50, 120, 250, 290]
                        },
                    }
                    if config.try_trigger_generation:
                        bos_message["try_trigger_generation"] = True
                    
                    await ws.send(json.dumps(bos_message))
                    stream_start_time = time.time()
                    logger.info(f"BOS sent with {len(text_chunk)} chars initial text")
                    first_text = False
                else:
                    # Subsequent text chunks
                    message = {"text": text_chunk}
                    if config.try_trigger_generation:
                        message["try_trigger_generation"] = True
                    await ws.send(json.dumps(message))
                    logger.debug(f"Sent text chunk: {len(text_chunk)} chars")
            
            # EOS message (empty text)
            await ws.send(json.dumps({"text": ""}))
            logger.info("EOS sent")
            
        except Exception as e:
            logger.error(f"Error sending text: {e}")
        finally:
            send_complete.set()
    
    async def receive_audio(ws):
        """Receive audio chunks from ElevenLabs."""
        nonlocal first_chunk_time, receive_error
        
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    
                    # Check for error first
                    if "error" in data:
                        error_msg = data["error"]
                        logger.error(f"ElevenLabs error: {error_msg}")
                        receive_error = error_msg
                        break
                    
                    # Process audio data
                    if "audio" in data and data["audio"]:
                        audio_bytes = base64.b64decode(data["audio"])
                        stats["chunks_received"] += 1
                        stats["total_audio_bytes"] += len(audio_bytes)
                        
                        # Track first chunk latency
                        if first_chunk_time is None and stream_start_time:
                            first_chunk_time = time.time()
                            latency_ms = (first_chunk_time - stream_start_time) * 1000
                            stats["first_chunk_latency_ms"] = latency_ms
                            
                            if latency_ms < 150:
                                logger.info(f"⚡ ULTRA-FAST FIRST CHUNK: {latency_ms:.0f}ms")
                            elif latency_ms < 300:
                                logger.info(f"✅ Fast first chunk: {latency_ms:.0f}ms")
                            else:
                                logger.warning(f"⚠️ Slow first chunk: {latency_ms:.0f}ms (target: <150ms)")
                        
                        # Build metadata
                        metadata = {
                            "chunk_index": stats["chunks_received"],
                            "is_final": data.get("isFinal", False),
                            "alignment": data.get("alignment"),
                            "normalizedAlignment": data.get("normalizedAlignment"),
                            "sample_rate": config.sample_rate,
                            "format": config.output_format,
                        }
                        
                        # Invoke callback
                        try:
                            result = audio_callback(audio_bytes, config.sample_rate, metadata)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            logger.error(f"Audio callback error: {e}")
                        
                        # Check for final chunk
                        if data.get("isFinal"):
                            logger.info(f"✅ Stream complete: {stats['chunks_received']} chunks, {stats['total_audio_bytes']} bytes")
                            break
                    
                    # Final message without audio
                    elif data.get("isFinal"):
                        logger.info(f"✅ Stream complete (no audio in final): {stats['chunks_received']} chunks")
                        break
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from ElevenLabs: {e}")
                    continue
                    
        except ConnectionClosed as e:
            if e.code == 1000:
                logger.info(f"WebSocket closed normally (code={e.code})")
            else:
                logger.warning(f"WebSocket closed: code={e.code}, reason={e.reason}")
        except asyncio.CancelledError:
            logger.info("Audio receive cancelled")
            raise
        except Exception as e:
            logger.error(f"Error receiving audio: {e}")
            receive_error = str(e)
    
    # Main connection logic
    try:
        logger.info(f"Connecting to ElevenLabs WebSocket...")
        
        async with websockets.connect(
            ws_url,
            additional_headers=headers,
            ping_interval=20,      # Keep-alive ping every 20s
            ping_timeout=10,       # Wait 10s for pong
            close_timeout=5,       # Wait 5s for close handshake
            max_size=10_000_000,   # Allow large audio messages (10MB)
            max_queue=64,          # Buffer up to 64 messages
        ) as ws:
            logger.info("ElevenLabs WebSocket connected")
            
            # Run send and receive concurrently
            send_task = asyncio.create_task(send_text(ws))
            receive_task = asyncio.create_task(receive_audio(ws))
            
            # Wait for both to complete
            await asyncio.gather(send_task, receive_task)
            
    except asyncio.TimeoutError:
        logger.error(f"WebSocket connection timeout")
        stats["error"] = "Connection timeout"
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        stats["error"] = str(e)
    finally:
        stats["total_time_ms"] = (time.time() - start_time) * 1000
        if receive_error:
            stats["error"] = receive_error
    
    return stats


# =============================================================================
# Class-based wrapper (Backward Compatibility)
# =============================================================================

class ElevenLabsStreamManager:
    """
    Manages WebSocket connection to ElevenLabs stream-input API.
    
    This class wraps the stateless stream_text_to_audio() function for
    backward compatibility with existing code.
    
    For new code, prefer using stream_text_to_audio() directly.
    """
    
    def __init__(self, config: TTSLabsConfig):
        """Initialize ElevenLabs stream manager."""
        self.config = config
        self.is_connected = False
        self.is_streaming = False
        
        # Performance metrics (populated after synthesis)
        self.first_chunk_time: Optional[float] = None
        self.chunks_received = 0
        self.total_audio_bytes = 0
        self.last_latency_ms: Optional[float] = None
        
        logger.info(f"ElevenLabs StreamManager initialized")
        logger.info(f"  Model: {config.elevenlabs_model_id.strip()}")
        logger.info(f"  Voice: {config.elevenlabs_voice_id}")
        logger.info(f"  Latency optimization: {config.optimize_streaming_latency}")
        logger.info(f"  Output format: {config.output_format}")
    
    async def connect(self) -> bool:
        """Mark as ready (connection is per-stream now)."""
        self.is_connected = True
        return True
    
    async def disconnect(self):
        """Mark as disconnected."""
        self.is_connected = False
        self.is_streaming = False
        logger.info("ElevenLabs StreamManager disconnected")
    
    async def synthesize_stream(
        self,
        text_generator: AsyncGenerator[str, None],
        audio_callback: Optional[Callable[[bytes, int, Dict[str, Any]], Any]] = None
    ) -> Dict[str, Any]:
        """
        Stream text from generator and receive audio chunks.
        
        Delegates to the stateless stream_text_to_audio() function.
        """
        self.is_streaming = True
        
        # Default callback that does nothing
        def noop_callback(audio_bytes, sample_rate, metadata):
            pass
        
        callback = audio_callback or noop_callback
        
        try:
            stats = await stream_text_to_audio(self.config, text_generator, callback)
            
            # Update metrics from stats
            self.chunks_received = stats.get("chunks_received", 0)
            self.total_audio_bytes = stats.get("total_audio_bytes", 0)
            self.last_latency_ms = stats.get("first_chunk_latency_ms")
            
            return stats
        finally:
            self.is_streaming = False
    
    async def synthesize_text(
        self,
        text: str,
        audio_callback: Optional[Callable[[bytes, int, Dict[str, Any]], Any]] = None
    ) -> Dict[str, Any]:
        """Synthesize a single text string."""
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
        }


# =============================================================================
# High-level Provider (for HTTP synthesis)
# =============================================================================

class ElevenLabsProvider:
    """
    High-level provider interface for ElevenLabs TTS.
    
    Provides a simple synthesize() method that collects all audio chunks
    and returns the complete audio bytes. Used by the HTTP endpoint.
    """
    
    def __init__(self, config: TTSLabsConfig):
        """Initialize provider."""
        self.config = config
        self._lock = asyncio.Lock()
    
    async def warmup(self):
        """
        Pre-warm is a no-op now since each synthesis uses a fresh connection.
        Kept for backward compatibility.
        """
        logger.info("ElevenLabs provider ready (warmup is no-op)")
    
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
        audio_chunks: list[bytes] = []
        
        def collect_audio(audio_bytes: bytes, sample_rate: int, metadata: Dict[str, Any]):
            audio_chunks.append(audio_bytes)
        
        async with self._lock:
            # Use custom voice if provided
            if voice and voice != self.config.elevenlabs_voice_id:
                temp_config = TTSLabsConfig.from_env()
                temp_config.elevenlabs_voice_id = voice
                config = temp_config
            else:
                config = self.config
            
            stats = await stream_text_to_audio(config, text, collect_audio)
            
            if stats.get("error"):
                logger.error(f"Synthesis failed: {stats['error']}")
                return b""
        
        return b"".join(audio_chunks)
    
    async def close(self):
        """Close provider (no-op, connections are per-stream)."""
        pass
