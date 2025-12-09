"""
ElevenLabs TTS WebSocket Client for Orchestrator

Direct async WebSocket client that connects to the tts-labs service's /api/v1/stream endpoint.
Optimized for ultra-low latency (<150ms first audio chunk) with prewarm support.

Usage:
    client = ElevenLabsTTSClient(tts_service_url, session_id)
    
    # Pre-warm connection when user starts speaking (VAD detected)
    await client.prewarm()
    
    # Stream text chunks from RAG/LLM and receive audio
    async for audio_bytes, metadata in client.stream_text_to_audio(text_generator):
        # Forward audio to playback destination
        await audio_sink(audio_bytes)
    
    # Cleanup
    await client.close()
"""

import asyncio
import base64
import json
import logging
import time
from typing import Optional, AsyncGenerator, Dict, Any, Tuple

import aiohttp
from aiohttp import WSMsgType

logger = logging.getLogger(__name__)


class ElevenLabsTTSClient:
    """
    Async WebSocket client for ElevenLabs TTS streaming via tts-labs service.
    
    Connects to tts-labs service's /api/v1/stream endpoint which handles
    the actual ElevenLabs WebSocket connection internally.
    
    Features:
    - Prewarm support: Pre-establish connection before first text arrives
    - Continuous streaming: Send text chunks as they arrive from RAG
    - Low-latency audio delivery: Receive audio chunks as they're generated
    """
    
    def __init__(
        self, 
        tts_service_url: str,
        session_id: str,
        sample_rate: int = 24000
    ):
        """
        Initialize ElevenLabs TTS client.
        
        Args:
            tts_service_url: Base URL of tts-labs service (e.g., http://tts-labs:8006)
            session_id: Unique session identifier for this connection
            sample_rate: Expected audio sample rate (default: 24000 for ElevenLabs)
        """
        self.tts_service_url = tts_service_url
        self.session_id = session_id
        self.sample_rate = sample_rate
        
        # Connection state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._is_connected = False
        self._is_prewarmed = False
        self._is_streaming = False
        
        # Metrics
        self._prewarm_time_ms: Optional[float] = None
        self._first_audio_time: Optional[float] = None
        self._stream_start_time: Optional[float] = None
        self._chunks_received = 0
        self._total_audio_bytes = 0
        
        # Build WebSocket URL
        ws_base = tts_service_url.replace("http://", "ws://").replace("https://", "wss://")
        self._ws_url = f"{ws_base}/api/v1/stream?session_id={session_id}"
        
        logger.info(f"ElevenLabsTTSClient initialized for session {session_id}")
        logger.debug(f"  WebSocket URL: {self._ws_url}")
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._is_connected and self._ws is not None
    
    @property
    def is_prewarmed(self) -> bool:
        """Check if connection is pre-warmed and ready for streaming."""
        return self._is_prewarmed
    
    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming text/audio."""
        return self._is_streaming
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection to tts-labs service.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self._is_connected:
            return True
        
        try:
            # Create aiohttp session if needed
            if self._session is None:
                self._session = aiohttp.ClientSession()
            
            logger.info(f"Connecting to TTS service: {self._ws_url}")
            
            self._ws = await asyncio.wait_for(
                self._session.ws_connect(
                    self._ws_url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ),
                timeout=10.0
            )
            
            # Wait for connection confirmation
            msg = await asyncio.wait_for(self._ws.receive(), timeout=5.0)
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "connected":
                    self._is_connected = True
                    logger.info(f"TTS WebSocket connected: {data}")
                    return True
            
            logger.warning(f"Unexpected connection response: {msg}")
            return False
            
        except asyncio.TimeoutError:
            logger.error("TTS WebSocket connection timeout")
            return False
        except Exception as e:
            logger.error(f"TTS WebSocket connection failed: {e}")
            return False
    
    async def prewarm(self) -> bool:
        """
        Pre-warm the ElevenLabs connection for ultra-low latency.
        
        Call this when VAD detects user speech to establish the connection
        before the RAG/LLM response is ready.
        
        Returns:
            True if prewarm successful, False otherwise
        """
        if self._is_prewarmed:
            logger.debug("Already prewarmed, skipping")
            return True
        
        # Ensure connected
        if not self._is_connected:
            if not await self.connect():
                return False
        
        try:
            prewarm_start = time.time()
            
            # Send prewarm message
            await self._ws.send_json({
                "type": "prewarm"
            })
            
            # Wait for prewarm confirmation
            msg = await asyncio.wait_for(self._ws.receive(), timeout=5.0)
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "prewarmed":
                    self._prewarm_time_ms = (time.time() - prewarm_start) * 1000
                    self._is_prewarmed = True
                    
                    server_prewarm_ms = data.get("prewarm_duration_ms", 0)
                    logger.info(f"⚡ TTS pre-warmed in {self._prewarm_time_ms:.0f}ms (server: {server_prewarm_ms:.0f}ms)")
                    return True
                elif data.get("type") == "error":
                    logger.error(f"Prewarm error: {data.get('message')}")
                    return False
            
            logger.warning(f"Unexpected prewarm response: {msg}")
            return False
            
        except asyncio.TimeoutError:
            logger.error("Prewarm timeout")
            return False
        except Exception as e:
            logger.error(f"Prewarm failed: {e}")
            return False
    
    async def stream_text_to_audio(
        self,
        text_generator: AsyncGenerator[str, None],
        on_first_audio: Optional[callable] = None
    ) -> AsyncGenerator[Tuple[bytes, Dict[str, Any]], None]:
        """
        Stream text chunks to ElevenLabs and yield audio chunks as they arrive.
        
        This is the main streaming method. Text chunks from the generator are
        sent to ElevenLabs as they arrive, and audio chunks are yielded
        as soon as they're received.
        
        Args:
            text_generator: Async generator yielding text chunks from RAG/LLM
            on_first_audio: Optional callback when first audio chunk arrives
            
        Yields:
            Tuple of (audio_bytes, metadata_dict)
        """
        if not self._is_connected:
            if not await self.connect():
                logger.error("Cannot stream: not connected")
                return
        
        self._is_streaming = True
        self._stream_start_time = time.time()
        self._first_audio_time = None
        self._chunks_received = 0
        self._total_audio_bytes = 0
        
        # Queue for audio chunks from receiver
        audio_queue: asyncio.Queue = asyncio.Queue()
        receiver_task: Optional[asyncio.Task] = None
        
        async def receive_audio():
            """Background task to receive audio from WebSocket."""
            stream_complete_received = False
            try:
                async for msg in self._ws:
                    if msg.type == WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        
                        if msg_type == "audio":
                            audio_b64 = data.get("data", "")
                            if audio_b64:
                                audio_bytes = base64.b64decode(audio_b64)
                                metadata = {
                                    "index": data.get("index", self._chunks_received),
                                    "sample_rate": data.get("sample_rate", self.sample_rate),
                                    "is_final": data.get("is_final", False)
                                }
                                await audio_queue.put((audio_bytes, metadata))
                                logger.debug(f"📥 Received audio chunk: {len(audio_bytes)} bytes")
                        
                        elif msg_type == "stream_complete":
                            logger.info("Stream complete signal received (continuing to drain remaining chunks)")
                            stream_complete_received = True
                            # Don't break immediately - continue receiving any remaining chunks
                            # The main loop will handle the final signal
                            # Set a flag and continue for a short time to drain
                            await asyncio.sleep(0.5)  # Brief wait for any final chunks
                            await audio_queue.put((None, {"is_final": True}))
                            break
                        
                        elif msg_type == "error":
                            logger.error(f"TTS error: {data.get('message')}")
                            await audio_queue.put((None, {"error": data.get("message")}))
                            break
                        
                        elif msg_type in ("prewarmed", "connected", "pong"):
                            # Ignore non-audio messages
                            pass
                        
                    elif msg.type == WSMsgType.CLOSED:
                        logger.info("WebSocket closed by server")
                        if not stream_complete_received:
                            # WebSocket closed unexpectedly
                            await audio_queue.put((None, {"is_final": True}))
                        break
                    elif msg.type == WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {msg}")
                        await audio_queue.put((None, {"error": "WebSocket error"}))
                        break
                        
            except asyncio.CancelledError:
                logger.debug("Receiver task cancelled")
                pass
            except Exception as e:
                logger.error(f"Receiver error: {e}")
            finally:
                # Ensure final signal is sent if not already sent
                if not stream_complete_received:
                    try:
                        await audio_queue.put((None, {"is_final": True}))
                    except Exception:
                        pass
        
        try:
            # Start receiver task
            receiver_task = asyncio.create_task(receive_audio())
            
            # Send text chunks as they arrive and yield audio immediately
            chunk_count = 0
            text_generator_task = None
            
            async def send_text_chunks():
                """Background task to send text chunks."""
                nonlocal chunk_count
                try:
                    async for text_chunk in text_generator:
                        if not text_chunk:
                            continue
                        
                        chunk_count += 1
                        
                        # Send chunk to TTS service
                        await self._ws.send_json({
                            "type": "stream_chunk",
                            "text": text_chunk,
                            "emotion": "helpful"
                        })
                        logger.debug(f"Sent chunk {chunk_count}: {len(text_chunk)} chars")
                except Exception as e:
                    logger.error(f"Error sending text chunks: {e}")
            
            # Start text sending task
            text_generator_task = asyncio.create_task(send_text_chunks())
            
            # Continuously yield audio chunks as they arrive (don't wait for text chunks)
            try:
                while True:
                    # Wait for audio with timeout to check if text generator is done
                    try:
                        audio_bytes, metadata = await asyncio.wait_for(
                            audio_queue.get(),
                            timeout=0.1  # Short timeout to check text task
                        )
                    except asyncio.TimeoutError:
                        # Check if text generator is done
                        if text_generator_task.done():
                            # Text sending complete, break to send stream_end
                            break
                        # Text still sending, continue waiting for audio
                        continue
                    
                    if audio_bytes is None:
                        if metadata.get("error"):
                            logger.error(f"Stream error: {metadata.get('error')}")
                            return
                        if metadata.get("is_final"):
                            # Stream complete signal
                            break
                        continue
                    
                    self._chunks_received += 1
                    self._total_audio_bytes += len(audio_bytes)
                    
                    # Track first audio latency
                    if self._first_audio_time is None:
                        self._first_audio_time = time.time()
                        latency_ms = (self._first_audio_time - self._stream_start_time) * 1000
                        
                        if latency_ms < 150:
                            logger.info(f"⚡ ULTRA-FAST FIRST AUDIO: {latency_ms:.0f}ms")
                        elif latency_ms < 300:
                            logger.info(f"✅ Fast first audio: {latency_ms:.0f}ms")
                        else:
                            logger.warning(f"⚠️ Slow first audio: {latency_ms:.0f}ms")
                        
                        if on_first_audio:
                            try:
                                await on_first_audio(latency_ms)
                            except Exception:
                                pass
                    
                    yield audio_bytes, metadata
                    
            finally:
                # Wait for text generator to finish
                if text_generator_task and not text_generator_task.done():
                    try:
                        await asyncio.wait_for(text_generator_task, timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning("Text generator task timeout")
                        text_generator_task.cancel()
            
            # Send end of stream
            await self._ws.send_json({
                "type": "stream_end"
            })
            logger.info(f"Stream end sent after {chunk_count} chunks")
            
            # Yield remaining audio chunks immediately (don't wait, drain queue)
            # Continue until we get the final signal or timeout
            remaining_timeout = 15.0  # Longer timeout for remaining chunks
            last_chunk_time = time.time()
            
            while True:
                try:
                    # Try to get chunk immediately, then wait with timeout
                    try:
                        audio_bytes, metadata = audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        # Queue empty, wait a bit for more chunks
                        elapsed = time.time() - last_chunk_time
                        if elapsed > remaining_timeout:
                            logger.warning(f"No audio chunks received for {elapsed:.1f}s, ending stream")
                            break
                        
                        try:
                            audio_bytes, metadata = await asyncio.wait_for(
                                audio_queue.get(),
                                timeout=0.5  # Short wait for next chunk
                            )
                        except asyncio.TimeoutError:
                            # No chunk arrived, check if we should continue waiting
                            elapsed = time.time() - last_chunk_time
                            if elapsed > remaining_timeout:
                                logger.warning(f"Timeout waiting for remaining audio chunks ({elapsed:.1f}s)")
                                break
                            continue
                    
                    if audio_bytes is None:
                        # Final signal or error
                        if metadata.get("is_final"):
                            logger.debug("Received final signal, ending stream")
                        break
                    
                    last_chunk_time = time.time()
                    self._chunks_received += 1
                    self._total_audio_bytes += len(audio_bytes)
                    
                    # Track first audio latency (in case no audio arrived during text sending)
                    if self._first_audio_time is None:
                        self._first_audio_time = time.time()
                        latency_ms = (self._first_audio_time - self._stream_start_time) * 1000
                        logger.info(f"First audio chunk: {latency_ms:.0f}ms")
                        
                        if on_first_audio:
                            try:
                                await on_first_audio(latency_ms)
                            except Exception:
                                pass
                    
                    yield audio_bytes, metadata
                    
                except Exception as e:
                    logger.error(f"Error yielding remaining chunks: {e}")
                    break
            
            total_time = (time.time() - self._stream_start_time) * 1000
            logger.info(f"✅ Stream complete: {self._chunks_received} chunks, {self._total_audio_bytes} bytes in {total_time:.0f}ms")
            
        except asyncio.CancelledError:
            logger.warning("Stream cancelled")
            raise
        except Exception as e:
            logger.error(f"Stream error: {e}")
            raise
        finally:
            self._is_streaming = False
            if receiver_task and not receiver_task.done():
                receiver_task.cancel()
                try:
                    await receiver_task
                except asyncio.CancelledError:
                    pass
    
    async def send_single_text(self, text: str) -> AsyncGenerator[Tuple[bytes, Dict[str, Any]], None]:
        """
        Send a single text for synthesis and yield audio chunks.
        
        Convenience method for non-streaming use cases.
        
        Args:
            text: Full text to synthesize
            
        Yields:
            Tuple of (audio_bytes, metadata_dict)
        """
        async def text_gen():
            yield text
        
        async for audio_bytes, metadata in self.stream_text_to_audio(text_gen()):
            yield audio_bytes, metadata
    
    async def close(self):
        """Close WebSocket connection and cleanup resources."""
        self._is_connected = False
        self._is_prewarmed = False
        self._is_streaming = False
        
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
            self._ws = None
        
        if self._session:
            try:
                await self._session.close()
            except Exception as e:
                logger.debug(f"Error closing session: {e}")
            self._session = None
        
        logger.info(f"ElevenLabsTTSClient closed for session {self.session_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        return {
            "session_id": self.session_id,
            "is_connected": self._is_connected,
            "is_prewarmed": self._is_prewarmed,
            "is_streaming": self._is_streaming,
            "prewarm_time_ms": self._prewarm_time_ms,
            "first_audio_latency_ms": (
                (self._first_audio_time - self._stream_start_time) * 1000
                if self._first_audio_time and self._stream_start_time else None
            ),
            "chunks_received": self._chunks_received,
            "total_audio_bytes": self._total_audio_bytes
        }


class ElevenLabsTTSPool:
    """
    Pool of pre-warmed ElevenLabs TTS clients for instant streaming.
    
    Maintains a pool of prewarmed connections that can be acquired
    immediately when RAG response starts, eliminating connection latency.
    """
    
    def __init__(
        self,
        tts_service_url: str,
        pool_size: int = 3,
        sample_rate: int = 24000
    ):
        """
        Initialize TTS client pool.
        
        Args:
            tts_service_url: Base URL of tts-labs service
            pool_size: Number of pre-warmed connections to maintain
            sample_rate: Audio sample rate
        """
        self.tts_service_url = tts_service_url
        self.pool_size = pool_size
        self.sample_rate = sample_rate
        
        self._pool: asyncio.Queue = asyncio.Queue()
        self._active_clients: Dict[str, ElevenLabsTTSClient] = {}
        self._lock = asyncio.Lock()
        self._session_counter = 0
        self._prewarm_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the pool and begin pre-warming connections."""
        logger.info(f"Starting ElevenLabs TTS pool with {self.pool_size} connections")
        self._prewarm_task = asyncio.create_task(self._maintain_pool())
    
    async def stop(self):
        """Stop the pool and close all connections."""
        if self._prewarm_task:
            self._prewarm_task.cancel()
            try:
                await self._prewarm_task
            except asyncio.CancelledError:
                pass
        
        # Close pooled clients
        while not self._pool.empty():
            try:
                client = self._pool.get_nowait()
                await client.close()
            except asyncio.QueueEmpty:
                break
        
        # Close active clients
        for client in list(self._active_clients.values()):
            await client.close()
        self._active_clients.clear()
        
        logger.info("ElevenLabs TTS pool stopped")
    
    async def _maintain_pool(self):
        """Background task to maintain pool of pre-warmed connections."""
        while True:
            try:
                # Check if we need more connections
                current_size = self._pool.qsize()
                needed = self.pool_size - current_size
                
                for _ in range(needed):
                    async with self._lock:
                        self._session_counter += 1
                        session_id = f"pool_{self._session_counter}_{int(time.time())}"
                    
                    client = ElevenLabsTTSClient(
                        self.tts_service_url,
                        session_id,
                        self.sample_rate
                    )
                    
                    if await client.prewarm():
                        await self._pool.put(client)
                        logger.debug(f"Added pre-warmed client to pool: {session_id}")
                    else:
                        await client.close()
                        logger.warning(f"Failed to prewarm client: {session_id}")
                
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pool maintenance error: {e}")
                await asyncio.sleep(5.0)
    
    async def acquire(self, timeout: float = 5.0) -> Optional[ElevenLabsTTSClient]:
        """
        Acquire a pre-warmed client from the pool.
        
        Args:
            timeout: Maximum time to wait for a client
            
        Returns:
            Pre-warmed ElevenLabsTTSClient, or None if timeout
        """
        try:
            client = await asyncio.wait_for(
                self._pool.get(),
                timeout=timeout
            )
            
            # Track active client
            async with self._lock:
                self._active_clients[client.session_id] = client
            
            return client
            
        except asyncio.TimeoutError:
            logger.warning("Timeout acquiring client from pool, creating new one")
            
            # Fallback: create a new client
            async with self._lock:
                self._session_counter += 1
                session_id = f"fallback_{self._session_counter}_{int(time.time())}"
            
            client = ElevenLabsTTSClient(
                self.tts_service_url,
                session_id,
                self.sample_rate
            )
            
            if await client.prewarm():
                async with self._lock:
                    self._active_clients[session_id] = client
                return client
            
            await client.close()
            return None
    
    async def release(self, client: ElevenLabsTTSClient):
        """
        Release a client back to the pool or close it.
        
        Args:
            client: Client to release
        """
        async with self._lock:
            if client.session_id in self._active_clients:
                del self._active_clients[client.session_id]
        
        # Close the client (don't reuse after streaming)
        await client.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": self._pool.qsize(),
            "active_clients": len(self._active_clients),
            "target_size": self.pool_size,
            "total_created": self._session_counter
        }
