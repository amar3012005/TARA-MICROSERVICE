"""
FastRTC Wrapper for STT/VAD Docker Service
==========================================

Streams audio from browser to Docker STT/VAD service with real-time transcription.
Uses Gradio FastRTC UI matching leibniz_fastrtc_wrapper.py for consistent UX.

Features:
- Direct audio streaming from browser to Docker service (no buffering)
- Real-time transcript logging in Docker console
- Enhanced pipeline visibility (VAD/STT fragments, Gemini connection status)
- Automatic reconnection on failures
- Same UI/UX as leibniz_fastrtc_wrapper.py

Environment Variables:
- STT_VAD_HOST: Docker service host (default: localhost)
- STT_VAD_PORT: Docker service port (default: 8001)
- FAST_RTC_PORT: FastRTC server port (default: 7860)
- GEMINI_API_KEY: Hardcoded for testing
"""

import asyncio
import logging
import os
import time
from typing import Optional, Tuple, Any

import numpy as np
import websockets
import json
import httpx

from fastrtc import Stream, AsyncStreamHandler

# Configure detailed logging for Docker console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Hardcode API key
GEMINI_API_KEY = "AIzaSyC6cvyEl4FNjIQCV_p5_2wJkOa1cUObFHU"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Configuration from environment
STT_VAD_HOST = os.getenv("STT_VAD_HOST", "localhost")
STT_VAD_PORT = int(os.getenv("STT_VAD_PORT", "8001"))
FAST_RTC_PORT = int(os.getenv("FAST_RTC_PORT", "7860"))

# WebSocket URLs
DOCKER_WS_URL = f"ws://{STT_VAD_HOST}:{STT_VAD_PORT}/api/v1/transcribe/stream"
HEALTH_URL = f"http://{STT_VAD_HOST}:{STT_VAD_PORT}/health"

# Global state
docker_ws: Optional[Any] = None  # Use Any to avoid deprecation warning
session_id = f"fastrtc_session_{int(time.time())}"
last_transcript_time = 0
transcript_count = 0
chunk_count = 0


def is_websocket_closed(ws) -> bool:
    """Check if WebSocket connection is closed (handles different websockets API versions)"""
    if ws is None:
        return True
    try:
        # Try new API first (websockets >= 10.0) - check close_code
        if hasattr(ws, 'close_code'):
            return ws.close_code is not None
        # Fall back to old API (websockets < 10.0) - check closed attribute
        if hasattr(ws, 'closed'):
            return ws.closed
        # If neither attribute exists, assume closed for safety
        return True
    except Exception:
        # Any error checking state means connection is likely closed
        return True


async def wait_for_service_ready(max_retries: int = 60, retry_delay: float = 1.0) -> bool:
    """
    Wait for Docker STT/VAD service to be ready.
    
    Increased retries to allow for Redis connection retries (up to 25 seconds)
    plus uvicorn startup time.
    """
    logger.info("⏳ Waiting for STT/VAD service to be ready...")
    logger.info(f"   Will check up to {max_retries} times (every {retry_delay}s)")
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(HEALTH_URL)
                if response.status_code == 200:
                    logger.info(f"✅ STT/VAD service is ready! (attempt {attempt + 1}/{max_retries})")
                    return True
        except Exception as e:
            if attempt < max_retries - 1:
                # Log every 5 attempts to avoid spam
                if (attempt + 1) % 5 == 0:
                    logger.info(f"⏳ Service not ready yet (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ Service failed to become ready after {max_retries} attempts")
                logger.error(f"   Last error: {e}")
                logger.error(f"   Health URL: {HEALTH_URL}")
                return False
    
    return False


async def connect_to_docker_service():
    """Connect to Docker STT/VAD WebSocket service."""
    global docker_ws, session_id

    try:
        # Update session ID before connecting
        session_id = f"fastrtc_session_{int(time.time())}"
        
        # Include session_id in WebSocket URL (required by Docker service)
        ws_url_with_session = f"{DOCKER_WS_URL}?session_id={session_id}"
        
        logger.info(f"🔌 Connecting to Docker STT/VAD service at {ws_url_with_session}...")
        
        # Use longer timeout for Docker network
        docker_ws = await asyncio.wait_for(
            websockets.connect(
                ws_url_with_session,
                ping_interval=20,  # Keep-alive ping
                ping_timeout=10,
                close_timeout=10,
            ),
            timeout=10.0
        )
        
        logger.info("=" * 70)
        logger.info(f"✅ Connected to Docker service | Session: {session_id}")
        logger.info("📡 WebSocket connection established | Ready for audio streaming")
        logger.info("=" * 70)

        # Start transcript receiver task
        asyncio.create_task(receive_transcripts())

        return True
    except asyncio.TimeoutError:
        logger.error(f"❌ WebSocket connection timeout after 10s")
        docker_ws = None
        return False
    except Exception as e:
        logger.error(f"❌ Failed to connect to Docker service: {e}")
        import traceback
        logger.error(traceback.format_exc())
        docker_ws = None
        return False


async def receive_transcripts():
    """Receive and log transcripts from Docker service with enhanced visibility."""
    global docker_ws, last_transcript_time, transcript_count

    try:
        logger.info("📡 Started transcript receiver | Listening for fragments...")
        
        while docker_ws and not is_websocket_closed(docker_ws):
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(docker_ws.recv(), timeout=1.0)
                data = json.loads(message)

                if data.get("type") == "fragment":
                    transcript = data.get("text", "").strip()
                    is_final = data.get("is_final", False)

                    if transcript:
                        transcript_count += 1
                        last_transcript_time = time.time()

                        # Enhanced logging with pipeline visibility
                        latency = time.time() - data.get("timestamp", time.time())
                        status = "✅ FINAL" if is_final else "🔄 PARTIAL"
                        
                        logger.info(
                            f"🎤 [{status}] Fragment #{transcript_count} | "
                            f"Text: '{transcript}' | "
                            f"Latency: {latency*1000:.0f}ms"
                        )

                elif data.get("type") == "error":
                    logger.error(f"❌ Docker service error: {data}")

                elif data.get("type") == "timeout":
                    logger.warning("⏱️ Docker service timeout - restarting session")
                    await restart_session()

                elif data.get("type") == "connected":
                    logger.info(f"✅ Session confirmed: {data.get('session_id', 'unknown')}")

            except asyncio.TimeoutError:
                continue  # No message received, continue loop

            except websockets.exceptions.ConnectionClosed:
                logger.warning("⚠️ Docker WebSocket connection closed")
                break

    except Exception as e:
        logger.error(f"❌ Transcript receiver error: {e}")


async def restart_session():
    """Restart the Docker service session."""
    global docker_ws

    logger.info("🔄 Restarting Docker service session...")

    # Close existing connection
    if docker_ws:
        try:
            await docker_ws.close()
        except:
            pass
        docker_ws = None

    # Wait before reconnecting
    await asyncio.sleep(1.0)

    # Reconnect
    success = await connect_to_docker_service()
    if success:
        logger.info("✅ Docker service session restarted successfully")
    else:
        logger.error("❌ Failed to restart Docker service session")


class FastRTCSTTHandler(AsyncStreamHandler):
    """
    FastRTC AsyncStreamHandler that streams audio directly to Docker STT/VAD service.
    
    Provides direct audio streaming without buffering for ultra-low latency.
    """
    
    def __init__(self):
        super().__init__()
        self._chunk_count = 0
        self._last_chunk_log = 0
        self._started = False
        self._ws_connected = False
        logger.info("🎙️ FastRTC STT Handler initialized")
        logger.info(f"   Handler instance: {id(self)}")
        
        # CRITICAL: Try to connect WebSocket immediately (non-blocking fallback)
        # This ensures connection is ready even if start_up() never runs due to WebRTC issues
        try:
            # Check if event loop is running before creating task
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No event loop running yet, will connect later
                pass
            
            if loop:
                loop.create_task(self._ensure_websocket_connection())
            else:
                # No loop yet, will connect in start_up() or receive()
                logger.debug("No event loop yet - will connect WebSocket later")
        except Exception as e:
            logger.debug(f"Could not start pre-connection task: {e}")
    
    async def _ensure_websocket_connection(self):
        """Ensure WebSocket is connected (called from __init__ as fallback)"""
        global docker_ws
        try:
            # Wait a moment for service to be ready
            await asyncio.sleep(2.0)
            
            if docker_ws is None or is_websocket_closed(docker_ws):
                logger.info("🔌 Pre-connecting WebSocket (from __init__ fallback)...")
                success = await connect_to_docker_service()
                if success:
                    self._ws_connected = True
                    logger.info("✅ WebSocket pre-connected (fallback)")
        except Exception as e:
            logger.debug(f"Pre-connection failed (will retry): {e}")
    
    async def start_up(self):
        """Called when WebRTC stream starts - initialize connection and pipeline."""
        global docker_ws, chunk_count
        
        try:
            self._started = True
            logger.info("=" * 70)
            logger.info("🚀 FastRTC stream started | Initializing STT/VAD pipeline...")
            logger.info(f"   Handler instance: {id(self)} | Started: {self._started}")
            logger.info("=" * 70)
            
            # Reset chunk counter for new stream
            chunk_count = 0
            
            # CRITICAL: Connect to Docker service immediately (don't wait)
            # This ensures WebSocket is ready even if WebRTC has issues
            logger.info("🔌 Connecting to Docker WebSocket service...")
            
            if docker_ws is None or is_websocket_closed(docker_ws):
                success = await connect_to_docker_service()
                if success:
                    self._ws_connected = True
                    logger.info("=" * 70)
                    logger.info("✅ Pipeline initialized | Ready for audio streaming")
                    logger.info("📊 Flow: Browser → FastRTC → WebSocket → VAD Manager → Gemini Live → STT")
                    logger.info("🎤 Start speaking - audio will trigger automatic pipeline processing")
                    logger.info("=" * 70)
                else:
                    logger.warning("⚠️ WebSocket connection failed - will retry on first audio chunk")
            else:
                self._ws_connected = True
                logger.info("✅ WebSocket already connected")
            
            logger.info("=" * 70)
            logger.info("✅ FastRTC handler ready")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"❌ Error in start_up(): {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Don't raise - allow connection to continue
    
    async def receive(self, audio: tuple) -> None:
        """
        CRITICAL FIX: Send audio directly to Docker without buffering for ultra-low latency.
        
        Args:
            audio: Tuple of (sample_rate: int, audio_array: np.ndarray) from FastRTC
        """
        global docker_ws, chunk_count
        
        if chunk_count == 0:
            logger.info("=" * 70)
            logger.info("🎤 AUDIO STREAMING STARTED")
            logger.info("Pipeline: Browser 🎤 → FastRTC → WebSocket → VAD Manager → Gemini Live → STT")
            logger.info("=" * 70)
        
        # Ensure WebSocket is connected
        if docker_ws is None or is_websocket_closed(docker_ws):
            logger.warning("🔄 WebSocket not connected, attempting reconnection...")
            logger.info("   This may happen if start_up() was not called (WebRTC connection issue)")
            success = await connect_to_docker_service()
            if success:
                self._ws_connected = True
                logger.info("✅ WebSocket reconnected | Ready for audio streaming")
            else:
                logger.error("❌ Failed to reconnect - audio will be dropped")
                return
        
        try:
            # Parse FastRTC audio format
            if not isinstance(audio, tuple) or len(audio) != 2:
                logger.warning(f"⚠️ Unexpected audio format: {type(audio)}")
                return
            
            sample_rate, audio_array = audio
            
            # Validate and normalize audio
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array, dtype=np.float32)
            
            # Handle multi-dimensional arrays
            if audio_array.ndim == 2:
                audio_array = audio_array.squeeze()
            elif audio_array.ndim > 2:
                audio_array = audio_array.flatten()
            
            # Ensure float32
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # CRITICAL: Normalize to [-1.0, 1.0]
            max_val = np.max(np.abs(audio_array))
            if max_val > 1.0:
                audio_array = audio_array / 32767.0
            
            # CRITICAL: Convert to 16-bit PCM bytes
            audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # CRITICAL: Send IMMEDIATELY (NO BUFFERING)
            try:
                await docker_ws.send(audio_bytes)
                chunk_count += 1
                
                if chunk_count == 1:
                    logger.info(f"✅ First audio chunk sent | Size: {len(audio_bytes)} bytes")
                elif chunk_count % 50 == 0:
                    logger.info(f"📤 Audio chunk #{chunk_count} | {len(audio_bytes)} bytes | {sample_rate}Hz")
                    
            except Exception as send_error:
                logger.error(f"❌ Failed to send audio: {send_error}")
                asyncio.create_task(restart_session())
                
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def emit(self):
        """
        Emit method required by AsyncStreamHandler.
        Returns silence since we're only receiving audio (no TTS output in this service).
        """
        # Return silence - this service only does STT, not TTS
        await asyncio.sleep(0.02)  # Prevent busy loop
        return (16000, np.zeros((1, 1600), dtype=np.int16))
    
    def copy(self) -> 'FastRTCSTTHandler':
        """Create a copy of this handler for FastRTC."""
        return FastRTCSTTHandler()
    
    async def shutdown(self) -> None:
        """Cleanup resources when stream closes."""
        global docker_ws
        
        logger.info("=" * 70)
        logger.info("🛑 FastRTC stream shutting down...")
        logger.info(f"   Handler instance: {id(self)} | Started: {self._started} | WS Connected: {self._ws_connected}")
        logger.info("=" * 70)
        
        if docker_ws:
            try:
                await docker_ws.close()
                logger.info("✅ WebSocket closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing WebSocket: {e}")
            docker_ws = None
            self._ws_connected = False
        
        self._started = False
        logger.info("✅ FastRTC stream closed")


# Create FastRTC stream function (matching leibniz_fastrtc_wrapper.py pattern)
def create_fastrtc_app():
    """
    Create FastRTC Gradio app for STT/VAD transcription.
    
    Returns:
        Gradio Interface configured for WebRTC audio input
    """
    # Create handler instance (fresh for each call)
    handler = FastRTCSTTHandler()
    
    # Create FastRTC stream (matching leibniz_fastrtc_wrapper.py)
    stream = Stream(
        handler=handler,
        modality="audio",
        mode="send-receive",  # Use send-receive even for receive-only (required for FastRTC to work)
        ui_args={
            "title": "Leibniz STT/VAD Transcription Service",
            "description": "Speak into your browser microphone. Audio streams directly to the STT/VAD service for real-time transcription. "
                         "Check Docker console logs for pipeline progress, speech detection, and transcript fragments."
        }
    )
    
    logger.info("✅ FastRTC app created")
    return stream.ui

# Use Gradio's launch method to serve the FastRTC UI
# stream.ui is a Gradio Blocks app that needs to be launched directly
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🚀 FastRTC STT/VAD Bridge Starting")
    logger.info("=" * 70)
    logger.info(f"📡 Docker STT/VAD service: {STT_VAD_HOST}:{STT_VAD_PORT}")
    logger.info(f"🌐 FastRTC server port: {FAST_RTC_PORT}")
    logger.info(f"🔑 Gemini API Key: {'*' * 20} (hardcoded)")
    logger.info("=" * 70)
    
    # Create FastRTC app (matching leibniz_fastrtc_wrapper.py pattern)
    # Create fresh app instance for launch
    app = create_fastrtc_app()
    
    # Wait for service to be ready in background (non-blocking)
    # Increased retries to allow for full service initialization
    def start_background_task():
        try:
            asyncio.run(wait_for_service_ready(max_retries=60, retry_delay=1.0))
        except:
            pass  # Silent failure - service check is optional
    
    import threading
    thread = threading.Thread(target=start_background_task, daemon=True)
    thread.start()
    
    # Launch Gradio app with HTTPS support (REQUIRED for microphone access)
    # Browsers require HTTPS for microphone access - use Gradio share to create HTTPS tunnel
    logger.info(f"🌐 Launching Gradio FastRTC UI on port {FAST_RTC_PORT}...")
    logger.info(f"📱 Local access: http://localhost:{FAST_RTC_PORT}")
    logger.info(f"🔒 HTTPS URL will be provided below - USE THAT URL for microphone access")
    logger.info(f"⚠️  IMPORTANT: Use the HTTPS URL (not localhost) for microphone access!")
    logger.info("=" * 70)
    
    # Add Windows/Docker Warning
    if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_ENVIRONMENT") == "true":
        logger.info("⚠️  WINDOWS DOCKER DETECTED")
        logger.info("   WebRTC (Gradio) may fail due to Docker NAT issues.")
        logger.info("   If the UI gets stuck at 'Connecting...', please use the Direct TCP Client:")
        logger.info(f"   👉 http://localhost:{STT_VAD_PORT}/client")
        logger.info("=" * 70)
    
    print("\n" + "="*70)
    print("⚠️  IMPORTANT: Media Device Access Instructions")
    print("="*70)
    print("1. Wait for the HTTPS URL below (starts with https://)")
    print("2. Copy and paste that HTTPS URL into your browser")
    print("3. DO NOT use http://localhost - browsers require HTTPS for microphone")
    print("4. When prompted, click 'Allow' for microphone access")
    print("="*70 + "\n")
    
    # Launch with HTTPS tunnel via Gradio share (matching leibniz_fastrtc_wrapper.py exactly)
    # CRITICAL: Use the HTTPS URL provided by Gradio, NOT localhost
    app.launch(
        server_name="0.0.0.0",
        server_port=FAST_RTC_PORT,
        share=True,  # Create HTTPS tunnel via Gradio (REQUIRED for microphone access)
        show_error=True
    )
