"""
Native WebRTC STT/VAD Application (Sarvam AI Saarika)
=====================================================

Implements the "Native WebRTC" architecture for ultra-low latency.
- Bypasses WebSocket bridge (TCP)
- Uses direct UDP/RTP media streams via FastRTC/aiortc
- Runs VAD/STT pipeline in-process (zero-hop audio processing)

Architecture:
Browser ==(UDP/RTP)==> [Docker: FastRTC -> VADManager -> Sarvam Saarika]

Requires:
- UDP ports exposed in Docker (e.g., 50000-50050)
- AIORTC_UDP_PORT_MIN/MAX environment variables
"""

import asyncio
import logging
import os
import time
import numpy as np
import gradio as gr
from fastrtc import Stream, AsyncStreamHandler

from config import VADConfig
from vad_manager import VADManager
from sarvam_client import SarvamSTTClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global VAD Manager
vad_manager = None
sarvam_client = None


class NativeAudioHandler(AsyncStreamHandler):
    """
    Receives WebRTC audio directly and passes to VAD Manager in-memory.
    Zero network overhead between reception and processing.
    """
    def __init__(self):
        super().__init__()
        self.session_id = f"native_{int(time.time())}"
        self.transcript_history = []
        logger.info(f"🎙️ Native WebRTC Handler initialized | Session: {self.session_id}")

    def copy(self):
        """Required by AsyncStreamHandler to clone the handler."""
        return NativeAudioHandler()

    async def start_up(self):
        """Initialize pipeline resources when stream starts."""
        global vad_manager, sarvam_client
        logger.info("=" * 60)
        logger.info(f"🚀 Native WebRTC Stream Started | Session: {self.session_id}")
        logger.info("   Protocol: UDP (RTP) - Ultra Low Latency")
        logger.info("=" * 60)
        
        # Initialize Sarvam client and VAD Manager if needed
        if vad_manager is None:
            config = VADConfig.from_env()
            logger.info("⚙️ Initializing Sarvam STT Client...")
            sarvam_client = SarvamSTTClient(
                api_key=config.sarvam_api_key,
                model=config.model_name,
                sample_rate=config.sample_rate,
                channels=config.channels,
                endpoint=config.sarvam_endpoint,
            )
            logger.info("⚙️ Initializing VAD Manager...")
            vad_manager = VADManager(config, None, sarvam_client=sarvam_client)
            logger.info("✅ VAD Manager ready")

    async def receive(self, audio: tuple) -> None:
        """
        Direct audio path: WebRTC -> Memory -> Sarvam Saarika
        """
        try:
            sample_rate, audio_array = audio
            
            # 1. Normalize Audio (FastRTC -> VAD format)
            # Ensure float32 and shape (N,)
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array, dtype=np.float32)
            
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()
                
            # Normalize to [-1.0, 1.0]
            max_val = np.max(np.abs(audio_array))
            if max_val > 1.0:
                audio_array = audio_array / 32767.0
                
            # Convert to 16-bit PCM bytes (16kHz required by Sarvam)
            # Resample if needed (simple decimation for now if 48k -> 16k)
            if sample_rate == 48000:
                audio_array = audio_array[::3] # Quick downsample 48->16k
            elif sample_rate == 24000:
                audio_array = audio_array[::2] # Quick downsample 24->16k
                
            pcm_bytes = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            
            # 2. Process In-Memory (Zero Hop)
            if vad_manager:
                transcript = await vad_manager.process_audio_chunk_streaming(
                    session_id=self.session_id,
                    audio_chunk=pcm_bytes,
                    streaming_callback=self.handle_transcript # Callback for results
                )
                
        except Exception as e:
            logger.error(f"❌ Audio process error: {e}")

    def handle_transcript(self, text: str, is_final: bool):
        """Callback from VAD Manager - just logs for now, UI update happens via emit if needed"""
        status = "✅ FINAL" if is_final else "🔄 PARTIAL"
        logger.info(f"📝 {status}: {text}")

    async def emit(self):
        """Can send audio back to browser (e.g. TTS). Returning None for now."""
        await asyncio.sleep(0.02)
        return None

    async def shutdown(self):
        logger.info(f"🛑 Stream stopped | Session: {self.session_id}")


def create_ui():
    """Create the Gradio UI"""
    stream = Stream(
        handler=NativeAudioHandler(),
        modality="audio",
        mode="send-receive",
        ui_args={
            "title": "Leibniz Native WebRTC (UDP) - Sarvam Saarika",
            "description": "Ultra-low latency direct UDP connection. Check Docker logs for transcripts."
        }
    )
    return stream.ui

if __name__ == "__main__":
    # CRITICAL: Configure AIORTC to use specific ports if behind NAT
    # These must match docker-compose ports
    # os.environ["AIORTC_UDP_PORT_MIN"] = "50000"
    # os.environ["AIORTC_UDP_PORT_MAX"] = "50050"
    
    logger.info("🚀 Starting Native WebRTC Service (UDP) with Sarvam Saarika")
    
    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True, # Required for HTTPS
        ssl_verify=False
    )
