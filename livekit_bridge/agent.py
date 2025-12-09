import asyncio
import base64
import json
import logging
import os
import time
from typing import Optional

import aiohttp
import numpy as np
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("livekit-bridge")

# Environment variables
STT_URL = os.getenv("STT_URL", "ws://localhost:8001/api/v1/transcribe/stream")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "ws://localhost:8004/orchestrate")

class BridgeAgent:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.stt_ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.orchestrator_ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.audio_source: Optional[rtc.AudioSource] = None
        self.audio_resampler: Optional[rtc.AudioResampler] = None
        self.session_id = f"livekit_{ctx.job.id}"
        self._shutdown = False
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.tts_sample_rate: int = 22050  # Default, will be updated from TTS messages
        self.audio_source_initialized: bool = False
        
        # Register event handlers BEFORE connecting to prevent race conditions
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Register event handlers before connecting to prevent race conditions"""
        # These will be called when room is connected
        pass  # Handlers registered in start() after ctx.connect()
    
    async def start(self):
        logger.info(f"🚀 Starting Bridge Agent for job {self.ctx.job.id}")
        
        # Connect to LiveKit room first (required before accessing local_participant)
        await self.ctx.connect()
        logger.info("✅ Connected to LiveKit room")
        
        # Initialize audio resampler for STT input (48kHz -> 16kHz)
        self.audio_resampler = rtc.AudioResampler(
            input_rate=48000,
            output_rate=16000,
            num_channels=1,
            quality=rtc.AudioResamplerQuality.MEDIUM
        )
        logger.info("✅ Audio resampler initialized (48kHz -> 16kHz)")
        
        # Connect to STT and Orchestrator with retry logic
        await self.connect_services_with_retry()
        
        # Setup audio output (TTS -> LiveKit) - will be reconfigured when we know actual sample rate
        # Initialize with default, will be updated when TTS sends first audio
        await self._initialize_audio_source(self.tts_sample_rate)
        
        # Listen to user audio (User -> STT)
        # Subscribe to all remote audio tracks manually
        for participant in self.ctx.room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.kind == rtc.TrackKind.KIND_AUDIO:
                    # Subscribe to the track if not already subscribed
                    if not publication.subscribed:
                        await publication.set_subscribed(True)
                    track = await publication.track
                    if track:
                        logger.info(f"🎤 Found user audio: {participant.identity}")
                        asyncio.create_task(self.handle_audio_stream(track))
        
        # Register event handlers for new tracks (already connected, so these will catch future tracks)
        @self.ctx.room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"🎤 Subscribed to user audio: {participant.identity}")
                asyncio.create_task(self.handle_audio_stream(track))
        
        @self.ctx.room.on("track_published")
        def on_track_published(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if publication.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"🎤 New audio track published by: {participant.identity}")
                asyncio.create_task(self._handle_track_published(publication, participant))

    async def _handle_track_published(self, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Handle track published event asynchronously"""
        try:
            await publication.set_subscribed(True)
            track = await publication.track
            if track:
                asyncio.create_task(self.handle_audio_stream(track))
        except Exception as e:
            logger.error(f"❌ Error handling published track: {e}")

        # Start background tasks
        asyncio.create_task(self.listen_to_stt())
        asyncio.create_task(self.listen_to_orchestrator())
        
        # Wait for shutdown
        try:
            # Keep the agent alive
            while not self._shutdown:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.cleanup()

    async def _initialize_audio_source(self, sample_rate: int):
        """Initialize or reconfigure AudioSource with proper queue size"""
        if self.audio_source_initialized and sample_rate == self.tts_sample_rate:
            return  # Already initialized with correct sample rate
        
        self.tts_sample_rate = sample_rate
        # Create AudioSource with queue buffer to prevent "InvalidState" errors
        self.audio_source = rtc.AudioSource(sample_rate, 1, queue_size_ms=2000)
        track = rtc.LocalAudioTrack.create_audio_track("agent-voice", self.audio_source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        
        # Only publish if not already published
        if not self.audio_source_initialized:
            try:
                publication = await self.ctx.room.local_participant.publish_track(track, options)
                self.audio_source_initialized = True
                logger.info(f"🎤 AudioSource initialized and published: {sample_rate}Hz, queue_size=2000ms")
            except Exception as e:
                logger.error(f"❌ Failed to publish audio track: {e}")
                raise
    
    async def connect_services_with_retry(self, max_retries: int = 3):
        """Connect to services with exponential backoff retry logic"""
        self.http_session = aiohttp.ClientSession()
        
        # Connect to STT with retry
        stt_url = f"{STT_URL}?session_id={self.session_id}"
        for attempt in range(max_retries):
            try:
                self.stt_ws = await self.http_session.ws_connect(stt_url)
                logger.info(f"✅ Connected to STT Service: {stt_url}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"⚠️ STT connection failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed to connect to STT after {max_retries} attempts: {e}")
                    raise

        # Connect to Orchestrator with retry
        orch_url = f"{ORCHESTRATOR_URL}?session_id={self.session_id}"
        for attempt in range(max_retries):
            try:
                self.orchestrator_ws = await self.http_session.ws_connect(orch_url)
                logger.info(f"✅ Connected to Orchestrator: {orch_url}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"⚠️ Orchestrator connection failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed to connect to Orchestrator after {max_retries} attempts: {e}")
                    raise

    async def handle_audio_stream(self, track: rtc.RemoteAudioTrack):
        """Stream audio from LiveKit to STT Service with proper resampling"""
        logger.info("🌊 Starting audio stream to STT...")
        
        stream = rtc.AudioStream(track)
        frame_count = 0
        bytes_sent = 0
        start_time = time.time()
        
        try:
            async for frame_event in stream:
                if self._shutdown:
                    break
                    
                frame = frame_event.frame
                
                # Log audio format details (first frame only)
                if frame_count == 0:
                    logger.info(f"📊 Audio format: {frame.sample_rate}Hz, {frame.num_channels}ch, "
                              f"{len(frame.data)} bytes/frame")
                
                # Validate format: PCM 16-bit little-endian, mono
                if frame.num_channels != 1:
                    logger.warning(f"⚠️ Expected mono audio, got {frame.num_channels} channels")
                
                # Use AudioResampler for high-quality resampling (48kHz -> 16kHz)
                if frame.sample_rate == 48000:
                    # Resample using LiveKit's AudioResampler
                    resampled_frames = self.audio_resampler.push(frame)
                    for resampled_frame in resampled_frames:
                        data = resampled_frame.data.tobytes()
                        if self.stt_ws and not self.stt_ws.closed:
                            try:
                                await self.stt_ws.send_bytes(data)
                                bytes_sent += len(data)
                                frame_count += 1
                            except Exception as e:
                                logger.error(f"Error sending audio to STT: {e}")
                                # Try to reconnect
                                await self._reconnect_stt()
                elif frame.sample_rate == 16000:
                    # Already at target rate, send directly
                    data = frame.data.tobytes()
                    if self.stt_ws and not self.stt_ws.closed:
                        try:
                            await self.stt_ws.send_bytes(data)
                            bytes_sent += len(data)
                            frame_count += 1
                        except Exception as e:
                            logger.error(f"Error sending audio to STT: {e}")
                            await self._reconnect_stt()
                else:
                    logger.warning(f"⚠️ Unexpected sample rate: {frame.sample_rate}Hz, using as-is")
                    data = frame.data.tobytes()
                    if self.stt_ws and not self.stt_ws.closed:
                        try:
                            await self.stt_ws.send_bytes(data)
                            bytes_sent += len(data)
                            frame_count += 1
                        except Exception as e:
                            logger.error(f"Error sending audio to STT: {e}")
                            await self._reconnect_stt()
            
            # Flush resampler on stream end
            if self.audio_resampler:
                remaining_frames = self.audio_resampler.flush()
                for frame in remaining_frames:
                    data = frame.data.tobytes()
                    if self.stt_ws and not self.stt_ws.closed:
                        try:
                            await self.stt_ws.send_bytes(data)
                            bytes_sent += len(data)
                        except Exception as e:
                            logger.error(f"Error sending flushed audio to STT: {e}")
        
        except Exception as e:
            logger.error(f"❌ Audio stream error: {e}", exc_info=True)
        finally:
            elapsed = time.time() - start_time
            if elapsed > 0:
                logger.info(f"📊 Audio stream stats: {frame_count} frames, {bytes_sent} bytes, "
                           f"{bytes_sent/elapsed:.0f} bytes/sec over {elapsed:.1f}s")
    
    async def _reconnect_stt(self):
        """Reconnect to STT service"""
        if self.stt_ws:
            try:
                await self.stt_ws.close()
            except:
                pass
        
        stt_url = f"{STT_URL}?session_id={self.session_id}"
        for attempt in range(3):
            try:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                self.stt_ws = await self.http_session.ws_connect(stt_url)
                logger.info(f"✅ Reconnected to STT Service")
                break
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"⚠️ STT reconnection attempt {attempt + 1}/3 failed: {e}")
                else:
                    logger.error(f"❌ Failed to reconnect to STT: {e}")

    async def listen_to_stt(self):
        """Receive transcripts from STT and forward to Orchestrator with reconnection"""
        while not self._shutdown:
            try:
                async for msg in self.stt_ws:
                    if self._shutdown:
                        break
                    
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        # Forward to Orchestrator
                        # Expected format by Orchestrator: {"type": "stt_fragment", "text": "...", "is_final": bool}
                        # STT service sends: {"type": "fragment", "text": "...", "is_final": bool}
                        
                        if data.get("type") == "fragment":
                            text = data.get("text", "")
                            is_final = data.get("is_final", False)
                            
                            # Send fragment to orchestrator
                            payload = {
                                "type": "stt_fragment",
                                "text": text,
                                "is_final": is_final,
                                "session_id": self.session_id
                            }
                            if self.orchestrator_ws and not self.orchestrator_ws.closed:
                                try:
                                    await self.orchestrator_ws.send_json(payload)
                                    logger.info(f"📝 Forwarded STT fragment (final={is_final}): {text[:50]}...")
                                except Exception as e:
                                    logger.error(f"Error sending to orchestrator: {e}")
                                    await self._reconnect_orchestrator()
                            
                            # If final, also send vad_end to trigger response generation
                            if is_final and text.strip():
                                vad_end_payload = {
                                    "type": "vad_end",
                                    "session_id": self.session_id
                                }
                                if self.orchestrator_ws and not self.orchestrator_ws.closed:
                                    try:
                                        await self.orchestrator_ws.send_json(vad_end_payload)
                                        logger.info(f"🤐 Sent VAD_END to trigger response")
                                    except Exception as e:
                                        logger.error(f"Error sending VAD_END: {e}")
                                        await self._reconnect_orchestrator()
                    
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"STT WebSocket error: {self.stt_ws.exception()}")
                        await self._reconnect_stt()
                        break
                    
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("STT WebSocket closed, reconnecting...")
                        await self._reconnect_stt()
                        break
                            
            except Exception as e:
                logger.error(f"STT Listener Error: {e}", exc_info=True)
                if not self._shutdown:
                    await asyncio.sleep(2)
                    await self._reconnect_stt()
    
    async def _reconnect_orchestrator(self):
        """Reconnect to Orchestrator service"""
        if self.orchestrator_ws:
            try:
                await self.orchestrator_ws.close()
            except:
                pass
        
        orch_url = f"{ORCHESTRATOR_URL}?session_id={self.session_id}"
        for attempt in range(3):
            try:
                await asyncio.sleep(2 ** attempt)
                self.orchestrator_ws = await self.http_session.ws_connect(orch_url)
                logger.info(f"✅ Reconnected to Orchestrator")
                break
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"⚠️ Orchestrator reconnection attempt {attempt + 1}/3 failed: {e}")
                else:
                    logger.error(f"❌ Failed to reconnect to Orchestrator: {e}")

    async def listen_to_orchestrator(self):
        """Receive TTS audio from Orchestrator and publish to LiveKit"""
        while not self._shutdown:
            try:
                async for msg in self.orchestrator_ws:
                    if self._shutdown:
                        break
                    
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        # Raw binary audio bytes (backward compatibility)
                        audio_data = msg.data
                        await self._process_audio_bytes(audio_data, self.tts_sample_rate)
                        
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        # JSON format with base64-encoded audio
                        try:
                            data = json.loads(msg.data)
                            msg_type = data.get("type")
                            
                            if msg_type == "audio":
                                # Extract base64-encoded audio
                                audio_b64 = data.get("data", "")
                                if audio_b64:
                                    audio_data = base64.b64decode(audio_b64)
                                    # Get sample rate from message (default to current or 22050)
                                    sample_rate = data.get("sample_rate", self.tts_sample_rate or 22050)
                                    
                                    # Warn if sample rate changed (can't reconfigure AudioSource after publishing)
                                    if sample_rate != self.tts_sample_rate and self.audio_source_initialized:
                                        logger.warning(f"⚠️ TTS sample rate mismatch: AudioSource={self.tts_sample_rate}Hz, "
                                                     f"Message={sample_rate}Hz. Frame chunking will use message rate.")
                                    
                                    await self._process_audio_bytes(audio_data, sample_rate)
                            
                            elif msg_type in ("connected", "ping", "pong"):
                                # Control messages, ignore
                                pass
                            
                            elif msg_type == "interrupted":
                                logger.info("⚡ Received interrupt signal from orchestrator")
                                # Could pause audio playback here if needed
                            
                            else:
                                logger.debug(f"Received orchestrator message: {msg_type}")
                        
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse orchestrator JSON: {e}")
                        except Exception as e:
                            logger.error(f"Error processing orchestrator message: {e}")
                    
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"Orchestrator WebSocket error: {self.orchestrator_ws.exception()}")
                        await self._reconnect_orchestrator()
                        break
                    
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("Orchestrator WebSocket closed, reconnecting...")
                        await self._reconnect_orchestrator()
                        break
                            
            except Exception as e:
                logger.error(f"Orchestrator Listener Error: {e}", exc_info=True)
                if not self._shutdown:
                    await asyncio.sleep(2)
                    await self._reconnect_orchestrator()
    
    async def _process_audio_bytes(self, audio_data: bytes, sample_rate: int):
        """Process audio bytes and send to LiveKit AudioSource"""
        if not self.audio_source:
            logger.warning("AudioSource not initialized, skipping audio")
            return
        
        try:
            # Convert bytes to Int16 array
            arr = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate frame chunk size dynamically (20ms frames)
            samples_per_frame = int(sample_rate * 0.02)
            
            # Chunk audio into frames
            for i in range(0, len(arr), samples_per_frame):
                chunk = arr[i:i+samples_per_frame]
                
                # Pad last chunk if needed
                if len(chunk) < samples_per_frame:
                    chunk = np.pad(chunk, (0, samples_per_frame - len(chunk)), mode='constant')
                
                # Create audio frame
                frame = rtc.AudioFrame.create(sample_rate, 1, samples_per_frame)
                # Copy data to frame
                np.copyto(np.frombuffer(frame.data, dtype=np.int16), chunk)
                
                # Capture frame (may raise InvalidState if queue is full)
                try:
                    await self.audio_source.capture_frame(frame)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to capture frame (queue may be full): {e}")
                    # Wait a bit and retry
                    await asyncio.sleep(0.01)
                    try:
                        await self.audio_source.capture_frame(frame)
                    except Exception as retry_err:
                        logger.error(f"❌ Failed to capture frame after retry: {retry_err}")
        
        except Exception as e:
            logger.error(f"Error processing audio bytes: {e}", exc_info=True)

    async def cleanup(self):
        self._shutdown = True
        if self.stt_ws:
            await self.stt_ws.close()
        if self.orchestrator_ws:
            await self.orchestrator_ws.close()
        if self.http_session:
            await self.http_session.close()

async def entrypoint(ctx: JobContext):
    agent = BridgeAgent(ctx)
    await agent.start()

if __name__ == "__main__":
    # Run the worker
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
