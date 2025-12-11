"""
Unified FastRTC Handler for Orchestrator
=========================================

Single FastRTC endpoint that handles both:
- Input: Microphone audio -> STT Service (WebSocket)
- Output: TTS audio -> Browser Speaker

Features:
- State-Aware Gating: Gates input based on Orchestrator FSM state
- Echo Gating: Suppresses input while agent is speaking (backup)
- Robust WebSocket management with auto-reconnect
- Tight integration with StateManager
- No session mismatches - single source of truth
- No AttributeErrors - defensive programming throughout

Reference:
    gemini_live_experiment/fastrtc_handler.py - STT handler pattern
    tts_sarvam/fastrtc_handler.py - TTS handler pattern
"""

import asyncio
import json
import logging
import os
import time
from enum import Enum
from typing import Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass, field

import numpy as np

try:
    from fastrtc import AsyncStreamHandler
    FASTRTC_AVAILABLE = True
except ImportError:
    FASTRTC_AVAILABLE = False
    AsyncStreamHandler = object  # Fallback for type hints

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketClientProtocol = None

# Event-driven architecture imports
try:
    from leibniz_agent.services.shared.events import VoiceEvent, EventTypes
    from leibniz_agent.services.shared.event_broker import EventBroker
    EVENTS_AVAILABLE = True
except ImportError:
    try:
        from shared.events import VoiceEvent, EventTypes
        from shared.event_broker import EventBroker
        EVENTS_AVAILABLE = True
    except ImportError:
        EVENTS_AVAILABLE = False
        VoiceEvent = None
        EventTypes = None
        EventBroker = None

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection state machine"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class OrchestratorState(Enum):
    """
    Mirror of orchestrator FSM states for gating decisions.
    Kept separate to avoid circular imports.
    """
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPT = "interrupt"


@dataclass
class UnifiedSessionState:
    """State for a unified FastRTC session"""
    session_id: str
    created_at: float = field(default_factory=time.time)
    
    # Audio state (local tracking)
    is_speaking: bool = False  # True when TTS audio is being played
    is_listening: bool = False  # True when actively capturing STT
    
    # Orchestrator FSM state (driven externally)
    orchestrator_state: OrchestratorState = OrchestratorState.IDLE
    
    # Connection states
    stt_connection_state: ConnectionState = ConnectionState.DISCONNECTED
    
    # Timing
    last_audio_input_time: float = 0.0
    last_audio_output_time: float = 0.0
    last_state_change_time: float = 0.0
    
    # Metrics
    chunks_received: int = 0
    chunks_sent: int = 0
    reconnect_count: int = 0
    
    @property
    def stt_connected(self) -> bool:
        """Backward-compatible property for STT connection status."""
        return self.stt_connection_state == ConnectionState.CONNECTED
    
    @stt_connected.setter
    def stt_connected(self, value: bool):
        """Backward-compatible setter."""
        if value:
            self.stt_connection_state = ConnectionState.CONNECTED
        else:
            self.stt_connection_state = ConnectionState.DISCONNECTED
    
    def should_accept_input(self) -> bool:
        """
        Determine if microphone input should be processed based on orchestrator state.
        
        Returns True only in states where user speech should be captured:
        - IDLE: Waiting for user to start (may capture for VAD trigger)
        - LISTENING: Actively capturing user speech
        - INTERRUPT: User is interrupting agent (barge-in)
        
        Returns False for:
        - THINKING: Agent is processing, ignore spurious audio
        - SPEAKING: Agent is talking, gate to prevent echo
        """
        return self.orchestrator_state in (
            OrchestratorState.IDLE,
            OrchestratorState.LISTENING,
            OrchestratorState.INTERRUPT,
        )


class UnifiedFastRTCHandler(AsyncStreamHandler if FASTRTC_AVAILABLE else object):
    """
    Unified FastRTC handler for bidirectional audio streaming.
    
    Handles:
    - Input: Browser mic -> Orchestrator -> STT Service (WebSocket)
    - Output: TTS Service -> Orchestrator -> Browser speaker
    
    State-Aware Gating:
    - LISTENING/IDLE/INTERRUPT: Accept microphone input
    - THINKING/SPEAKING: Gate microphone input (prevent self-interruption)
    
    Echo Gating (backup):
    - When is_speaking=True, input audio is suppressed even if state allows
    """
    
    # Class-level registry for active instances
    active_instances: Dict[str, 'UnifiedFastRTCHandler'] = {}
    
    # Callbacks for orchestrator integration
    on_stt_transcript: Optional[Callable[[str, str, bool], None]] = None  # (session_id, text, is_final)
    on_vad_event: Optional[Callable[[str, str], None]] = None  # (session_id, event_type)
    on_connection_change: Optional[Callable[[str, bool], None]] = None  # (session_id, connected)
    
    # Class-level lock for thread-safe operations on active_instances
    _instances_lock: asyncio.Lock = None  # Initialized lazily
    
    # Event broker for event-driven architecture (shared across instances)
    event_broker: Optional["EventBroker"] = None
    
    def __init__(
        self,
        stt_ws_url: str = "ws://localhost:8001/api/v1/transcribe/stream",
        sample_rate_in: int = 16000,
        sample_rate_out: int = 24000,
        echo_gate_enabled: bool = True,
        echo_gate_tail_ms: int = 200,  # Extra time to gate after audio stops
    ):
        """
        Initialize Unified FastRTC Handler.
        
        Args:
            stt_ws_url: WebSocket URL for STT service
            sample_rate_in: Input sample rate (microphone)
            sample_rate_out: Output sample rate (TTS)
            echo_gate_enabled: Enable echo gating
            echo_gate_tail_ms: Additional gating time after TTS stops
        """
        if FASTRTC_AVAILABLE:
            super().__init__()
        
        # Configuration
        self.stt_ws_url = stt_ws_url
        self.sample_rate_in = sample_rate_in
        self.sample_rate_out = sample_rate_out
        self.echo_gate_enabled = echo_gate_enabled
        self.echo_gate_tail_ms = echo_gate_tail_ms
        
        # Session state
        self.session_id = f"unified_{int(time.time() * 1000)}"
        self.state = UnifiedSessionState(session_id=self.session_id)
        
        # Audio queues
        self._audio_out_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._audio_buffer = bytearray()  # Buffer for input audio
        self._buffer_limit = 3200  # ~100ms at 16kHz mono 16-bit
        
        # STT WebSocket connection
        self._stt_ws: Optional[WebSocketClientProtocol] = None
        self._stt_receive_task: Optional[asyncio.Task] = None
        self._stt_reconnect_task: Optional[asyncio.Task] = None
        self._stt_keepalive_task: Optional[asyncio.Task] = None
        self._keepalive_interval = 20.0  # Send keepalive every 20 seconds (before 30s timeout)
        self._stt_lock = asyncio.Lock()
        
        # Output state
        self._current_audio_chunk: Optional[Tuple[np.ndarray, int]] = None
        self._chunk_position: int = 0
        self._chunk_size_samples: int = 960  # ~40ms at 24kHz
        self._last_emit_time: float = 0.0
        
        # Audio playback duration tracking (for FastRTC fallback when browser doesn't send events)
        self.total_audio_duration_ms: float = 0.0
        self.playback_start_time: Optional[float] = None
        
        # Lifecycle
        self._started = False
        self._shutting_down = False
        
        logger.info(f"UnifiedFastRTCHandler initialized | session={self.session_id}")
        logger.info(f"  STT URL: {self.stt_ws_url}")
        logger.info(f"  Echo Gate: {self.echo_gate_enabled} (tail={self.echo_gate_tail_ms}ms)")
        
        # Verify STT URL points to stt_sarvam
        if "stt-vad" not in self.stt_ws_url and "stt_sarvam" not in self.stt_ws_url and "tara-stt-vad" not in self.stt_ws_url:
            logger.warning(f"⚠️ STT URL may not be stt_sarvam: {self.stt_ws_url}")
        else:
            logger.info(f"✅ STT URL verified: using stt_sarvam service")
    
    # =========================================================================
    # FastRTC Lifecycle Methods
    # =========================================================================
    
    async def start_up(self):
        """Called when WebRTC stream starts."""
        self._started = True
        self._shutting_down = False
        
        # Register this instance
        UnifiedFastRTCHandler.active_instances[self.session_id] = self
        
        logger.info("=" * 70)
        logger.info(f"UNIFIED FASTRTC STARTED | Session: {self.session_id}")
        logger.info("=" * 70)
        
        # Connect to STT service
        await self._connect_stt()
        
        # Notify orchestrator
        if UnifiedFastRTCHandler.on_connection_change:
            try:
                await self._safe_callback(
                    UnifiedFastRTCHandler.on_connection_change,
                    self.session_id, True
                )
            except Exception as e:
                logger.error(f"Connection callback error: {e}")
        
        logger.info(f"Unified FastRTC ready | Input: {self.sample_rate_in}Hz | Output: {self.sample_rate_out}Hz")
    
    async def shutdown(self):
        """Clean up when stream closes."""
        self._shutting_down = True
        self._started = False
        
        logger.info("=" * 70)
        logger.info(f"UNIFIED FASTRTC SHUTDOWN | Session: {self.session_id}")
        logger.info("=" * 70)
        
        # Disconnect STT
        await self._disconnect_stt()
        
        # Unregister
        if self.session_id in UnifiedFastRTCHandler.active_instances:
            del UnifiedFastRTCHandler.active_instances[self.session_id]
        
        # Clear queues
        while not self._audio_out_queue.empty():
            try:
                self._audio_out_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Notify orchestrator
        if UnifiedFastRTCHandler.on_connection_change:
            try:
                await self._safe_callback(
                    UnifiedFastRTCHandler.on_connection_change,
                    self.session_id, False
                )
            except Exception as e:
                logger.error(f"Disconnection callback error: {e}")
        
        logger.info(f"Unified FastRTC shutdown complete | Session: {self.session_id}")
    
    def copy(self) -> 'UnifiedFastRTCHandler':
        """Create a copy for new connections."""
        return UnifiedFastRTCHandler(
            stt_ws_url=self.stt_ws_url,
            sample_rate_in=self.sample_rate_in,
            sample_rate_out=self.sample_rate_out,
            echo_gate_enabled=self.echo_gate_enabled,
            echo_gate_tail_ms=self.echo_gate_tail_ms,
        )
    
    # =========================================================================
    # Audio Input (Microphone -> STT)
    # =========================================================================
    
    async def receive(self, audio: tuple) -> None:
        """
        Receive audio from browser microphone.
        
        Implements state-aware and echo gating: suppresses audio when agent
        is in THINKING/SPEAKING states or when TTS audio is playing.
        
        Args:
            audio: Tuple of (sample_rate: int, audio_array: np.ndarray)
        """
        # Guard: Check lifecycle state
        if not self._started or self._shutting_down:
            return
        
        # Guard: Audio buffer must exist
        if self._audio_buffer is None:
            self._audio_buffer = bytearray()
        
        # State-Aware + Echo Gating Check
        if self._should_gate_input():
            # Audio is gated - suppress input
            return
        
        try:
            # Parse audio tuple
            if audio is None:
                return
            
            if not isinstance(audio, (tuple, list)) or len(audio) != 2:
                logger.warning(f"[{self.session_id}] Unexpected audio format: {type(audio)}")
                return
            
            sample_rate, audio_array = audio
            
            # Validate sample_rate
            if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
                logger.warning(f"[{self.session_id}] Invalid sample rate: {sample_rate}")
                return
            
            sample_rate = int(sample_rate)
            
            # Ensure numpy array
            if audio_array is None:
                return
            
            if not isinstance(audio_array, np.ndarray):
                try:
                    audio_array = np.array(audio_array, dtype=np.float32)
                except Exception as e:
                    logger.warning(f"[{self.session_id}] Failed to convert audio to array: {e}")
                    return
            
            # Handle empty arrays
            if audio_array.size == 0:
                return
            
            # Handle multi-dimensional arrays
            if audio_array.ndim == 2:
                audio_array = audio_array.squeeze()
            elif audio_array.ndim > 2:
                audio_array = audio_array.flatten()
            
            # Resample to 16kHz if needed (STT expects 16kHz)
            if sample_rate != self.sample_rate_in:
                if sample_rate > self.sample_rate_in and sample_rate % self.sample_rate_in == 0:
                    step = int(sample_rate / self.sample_rate_in)
                    audio_array = audio_array[::step]
                else:
                    # Naive resampling for non-integer ratios
                    new_length = int(len(audio_array) * self.sample_rate_in / sample_rate)
                    if new_length > 0:
                        indices = np.linspace(0, len(audio_array) - 1, new_length).astype(int)
                        audio_array = audio_array[indices]
                    else:
                        return  # Can't resample to zero length
            
            # Convert to int16 PCM
            if audio_array.dtype == np.int16:
                audio_int16 = audio_array
            elif audio_array.dtype in (np.float32, np.float64):
                max_val = np.max(np.abs(audio_array)) if audio_array.size > 0 else 0
                if max_val > 1.0:
                    audio_int16 = np.clip(audio_array, -32768, 32767).astype(np.int16)
                else:
                    audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
            else:
                audio_array = audio_array.astype(np.float32)
                max_val = np.max(np.abs(audio_array)) if audio_array.size > 0 else 0
                if max_val > 1.0:
                    audio_int16 = np.clip(audio_array, -32768, 32767).astype(np.int16)
                else:
                    audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
            
            audio_bytes = audio_int16.tobytes()
            
            # Buffer chunks for efficiency
            self._audio_buffer.extend(audio_bytes)
            
            # Send when buffer is full
            if len(self._audio_buffer) >= self._buffer_limit:
                chunk_to_send = bytes(self._audio_buffer)
                self._audio_buffer.clear()
                
                await self._send_to_stt(chunk_to_send)
                self.state.chunks_sent += 1
                self.state.last_audio_input_time = time.time()
                
        except Exception as e:
            logger.error(f"[{self.session_id}] Audio receive error: {type(e).__name__}: {e}")
    
    def _should_gate_input(self) -> bool:
        """
        Check if input should be gated.
        
        Uses two-layer gating:
        1. State-based gating: Block input in THINKING/SPEAKING states
        2. Echo gating: Block input while TTS audio is playing (with tail)
        
        Returns:
            True if input should be blocked, False if input should be processed
        """
        # Initialize last gate state tracking if needed
        if not hasattr(self, '_last_gate_state'):
            self._last_gate_state = None
        
        # Layer 1: State-based gating (primary)
        # Only accept input in IDLE, LISTENING, or INTERRUPT states
        if not self.state.should_accept_input():
            should_gate = True
            gate_reason = f"state: {self.state.orchestrator_state.value}"
        else:
            # Layer 2: Echo gating (backup, in case state hasn't updated yet)
            if self.echo_gate_enabled:
                # Gate if actively speaking (TTS audio in queue/playing)
                if self.state.is_speaking:
                    should_gate = True
                    gate_reason = "TTS speaking"
                # Gate for tail duration after speaking stops
                elif self.state.last_audio_output_time > 0:
                    elapsed_ms = (time.time() - self.state.last_audio_output_time) * 1000
                    if elapsed_ms < self.echo_gate_tail_ms:
                        should_gate = True
                        gate_reason = f"echo tail ({elapsed_ms:.0f}ms < {self.echo_gate_tail_ms}ms)"
                    else:
                        should_gate = False
                        gate_reason = None
                else:
                    should_gate = False
                    gate_reason = None
            else:
                should_gate = False
                gate_reason = None
        
        # Log state changes for visibility
        if self._last_gate_state != should_gate:
            if should_gate:
                logger.info(f"[{self.session_id}] Mic gating: GATED ({gate_reason}) (state: {self.state.orchestrator_state.value})")
            else:
                logger.info(f"[{self.session_id}] Mic gating: OPEN (state: {self.state.orchestrator_state.value})")
            self._last_gate_state = should_gate
        
        return should_gate
    
    # =========================================================================
    # Audio Output (TTS -> Speaker)
    # =========================================================================
    
    async def emit(self) -> Tuple[int, np.ndarray]:
        """
        Emit audio to browser speaker.
        
        Returns:
            Tuple of (sample_rate, audio_array)
        """
        silence = np.zeros(self._chunk_size_samples, dtype=np.int16)
        sleep_interval = self._chunk_size_samples / self.sample_rate_out
        
        # Ensure queue exists
        if self._audio_out_queue is None:
            self._audio_out_queue = asyncio.Queue(maxsize=500)
        
        # Try to get audio from current chunk or queue
        try:
            # If we have a current chunk, continue from it
            if self._current_audio_chunk is not None:
                try:
                    audio_data, sample_rate = self._current_audio_chunk
                    
                    remaining = len(audio_data) - self._chunk_position
                    if remaining > 0:
                        take_samples = min(self._chunk_size_samples, remaining)
                        chunk = audio_data[self._chunk_position:self._chunk_position + take_samples]
                        self._chunk_position += take_samples
                        
                        # Update speaking state
                        self.state.is_speaking = True
                        self.state.last_audio_output_time = time.time()
                        self._last_emit_time = time.time()
                        
                        return (sample_rate, chunk)
                    else:
                        # Current chunk exhausted, get next
                        self._current_audio_chunk = None
                        self._chunk_position = 0
                except (TypeError, ValueError) as e:
                    logger.warning(f"[{self.session_id}] Invalid current audio chunk: {e}")
                    self._current_audio_chunk = None
                    self._chunk_position = 0
            
            # Try to get next chunk from queue (non-blocking)
            try:
                queue_item = self._audio_out_queue.get_nowait()
                
                # Validate queue item
                if queue_item is None or not isinstance(queue_item, tuple) or len(queue_item) != 2:
                    logger.warning(f"[{self.session_id}] Invalid queue item format")
                    await asyncio.sleep(sleep_interval)
                    return (self.sample_rate_out, silence)
                
                audio_data, sample_rate = queue_item
                
                # Ensure numpy array
                if not isinstance(audio_data, np.ndarray):
                    try:
                        audio_data = np.frombuffer(audio_data, dtype=np.int16)
                    except Exception as e:
                        logger.warning(f"[{self.session_id}] Failed to convert audio data: {e}")
                        await asyncio.sleep(sleep_interval)
                        return (self.sample_rate_out, silence)
                
                if audio_data.size == 0:
                    await asyncio.sleep(sleep_interval)
                    return (self.sample_rate_out, silence)
                
                if audio_data.ndim > 1:
                    audio_data = audio_data.flatten()
                
                self._current_audio_chunk = (audio_data, sample_rate)
                self._chunk_position = 0
                
                # Take first chunk
                take_samples = min(self._chunk_size_samples, len(audio_data))
                chunk = audio_data[:take_samples]
                self._chunk_position = take_samples
                
                # Track if this is the start of playback
                was_speaking = self.state.is_speaking
                
                # Update speaking state
                self.state.is_speaking = True
                self.state.last_audio_output_time = time.time()
                self._last_emit_time = time.time()
                
                # Emit playback_started event if transitioning to speaking
                if not was_speaking:
                    # Reset duration tracking for new playback sequence
                    self.total_audio_duration_ms = 0.0
                    self.playback_start_time = time.time()
                    asyncio.create_task(self.emit_playback_started(self._audio_out_queue.qsize() + 1))
                
                return (sample_rate, chunk)
                
            except asyncio.QueueEmpty:
                # No audio available - check if we should clear speaking state
                if self.state.is_speaking:
                    # Use calculated audio duration instead of 100ms grace period
                    # This is a fallback for FastRTC when browser doesn't send playback_done events
                    if self.playback_start_time and self.total_audio_duration_ms > 0:
                        expected_end_time = self.playback_start_time + (self.total_audio_duration_ms / 1000)
                        current_time = time.time()
                        
                        # Wait for expected duration + 200ms buffer for processing/network delays
                        if current_time >= expected_end_time + 0.2:
                            self.state.is_speaking = False
                            actual_duration_ms = (current_time - self.playback_start_time) * 1000
                            
                            # Emit playback_done event (fallback for FastRTC)
                            asyncio.create_task(self.emit_playback_done(
                                total_chunks=self.state.chunks_received,
                                duration_ms=actual_duration_ms
                            ))
                            
                            # Reset tracking for next audio sequence
                            self.total_audio_duration_ms = 0.0
                            self.playback_start_time = None
                    elif time.time() - self._last_emit_time > 0.5:
                        # Fallback: if no duration tracking, use 500ms grace period (longer than before)
                        self.state.is_speaking = False
                        logger.debug(f"[{self.session_id}] Cleared speaking state (fallback grace period)")
                
                await asyncio.sleep(sleep_interval)
                return (self.sample_rate_out, silence)
                
        except Exception as e:
            logger.error(f"[{self.session_id}] Audio emit error: {type(e).__name__}: {e}")
            await asyncio.sleep(sleep_interval)
            return (self.sample_rate_out, silence)
    
    # =========================================================================
    # Public API for Orchestrator
    # =========================================================================
    
    async def enqueue_audio(self, audio_bytes: bytes, sample_rate: int = 24000):
        """
        Enqueue audio for playback (called by orchestrator for TTS).
        
        Args:
            audio_bytes: Raw audio bytes (int16 PCM)
            sample_rate: Sample rate of audio
        """
        if self._shutting_down:
            return
        
        # Ensure queue exists
        if self._audio_out_queue is None:
            self._audio_out_queue = asyncio.Queue(maxsize=500)
        
        # Validate input
        if audio_bytes is None or len(audio_bytes) == 0:
            return
        
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).copy()
            
            if audio_array.size == 0:
                return
            
            # Calculate duration for this chunk (16-bit = 2 bytes per sample)
            chunk_samples = len(audio_bytes) // 2
            chunk_duration_ms = (chunk_samples / sample_rate) * 1000
            self.total_audio_duration_ms += chunk_duration_ms
            
            # Track when playback starts (first chunk)
            if self.playback_start_time is None:
                self.playback_start_time = time.time()
            
            # Put in queue (drop oldest if full)
            try:
                self._audio_out_queue.put_nowait((audio_array, sample_rate))
                self.state.chunks_received += 1
            except asyncio.QueueFull:
                # Drop oldest and add new (leaky bucket)
                try:
                    self._audio_out_queue.get_nowait()
                    self._audio_out_queue.put_nowait((audio_array, sample_rate))
                    logger.debug(f"[{self.session_id}] Audio queue full - dropped oldest chunk")
                except asyncio.QueueEmpty:
                    pass
                    
        except Exception as e:
            logger.error(f"[{self.session_id}] Enqueue audio error: {type(e).__name__}: {e}")
    
    @classmethod
    async def broadcast_audio(cls, audio_bytes: bytes, sample_rate: int = 24000):
        """Broadcast audio to all active instances with robust error handling."""
        if not cls.active_instances:
            logger.debug("No active UnifiedFastRTC instances for broadcast")
            return
        
        # Validate input
        if audio_bytes is None or len(audio_bytes) == 0:
            return
        
        # Iterate over a copy to avoid modification during iteration
        instances_copy = list(cls.active_instances.items())
        
        for session_id, instance in instances_copy:
            try:
                if instance is not None and not instance._shutting_down:
                    await instance.enqueue_audio(audio_bytes, sample_rate)
            except AttributeError as e:
                logger.warning(f"Broadcast to {session_id} skipped - instance invalid: {e}")
            except Exception as e:
                logger.error(f"Broadcast to {session_id} failed: {type(e).__name__}: {e}")
    
    def set_speaking(self, is_speaking: bool):
        """Set speaking state (for manual control)."""
        self.state.is_speaking = is_speaking
        if is_speaking:
            self.state.last_audio_output_time = time.time()
    
    def update_orchestrator_state(self, state_value: str):
        """
        Update the orchestrator FSM state for this handler.
        
        Called by the orchestrator when state transitions occur.
        This controls input gating behavior.
        
        Args:
            state_value: State string value (e.g., "listening", "speaking")
        """
        try:
            new_state = OrchestratorState(state_value.lower())
            old_state = self.state.orchestrator_state
            
            if new_state != old_state:
                self.state.orchestrator_state = new_state
                self.state.last_state_change_time = time.time()
                
                logger.info(
                    f"[{self.session_id}] Orchestrator state: "
                    f"{old_state.value.upper()} → {new_state.value.upper()}"
                )
                
                # Auto-set is_listening based on state
                self.state.is_listening = new_state in (
                    OrchestratorState.LISTENING,
                    OrchestratorState.IDLE,
                )
                
                # Manage keepalive task based on state (schedule async cleanup if needed)
                if new_state in (OrchestratorState.LISTENING, OrchestratorState.IDLE):
                    # Start keepalive if not running
                    if self._stt_keepalive_task is None or self._stt_keepalive_task.done():
                        self._stt_keepalive_task = asyncio.create_task(self._stt_keepalive_loop())
                        logger.debug(f"[{self.session_id}] STT keepalive task started")
                else:
                    # Stop keepalive during THINKING/SPEAKING (schedule async cancellation)
                    if self._stt_keepalive_task and not self._stt_keepalive_task.done():
                        self._stt_keepalive_task.cancel()
                        # Schedule cleanup task (non-blocking)
                        asyncio.create_task(self._cleanup_keepalive_task())
                        logger.debug(f"[{self.session_id}] STT keepalive task cancellation scheduled")
                
        except ValueError:
            logger.warning(f"[{self.session_id}] Unknown orchestrator state: {state_value}")
    
    @classmethod
    async def broadcast_state_change(cls, state_value: str):
        """
        Broadcast orchestrator state change to all active handler instances.
        
        Called by the orchestrator when state transitions occur.
        
        Args:
            state_value: State string value (e.g., "listening", "speaking")
        """
        if not cls.active_instances:
            logger.debug("No active UnifiedFastRTC instances for state broadcast")
            return
        
        # Iterate over a copy to avoid modification during iteration
        for session_id, instance in list(cls.active_instances.items()):
            try:
                if instance is not None:
                    instance.update_orchestrator_state(state_value)
            except Exception as e:
                logger.error(f"State broadcast to {session_id} failed: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current handler state."""
        return {
            "session_id": self.session_id,
            "is_speaking": self.state.is_speaking,
            "is_listening": self.state.is_listening,
            "orchestrator_state": self.state.orchestrator_state.value,
            "stt_connected": self.state.stt_connected,
            "stt_connection_state": self.state.stt_connection_state.value,
            "queue_size": self._audio_out_queue.qsize() if self._audio_out_queue else 0,
            "chunks_sent": self.state.chunks_sent,
            "chunks_received": self.state.chunks_received,
            "reconnect_count": self.state.reconnect_count,
            "uptime_s": time.time() - self.state.created_at,
        }
    
    @classmethod
    def get_all_states(cls) -> Dict[str, Dict[str, Any]]:
        """Get state of all active handler instances."""
        states = {}
        for session_id, instance in list(cls.active_instances.items()):
            try:
                if instance is not None:
                    states[session_id] = instance.get_state()
            except Exception as e:
                states[session_id] = {"error": str(e)}
        return states
    
    # =========================================================================
    # Event-Driven WebRTC Events
    # =========================================================================
    
    @classmethod
    def set_event_broker(cls, broker: "EventBroker"):
        """Set the event broker for WebRTC event emission."""
        cls.event_broker = broker
        logger.info("EventBroker set for UnifiedFastRTCHandler")
    
    async def _emit_webrtc_event(self, event_type: str, payload: Dict[str, Any] = None):
        """
        Emit a WebRTC event to Redis Stream for orchestrator coordination.
        
        Args:
            event_type: Type of event (playback_started, playback_done, barge_in)
            payload: Additional event payload
        """
        if not EVENTS_AVAILABLE or not self.event_broker:
            return
        
        try:
            event = VoiceEvent(
                event_type=event_type,
                session_id=self.session_id,
                source="unified_fastrtc",
                payload=payload or {},
                metadata={
                    "orchestrator_state": self.state.orchestrator_state.value,
                    "is_speaking": self.state.is_speaking,
                    "queue_size": self._audio_out_queue.qsize() if self._audio_out_queue else 0
                }
            )
            
            stream_key = f"voice:webrtc:session:{self.session_id}"
            await self.event_broker.publish(stream_key, event)
            logger.debug(f"[{self.session_id}] Emitted {event_type} to {stream_key}")
            
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to emit WebRTC event: {e}")
    
    async def emit_playback_started(self, chunk_count: int = 1):
        """Emit playback_started event when TTS audio begins playing."""
        await self._emit_webrtc_event(
            EventTypes.PLAYBACK_STARTED if EVENTS_AVAILABLE else "voice.webrtc.playback_started",
            payload={
                "chunk_count": chunk_count,
                "timestamp": time.time()
            }
        )
    
    async def emit_playback_done(self, total_chunks: int = 0, duration_ms: float = 0):
        """Emit playback_done event when TTS audio finishes playing."""
        await self._emit_webrtc_event(
            EventTypes.PLAYBACK_DONE if EVENTS_AVAILABLE else "voice.webrtc.playback_done",
            payload={
                "total_chunks": total_chunks,
                "duration_ms": duration_ms,
                "timestamp": time.time()
            }
        )
    
    async def emit_barge_in(self, reason: str = "user_speaking"):
        """
        Emit barge_in event when user interrupts during TTS playback.
        
        This is critical for the orchestrator to transition to INTERRUPT state.
        """
        await self._emit_webrtc_event(
            EventTypes.BARGE_IN if EVENTS_AVAILABLE else "voice.webrtc.barge_in",
            payload={
                "reason": reason,
                "was_speaking": self.state.is_speaking,
                "timestamp": time.time()
            }
        )
        logger.info(f"[{self.session_id}] Barge-in event emitted: {reason}")
    
    @classmethod
    async def broadcast_barge_in(cls, reason: str = "user_speaking"):
        """Broadcast barge-in event to all active handler instances."""
        if not cls.active_instances:
            return
        
        for session_id, instance in list(cls.active_instances.items()):
            try:
                if instance is not None and instance.state.is_speaking:
                    await instance.emit_barge_in(reason)
            except Exception as e:
                logger.error(f"Barge-in broadcast to {session_id} failed: {e}")
    
    # =========================================================================
    # STT WebSocket Management
    # =========================================================================
    
    def _is_websocket_valid(self, ws) -> bool:
        """
        Check if WebSocket is valid and ready for use.
        
        Returns True if WebSocket appears valid, False otherwise.
        This is a best-effort check - the receive loop will handle actual connection issues.
        """
        if ws is None:
            return False
        
        try:
            # Check if it has required methods/attributes
            if not hasattr(ws, 'recv'):
                return False
            
            # Try to check closed status safely - if we can't determine, assume valid
            # The receive loop will catch actual connection problems
            if hasattr(ws, 'closed'):
                try:
                    # If closed property exists and is True, WebSocket is not valid
                    if ws.closed:
                        return False
                except (AttributeError, RuntimeError):
                    # If accessing closed raises, assume valid (let receive loop handle it)
                    pass
            
            return True
        except Exception:
            # On any exception, assume invalid to be safe
            return False
    
    async def _connect_stt(self):
        """Connect to STT service WebSocket with robust error handling."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets library not available - STT disabled")
            return
        
        # Check if lock exists (handle edge case during initialization)
        if self._stt_lock is None:
            self._stt_lock = asyncio.Lock()
        
        async with self._stt_lock:
            # Check if already connected
            if self._is_websocket_valid(self._stt_ws):
                try:
                    if hasattr(self._stt_ws, 'closed') and not self._stt_ws.closed:
                        return  # Already connected
                except (AttributeError, RuntimeError):
                    # WebSocket object may be in invalid state
                    self._stt_ws = None
            
            # Update connection state
            self.state.stt_connection_state = ConnectionState.CONNECTING
            
            try:
                # Build URL with session ID
                url = f"{self.stt_ws_url}?session_id={self.session_id}"
                
                logger.info(f"[{self.session_id}] Connecting to STT service: {url}")
                
                self._stt_ws = await asyncio.wait_for(
                    websockets.connect(
                        url,
                        ping_interval=20,
                        ping_timeout=10,
                        close_timeout=5,
                    ),
                    timeout=15.0  # Connection timeout
                )
                
                # Verify WebSocket is ready using helper method
                if not self._is_websocket_valid(self._stt_ws):
                    logger.warning(f"[{self.session_id}] WebSocket validation failed after connection, but continuing")
                    # Don't raise - let the receive loop handle any connection issues
                
                self.state.stt_connection_state = ConnectionState.CONNECTED
                logger.info(f"[{self.session_id}] STT WebSocket connected and validated")
                
                # Start receive task only if WebSocket is valid
                if self._is_websocket_valid(self._stt_ws) and (self._stt_receive_task is None or self._stt_receive_task.done()):
                    self._stt_receive_task = asyncio.create_task(
                        self._stt_receive_loop(),
                        name=f"stt_receive_{self.session_id}"
                    )
                
            except asyncio.TimeoutError:
                logger.error(f"[{self.session_id}] STT connection timeout")
                self.state.stt_connection_state = ConnectionState.DISCONNECTED
                self._stt_ws = None
                
                # Schedule reconnect
                if not self._shutting_down:
                    self._schedule_stt_reconnect(delay=3.0)
                    
            except ConnectionRefusedError:
                logger.error(f"[{self.session_id}] STT connection refused - service may be down")
                self.state.stt_connection_state = ConnectionState.DISCONNECTED
                self._stt_ws = None
                
                # Schedule reconnect with longer delay
                if not self._shutting_down:
                    self._schedule_stt_reconnect(delay=5.0)
                    
            except Exception as e:
                logger.error(f"[{self.session_id}] STT connection failed: {type(e).__name__}: {e}")
                self.state.stt_connection_state = ConnectionState.DISCONNECTED
                self._stt_ws = None
                
                # Schedule reconnect
                if not self._shutting_down:
                    self._schedule_stt_reconnect()
    
    async def _disconnect_stt(self):
        """Disconnect from STT service with robust cleanup."""
        # Ensure lock exists
        if self._stt_lock is None:
            self._stt_lock = asyncio.Lock()
        
        async with self._stt_lock:
            # Cancel keepalive task
            if self._stt_keepalive_task is not None:
                try:
                    if not self._stt_keepalive_task.done():
                        self._stt_keepalive_task.cancel()
                        try:
                            await asyncio.wait_for(self._stt_keepalive_task, timeout=2.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                except Exception as e:
                    logger.debug(f"[{self.session_id}] Error cancelling keepalive task: {e}")
            self._stt_keepalive_task = None
            
            # Cancel receive task
            if self._stt_receive_task is not None:
                try:
                    if not self._stt_receive_task.done():
                        self._stt_receive_task.cancel()
                        try:
                            await asyncio.wait_for(self._stt_receive_task, timeout=2.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                except Exception as e:
                    logger.debug(f"[{self.session_id}] Error cancelling receive task: {e}")
            self._stt_receive_task = None
            
            # Cancel reconnect task
            if self._stt_reconnect_task is not None:
                try:
                    if not self._stt_reconnect_task.done():
                        self._stt_reconnect_task.cancel()
                        try:
                            await asyncio.wait_for(self._stt_reconnect_task, timeout=2.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                except Exception as e:
                    logger.debug(f"[{self.session_id}] Error cancelling reconnect task: {e}")
            self._stt_reconnect_task = None
            
            # Close WebSocket
            if self._stt_ws is not None:
                try:
                    await asyncio.wait_for(self._stt_ws.close(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.session_id}] STT WebSocket close timeout")
                except Exception as e:
                    logger.debug(f"[{self.session_id}] Error closing STT WebSocket: {e}")
            self._stt_ws = None
            self.state.stt_connection_state = ConnectionState.DISCONNECTED
    
    def _schedule_stt_reconnect(self, delay: float = 2.0):
        """Schedule STT reconnection attempt with exponential backoff."""
        if self._shutting_down:
            return
        
        # Check if already scheduled
        if self._stt_reconnect_task is not None:
            try:
                if not self._stt_reconnect_task.done():
                    return  # Already scheduled
            except Exception:
                pass
        
        # Update connection state
        self.state.stt_connection_state = ConnectionState.RECONNECTING
        
        # Calculate backoff delay (max 30 seconds)
        actual_delay = min(delay * (1.5 ** min(self.state.reconnect_count, 5)), 30.0)
        self.state.reconnect_count += 1
        
        logger.info(
            f"[{self.session_id}] Scheduling STT reconnect in {actual_delay:.1f}s "
            f"(attempt #{self.state.reconnect_count})"
        )
        
        async def reconnect():
            try:
                await asyncio.sleep(actual_delay)
                if not self._shutting_down:
                    await self._connect_stt()
                    # Reset reconnect count on successful connection
                    if self.state.stt_connection_state == ConnectionState.CONNECTED:
                        self.state.reconnect_count = 0
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[{self.session_id}] Reconnect task error: {e}")
        
        self._stt_reconnect_task = asyncio.create_task(
            reconnect(),
            name=f"stt_reconnect_{self.session_id}"
        )
    
    async def _send_to_stt(self, audio_bytes: bytes):
        """Send audio to STT service with robust error handling."""
        # Check WebSocket exists and is connected
        if self._stt_ws is None:
            if not self._shutting_down:
                self._schedule_stt_reconnect()
            return
        
        try:
            # Check if closed (with AttributeError protection)
            if self._stt_ws.closed:
                if not self._shutting_down:
                    self._schedule_stt_reconnect()
                return
        except AttributeError:
            # WebSocket in invalid state
            self._stt_ws = None
            if not self._shutting_down:
                self._schedule_stt_reconnect()
            return
        
        try:
            await self._stt_ws.send(audio_bytes)
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"[{self.session_id}] STT WebSocket closed during send: {e.code}")
            self.state.stt_connection_state = ConnectionState.DISCONNECTED
            self._stt_ws = None
            if not self._shutting_down:
                self._schedule_stt_reconnect()
        except AttributeError as e:
            logger.warning(f"[{self.session_id}] STT WebSocket AttributeError: {e}")
            self._stt_ws = None
            if not self._shutting_down:
                self._schedule_stt_reconnect()
        except Exception as e:
            logger.error(f"[{self.session_id}] STT send error: {type(e).__name__}: {e}")
    
    async def _stt_receive_loop(self):
        """Background task to receive STT events with robust error handling."""
        logger.info(f"[{self.session_id}] STT receive loop started")
        
        # Wait a tiny bit to ensure WebSocket is fully initialized
        await asyncio.sleep(0.1)
        
        # Capture local reference to avoid race conditions
        ws = self._stt_ws
        
        try:
            while not self._shutting_down:
                # Check WebSocket is valid using helper method
                if not self._is_websocket_valid(self._stt_ws):
                    logger.warning(f"[{self.session_id}] STT WebSocket invalid, exiting receive loop")
                    break
                
                # Update local reference
                ws = self._stt_ws
                
                # Check if WebSocket is closed
                try:
                    if hasattr(ws, 'closed') and ws.closed:
                        logger.warning(f"[{self.session_id}] STT WebSocket closed, exiting receive loop")
                        break
                except (AttributeError, RuntimeError) as e:
                    logger.debug(f"[{self.session_id}] Error checking WebSocket closed status: {e}")
                    # Continue anyway - might be a false alarm
                
                try:
                    # Use instance variable directly to avoid stale references
                    if self._stt_ws is None or self._stt_ws != ws:
                        ws = self._stt_ws
                        if ws is None:
                            logger.warning(f"[{self.session_id}] STT WebSocket became None, exiting receive loop")
                            break
                    
                    message = await asyncio.wait_for(
                        ws.recv(),
                        timeout=30.0
                    )
                    
                    # Parse message
                    if isinstance(message, str):
                        try:
                            data = json.loads(message)
                            await self._handle_stt_message(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"[{self.session_id}] Invalid JSON from STT: {e}")
                    else:
                        # Binary data (unexpected for STT)
                        logger.debug(f"[{self.session_id}] Received binary STT data: {len(message)} bytes")
                        
                except asyncio.TimeoutError:
                    # Send ping to keep alive
                    try:
                        if ws is not None and not ws.closed:
                            pong = await ws.ping()
                            await asyncio.wait_for(pong, timeout=5.0)
                        else:
                            break
                    except Exception as ping_err:
                        logger.warning(f"[{self.session_id}] STT ping failed: {ping_err}")
                        break
                        
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"[{self.session_id}] STT WebSocket closed: code={e.code}")
                    break
                
                except AttributeError as e:
                    logger.warning(f"[{self.session_id}] STT WebSocket AttributeError: {e}")
                    break
                    
                except Exception as e:
                    logger.error(f"[{self.session_id}] STT receive error: {type(e).__name__}: {e}")
                    await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            logger.info(f"[{self.session_id}] STT receive loop cancelled")
            raise
        
        finally:
            self.state.stt_connection_state = ConnectionState.DISCONNECTED
            # Don't reconnect if:
            # 1. Shutting down
            # 2. Agent is SPEAKING (timeout expected)
            # 3. In LISTENING/IDLE (timeout expected, keepalive should prevent but handle gracefully)
            if not self._shutting_down:
                if self.state.orchestrator_state == OrchestratorState.SPEAKING:
                    logger.debug(f"[{self.session_id}] Skipping STT reconnect - agent is speaking")
                elif self.state.orchestrator_state in (OrchestratorState.LISTENING, OrchestratorState.IDLE):
                    # In LISTENING, timeout is expected - reconnect but with longer delay
                    logger.debug(f"[{self.session_id}] STT timeout during LISTENING - reconnecting with delay")
                    self._schedule_stt_reconnect(delay=5.0)  # Longer delay
                else:
                    # THINKING or other states - reconnect normally
                    self._schedule_stt_reconnect()
    
    async def _handle_stt_message(self, data: Dict[str, Any]):
        """Handle incoming STT message."""
        msg_type = data.get("type", "")
        
        if msg_type == "fragment":
            text = data.get("text", "")
            is_final = data.get("is_final", False)
            
            if text:
                logger.info(f"STT {'FINAL' if is_final else 'partial'}: {text[:100]}")
                
                # Call transcript callback
                if UnifiedFastRTCHandler.on_stt_transcript:
                    try:
                        await self._safe_callback(
                            UnifiedFastRTCHandler.on_stt_transcript,
                            self.session_id, text, is_final
                        )
                    except Exception as e:
                        logger.error(f"STT transcript callback error: {e}")
        
        elif msg_type == "connected":
            logger.info(f"STT service confirmed connection: {data.get('session_id')}")
        
        elif msg_type == "timeout":
            # During LISTENING/IDLE, timeout is expected (no user speaking)
            # During SPEAKING, timeout is expected (no audio input while agent talks)
            if self.state.orchestrator_state in (OrchestratorState.LISTENING, OrchestratorState.IDLE, OrchestratorState.SPEAKING):
                logger.debug(f"[{self.session_id}] STT timeout ignored - state: {self.state.orchestrator_state.value}")
                return  # Don't trigger reconnect or VAD callback
            
            # Only treat as error in THINKING state (unexpected)
            logger.warning(f"[{self.session_id}] STT session timeout (state: {self.state.orchestrator_state.value})")
            if UnifiedFastRTCHandler.on_vad_event:
                try:
                    await self._safe_callback(
                        UnifiedFastRTCHandler.on_vad_event,
                        self.session_id, "timeout"
                    )
                except Exception as e:
                    logger.error(f"VAD event callback error: {e}")
        
        elif msg_type == "error":
            logger.error(f"STT error: {data.get('text', data.get('message', 'Unknown'))}")
        
        else:
            logger.debug(f"STT message type '{msg_type}': {data}")
    
    async def _safe_callback(self, callback, *args):
        """Safely execute a callback (sync or async)."""
        if callback is None:
            return
        
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")
    
    async def _stt_keepalive_loop(self):
        """Send periodic silent audio chunks to STT during LISTENING state to prevent timeout."""
        while not self._shutting_down:
            await asyncio.sleep(self._keepalive_interval)
            
            # Only send keepalive during LISTENING or IDLE states
            if self.state.orchestrator_state in (OrchestratorState.LISTENING, OrchestratorState.IDLE):
                if self._is_websocket_valid(self._stt_ws):
                    # Send silent audio chunk (3200 bytes = ~100ms at 16kHz mono 16-bit)
                    silent_chunk = b'\x00' * 3200
                    try:
                        await self._send_to_stt(silent_chunk)
                        logger.debug(f"[{self.session_id}] Sent STT keepalive chunk")
                    except Exception as e:
                        logger.debug(f"[{self.session_id}] Keepalive send failed: {e}")
            else:
                # State changed, exit loop (will be restarted if needed)
                logger.debug(f"[{self.session_id}] Keepalive loop exiting - state changed to {self.state.orchestrator_state.value}")
                break
    
    async def _cleanup_keepalive_task(self):
        """Clean up keepalive task after cancellation."""
        if self._stt_keepalive_task:
            try:
                await asyncio.wait_for(self._stt_keepalive_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._stt_keepalive_task = None


# Factory function for creating handler with config
def create_unified_handler(
    stt_ws_url: Optional[str] = None,
    echo_gate_enabled: bool = True,
) -> UnifiedFastRTCHandler:
    """
    Create a UnifiedFastRTCHandler with configuration from environment.
    
    Environment variables:
        STT_SERVICE_URL: Base URL for STT service (default: http://localhost:8001)
        UNIFIED_ECHO_GATE: Enable echo gating (default: true)
    """
    if stt_ws_url is None:
        base_url = os.getenv("STT_SERVICE_URL", "http://localhost:8001")
        # Convert HTTP URL to WebSocket URL
        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        stt_ws_url = f"{ws_url}/api/v1/transcribe/stream"
    
    return UnifiedFastRTCHandler(
        stt_ws_url=stt_ws_url,
        echo_gate_enabled=echo_gate_enabled,
    )
