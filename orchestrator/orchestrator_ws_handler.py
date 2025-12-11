"""
Unified WebSocket Handler for Orchestrator

Single bidirectional WebSocket connection handles:
- Audio input (microphone chunks from browser)
- Audio output (TTS streaming to browser)
- State synchronization
- Interrupt handling (barge-in)

This replaces the fragmented FastRTC STT/TTS connection pattern with a single /orchestrate endpoint.
"""

import asyncio
import json
import logging
import time
import base64
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .state_manager import StateManager, State
from .config import OrchestratorConfig
from .dialogue_manager import DialogueManager

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorSession:
    """Single unified session for a WebSocket connection"""
    session_id: str
    websocket: WebSocket
    state_manager: StateManager
    created_at: float = field(default_factory=time.time)
    
    # Audio state
    tts_task: Optional[asyncio.Task] = None
    stt_ws: Optional[aiohttp.ClientWebSocketResponse] = None
    stt_session: Optional[aiohttp.ClientSession] = None
    
    # Metrics
    audio_chunks_received: int = 0
    audio_chunks_sent: int = 0
    last_activity: float = field(default_factory=time.time)
    
    # Timeout tracking
    timeout_count: int = 0


class OrchestratorWSHandler:
    """
    Unified WebSocket handler for orchestrator.
    
    Single connection handles:
    - Audio input (microphone chunks)
    - Audio output (TTS streaming)
    - State synchronization
    - Interrupt handling
    """
    
    MAX_TIMEOUT_PROMPTS = 3
    TIMEOUT_SECONDS = 10.0
    
    def __init__(self, 
                 dialogue_manager: DialogueManager,
                 config: OrchestratorConfig,
                 redis_client=None,
                 event_broker=None):
        self.dialogue_manager = dialogue_manager
        self.config = config
        self.redis_client = redis_client
        self.event_broker = event_broker
        self.sessions: Dict[str, OrchestratorSession] = {}
        
        # Callbacks for external integration
        self.on_stt_final: Optional[Callable] = None
    
    async def handle_connection(self, 
                               websocket: WebSocket,
                               session_id: Optional[str] = None):
        """
        Main WebSocket connection handler.
        
        Replaces the fragmented logic with a single unified handler.
        """
        logger.info("=" * 70)
        logger.info(f"🔌 New WebSocket connection to /ws endpoint")
        logger.info(f"   Client: {websocket.client}")
        logger.info(f"   Session ID (provided): {session_id}")
        logger.info("=" * 70)
        
        await websocket.accept()
        logger.info("✅ WebSocket accepted")
        
        # Create or reuse session
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.websocket = websocket  # Update connection
            logger.info(f"[{session_id}] Reconnected to existing session")
        else:
            session = await self._create_session(websocket)
            session_id = session.session_id
            logger.info(f"[{session_id}] ✅ New session created")
        
        # Send session ready notification
        await self._send_json(websocket, {
            "type": "session_ready",
            "session_id": session_id,
            "timestamp": time.time()
        })
        logger.info(f"[{session_id}] 📤 Sent session_ready message")
        
        # Connect to STT service immediately
        logger.info(f"[{session_id}] 🔌 Establishing STT connection...")
        logger.info(f"[{session_id}]    STT Service URL from config: {self.config.stt_service_url}")
        await self._connect_stt(session)
        
        if not session.stt_ws:
            logger.error(f"[{session_id}] ❌ Failed to establish STT connection - audio will not be transcribed")
            logger.error(f"[{session_id}]    Check STT service is running at: {self.config.stt_service_url}")
        else:
            logger.info(f"[{session_id}] ✅ STT connection ready and active")
        
        # Start STT receive loop
        logger.info(f"[{session_id}] 🚀 Starting STT receive loop...")
        stt_receive_task = asyncio.create_task(self._stt_receive_loop(session))
        
        # Start timeout monitor
        timeout_task = asyncio.create_task(self._timeout_monitor(session))
        
        try:
            # Main message loop
            logger.info(f"[{session_id}] 📡 Entering main message loop - waiting for client messages...")
            while True:
                try:
                    data = await websocket.receive_json()
                    logger.debug(f"[{session_id}] 📨 Received message: {data.get('type', 'unknown')}")
                    await self._route_message(session, data)
                except json.JSONDecodeError:
                    # Handle binary audio data
                    logger.debug(f"[{session_id}] 📨 Received binary audio data")
                    raw = await websocket.receive_bytes()
                    await self._handle_binary_audio(session, raw)
        
        except WebSocketDisconnect:
            logger.info(f"[{session_id}] Client disconnected")
        
        except Exception as e:
            logger.error(f"[{session_id}] WebSocket error: {e}", exc_info=True)
        
        finally:
            # Cancel background tasks
            stt_receive_task.cancel()
            timeout_task.cancel()
            try:
                await stt_receive_task
            except asyncio.CancelledError:
                pass
            try:
                await timeout_task
            except asyncio.CancelledError:
                pass
            
            await self._cleanup_session(session)
    
    async def _route_message(self, 
                           session: OrchestratorSession, 
                           msg: Dict[str, Any]):
        """Route incoming message to appropriate handler"""
        msg_type = msg.get("type")
        
        if msg_type == "audio_chunk":
            await self._handle_audio_chunk(session, msg)
        
        elif msg_type == "playback_done":
            await self._handle_playback_done(session, msg)
        
        elif msg_type == "interrupt":
            await self._handle_interrupt(session, msg)
        
        elif msg_type == "state_sync_request":
            await self._handle_state_sync(session)
        
        elif msg_type == "start_session":
            await self._handle_start_session(session, msg)
        
        elif msg_type == "end_session":
            await self._handle_end_session(session)
        
        else:
            logger.warning(f"[{session.session_id}] Unknown message type: {msg_type}")
    
    async def _handle_audio_chunk(self, 
                                  session: OrchestratorSession, 
                                  msg: Dict[str, Any]):
        """
        Handle incoming audio chunk from browser microphone.
        
        Streams audio to STT service via WebSocket.
        """
        state_mgr = session.state_manager
        
        # Check state - ignore if not listening
        if state_mgr.state not in [State.IDLE, State.LISTENING, State.INTERRUPT]:
            logger.debug(f"[{session.session_id}] Ignoring audio in {state_mgr.state.value} state")
            return
        
        session.last_activity = time.time()
        session.audio_chunks_received += 1
        
        try:
            # Decode audio
            audio_b64 = msg.get("data", "")
            if not audio_b64:
                logger.warning(f"[{session.session_id}] Empty audio chunk received")
                return
                
            audio_bytes = base64.b64decode(audio_b64)
            
            if session.audio_chunks_received == 1:
                logger.info(f"[{session.session_id}] 🎤 First audio chunk received ({len(audio_bytes)} bytes)")
            
            # Ensure STT connection exists
            if not session.stt_ws or session.stt_ws.closed:
                logger.warning(f"[{session.session_id}] ⚠️ STT not connected, attempting connection...")
                await self._connect_stt(session)
            
            # Stream to STT service
            if session.stt_ws and not session.stt_ws.closed:
                await session.stt_ws.send_bytes(audio_bytes)
                if session.audio_chunks_received == 1:
                    logger.info(f"[{session.session_id}] ✅ First audio chunk sent to STT service")
                elif session.audio_chunks_received % 50 == 0:  # Log every 50 chunks
                    logger.info(f"[{session.session_id}] 📊 Sent {session.audio_chunks_received} audio chunks to STT")
            else:
                logger.error(f"[{session.session_id}] ❌ STT WebSocket not available, cannot send audio")
                logger.error(f"[{session.session_id}]    STT connection state: ws={session.stt_ws}, closed={session.stt_ws.closed if session.stt_ws else 'N/A'}")
        
        except Exception as e:
            logger.error(f"[{session.session_id}] ❌ Audio chunk error: {e}", exc_info=True)
    
    async def _handle_binary_audio(self, session: OrchestratorSession, audio_bytes: bytes):
        """Handle raw binary audio data"""
        state_mgr = session.state_manager
        
        if state_mgr.state not in [State.IDLE, State.LISTENING, State.INTERRUPT]:
            logger.debug(f"[{session.session_id}] Ignoring binary audio in {state_mgr.state.value} state")
            return
        
        session.last_activity = time.time()
        session.audio_chunks_received += 1
        
        try:
            if session.audio_chunks_received == 1:
                logger.info(f"[{session.session_id}] 🎤 First binary audio chunk received ({len(audio_bytes)} bytes)")
            
            # Ensure STT connection exists
            if not session.stt_ws or session.stt_ws.closed:
                logger.warning(f"[{session.session_id}] ⚠️ STT not connected for binary audio, connecting...")
                await self._connect_stt(session)
            
            if session.stt_ws and not session.stt_ws.closed:
                await session.stt_ws.send_bytes(audio_bytes)
                if session.audio_chunks_received == 1:
                    logger.info(f"[{session.session_id}] ✅ First binary audio chunk sent to STT")
                elif session.audio_chunks_received % 50 == 0:
                    logger.info(f"[{session.session_id}] 📊 Sent {session.audio_chunks_received} binary audio chunks to STT")
            else:
                logger.error(f"[{session.session_id}] ❌ STT WebSocket not available for binary audio")
        except Exception as e:
            logger.error(f"[{session.session_id}] ❌ Binary audio error: {e}", exc_info=True)
    
    async def _connect_stt(self, session: OrchestratorSession):
        """Connect to STT service WebSocket with enhanced logging"""
        try:
            # Build STT WebSocket URL
            stt_ws_url = self.config.stt_service_url.replace("http://", "ws://").replace("https://", "wss://")
            stt_ws_url = f"{stt_ws_url}/api/v1/transcribe/stream?session_id={session.session_id}"
            
            logger.info(f"[{session.session_id}] " + "=" * 50)
            logger.info(f"[{session.session_id}] 🔌 Connecting to STT service...")
            logger.info(f"[{session.session_id}]    URL: {stt_ws_url}")
            logger.info(f"[{session.session_id}]    Config STT URL: {self.config.stt_service_url}")
            
            # Create aiohttp session if needed
            if session.stt_session is None:
                session.stt_session = aiohttp.ClientSession()
                logger.debug(f"[{session.session_id}] Created new aiohttp ClientSession")
            
            # Connect to STT WebSocket with timeout
            session.stt_ws = await asyncio.wait_for(
                session.stt_session.ws_connect(stt_ws_url),
                timeout=10.0
            )
            
            logger.info(f"[{session.session_id}] ✅ STT WebSocket connected successfully")
            logger.info(f"[{session.session_id}]    Ready to receive transcripts")
            logger.info(f"[{session.session_id}] " + "=" * 50)
        
        except asyncio.TimeoutError:
            logger.error(f"[{session.session_id}] ❌ STT connection timeout (10s) - service may be unavailable")
            logger.error(f"[{session.session_id}]    Check STT service at: {self.config.stt_service_url}")
            session.stt_ws = None
        except aiohttp.ClientError as e:
            logger.error(f"[{session.session_id}] ❌ STT connection error: {type(e).__name__}: {e}")
            session.stt_ws = None
        except Exception as e:
            logger.error(f"[{session.session_id}] ❌ Failed to connect STT: {e}", exc_info=True)
            session.stt_ws = None
    
    async def _stt_receive_loop(self, session: OrchestratorSession):
        """Receive transcripts from STT service with timeout and comprehensive logging"""
        logger.info(f"[{session.session_id}] 🔄 STT receive loop started")
        message_count = 0
        reconnect_delay = 1.0
        max_reconnect_delay = 30.0
        
        while True:
            try:
                # Check WebSocket connection state
                if not session.stt_ws or session.stt_ws.closed:
                    logger.warning(f"[{session.session_id}] STT WebSocket not connected, reconnecting in {reconnect_delay:.1f}s...")
                    await asyncio.sleep(reconnect_delay)
                    await self._connect_stt(session)
                    # Exponential backoff
                    reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                    continue
                
                # Reset reconnect delay on successful connection
                reconnect_delay = 1.0
                
                # Add timeout to prevent indefinite blocking
                try:
                    msg = await asyncio.wait_for(session.stt_ws.receive(), timeout=30.0)
                except asyncio.TimeoutError:
                    # 30s timeout - connection still alive, just no messages
                    logger.debug(f"[{session.session_id}] STT receive timeout (30s) - connection still alive, no speech detected")
                    continue
                
                message_count += 1
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type", "")
                    
                    # Log ALL messages at INFO level for debugging
                    logger.info(f"[{session.session_id}] 📨 STT message #{message_count}: type={msg_type}, data={json.dumps(data)[:200]}")
                    
                    # Handle different STT message types
                    if msg_type == "connected":
                        logger.info(f"[{session.session_id}] ✅ STT service confirmed connection")
                    elif msg_type == "fragment":
                        # STT sends fragments with text and is_final
                        text = data.get("text", "")
                        if text and text.strip():
                            await self._handle_stt_result(session, data)
                        else:
                            logger.debug(f"[{session.session_id}] Empty fragment received (ignored)")
                    elif msg_type == "capture_started":
                        logger.info(f"[{session.session_id}] 🎙️ STT capture started - audio processing active")
                    elif msg_type == "timeout":
                        logger.warning(f"[{session.session_id}] ⏰ STT timeout - no speech detected for idle period")
                    elif msg_type == "error":
                        logger.error(f"[{session.session_id}] ❌ STT error: {data.get('text', data.get('message', 'Unknown'))}")
                    else:
                        # Log unknown message types with full payload for debugging
                        logger.warning(f"[{session.session_id}] ⚠️ Unknown STT message type: {msg_type}, full data: {json.dumps(data)}")
                
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    logger.info(f"[{session.session_id}] 📦 Received binary data from STT: {len(msg.data)} bytes (unexpected)")
                
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning(f"[{session.session_id}] 🔌 STT WebSocket closed by server, reconnecting...")
                    session.stt_ws = None
                    await asyncio.sleep(reconnect_delay)
                    await self._connect_stt(session)
                    reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"[{session.session_id}] ❌ STT WebSocket error, reconnecting...")
                    session.stt_ws = None
                    await asyncio.sleep(reconnect_delay)
                    await self._connect_stt(session)
                    reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
            
            except asyncio.CancelledError:
                logger.info(f"[{session.session_id}] 🛑 STT receive loop cancelled (received {message_count} messages total)")
                break
            except json.JSONDecodeError as e:
                logger.error(f"[{session.session_id}] ❌ Invalid JSON from STT: {e}")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"[{session.session_id}] ❌ STT receive error: {e}", exc_info=True)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
    
    async def _handle_stt_result(self, session: OrchestratorSession, data: Dict[str, Any]):
        """Handle STT transcription result"""
        text = data.get("text", "")
        is_final = data.get("is_final", False)
        
        if not text or not text.strip():
            return
        
        # Update activity
        session.last_activity = time.time()
        session.timeout_count = 0  # Reset timeout on user activity
        
        logger.info(f"[{session.session_id}] 📝 STT {'FINAL' if is_final else 'fragment'}: {text[:100]}")
        
        # Send transcript to browser
        await self._send_json(session.websocket, {
            "type": "transcript",
            "text": text,
            "is_final": is_final,
            "timestamp": time.time()
        })
        
        if is_final:
            logger.info(f"[{session.session_id}] ✅ STT FINAL transcript received: {text[:100]}")
            await self._process_user_input(session, text)
    
    async def _process_user_input(self, 
                                  session: OrchestratorSession, 
                                  text: str):
        """
        Process final STT result - trigger Intent+RAG+TTS pipeline.
        """
        state_mgr = session.state_manager
        
        if not text.strip():
            return
        
        # Transition to THINKING
        await state_mgr.transition(State.THINKING, trigger="stt_final", data={"text": text})
        await self._broadcast_state(session, State.THINKING)
        
        # Cancel any ongoing TTS
        if session.tts_task and not session.tts_task.done():
            session.tts_task.cancel()
            try:
                await session.tts_task
            except asyncio.CancelledError:
                pass
        
        try:
            # Stream response and TTS (will transition to SPEAKING when response is ready)
            session.tts_task = asyncio.create_task(
                self._stream_response_and_tts(session, text)
            )
            
            await session.tts_task
        
        except asyncio.CancelledError:
            logger.info(f"[{session.session_id}] Processing cancelled (user interrupt)")
        
        except Exception as e:
            logger.error(f"[{session.session_id}] Processing error: {e}", exc_info=True)
            await state_mgr.transition(State.LISTENING, trigger="error")
            await self._broadcast_state(session, State.LISTENING)
    
    async def _stream_response_and_tts(self, 
                                       session: OrchestratorSession, 
                                       user_text: str):
        """
        Stream LLM response and TTS audio to browser.
        """
        from .parallel_pipeline import process_intent_rag_llm
        
        response_text = ""
        state_mgr = session.state_manager
        
        try:
            # Stream from RAG/LLM
            async for token in process_intent_rag_llm(
                user_text=user_text,
                session_id=session.session_id,
                intent_url=self.config.intent_service_url,
                rag_url=self.config.rag_service_url
            ):
                response_text += token
                
                # Send text chunk to browser
                await self._send_json(session.websocket, {
                    "type": "agent_response",
                    "text": token,
                    "is_streaming": True,
                    "timestamp": time.time()
                })
            
            # Transition to SPEAKING when response is ready
            await state_mgr.transition(State.SPEAKING, trigger="response_ready", data={"response": response_text})
            await self._broadcast_state(session, State.SPEAKING)
            
            # Signal text complete
            await self._send_json(session.websocket, {
                "type": "agent_response",
                "text": "",
                "is_streaming": False,
                "is_complete": True,
                "full_text": response_text,
                "timestamp": time.time()
            })
            
            # Stream TTS audio
            await self._stream_tts_to_browser(session, response_text)
            
        except asyncio.CancelledError:
            logger.info(f"[{session.session_id}] Response streaming cancelled")
            raise
    
    async def _stream_tts_to_browser(self, 
                                     session: OrchestratorSession, 
                                     text: str):
        """
        Stream TTS audio to browser.
        """
        chunk_counter = 0
        
        try:
            # Connect to TTS service
            tts_ws_url = self.config.tts_service_url.replace("http://", "ws://").replace("https://", "wss://")
            tts_ws_url = f"{tts_ws_url}/api/v1/stream?session_id={session.session_id}"
            
            async with aiohttp.ClientSession() as tts_session:
                async with tts_session.ws_connect(tts_ws_url) as tts_ws:
                    # Send synthesis request
                    await tts_ws.send_json({
                        "type": "synthesize",
                        "text": text,
                        "emotion": "helpful"
                    })
                    
                    # Receive and forward audio chunks
                    async for msg in tts_ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            msg_type = data.get("type")
                            
                            if msg_type == "audio":
                                audio_b64 = data.get("data", "")
                                if audio_b64:
                                    await self._send_json(session.websocket, {
                                        "type": "audio_chunk",
                                        "audio": audio_b64,
                                        "chunk_id": f"{session.session_id}_chunk_{chunk_counter}",
                                        "is_final": False,
                                        "timestamp": time.time()
                                    })
                                    session.audio_chunks_sent += 1
                                    chunk_counter += 1
                            
                            elif msg_type == "complete":
                                break
                        
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            
            # Signal TTS completion
            await self._send_json(session.websocket, {
                "type": "audio_chunk",
                "chunk_id": f"{session.session_id}_chunk_{chunk_counter}",
                "is_final": True,
                "timestamp": time.time()
            })
            
            logger.info(f"[{session.session_id}] TTS streaming complete ({chunk_counter} chunks)")
        
        except asyncio.CancelledError:
            logger.info(f"[{session.session_id}] TTS cancelled (user interrupt)")
            raise
        
        except Exception as e:
            logger.error(f"[{session.session_id}] TTS streaming error: {e}")
    
    async def _handle_playback_done(self, 
                                    session: OrchestratorSession, 
                                    msg: Dict[str, Any]):
        """
        Handle browser confirming playback completion.
        
        This is the KEY to synchronization - server waits for browser confirmation.
        """
        state_mgr = session.state_manager
        session.last_activity = time.time()
        
        if state_mgr.state != State.SPEAKING:
            logger.debug(f"[{session.session_id}] Playback done in non-SPEAKING state (ignored)")
            return
        
        logger.info(f"[{session.session_id}] ✅ Playback DONE (browser confirmed)")
        
        # Transition back to LISTENING
        await state_mgr.transition(State.LISTENING, trigger="playback_done")
        await self._broadcast_state(session, State.LISTENING)
    
    async def _handle_interrupt(self, 
                               session: OrchestratorSession, 
                               msg: Dict[str, Any]):
        """
        Handle user interrupting agent (barge-in).
        
        Server must respond immediately - cancel TTS and clear output queue.
        """
        state_mgr = session.state_manager
        session.last_activity = time.time()
        
        logger.info(f"[{session.session_id}] 🔴 INTERRUPT detected")
        
        # Cancel TTS task
        if session.tts_task and not session.tts_task.done():
            session.tts_task.cancel()
            try:
                await session.tts_task
            except asyncio.CancelledError:
                pass
        
        # Notify browser to stop playback
        await self._send_json(session.websocket, {
            "type": "playback_control",
            "action": "stop",
            "reason": "user_interrupt",
            "timestamp": time.time()
        })
        
        # Transition: SPEAKING → INTERRUPT → LISTENING
        if state_mgr.state == State.SPEAKING:
            await state_mgr.transition(State.INTERRUPT, trigger="user_interrupt")
            await asyncio.sleep(0.05)  # Brief pause
            await state_mgr.transition(State.LISTENING, trigger="interrupt_complete")
        
        await self._broadcast_state(session, State.LISTENING)
    
    async def _handle_state_sync(self, session: OrchestratorSession):
        """Sync current state to client"""
        await self._broadcast_state(session, session.state_manager.state)
    
    async def _handle_start_session(self, session: OrchestratorSession, msg: Dict[str, Any]):
        """Handle session start - play intro greeting"""
        mode = msg.get("mode", "conversation")
        logger.info(f"[{session.session_id}] Session started in {mode} mode")
        
        # Transition to SPEAKING for intro
        await session.state_manager.transition(
            State.SPEAKING, 
            trigger="intro_start",
            data={"response": self.config.intro_greeting}
        )
        await self._broadcast_state(session, State.SPEAKING)
        
        # Stream intro TTS
        session.tts_task = asyncio.create_task(
            self._stream_tts_to_browser(session, self.config.intro_greeting)
        )
    
    async def _handle_end_session(self, session: OrchestratorSession):
        """Client requests session end"""
        logger.info(f"[{session.session_id}] Session end requested")
        
        # Play goodbye if configured
        if self.dialogue_manager:
            from .dialogue_manager import DialogueType
            exit_asset = self.dialogue_manager.get_random_exit()
            if exit_asset:
                await session.state_manager.transition(
                    State.SPEAKING,
                    trigger="session_ending",
                    data={"response": exit_asset.text}
                )
                await self._broadcast_state(session, State.SPEAKING)
                await self._stream_tts_to_browser(session, exit_asset.text)
        
        await session.websocket.close()
    
    async def _timeout_monitor(self, session: OrchestratorSession):
        """Monitor for user inactivity"""
        while True:
            try:
                await asyncio.sleep(1.0)
                
                state_mgr = session.state_manager
                if state_mgr.state != State.LISTENING:
                    continue
                
                # Check timeout
                elapsed = time.time() - session.last_activity
                if elapsed >= self.TIMEOUT_SECONDS:
                    session.timeout_count += 1
                    session.last_activity = time.time()  # Reset
                    
                    logger.info(f"[{session.session_id}] Timeout {session.timeout_count}/{self.MAX_TIMEOUT_PROMPTS}")
                    
                    if session.timeout_count >= self.MAX_TIMEOUT_PROMPTS:
                        # End session
                        await self._handle_end_session(session)
                        break
                    else:
                        # Play timeout prompt
                        if self.dialogue_manager:
                            timeout_asset = self.dialogue_manager.get_timeout_prompt()
                            await state_mgr.transition(
                                State.SPEAKING,
                                trigger="timeout_detected",
                                data={"response": timeout_asset.text}
                            )
                            await self._broadcast_state(session, State.SPEAKING)
                            session.tts_task = asyncio.create_task(
                                self._stream_tts_to_browser(session, timeout_asset.text)
                            )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{session.session_id}] Timeout monitor error: {e}")
    
    async def _broadcast_state(self, 
                              session: OrchestratorSession, 
                              state: State):
        """Notify browser of state change"""
        await self._send_json(session.websocket, {
            "type": "state_update",
            "state": state.value,
            "timestamp": time.time()
        })
    
    async def _send_json(self, websocket: WebSocket, data: Dict[str, Any]):
        """Safe JSON send"""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.debug(f"Failed to send JSON: {e}")
    
    async def _create_session(self, websocket: WebSocket) -> OrchestratorSession:
        """Create new orchestrator session"""
        session_id = f"ws_session_{int(time.time() * 1000) % 1000000}"
        
        state_manager = StateManager(
            session_id=session_id,
            redis_client=self.redis_client,
            broker=self.event_broker
        )
        await state_manager.initialize()
        
        session = OrchestratorSession(
            session_id=session_id,
            websocket=websocket,
            state_manager=state_manager
        )
        
        self.sessions[session_id] = session
        return session
    
    async def _cleanup_session(self, session: OrchestratorSession):
        """Clean up session resources"""
        # Cancel TTS
        if session.tts_task and not session.tts_task.done():
            session.tts_task.cancel()
            try:
                await session.tts_task
            except asyncio.CancelledError:
                pass
        
        # Close STT connection
        if session.stt_ws and not session.stt_ws.closed:
            await session.stt_ws.close()
        
        if session.stt_session:
            await session.stt_session.close()
        
        # Remove from registry
        if session.session_id in self.sessions:
            del self.sessions[session.session_id]
        
        logger.info(f"[{session.session_id}] Session cleaned up")
