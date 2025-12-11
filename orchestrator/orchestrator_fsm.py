"""
Event-Driven Orchestrator FSM.

Consumes events from Redis Streams and drives the conversation state machine.
Handles parallel processing of Intent and RAG, and coordinates TTS streaming.
"""

import asyncio
import logging
import json
import uuid
import time
from typing import Dict, Any, Optional, List

from redis.asyncio import Redis
from leibniz_agent.services.shared.events import VoiceEvent, EventTypes
from leibniz_agent.services.shared.event_broker import EventBroker
from leibniz_agent.services.orchestrator.state_manager import StateManager, State
from leibniz_agent.services.orchestrator.stt_event_handler import STTEventHandler
from leibniz_agent.services.orchestrator.config import OrchestratorConfig
from leibniz_agent.services.orchestrator.dialogue_manager import DialogueManager
from leibniz_agent.services.orchestrator import app as orchestrator_app
from .structured_logger import StructuredLogger

logger = logging.getLogger(__name__)

class OrchestratorFSM:
    def __init__(
        self, 
        session_id: str, 
        redis_client: Redis, 
        broker: EventBroker,
        config: Optional[OrchestratorConfig] = None
    ):
        self.session_id = session_id
        self.redis = redis_client
        self.broker = broker
        self.state_mgr = StateManager(session_id, redis_client, broker)
        self.config = config or OrchestratorConfig.from_env()
        self.dialogue_manager = DialogueManager(tara_mode=self.config.tara_mode)
        self.running = False
        self._task = None
        self._output_handler = None
        self.structured_logger = StructuredLogger(logger)

    def set_output_handler(self, handler):
        """Set handler for outgoing messages (e.g. to WebSocket)."""
        self._output_handler = handler

    async def _emit_output(self, data: Dict[str, Any]):
        """Emit data to output handler if set."""
        if self._output_handler:
            if asyncio.iscoroutinefunction(self._output_handler):
                await self._output_handler(data)
            else:
                self._output_handler(data)

        
    async def start(self):
        """Start the FSM event loop."""
        await self.state_mgr.initialize()
        self.running = True
        self._task = asyncio.create_task(self.run())
        logger.info(f"[{self.session_id}] Orchestrator FSM started")

    async def stop(self):
        """Stop the FSM event loop."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"[{self.session_id}] Orchestrator FSM stopped")

    async def run(self):
        """Main event loop – consumes from broker, updates FSM."""
        # Subscribe to relevant streams for this session
        streams = {
            f"voice:stt:session:{self.session_id}": "$",
            f"voice:intent:session:{self.session_id}": "$",
            f"voice:rag:session:{self.session_id}": "$",
            f"voice:tts:session:{self.session_id}": "$",
            f"voice:webrtc:session:{self.session_id}": "$",
        }
        
        while self.running:
            try:
                # Block read from Redis Streams (new messages only)
                messages = await self.broker.consume(streams, block=100)
                
                if not messages:
                    continue
                    
                for stream_key, message_list in messages:
                    # Update last ID for next read
                    if message_list:
                        streams[stream_key] = message_list[-1][0]
                        
                    for message_id, data in message_list:
                        try:
                            # Parse event - Redis returns dict of strings/bytes
                            # Our EventBroker stores the event as fields
                            # But wait, VoiceEvent.to_redis_dict flattens it.
                            # So data IS the event structure.
                            
                            # However, 'data' from xread is {field: value}.
                            # VoiceEvent.from_redis_dict handles this.
                            event = VoiceEvent.from_redis_dict(data)
                            await self.handle_event(event)
                            
                            # Use orchestrator-group for ack if we were using consumer groups
                            # Here we are using XREAD (fan-out per session instance)
                            # so no XACK needed unless we use groups.
                            # The plan says "OrchestratorFSM: consumes all events".
                            # Since Orchestrator is singleton PER SESSION, XREAD is fine.
                            
                        except Exception as e:
                            logger.error(f"[{self.session_id}] Error handling event {message_id}: {e}", exc_info=True)
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.session_id}] FSM Loop Error: {e}")
                await asyncio.sleep(1.0)

    async def handle_event(self, event: VoiceEvent):
        """Dispatch event to appropriate handler."""
        try:
            event.validate_payload()
        except ValueError as e:
            logger.error(f"[{self.session_id}] Invalid event payload: {e}")
            return

        # Structured event log for observability
        self.structured_logger.event_received(
            self.session_id,
            event_type=event.event_type,
            payload=event.payload,
        )

        handlers = {
            EventTypes.STT_FINAL: self.on_stt_final,
            EventTypes.INTENT_DETECTED: self.on_intent_detected,
            EventTypes.RAG_ANSWER_READY: self.on_rag_answer,
            EventTypes.TTS_CHUNK_READY: self.on_tts_chunk,
            EventTypes.PLAYBACK_DONE: self.on_playback_done,
            EventTypes.BARGE_IN: self.on_barge_in,
        }
        
        handler = handlers.get(event.event_type)
        if handler:
            await handler(event)

    async def on_stt_final(self, event: VoiceEvent):
        """Handle final STT event → trigger Intent+RAG via STTEventHandler."""
        text = event.payload.get("text", "")
        is_final = event.payload.get("is_final", False)
        
        if not text:
            return

        # Check if this event was already processed by WebSocket?
        # The WebSocket handler doesn't emit STT_FINAL to Redis yet (based on app.py analysis)
        # It handles it locally.
        # But if we receive this from Redis (e.g. from an external component), we should process it.
        # Ideally, we should unify so WebSocket ALSO emits to Redis and FSM handles it.
        # However, to preserve latency, the WebSocket often does it directly.
        # We can detect if we should process it by checking if it's already in THINKING state?
        # But STTEventHandler handles state checks.
        
        logger.info(f"[{self.session_id}] 🎤 STT Final (Stream): '{text}'")
        
        # Instantiate unified handler
        handler = STTEventHandler(
            self.session_id, 
            self.state_mgr, 
            self.config,
            self.dialogue_manager
        )
        
        # Define filler callbacks similar to app.py
        # NOTE: This duplicates the filler logic from app.py slightly.
        # In a perfect world, we'd extract the filler logic into a shared helper.
        # For Phase 3, we'll implement it here to ensure parity.
        
        async def on_thinking():
            # Trigger fillers via app.py's task helper if possible, 
            # or directly if we have access to the websocket/streamer.
            # FSM doesn't have direct access to 'websocket' object easily unless passed.
            # But it has _output_handler.
            # The filler tasks need to stream audio.
            # This part is tricky without the full app.py context (websocket).
            # If the session was auto-created (no websocket), TTS streams directly to Service.
            pass

        try:
            # Process
            result = await handler.handle_stt_final(
                text, 
                is_final, 
                source="redis_stream",
                on_thinking=on_thinking
            )
            
            # Handle Result (Generator)
            if result:
                # We need to stream this generator to the TTS service
                # app.py uses stream_tts_from_generator.
                # We can call that if we import app.
                # However, app.py functions often require 'websocket'.
                # For auto-sessions, websocket is None.
                
                # We need to locate the websocket if it exists.
                session_data = orchestrator_app.active_sessions.get(self.session_id)
                websocket = session_data.get("websocket") if session_data else None
                
                # Use the session task helper from app.py to manage the stream task
                stream_task = await orchestrator_app.replace_session_task(
                    self.session_id,
                    orchestrator_app.stream_tts_from_generator(
                        self.session_id, 
                        result, 
                        websocket, 
                        self.state_mgr
                    ),
                    reason="fsm_response_stream"
                )
                
                if stream_task:
                    await stream_task
                    
                await self.state_mgr.transition(State.LISTENING, "interaction_complete", {})
                
        except Exception as e:
            logger.error(f"[{self.session_id}] FSM STT Error: {e}", exc_info=True)
            await self.state_mgr.transition(State.IDLE, "error", {"error": str(e)})

    async def on_intent_detected(self, event: VoiceEvent):
        # Legacy/Parallel specific - STTEventHandler handles this internally now
        pass

    async def on_rag_answer(self, event: VoiceEvent):
        # Legacy/Parallel specific - STTEventHandler handles this internally now
        pass

    async def on_tts_chunk(self, event: VoiceEvent):
        """
        Handle TTS audio chunk.
        Forward to Client (WebSocket/WebRTC).
        """
        # Forward the payload directly to the client
        # The payload should contain "audio_base64" and "index" etc.
        await self._emit_output(event.payload)


    async def on_playback_done(self, event: VoiceEvent):
        """Handle WebRTC playback completion → back to LISTENING."""
        logger.info(f"[{self.session_id}] 🔊 Playback Done")
        await self.state_mgr.transition(State.LISTENING, "playback_done", {})

    async def on_barge_in(self, event: VoiceEvent):
        """Handle user interruption → cancel TTS, back to LISTENING."""
        logger.info(f"[{self.session_id}] ⚡ Barge-in")
        
        # Cancel TTS
        cancel_evt = VoiceEvent(
            event_type=EventTypes.TTS_CANCEL,
            session_id=self.session_id,
            source="orchestrator",
            payload={}
        )
        await self.broker.publish(f"voice:tts:session:{self.session_id}", cancel_evt)
        
        # Transition
        await self.state_mgr.transition(State.INTERRUPT, "barge_in", {})
        await asyncio.sleep(0.1)
        await self.state_mgr.transition(State.LISTENING, "ready_after_interrupt", {})
