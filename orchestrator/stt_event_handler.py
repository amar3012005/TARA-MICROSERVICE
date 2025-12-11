import asyncio
import logging
import time
from typing import Optional, Dict, Any, Union, Callable

from leibniz_agent.services.orchestrator.config import OrchestratorConfig
from leibniz_agent.services.orchestrator.state_manager import StateManager, State
from leibniz_agent.services.orchestrator.parallel_pipeline import (
    process_intent_rag_llm,
    process_rag_incremental,
)
from leibniz_agent.services.orchestrator.dialogue_manager import DialogueManager
from .structured_logger import StructuredLogger

logger = logging.getLogger(__name__)


class STTEventHandler:
    """
    Single point of truth for STT event processing.

    Routes to:
    1. Parallel Intent+RAG
    2. State transitions
    3. Filler management (deferred to caller via return values/side effects on session)
    """

    def __init__(
        self,
        session_id: str,
        state_mgr: StateManager,
        config: OrchestratorConfig,
        dialogue_manager: Optional[DialogueManager] = None,
    ):
        self.session_id = session_id
        self.state_mgr = state_mgr
        self.config = config
        self.dialogue_manager = dialogue_manager
        self.structured_logger = StructuredLogger(logger)

    async def handle_stt_final(
        self,
        text: str,
        is_final: bool,
        source: str = "unknown",  # "websocket", "redis_pubsub", "redis_stream"
        on_thinking: Optional[Callable[[], Any]] = None,
    ) -> Optional[Any]:
        """
        Process final STT result.

        Returns:
            - A generator (async iterator) for TTS streaming if processing succeeded.
            - None if processing failed, text was empty, or exit keyword was detected (handled internally).
        """
        # 1. Validate
        if not text or not text.strip():
            logger.warning(f"[{self.session_id}] Empty STT text, ignoring")
            return None

        # Redundant check if pre-validated by caller/FSM, but safe to keep
        if not is_final:
            logger.debug(f"[{self.session_id}] Partial STT (ignoring): {text[:50]}...")
            return None

        # 2. Log source for debugging (structured + human-readable)
        logger.info("=" * 70)
        logger.info(f"🤐 Processing STT event (source={source})")
        logger.info(f"📝 Text: {text}")
        if self.config.tara_mode:
            logger.info(f"🇮🇳 TARA MODE: Direct RAG (skip Intent)")
        logger.info("=" * 70)
        self.structured_logger.event_received(
            self.session_id,
            event_type="voice.stt.final",
            payload={"text": text, "is_final": is_final, "source": source},
        )

        # 3. Validate state (must be LISTENING)
        if self.state_mgr.state != State.LISTENING:
            logger.warning(
                f"[{self.session_id}] Cannot process STT in {self.state_mgr.state.value} state"
            )
            return None

        # 4. Check for exit keywords
        if await self._check_exit_keywords(text):
            return None  # Exit handled internally

        # 5. Transition to THINKING
        await self.state_mgr.transition(State.THINKING, "stt_received", {"text": text})
        
        # Trigger on_thinking callback (e.g. for fillers)
        if on_thinking:
            try:
                res = on_thinking()
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
            except Exception as e:
                logger.error(f"[{self.session_id}] on_thinking callback error: {e}")

        # 6. Process Intent/RAG
        try:
            start_time = time.time()
            result = await self._process_pipeline(text)
            duration_ms = (time.time() - start_time) * 1000.0
            self.structured_logger.latency_recorded(
                self.session_id,
                operation="stt_pipeline",
                duration_ms=duration_ms,
                extra={"source": source},
            )
            if result:
                # 7. Transition to SPEAKING
                # Note: The actual audio streaming happens in the caller (app.py)
                # but we signal that we are ready to speak.
                await self.state_mgr.transition(
                    State.SPEAKING, 
                    "response_ready", 
                    {"text": text}  # We don't have the full response text yet if it's a generator
                )
            return result
        except Exception as e:
            logger.error(f"[{self.session_id}] Processing error: {e}", exc_info=True)
            await self.state_mgr.transition(State.IDLE, "error", {"error": str(e)})
            return None

    async def _check_exit_keywords(self, text: str) -> bool:
        """Check for exit keywords and handle session termination if found."""
        text_lower = text.lower().strip()
        exit_keywords = []
        
        if self.dialogue_manager and self.dialogue_manager.exit_keywords:
            exit_keywords.extend(self.dialogue_manager.exit_keywords)
        if hasattr(self.config, "exit_keywords"):
            exit_keywords.extend(
                kw.strip().lower() for kw in self.config.exit_keywords if kw.strip()
            )
        
        exit_keywords = sorted(set(exit_keywords))
        
        if any(keyword in text_lower for keyword in exit_keywords):
            logger.info("=" * 70)
            logger.info(f"🚪 EXIT DETECTED: User said '{text}'")
            logger.info("=" * 70)
            
            # Transition to SPEAKING for exit message
            await self.state_mgr.transition(State.SPEAKING, "exit_detected", {})
            
            # The caller (app.py) handles the actual TTS playback and cleanup 
            # for exit scenarios, but since we want to unify, we might need 
            # to signal this specific result type.
            # For now, we'll return a special marker or handle it here?
            # Ideally, STTEventHandler handles the LOGIC, but app.py handles 
            # the TASKS (TTS/Stream).
            
            # Refactoring note: existing app.py code handles exit TTS and cleanup inline.
            # To preserve behavior without massive changes, we might want to return 
            # a specific result indicating "EXIT".
            return True # Signaled exit
            
        return False

    async def _process_pipeline(self, text: str) -> Optional[Any]:
        """Execute the configured RAG/Intent pipeline."""
        start_time = time.time()
        generator = None

        if self.config.skip_intent_service or self.config.tara_mode:
            if self.config.rag_service_url:
                logger.info("🇮🇳 TARA: Incremental RAG processing (using buffered context)...")
                generator = process_rag_incremental(
                    user_text=text,
                    session_id=self.session_id,
                    rag_url=self.config.rag_service_url,
                    is_final=True,
                    language=self.config.response_language,
                    organization=self.config.organization_name
                )
            else:
                logger.error("❌ TARA mode requires RAG service URL")
                await self.state_mgr.transition(State.IDLE, "error", {"error": "RAG service not configured"})
                return None
        else:
            # Standard mode
            if self.config.rag_service_url:
                logger.info("⚡ Starting parallel Intent+RAG processing...")
            else:
                logger.info("⚡ Starting Intent processing (RAG not configured)...")
            
            generator = process_intent_rag_llm(
                text, 
                self.session_id,
                self.config.intent_service_url,
                self.config.rag_service_url
            )
            
        return generator

