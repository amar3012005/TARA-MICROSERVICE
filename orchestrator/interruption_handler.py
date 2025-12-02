"""
Barge-in detection and TTS cancellation

Handles user interruptions during TTS playback.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class InterruptionHandler:
    """Handles barge-in (user interrupts during TTS)"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.interrupted = False
    
    async def detect_barge_in(self, vad_confidence: float) -> bool:
        """Detect if user started speaking during TTS"""
        if vad_confidence > 0.7:
            self.interrupted = True
            logger.warning(f"⚡ Barge-in detected for {self.session_id}")
            return True
        return False
    
    async def handle_interruption(self, state_mgr):
        """Handle interruption logic"""
        logger.info(f"🔄 Resetting state after interruption")
        state_mgr.context.text_buffer = []
        state_mgr.context.llm_response = None
        self.interrupted = False







