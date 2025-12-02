"""
VAD Manager for Sarvam AI Saarika-based STT microservice.

Accumulates PCM audio chunks from WebSocket sessions, performs lightweight
energy-based voice activity detection, and forwards speech segments to Sarvam's
speech-to-text endpoint for transcription.
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

try:
    import redis.asyncio as redis  # type: ignore
except ImportError:  # pragma: no cover
    redis = None

from config import VADConfig
from sarvam_client import SarvamSTTClient, SarvamTranscriptionResult
from utils import normalize_english_transcript

logger = logging.getLogger(__name__)

StreamingCallback = Optional[Callable[[str, bool], None]]


class VADManager:
    """Stateful voice activity manager orchestrating Sarvam STT calls."""

    def __init__(
        self,
        config: VADConfig,
        redis_client: Optional[Any] = None,
        *,
        sarvam_client: Optional[SarvamSTTClient] = None,
    ):
        self.config = config
        self.redis_client = redis_client
        self.sarvam_client = sarvam_client

        self.conversation_state = "idle"
        self.is_agent_speaking = False
        self.is_listening = False

        self.consecutive_timeouts = 0
        self.capture_count = 0
        self.total_capture_time = 0.0

        self._session_states: Dict[str, Dict[str, Any]] = {}

    async def capture_speech_streaming(
        self,
        session_id: str,
        audio_queue: asyncio.Queue,
        streaming_callback: StreamingCallback = None,
    ) -> Optional[str]:
        """Capture speech from a queue and stream transcripts when ready."""

        capture_start = time.time()
        final_transcript = None
        self.is_listening = True
        self.conversation_state = "listening"

        try:
            while self.is_listening:
                try:
                    audio_chunk = await asyncio.wait_for(
                        audio_queue.get(),
                        timeout=self.config.start_timeout_s,
                    )
                except asyncio.TimeoutError:
                    logger.warning("⏱️ [%s] Capture timeout", session_id)
                    self.consecutive_timeouts += 1
                    break

                transcript = await self.process_audio_chunk_streaming(
                    session_id,
                    audio_chunk,
                    streaming_callback,
                )
                if transcript:
                    final_transcript = transcript
        finally:
            leftover = await self._flush_session(
                session_id,
                streaming_callback,
                mark_final=True,
                reason="capture_complete",
                force=True,
            )
            final_transcript = leftover or final_transcript

            self.capture_count += 1
            self.total_capture_time += time.time() - capture_start

            self.is_listening = False
            self.conversation_state = "idle"

        return final_transcript

    async def set_agent_speaking_state(self, is_speaking: bool, context: str = ""):
        """Update agent speaking flag (used for barge-in decisions)."""
        self.is_agent_speaking = is_speaking
        if self.config.log_state_transitions:
            logger.info("Agent speaking=%s context=%s", is_speaking, context)

    async def process_audio_chunk_streaming(
        self,
        session_id: str,
        audio_chunk: bytes,
        streaming_callback: StreamingCallback = None,
    ) -> Optional[str]:
        """Handle a PCM audio chunk and emit transcripts when boundaries are met."""
        if not audio_chunk:
            return None

        if not self.sarvam_client:
            logger.error("Sarvam client is not initialized")
            return None

        state = self._session_states.setdefault(session_id, self._create_session_state())
        state["buffer"].extend(audio_chunk)
        chunk_ms = (len(audio_chunk) / (self.config.sample_rate * 2)) * 1000
        state["buffer_duration_ms"] += chunk_ms

        now = time.time()
        energy = self._measure_energy(audio_chunk)

        if energy >= self.config.energy_activation:
            state["speech_active"] = True
            state["last_voice_ts"] = now

            if state["buffer_duration_ms"] >= self.config.partial_flush_ms:
                return await self._flush_session(
                    session_id,
                    streaming_callback,
                    mark_final=False,
                    reason="partial_flush",
                )
        else:
            if state["speech_active"] and state["last_voice_ts"]:
                silence_duration = now - state["last_voice_ts"]
                if silence_duration >= self.config.silence_timeout:
                    state["speech_active"] = False
                    result = await self._flush_session(
                        session_id,
                        streaming_callback,
                        mark_final=True,
                        reason="silence",
                    )
                    if result:
                        return result

        if state["buffer_duration_ms"] >= self.config.max_buffer_ms:
            return await self._flush_session(
                session_id,
                streaming_callback,
                mark_final=False,
                reason="max_buffer",
            )

        return None

    async def _flush_session(
        self,
        session_id: str,
        streaming_callback: StreamingCallback,
        *,
        mark_final: bool,
        reason: str,
        force: bool = False,
    ) -> Optional[str]:
        state = self._session_states.get(session_id)
        if not state or not state["buffer"]:
            return None

        if not force and state["buffer_duration_ms"] < self.config.min_speech_ms:
            # Not enough context to call STT yet
            self._reset_buffer(state)
            return None

        audio_bytes = bytes(state["buffer"])
        self._reset_buffer(state)

        sarvam_language = self._resolve_language_code()
        result = await self.sarvam_client.transcribe(
            audio_bytes,
            language_code=sarvam_language,
        )
        if not result or not result.text:
            logger.debug("[%s] No transcript returned (%s)", session_id, reason)
            return None

        normalized = self._normalize_transcript(result)
        await self._emit_transcript(
            session_id,
            normalized,
            streaming_callback,
            is_final=mark_final,
            language_code=result.language_code,
        )

        if mark_final:
            self._session_states.pop(session_id, None)
            self.consecutive_timeouts = 0

        return normalized

    def reset_streaming_state(self):
        logger.info("Resetting Sarvam streaming state")
        self._session_states.clear()
        self.consecutive_timeouts = 0

    def get_performance_metrics(self) -> Dict[str, Any]:
        avg_capture_ms = (
            (self.total_capture_time / self.capture_count) * 1000
            if self.capture_count
            else 0.0
        )
        return {
            "total_captures": self.capture_count,
            "avg_capture_time_ms": round(avg_capture_ms, 2),
            "consecutive_timeouts": self.consecutive_timeouts,
        }

    def _create_session_state(self) -> Dict[str, Any]:
        return {
            "buffer": bytearray(),
            "buffer_duration_ms": 0.0,
            "speech_active": False,
            "last_voice_ts": 0.0,
        }

    def _reset_buffer(self, state: Dict[str, Any]):
        state["buffer"] = bytearray()
        state["buffer_duration_ms"] = 0.0

    def _measure_energy(self, audio_chunk: bytes) -> float:
        if not audio_chunk:
            return 0.0
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        if audio_array.size == 0:
            return 0.0
        return float(np.mean(np.abs(audio_array)))

    def _resolve_language_code(self) -> str:
        if self.config.enable_language_detection or self.config.language_code == "unknown":
            return "unknown"
        return self.config.language_code

    def _normalize_transcript(self, result: SarvamTranscriptionResult) -> str:
        text = (result.text or "").strip()
        if not text:
            return ""
        language_code = (result.language_code or "").lower()
        if language_code.startswith("en"):
            return normalize_english_transcript(text)
        return text

    async def _emit_transcript(
        self,
        session_id: str,
        text: str,
        streaming_callback: StreamingCallback,
        *,
        is_final: bool,
        language_code: Optional[str],
    ):
        if streaming_callback:
            try:
                streaming_callback(text, is_final)
            except Exception as callback_error:  # pragma: no cover
                logger.error("Streaming callback error: %s", callback_error)

        if not is_final or not text:
            return

        if self.redis_client:
            try:
                event_data = {
                    "text": text,
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "language_code": language_code,
                    "source": "stt-sarvam",
                }
                await self.redis_client.publish(
                    "leibniz:events:stt",
                    json.dumps(event_data),
                )
                logger.info("📢 Published transcript to Redis channel 'leibniz:events:stt'")
            except Exception as err:  # pragma: no cover
                logger.warning("Failed to publish transcript to Redis: %s", err)
