"""
Sarvam AI Saarika STT Client

Wraps Sarvam AI's speech-to-text REST API with async-friendly helpers.
Reference: https://docs.sarvam.ai/api-reference-docs/getting-started/models/saarika
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SarvamTranscriptionResult:
    """
    Normalized transcription payload returned by SarvamSTTClient.
    """

    text: str
    language_code: str
    raw_response: Dict[str, Any]


class SarvamSTTClient:
    """
    Thin async wrapper around Sarvam AI's speech-to-text endpoint.

    Converts raw PCM audio into WAV, forwards the request, and keeps basic
    observability metrics so /metrics and /health can report client status.
    """

    DEFAULT_ENDPOINT = "https://api.sarvam.ai/speech-to-text"

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        sample_rate: int,
        channels: int = 1,
        *,
        timeout_seconds: float = 45.0,
        endpoint: Optional[str] = None,
    ):
        self._mock_mode = not bool(api_key)
        if self._mock_mode:
            logger.warning(
                "SARVAM_API_SUBSCRIPTION_KEY is not set. "
                "SarvamSTTClient will return mock transcripts."
            )

        self.api_key = api_key or "mock-api-key"
        self.model = model
        self.sample_rate = sample_rate
        self.channels = channels
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT

        timeout = httpx.Timeout(timeout_seconds, read=timeout_seconds)
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
        self._client = httpx.AsyncClient(timeout=timeout, limits=limits)

        self._stats = {
            "total_requests": 0,
            "total_failures": 0,
            "total_latency_ms": 0.0,
            "last_latency_ms": 0.0,
            "last_status": "idle",
            "last_error": None,
            "last_language": None,
            "last_request_id": None,
        }
        self._lock = asyncio.Lock()

    async def aclose(self):
        await self._client.aclose()

    def get_stats(self) -> Dict[str, Any]:
        avg_latency = (
            self._stats["total_latency_ms"] / self._stats["total_requests"]
            if self._stats["total_requests"]
            else 0.0
        )
        return {
            "total_requests": self._stats["total_requests"],
            "total_failures": self._stats["total_failures"],
            "last_latency_ms": round(self._stats["last_latency_ms"], 2),
            "avg_latency_ms": round(avg_latency, 2),
            "last_status": self._stats["last_status"],
            "last_error": self._stats["last_error"],
            "last_language": self._stats["last_language"],
            "last_request_id": self._stats["last_request_id"],
        }

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        language_code: str = "unknown",
    ) -> Optional[SarvamTranscriptionResult]:
        """
        Send audio to Sarvam AI for transcription.

        Args:
            audio_bytes: Raw 16-bit PCM audio.
            language_code: Language hint. Use "unknown" to enable automatic LID.
        """
        if not audio_bytes:
            logger.warning("SarvamSTTClient.transcribe called with empty audio payload")
            return None

        if self._mock_mode:
            mock_text = "Mock Sarvam transcription response (no API key configured)."
            await self._update_stats(
                success=True,
                latency_ms=1.0,
                language_code=language_code or "en-IN",
                request_id="mock-request",
            )
            return SarvamTranscriptionResult(
                text=mock_text,
                language_code=language_code or "en-IN",
                raw_response={"mock": True, "text": mock_text},
            )

        wav_bytes = self._pcm_to_wav(audio_bytes)
        files = {"file": ("capture.wav", wav_bytes, "audio/wav")}
        data = {"model": self.model, "language_code": language_code or "unknown"}
        headers = {"api-subscription-key": self.api_key}

        start = time.perf_counter()
        try:
            response = await self._client.post(self.endpoint, data=data, files=files, headers=headers)
            latency_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            payload = response.json()

            transcript = (payload.get("transcript") or "").strip()
            detected_lang = payload.get("language_code") or language_code or "unknown"

            await self._update_stats(
                success=True,
                latency_ms=latency_ms,
                language_code=detected_lang,
                request_id=payload.get("request_id"),
            )

            if not transcript:
                logger.info("Sarvam response did not contain transcript text")
                return None

            return SarvamTranscriptionResult(
                text=transcript,
                language_code=detected_lang,
                raw_response=payload,
            )

        except httpx.HTTPStatusError as http_err:
            await self._update_stats(success=False, latency_ms=0.0, error=str(http_err))
            logger.error(
                "Sarvam transcription failed | status=%s | detail=%s",
                http_err.response.status_code if http_err.response else "unknown",
                http_err.response.text if http_err.response else "",
            )
        except Exception as exc:
            await self._update_stats(success=False, latency_ms=0.0, error=str(exc))
            logger.exception("Unexpected Sarvam transcription failure: %s", exc)

        return None

    async def _update_stats(
        self,
        *,
        success: bool,
        latency_ms: float,
        language_code: Optional[str] = None,
        request_id: Optional[str] = None,
        error: Optional[str] = None,
    ):
        async with self._lock:
            self._stats["total_requests"] += 1
            if success:
                self._stats["total_latency_ms"] += latency_ms
                self._stats["last_latency_ms"] = latency_ms
                self._stats["last_status"] = "success"
                self._stats["last_language"] = language_code
                self._stats["last_request_id"] = request_id
                self._stats["last_error"] = None
            else:
                self._stats["total_failures"] += 1
                self._stats["last_status"] = "error"
                self._stats["last_error"] = error

    def _pcm_to_wav(self, audio_bytes: bytes) -> bytes:
        """Convert raw PCM bytes (16-bit mono) to WAV for Sarvam API."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_bytes)
        buffer.seek(0)
        return buffer.read()

