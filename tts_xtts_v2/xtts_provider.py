"""XTTS-v2 Native Streaming Provider.

Provides ultra-low latency streaming synthesis using Coqui XTTS-v2's
inference_stream API with CUDA acceleration.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

logger = logging.getLogger(__name__)


@dataclass
class XTTSProviderConfig:
    """Configuration for XTTS provider."""

    model_dir: Path
    device: str = "cuda"
    stream_chunk_tokens: int = 20
    max_buffer_chunks: int = 8


class XTTSNativeProvider:
    """Native XTTS-v2 streaming provider with speaker latent caching."""

    def __init__(self, config: XTTSProviderConfig):
        self.config = config
        self.sample_rate = 24000
        self._speaker_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._speaker_lock = asyncio.Lock()

        model_path = config.model_dir.expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"XTTS model dir not found: {model_path}. Set XTTS_MODEL_DIR env var"
            )

        logger.info("🔄 Loading XTTS-v2 native model from %s", model_path)
        xtts_config = XttsConfig()
        xtts_config.load_json(str(model_path / "config.json"))

        self.model = Xtts.init_from_config(xtts_config)
        self.model.load_checkpoint(
            xtts_config,
            checkpoint_dir=str(model_path),
            use_deepspeed=False,
            eval=True,
        )

        if config.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU")
            self.config.device = "cpu"

        self.model.to(self.config.device)
        self.model.eval()
        logger.info("✅ XTTS model ready on %s", self.config.device)

    async def synthesize_to_bytes(
        self,
        text: str,
        *,
        language: str,
        speaker_wav: str,
        stream_chunk_tokens: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> bytes:
        """Synthesize entire utterance and return PCM bytes."""

        pcm = bytearray()
        async for chunk in self.stream_text(
            text,
            language=language,
            speaker_wav=speaker_wav,
            stream_chunk_tokens=stream_chunk_tokens,
            cancel_event=cancel_event,
        ):
            pcm.extend(chunk)
        return bytes(pcm)

    async def stream_text(
        self,
        text: str,
        *,
        language: str,
        speaker_wav: str,
        stream_chunk_tokens: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> AsyncIterator[bytes]:
        """Stream audio chunks (int16 PCM bytes) for the provided text."""

        tokens_per_chunk = stream_chunk_tokens or self.config.stream_chunk_tokens
        stop_event = cancel_event or threading.Event()
        producer_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(
            maxsize=self.config.max_buffer_chunks
        )

        gpt_latent, speaker_embedding = await self._get_speaker_latents(speaker_wav)

        loop = asyncio.get_event_loop()
        first_chunk_emitted = asyncio.Event()

        def _producer():
            try:
                generator = self.model.inference_stream(
                    text,
                    language,
                    gpt_latent,
                    speaker_embedding,
                    stream_chunk_size=tokens_per_chunk,
                    enable_text_splitting=True,
                )

                for chunk in generator:
                    if stop_event.is_set():
                        break

                    chunk_array = self._tensor_to_numpy(chunk)

                    try:
                        producer_queue.put(chunk_array, timeout=0.5)
                    except queue.Full:
                        # backpressure: wait until consumer drains
                        while not stop_event.is_set():
                            try:
                                producer_queue.put(chunk_array, timeout=0.5)
                                break
                            except queue.Full:
                                continue

                    if not first_chunk_emitted.is_set():
                        loop.call_soon_threadsafe(first_chunk_emitted.set)

            except Exception as exc:  # pragma: no cover
                logger.exception("XTTS producer error: %s", exc)
                producer_queue.put(exc)
            finally:
                producer_queue.put(None)

        threading.Thread(target=_producer, daemon=True).start()

        if stop_event.is_set():
            return

        await first_chunk_emitted.wait()

        while True:
            item = await asyncio.to_thread(producer_queue.get)
            if item is None or stop_event.is_set():
                break
            if isinstance(item, Exception):
                raise item

            pcm_bytes = self._float_to_int16_bytes(item)
            yield pcm_bytes

    async def _get_speaker_latents(
        self, speaker_wav: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        speaker_wav = os.path.abspath(speaker_wav)
        if speaker_wav in self._speaker_cache:
            return self._speaker_cache[speaker_wav]

        async with self._speaker_lock:
            if speaker_wav in self._speaker_cache:
                return self._speaker_cache[speaker_wav]

            logger.info("🎙️ Computing speaker latents for %s", speaker_wav)
            gpt_latent, speaker_embedding = await asyncio.to_thread(
                self.model.get_conditioning_latents,
                audio_path=[speaker_wav],
            )
            gpt_latent = gpt_latent.to(self.config.device)
            speaker_embedding = speaker_embedding.to(self.config.device)

            self._speaker_cache[speaker_wav] = (gpt_latent, speaker_embedding)
            return gpt_latent, speaker_embedding

    @staticmethod
    def _tensor_to_numpy(chunk) -> np.ndarray:
        if isinstance(chunk, torch.Tensor):
            return chunk.detach().cpu().numpy().astype(np.float32)
        if isinstance(chunk, np.ndarray):
            return chunk.astype(np.float32)
        return np.array(chunk, dtype=np.float32)

    def _float_to_int16_bytes(self, chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        int16 = (chunk * 32767).astype(np.int16)
        return int16.tobytes()