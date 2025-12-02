#!/usr/bin/env python3
"""Simple microphone-driven Whisper transcription demo.

Streams audio from the local microphone into the STT Local pipeline and prints
partial and final transcripts in the terminal. Press Ctrl+C to stop.
"""

import argparse
import asyncio
import signal
import time
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "sounddevice is required for microphone streaming. Install it via "
        "`pip install sounddevice` before running this script."
    ) from exc

from config import STTLocalConfig
from stt_manager import STTManager


def _print_devices() -> None:
    """List available audio input devices."""
    devices = sd.query_devices()
    print("\nAvailable audio input devices:\n" + "-" * 60)
    for idx, info in enumerate(devices):
        if info.get("max_input_channels", 0) > 0:
            default = " (default)" if idx == sd.default.device[0] else ""
            print(f"[{idx:>2}] {info['name']}{default}")
            print(f"     max_channels={info['max_input_channels']}, "
                  f"default_samplerate={info['default_samplerate']:.0f} Hz")
    print("-" * 60)


def _configure_device(device: Optional[int]) -> None:
    """Set the default input device if provided."""
    if device is not None:
        sd.default.device = (device, None)


def _ensure_cuda_compat(config: STTLocalConfig) -> STTLocalConfig:
    """Ensure config matches actual CUDA availability."""
    try:
        import torch
    except ImportError:
        config.whisper_device = "cpu"
        config.whisper_compute_type = "float32"
        config.use_gpu = False
        return config

    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1).cuda()
            del _
            torch.cuda.empty_cache()
            return config
        except (SystemError, OSError, RuntimeError):
            pass

    config.whisper_device = "cpu"
    config.whisper_compute_type = "float32"
    config.use_gpu = False
    return config


def _pcm_bytes_from_chunk(chunk: np.ndarray) -> bytes:
    """Convert float32 audio chunk to 16-bit PCM bytes."""
    mono = chunk if chunk.ndim == 1 else chunk[:, 0]
    clipped = np.clip(mono, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream microphone audio through the STT Local pipeline."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index (see --list-devices)."
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Chunk size in milliseconds (default: 100)."
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available devices and exit."
    )
    return parser


async def _stream_microphone(config: STTLocalConfig, device: Optional[int], chunk_ms: int) -> None:
    """Capture microphone audio and feed it into STTManager."""
    config = _ensure_cuda_compat(config)
    manager = STTManager(config, redis_client=None)
    session_id = f"local_mic_{int(time.time())}"

    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
    stop_event = asyncio.Event()

    def streaming_callback(text: str, is_final: bool) -> None:
        tag = "FINAL" if is_final else "PARTIAL"
        print(f"[{tag}] {text}")

    def audio_callback(indata, _frames, _time_info, status):
        if status:
            print(f"[AUDIO WARNING] {status}")
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())

    _configure_device(device)
    samplerate = config.sample_rate
    blocksize = max(int(samplerate * (chunk_ms / 1000.0)), 1)

    def _stop_handler(_sig, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    print("\n🎙️  Microphone streaming started. Speak into the mic...")
    print("    Press Ctrl+C to stop.\n")

    with sd.InputStream(
        samplerate=samplerate,
        blocksize=blocksize,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    ):
        while not stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if chunk.size == 0:
                continue

            pcm_bytes = _pcm_bytes_from_chunk(chunk)
            await manager.process_audio_chunk_streaming(
                session_id=session_id,
                audio_chunk=pcm_bytes,
                streaming_callback=streaming_callback,
            )

    print("\n🛑 Streaming stopped. Goodbye!\n")


async def _async_main(args: argparse.Namespace) -> None:
    config = STTLocalConfig.from_env()
    if args.list_devices:
        _print_devices()
        return
    await _stream_microphone(config, args.device, args.chunk_ms)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
