"""
Utility script to pre-synthesize all dialogue manager phrases using TTS Sarvam.

Reads language-specific JSON configs:
  - orchestrator/assets/dialogues_te.json
  - orchestrator/assets/dialogues_en.json

For each dialogue entry that defines an "audio_file", this script:
  - Calls the TTS Sarvam HTTP endpoint (/api/v1/synthesize) via the running
    TTS microservice (default: http://localhost:2005)
  - Receives base64-encoded 16-bit PCM audio
  - Wraps it in a WAV container and saves to orchestrator/assets/audio/<audio_file>

If a target WAV file already exists, it is skipped (no overwrite by default).

USAGE (from repo root):
  # Telugu / TARA mode (default)
  python -m orchestrator.generate_dialogue_audio_sarvam --lang te

  # English
  python -m orchestrator.generate_dialogue_audio_sarvam --lang en
"""

import asyncio
import base64
import json
import os
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "orchestrator" / "assets"
AUDIO_DIR = ASSETS_DIR / "audio"

# Default TTS Sarvam HTTP endpoint (container mapped: 2005 -> 8025)
DEFAULT_TTS_BASE_URL = os.getenv("DIALOGUE_TTS_URL", "http://localhost:2005")
TTS_SYNTH_ENDPOINT = f"{DEFAULT_TTS_BASE_URL}/api/v1/synthesize"


@dataclass
class DialogueTarget:
    text: str
    emotion: str
    audio_file: str

    @property
    def output_path(self) -> Path:
        return AUDIO_DIR / self.audio_file


def load_dialogue_targets(lang: str) -> List[DialogueTarget]:
    """
    Load dialogue entries that require audio for a given language.

    Args:
        lang: 'te' or 'en'
    """
    if lang.lower().startswith("te"):
        json_path = ASSETS_DIR / "dialogues_te.json"
    else:
        json_path = ASSETS_DIR / "dialogues_en.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Dialogue JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dialogues = data.get("dialogues", {})
    targets: List[DialogueTarget] = []

    # We care about any group that defines an "audio_file"
    for group_name, items in dialogues.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            audio_file = item.get("audio_file")
            if not audio_file:
                # This dialogue is text-only; nothing to pre-synthesize
                continue
            emotion = item.get("emotion") or "neutral"
            targets.append(DialogueTarget(text=text, emotion=emotion, audio_file=audio_file))

    return targets


async def synthesize_one(
    session: aiohttp.ClientSession,
    target: DialogueTarget,
    overwrite: bool = False,
) -> Tuple[DialogueTarget, bool]:
    """
    Synthesize a single dialogue entry via TTS Sarvam and write WAV file.

    Returns:
        (target, success)
    """
    out_path = target.output_path

    if out_path.exists() and not overwrite:
        print(f"[SKIP] {out_path.name} already exists")
        return target, True

    payload = {
        "text": target.text,
        "emotion": target.emotion,
        # voice/language omitted -> service uses its defaults (TARA config)
    }

    try:
        async with session.post(TTS_SYNTH_ENDPOINT, json=payload, timeout=60) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"[FAIL] {out_path.name}: HTTP {resp.status} -> {text}")
                return target, False

            data = await resp.json()
    except Exception as exc:
        print(f"[FAIL] {out_path.name}: request error: {exc}")
        return target, False

    if not data.get("success"):
        print(f"[FAIL] {out_path.name}: {data.get('error')}")
        return target, False

    audio_b64 = data.get("audio_data")
    sample_rate = data.get("sample_rate") or 22050

    if not audio_b64:
        print(f"[FAIL] {out_path.name}: no audio_data in response")
        return target, False

    try:
        pcm_bytes = base64.b64decode(audio_b64)
    except Exception as exc:
        print(f"[FAIL] {out_path.name}: base64 decode error: {exc}")
        return target, False

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap raw 16-bit PCM mono into a WAV container
    try:
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm_bytes)
    except Exception as exc:
        print(f"[FAIL] {out_path.name}: failed to write WAV: {exc}")
        return target, False

    duration_ms = len(pcm_bytes) / (int(sample_rate) * 2) * 1000.0
    print(f"[OK]   {out_path.name}  ({duration_ms:.0f} ms, sr={sample_rate})")
    return target, True


async def main_async(lang: str, overwrite: bool = False) -> None:
    targets = load_dialogue_targets(lang)
    if not targets:
        print(f"No dialogue entries with audio_file found for lang={lang}")
        return

    print(f"Using TTS endpoint: {TTS_SYNTH_ENDPOINT}")
    print(f"Found {len(targets)} dialogue entries with audio_file for lang={lang}")
    print(f"Output directory: {AUDIO_DIR}")
    print(f"Overwrite existing: {overwrite}")
    print("-" * 60)

    async with aiohttp.ClientSession() as session:
        results = []
        # Sequential for simplicity & stability; can be parallelized if needed
        for target in targets:
            res = await synthesize_one(session, target, overwrite=overwrite)
            results.append(res)

    successes = sum(1 for _t, ok in results if ok)
    failures = len(results) - successes
    print("-" * 60)
    print(f"Completed. Success: {successes}, Failed: {failures}")


def parse_args(argv: Optional[List[str]] = None) -> Tuple[str, bool]:
    """
    Minimal CLI arg parsing.

    Supported:
      --lang te|en   (default: te)
      --overwrite    (overwrite existing WAV files)
    """
    if argv is None:
        argv = sys.argv[1:]

    lang = "te"
    overwrite = False

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--lang" and i + 1 < len(argv):
            lang = argv[i + 1]
            i += 2
        elif arg == "--overwrite":
            overwrite = True
            i += 1
        else:
            print(f"Unknown argument: {arg}")
            i += 1

    return lang, overwrite


def main() -> None:
    lang, overwrite = parse_args()
    try:
        asyncio.run(main_async(lang, overwrite))
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()

