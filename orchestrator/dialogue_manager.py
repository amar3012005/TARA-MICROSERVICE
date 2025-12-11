"""
Dialogue Manager for pre-defined audio/text dialogues

Manages intro, exit, filler phrases, timeout prompts, and unclear prompts with
support for pre-synthesized audio files and emotional context, driven by
language-specific JSON configuration.
"""

import os
import json
import random
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DialogueType(Enum):
    """Types of pre-defined dialogues"""
    INTRO = "intro"
    EXIT = "exit"
    FILLER_IMMEDIATE = "filler_immediate"
    FILLER_LATENCY = "filler_latency"
    TIMEOUT = "timeout"
    UNCLEAR = "unclear"


@dataclass
class DialogueAsset:
    """Represents a dialogue asset with text, emotion, and optional audio"""
    text: str
    emotion: str = "helpful"
    audio_path: Optional[str] = None
    trigger: Optional[str] = None  # e.g., 'immediate', 'delay_ms:1500', 'timeout_ms:10000'
    keywords: Optional[List[str]] = None

    def has_audio(self) -> bool:
        """Check if audio file exists"""
        if not self.audio_path:
            return False
        return os.path.exists(self.audio_path)


class DialogueManager:
    """
    Manages pre-defined dialogues for intro, exit, fillers (immediate/latency),
    timeout, and unclear scenarios.

    Configuration is loaded from language-specific JSON files:
    - orchestrator/assets/dialogues_en.json
    - orchestrator/assets/dialogues_te.json

    Supports both pre-synthesized audio files and text-to-speech fallback.
    """

    def __init__(self, assets_dir: Optional[str] = None, tara_mode: bool = True):
        """
        Initialize Dialogue Manager

        Args:
            assets_dir: Base assets directory (default: orchestrator/assets/)
            tara_mode: If True, uses Telugu config; else English
        """
        # Determine base assets directory
        if assets_dir:
            self.base_assets_dir = Path(assets_dir)
        else:
            # Default to orchestrator/assets/
            script_dir = Path(__file__).parent
            self.base_assets_dir = script_dir / "assets"

        # Audio assets live under assets/audio
        self.audio_dir = self.base_assets_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.tara_mode = tara_mode
        self.language = "te" if tara_mode else "en"
        self._dialogue_cache: Dict[DialogueType, List[DialogueAsset]] = {}
        self.exit_keywords: List[str] = []

        # Load dialogue configurations (JSON first, fallback to legacy defaults)
        self._load_dialogues()

        logger.info(f"DialogueManager initialized | TARA mode: {tara_mode} | lang: {self.language}")
        logger.info(f"  Assets directory: {self.base_assets_dir}")
        logger.info(f"  Audio directory:  {self.audio_dir}")

    # ------------------------------------------------------------------
    # Loading configuration
    # ------------------------------------------------------------------
    def _load_dialogues(self):
        """Load dialogue configurations from JSON, with safe fallback."""
        json_loaded = self._load_from_json()
        if not json_loaded:
            logger.warning("Dialogue JSON not found or invalid. Falling back to built-in defaults.")
            self._load_legacy_defaults()

        # Log loaded dialogues
        for dtype, assets in self._dialogue_cache.items():
            audio_count = sum(1 for a in assets if a.has_audio())
            logger.info(f"  Loaded {len(assets)} {dtype.value} dialogue(s) ({audio_count} with audio files)")

    def _load_from_json(self) -> bool:
        """Attempt to load dialogues from JSON configuration."""
        # Pick JSON file based on language / TARA mode
        config_filename = "dialogues_te.json" if self.tara_mode else "dialogues_en.json"
        config_path = self.base_assets_dir / config_filename

        if not config_path.exists():
            logger.warning(f"Dialogue config JSON not found at {config_path}")
            return False

        try:
            with config_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            logger.error(f"Failed to load dialogue config JSON {config_path}: {exc}")
            return False

        dialogues: Dict[str, Any] = raw.get("dialogues", {})

        def _load_group(key: str, dtype: DialogueType):
            items = dialogues.get(key, [])
            assets: List[DialogueAsset] = []
            for item in items:
                text = item.get("text", "").strip()
                if not text:
                    continue
                emotion = item.get("emotion", "neutral")
                audio_file = item.get("audio_file")
                audio_path = str(self.audio_dir / audio_file) if audio_file else None
                trigger = item.get("trigger")
                keywords = item.get("keywords")
                # Normalize keywords to lowercase list if present
                if keywords:
                    keywords = [str(k).strip().lower() for k in keywords if str(k).strip()]
                assets.append(
                    DialogueAsset(
                        text=text,
                        emotion=emotion,
                        audio_path=audio_path,
                        trigger=trigger,
                        keywords=keywords,
                    )
                )
            if assets:
                self._dialogue_cache[dtype] = assets

        # Map JSON keys to DialogueType
        _load_group("intro", DialogueType.INTRO)
        _load_group("exit", DialogueType.EXIT)
        _load_group("filler_immediate", DialogueType.FILLER_IMMEDIATE)
        _load_group("filler_latency", DialogueType.FILLER_LATENCY)
        _load_group("timeout", DialogueType.TIMEOUT)
        _load_group("unclear", DialogueType.UNCLEAR)

        # Collect exit keywords from exit group
        exit_assets = self._dialogue_cache.get(DialogueType.EXIT, [])
        keywords: List[str] = []
        for asset in exit_assets:
            if asset.keywords:
                keywords.extend(asset.keywords)
        # Deduplicate
        self.exit_keywords = sorted(set(k.lower() for k in keywords))

        return True

    def _load_legacy_defaults(self):
        """Fallback hardcoded configuration (kept for robustness)."""
        if self.tara_mode:
            # Telugu defaults
            self._dialogue_cache[DialogueType.INTRO] = [
                DialogueAsset(
                    text="నమస్కారం అండి! నేను TARA, TASK యొక్క కస్టమర్ సర్వీస్ ఏజెంట్. మీకు ఎలా సహాయం చేయగలను?",
                    emotion="warm",
                    audio_path=str(self.audio_dir / "intro.wav"),
                )
            ]

            self._dialogue_cache[DialogueType.EXIT] = [
                DialogueAsset(
                    text="ధన్యవాదాలు! మీకు మరో సహాయం కావాలంటే, మళ్లీ కాల్ చేయండి. మంచి రోజు!",
                    emotion="grateful",
                    audio_path=str(self.audio_dir / "exit.wav"),
                ),
                DialogueAsset(
                    text="సంతోషంగా! మీకు మరో సహాయం కావాలంటే, మాకు తెలపండి. శుభం!",
                    emotion="friendly",
                    audio_path=str(self.audio_dir / "exit_alt.wav"),
                ),
            ]

            self._dialogue_cache[DialogueType.FILLER_IMMEDIATE] = [
                DialogueAsset(text="సరే.", emotion="neutral"),
                DialogueAsset(text="అలాగే.", emotion="neutral"),
            ]

            self._dialogue_cache[DialogueType.FILLER_LATENCY] = [
                DialogueAsset(
                    text="చూస్తున్నాను, ఒక్క క్షణం...", emotion="thinking", audio_path=str(self.audio_dir / "filler_thinking.wav")
                )
            ]

            self._dialogue_cache[DialogueType.TIMEOUT] = [
                DialogueAsset(
                    text="మీరు అక్కడ ఉన్నారా? నేను మీకు సహాయం చేయగలను.",
                    emotion="concerned",
                    audio_path=str(self.audio_dir / "timeout.wav"),
                )
            ]

            self._dialogue_cache[DialogueType.UNCLEAR] = [
                DialogueAsset(text="క్షమించండి, బాగా వినపడలేదు. మరోసారి చెప్పగలరా?", emotion="neutral")
            ]
        else:
            # English defaults
            self._dialogue_cache[DialogueType.INTRO] = [
                DialogueAsset(
                    text="Hello! I'm TARA, the customer service agent for TASK. How can I help you today?",
                    emotion="warm",
                    audio_path=str(self.audio_dir / "intro.wav"),
                )
            ]

            self._dialogue_cache[DialogueType.EXIT] = [
                DialogueAsset(
                    text="Thank you! If you need any more assistance, please call again. Have a great day!",
                    emotion="grateful",
                    audio_path=str(self.audio_dir / "exit.wav"),
                ),
                DialogueAsset(
                    text="Glad I could help! If you need anything else, just let us know. Take care!",
                    emotion="friendly",
                    audio_path=str(self.audio_dir / "exit_alt.wav"),
                ),
            ]

            self._dialogue_cache[DialogueType.FILLER_IMMEDIATE] = [
                DialogueAsset(text="Okay.", emotion="neutral"),
                DialogueAsset(text="Got it.", emotion="neutral"),
            ]

            self._dialogue_cache[DialogueType.FILLER_LATENCY] = [
                DialogueAsset(
                    text="Let me think about that...",
                    emotion="thinking",
                    audio_path=str(self.audio_dir / "filler_thinking.wav"),
                )
            ]

            self._dialogue_cache[DialogueType.TIMEOUT] = [
                DialogueAsset(
                    text="Are you still there? I'm here to help you.",
                    emotion="concerned",
                    audio_path=str(self.audio_dir / "timeout.wav"),
                )
            ]

            self._dialogue_cache[DialogueType.UNCLEAR] = [
                DialogueAsset(text="Sorry, I didn't catch that clearly. Could you say it again?", emotion="neutral")
            ]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_asset(self, dialogue_type: DialogueType) -> DialogueAsset:
        """
        Get a dialogue asset of the specified type.
        
        Args:
            dialogue_type: Type of dialogue to retrieve
            
        Returns:
            DialogueAsset with text, emotion, and optional audio path
        """
        assets = self._dialogue_cache.get(dialogue_type, [])
        if not assets:
            # Fallback to default text
            logger.warning(f"No {dialogue_type.value} dialogue found, using fallback")
            return DialogueAsset(
                text=f"[{dialogue_type.value} dialogue]",
                emotion="neutral"
            )
        
        # Default behavior: pick first asset (deterministic for intro/timeout)
        return assets[0]

    def get_random_exit(self) -> DialogueAsset:
        """Get a random exit dialogue."""
        exits = self._dialogue_cache.get(DialogueType.EXIT, [])
        if not exits:
            logger.warning("No exit dialogues found, using fallback")
            return DialogueAsset(text="Thank you! Goodbye!", emotion="grateful")
        return random.choice(exits)

    def get_immediate_filler(self) -> DialogueAsset:
        """Get an immediate filler (acknowledge right after user stops)."""
        fillers = self._dialogue_cache.get(DialogueType.FILLER_IMMEDIATE, [])
        if not fillers:
            logger.warning("No immediate fillers found, using fallback")
            return DialogueAsset(text="Okay.", emotion="neutral")
        return random.choice(fillers)

    def get_latency_filler(self) -> DialogueAsset:
        """Get a latency-covering filler (after 1.5–2s)."""
        fillers = self._dialogue_cache.get(DialogueType.FILLER_LATENCY, [])
        if not fillers:
            logger.warning("No latency fillers found, using fallback")
            return DialogueAsset(text="Let me check that for you...", emotion="thinking")
        return random.choice(fillers)

    def get_timeout_prompt(self) -> DialogueAsset:
        """Get a timeout prompt (no user input for N seconds)."""
        timeouts = self._dialogue_cache.get(DialogueType.TIMEOUT, [])
        if not timeouts:
            logger.warning("No timeout prompts found, using fallback")
            return DialogueAsset(text="Are you still there?", emotion="concerned")
        return random.choice(timeouts)

    def get_unclear_prompt(self) -> DialogueAsset:
        """Get an 'unclear input' prompt."""
        unclear = self._dialogue_cache.get(DialogueType.UNCLEAR, [])
        if not unclear:
            logger.warning("No unclear prompts found, using fallback")
            return DialogueAsset(text="Sorry, I didn't catch that. Could you repeat?", emotion="neutral")
        return random.choice(unclear)

    def add_custom_dialogue(self, dialogue_type: DialogueType, asset: DialogueAsset):
        """Add a custom dialogue asset (for runtime configuration)."""
        if dialogue_type not in self._dialogue_cache:
            self._dialogue_cache[dialogue_type] = []
        self._dialogue_cache[dialogue_type].append(asset)
        logger.info(f"Added custom {dialogue_type.value} dialogue: {asset.text[:50]}...")




