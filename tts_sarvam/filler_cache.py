"""
Filler Phrase Cache for Ultra-Low Latency TTS

Pre-synthesizes common filler phrases and responses for instant playback.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from .sarvam_provider import SarvamProvider
from .audio_cache import AudioCache

logger = logging.getLogger(__name__)

class FillerPhraseCache:
    """Manages pre-synthesized filler phrases for instant playback"""

    # Common filler phrases for ultra-low latency
    FILLER_PHRASES = {
        "thinking": [
            "Let me think about that...",
            "Give me a moment...",
            "Processing your request...",
            "Let me check that for you..."
        ],
        "searching": [
            "Searching my knowledge base...",
            "Looking up the information...",
            "Let me find that for you..."
        ],
        "acknowledgment": [
            "I understand...",
            "Got it...",
            "Alright..."
        ],
        "transition": [
            "Additionally...",
            "Furthermore...",
            "Also..."
        ],
        "completion": [
            "Here's what I found...",
            "Based on the information...",
            "According to the data..."
        ]
    }

    def __init__(self, provider: SarvamProvider, cache: AudioCache, auto_preload: bool = True):
        self.provider = provider
        self.cache = cache
        self.cache_file = Path("/app/audio_cache/filler_phrases.json")
        self.phrase_cache: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False
        self.auto_preload = auto_preload

    async def initialize(self) -> bool:
        """Initialize the filler phrase cache"""
        try:
            # Load existing cache if available
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.phrase_cache = json.load(f)
                logger.info(f"✅ Loaded {len(self.phrase_cache)} cached filler phrases")

            # Pre-synthesize missing phrases if auto_preload is enabled
            if self.auto_preload:
                await self._preload_common_phrases()

            self.is_initialized = True
            logger.info(f"✅ Filler phrase cache initialized with {len(self.phrase_cache)} phrases")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize filler cache: {e}")
            return False

    async def _preload_common_phrases(self):
        """Pre-synthesize common filler phrases"""
        logger.info("🎤 Pre-synthesizing filler phrases for ultra-low latency...")

        phrases_to_synthesize = []
        for category, phrases in self.FILLER_PHRASES.items():
            for phrase in phrases:
                if phrase not in self.phrase_cache:
                    phrases_to_synthesize.append((phrase, category))

        if not phrases_to_synthesize:
            logger.info("✅ All filler phrases already cached")
            return

        logger.info(f"📝 Synthesizing {len(phrases_to_synthesize)} new filler phrases...")

        # Synthesize in batches to avoid overwhelming the API
        batch_size = 3
        for i in range(0, len(phrases_to_synthesize), batch_size):
            batch = phrases_to_synthesize[i:i+batch_size]

            tasks = []
            for phrase, category in batch:
                task = self._synthesize_and_cache_phrase(phrase, category)
                tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

            # Small delay between batches
            await asyncio.sleep(0.1)

        # Save cache to disk
        self._save_cache()
        logger.info(f"✅ Pre-synthesized {len(phrases_to_synthesize)} filler phrases")

    async def _synthesize_and_cache_phrase(self, phrase: str, category: str):
        """Synthesize a single phrase and cache it"""
        try:
            start_time = time.time()

            # Synthesize the phrase
            audio_bytes = await self.provider.synthesize(
                text=phrase,
                speaker=self.provider.speaker,
                language=self.provider.language
            )

            synthesis_time = time.time() - start_time

            # Store in cache
            cache_key = f"filler_{hash(phrase) % 10000:04d}"
            metadata = {
                "phrase": phrase,
                "category": category,
                "synthesis_time": synthesis_time,
                "cached_at": time.time(),
                "audio_length_bytes": len(audio_bytes)
            }

            # Try to cache in audio cache system
            cache_path = None
            if self.cache:
                try:
                    # Save to temp file first
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    
                    try:
                        cache_path = self.cache.cache_audio(
                            text=phrase,
                            voice=self.provider.speaker,
                            language=self.provider.language,
                            provider="sarvam",
                            emotion="neutral",
                            audio_file=tmp_path
                        )
                        metadata["cache_path"] = str(cache_path)
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                            
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save to audio cache: {e}")

            # Store metadata
            self.phrase_cache[phrase] = metadata

            logger.debug(f"✅ Cached filler phrase: '{phrase}' ({synthesis_time:.2f}s)")

        except Exception as e:
            logger.error(f"❌ Failed to synthesize filler phrase '{phrase}': {e}")

    def _save_cache(self):
        """Save cache metadata to disk"""
        try:
            self.cache_file.parent.mkdir(exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.phrase_cache, f, indent=2)
            logger.debug("💾 Saved filler phrase cache to disk")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save filler cache: {e}")

    async def get_filler_audio(self, category: str) -> Optional[bytes]:
        """Get a random filler phrase audio for the given category"""
        if not self.is_initialized:
            return None

        phrases = self.FILLER_PHRASES.get(category, [])
        if not phrases:
            return None

        # Try to get a cached phrase
        for phrase in phrases:
            if phrase in self.phrase_cache:
                metadata = self.phrase_cache[phrase]

                # Try to load from audio cache first
                if "cache_path" in metadata and self.cache:
                    try:
                        return self.cache.get_cached_audio(metadata["cache_path"])
                    except Exception:
                        pass

                # Fallback: synthesize on-demand (shouldn't happen if preloaded)
                try:
                    audio_bytes = await self.provider.synthesize(
                        text=phrase,
                        speaker=self.provider.speaker,
                        language=self.provider.language
                    )
                    logger.debug(f"⚡ Synthesized cached filler: '{phrase}'")
                    return audio_bytes
                except Exception as e:
                    logger.warning(f"❌ Failed to synthesize cached filler '{phrase}': {e}")

        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_phrases = sum(len(phrases) for phrases in self.FILLER_PHRASES.values())
        cached_phrases = len(self.phrase_cache)

        return {
            "total_phrases": total_phrases,
            "cached_phrases": cached_phrases,
            "cache_hit_rate": cached_phrases / total_phrases if total_phrases > 0 else 0,
            "cache_file": str(self.cache_file),
            "initialized": self.is_initialized
        }

    async def preload_category(self, category: str):
        """Preload all phrases for a specific category"""
        if category not in self.FILLER_PHRASES:
            logger.warning(f"⚠️ Unknown filler category: {category}")
            return

        phrases = self.FILLER_PHRASES[category]
        logger.info(f"🔄 Preloading {len(phrases)} phrases for category '{category}'...")

        tasks = []
        for phrase in phrases:
            if phrase not in self.phrase_cache:
                tasks.append(self._synthesize_and_cache_phrase(phrase, category))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self._save_cache()

        logger.info(f"✅ Category '{category}' preloaded")

