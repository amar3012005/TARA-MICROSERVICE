"""
TTS Audio Cache Module

Extracted from leibniz_tts.py (TTSCache class, lines 254-471) for microservice deployment.

Implements MD5-based caching with:
    - MD5 cache keys from synthesis parameters
    - JSON cache index with metadata
    - LRU cleanup (500 entries max by default)
    - Atomic file writes with Windows retry logic
    - Cache statistics tracking

Reference:
    leibniz_agent/leibniz_tts.py - Original TTSCache class (lines 254-471)
"""

import os
import json
import time
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Cache constants
CACHE_INDEX_FILE = "cache_index.json"


class AudioCache:
    """
    TTS audio cache with MD5-based keys and LRU cleanup.
    
    Features:
        - MD5 cache keys from synthesis parameters (text+voice+language+provider+emotion)
        - JSON cache index with metadata (text, voice, created, last_accessed, file_size)
        - LRU cleanup: Remove oldest when size > max_size
        - Atomic file writes with Windows retry logic
        - Cache statistics: hits, misses, hit_rate
    
    Cache Structure:
        cache_dir/
            ├── cache_index.json - Metadata for all cached files
            ├── {md5_hash_1}.wav - Cached audio file
            ├── {md5_hash_2}.wav - Cached audio file
            └── ...
    
    Cache Key Generation:
        MD5(text + voice + language + provider + emotion)
        Example: MD5("Hello world_en-US-Neural2-F_en-US_google_helpful")
    
    LRU Cleanup:
        When cache size exceeds max_size:
        1. Sort entries by last_accessed time (oldest first)
        2. Remove oldest entries until size <= max_size
        3. Delete both cache file and index entry
    """
    
    def __init__(self, cache_dir: str, max_size: int = 500):
        """
        Initialize TTS audio cache.
        
        Args:
            cache_dir: Directory for cached audio files
            max_size: Maximum number of cached entries (default: 500)
        """
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.cache_index: Dict[str, Dict[str, Any]] = {}
        self.index_file = self.cache_dir / CACHE_INDEX_FILE
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cache index
        self._load_cache_index()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
        logger.info(f" Audio cache initialized: {len(self.cache_index)} entries, max_size={max_size}")
    
    def _load_cache_index(self):
        """
        Load cache index from JSON file.
        
        Reads cache_index.json and populates self.cache_index.
        If file doesn't exist or is corrupted, starts with empty index.
        """
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.cache_index = json.load(f)
                logger.info(f" Loaded TTS cache index: {len(self.cache_index)} entries")
            except Exception as e:
                logger.warning(f"️ Failed to load cache index: {e}. Starting with empty cache.")
                self.cache_index = {}
        else:
            self.cache_index = {}
            logger.info(" No existing cache index found. Starting fresh.")
    
    def _save_cache_index(self):
        """
        Save cache index to JSON file.
        
        Writes self.cache_index to cache_index.json with atomic write pattern
        (write to temp file, then rename to avoid corruption).
        """
        try:
            # Atomic write: temp file then rename
            temp_file = self.index_file.with_suffix('.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, indent=2)
            
            # Rename (atomic on most systems)
            # Windows may require retry if file locked
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    temp_file.replace(self.index_file)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
                    else:
                        raise
        
        except Exception as e:
            logger.error(f"️ Failed to save cache index: {e}")
    
    def get_cache_key(
        self,
        text: str,
        voice: str,
        language: str,
        provider: str,
        emotion: str = "neutral"
    ) -> str:
        """
        Generate MD5 cache key from synthesis parameters.
        
        Cache key includes all parameters that affect audio output:
        - text: The text to synthesize
        - voice: Voice name/ID (provider-specific)
        - language: Language code (e.g., "en-US")
        - provider: Provider name (google, elevenlabs, gemini, xtts_local, mock)
        - emotion: Emotion descriptor (helpful, excited, calm, etc.)
        
        Args:
            text: Text to synthesize
            voice: Voice name/ID
            language: Language code
            provider: Provider name
            emotion: Emotion type
        
        Returns:
            MD5 hash string (32 hex characters)
        
        Example:
            >>> cache.get_cache_key("Hello", "en-US-Neural2-F", "en-US", "google", "helpful")
            "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        """
        cache_string = f"{text}_{voice}_{language}_{provider}_{emotion}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_cached_audio(
        self,
        text: str,
        voice: str,
        language: str,
        provider: str,
        emotion: str = "neutral"
    ) -> Optional[str]:
        """
        Get cached audio file path if exists.
        
        Checks:
        1. Cache key exists in index
        2. Audio file exists on disk
        3. Updates last_accessed timestamp
        
        Args:
            text: Text to synthesize
            voice: Voice name/ID
            language: Language code
            provider: Provider name
            emotion: Emotion type
        
        Returns:
            Path to cached audio file (str) or None if not cached
        """
        cache_key = self.get_cache_key(text, voice, language, provider, emotion)
        
        # Check if in index
        if cache_key not in self.cache_index:
            self.misses += 1
            return None
        
        # Check if file exists
        cache_file = self.cache_dir / f"{cache_key}.wav"
        if not cache_file.exists():
            # Remove from index if file missing
            logger.warning(f"️ Cache file missing for key {cache_key}. Removing from index.")
            del self.cache_index[cache_key]
            self._save_cache_index()
            self.misses += 1
            return None
        
        # Update last accessed time
        self.cache_index[cache_key]['last_accessed'] = time.time()
        self._save_cache_index()
        
        self.hits += 1
        logger.debug(f" Cache HIT: {cache_key} ({text[:50]}...)")
        return str(cache_file)
    
    def cache_audio(
        self,
        text: str,
        voice: str,
        language: str,
        provider: str,
        emotion: str,
        audio_file: str
    ) -> str:
        """
        Cache audio file with metadata.
        
        Process:
        1. Generate cache key
        2. Copy audio file to cache directory
        3. Update cache index with metadata
        4. Perform LRU cleanup if over max_size
        5. Save index to disk
        
        Args:
            text: Text to synthesize
            voice: Voice name/ID
            language: Language code
            provider: Provider name
            emotion: Emotion type
            audio_file: Path to audio file to cache
        
        Returns:
            Path to cached file (str)
        
        Raises:
            Exception: If file copy fails
        """
        cache_key = self.get_cache_key(text, voice, language, provider, emotion)
        cache_file = self.cache_dir / f"{cache_key}.wav"
        
        # Copy file to cache
        try:
            shutil.copy2(audio_file, cache_file)
        except Exception as e:
            logger.error(f" Failed to cache audio file: {e}")
            raise
        
        # Update index
        self.cache_index[cache_key] = {
            'text': text[:100],  # Store first 100 chars only
            'voice': voice,
            'language': language,
            'provider': provider,
            'emotion': emotion,
            'created': time.time(),
            'last_accessed': time.time(),
            'file_size': cache_file.stat().st_size
        }
        
        # Cleanup if over max size
        if len(self.cache_index) > self.max_size:
            self._cleanup_cache()
        
        # Save index
        self._save_cache_index()
        
        logger.debug(f" Cached audio: {cache_key} ({text[:50]}...) - {cache_file.stat().st_size} bytes")
        return str(cache_file)
    
    def _cleanup_cache(self):
        """
        Cleanup old cache entries using LRU (Least Recently Used).
        
        Process:
        1. Sort entries by last_accessed time (oldest first)
        2. Calculate how many entries to remove (current_size - max_size)
        3. Delete oldest entries (both file and index entry)
        4. Log cleanup summary
        """
        # Sort by last accessed time (oldest first)
        sorted_entries = sorted(
            self.cache_index.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Remove oldest entries
        entries_to_remove = len(sorted_entries) - self.max_size
        if entries_to_remove > 0:
            for cache_key, _ in sorted_entries[:entries_to_remove]:
                # Delete file
                cache_file = self.cache_dir / f"{cache_key}.wav"
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"️ Failed to delete cache file {cache_key}: {e}")
                
                # Remove from index
                del self.cache_index[cache_key]
            
            logger.info(f" Cleaned up {entries_to_remove} old cache entries (LRU)")
    
    def clear_cache(self):
        """
        Clear all cached audio files.
        
        Deletes:
        - All .wav files in cache directory
        - All entries from cache index
        - Resets statistics (hits, misses)
        """
        cleared_count = 0
        
        for cache_key in list(self.cache_index.keys()):
            cache_file = self.cache_dir / f"{cache_key}.wav"
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except Exception as e:
                    logger.warning(f"️ Failed to delete cache file {cache_key}: {e}")
        
        self.cache_index = {}
        self._save_cache_index()
        self.hits = 0
        self.misses = 0
        
        logger.info(f" Cache cleared: {cleared_count} files deleted")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics:
                - size: Current number of cached entries
                - max_size: Maximum allowed entries
                - hits: Number of cache hits
                - misses: Number of cache misses
                - hit_rate: Ratio of hits to total requests (0.0-1.0)
                - total_requests: Total cache lookups (hits + misses)
        
        Example:
            >>> cache.get_stats()
            {
                'size': 247,
                'max_size': 500,
                'hits': 1832,
                'misses': 468,
                'hit_rate': 0.7965,
                'total_requests': 2300
            }
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache_index),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
