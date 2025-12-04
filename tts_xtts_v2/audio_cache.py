"""
Audio Cache

MD5-based caching with LRU cleanup and TTL enforcement.
Simplified from leibniz_tts.py (lines 182-418).
"""

import json
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

CACHE_INDEX_FILE = "cache_index.json"
CACHE_TTL_DAYS = 30


class AudioCache:
    """
    TTS cache with MD5-based keys and LRU cleanup.
    """
    
    def __init__(self, cache_dir: str, max_size: int = 500):
        """
        Initialize audio cache.
        
        Args:
            cache_dir: Directory for cached audio files
            max_size: Maximum number of cached entries
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
    
    def _load_cache_index(self):
        """Load cache index from JSON file"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.cache_index = json.load(f)
                logger.info(f"Loaded TTS cache index: {len(self.cache_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self.cache_index = {}
        else:
            self.cache_index = {}
    
    def _save_cache_index(self):
        """Save cache index to JSON file"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
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
        
        Args:
            text: Text to synthesize
            voice: Voice name/ID
            language: Language code
            provider: Provider name
            emotion: Emotion type
            
        Returns:
            MD5 hash string
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
        Get cached audio file if exists.
        
        Args:
            text: Text to synthesize
            voice: Voice name/ID
            language: Language code
            provider: Provider name
            emotion: Emotion type
            
        Returns:
            Path to cached file or None
        """
        cache_key = self.get_cache_key(text, voice, language, provider, emotion)
        
        # Check if in index
        if cache_key not in self.cache_index:
            self.misses += 1
            return None
        
        entry = self.cache_index[cache_key]
        
        # Check TTL enforcement - remove expired entries
        current_time = time.time()
        if 'created' in entry:
            age_days = (current_time - entry['created']) / (24 * 3600)
            if age_days > CACHE_TTL_DAYS:
                # Remove expired entry
                cache_file = self.cache_dir / f"{cache_key}.wav"
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass  # Best effort cleanup
                
                del self.cache_index[cache_key]
                self._save_cache_index()
                self.misses += 1
                return None
        
        # Check if file exists
        cache_file = self.cache_dir / f"{cache_key}.wav"
        if not cache_file.exists():
            # Remove from index if file missing
            del self.cache_index[cache_key]
            self._save_cache_index()
            self.misses += 1
            return None
        
        # Update last accessed time
        self.cache_index[cache_key]['last_accessed'] = current_time
        self._save_cache_index()
        
        self.hits += 1
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
        Cache audio file.
        
        Args:
            text: Text to synthesize
            voice: Voice name/ID
            language: Language code
            provider: Provider name
            emotion: Emotion type
            audio_file: Path to audio file to cache
            
        Returns:
            Path to cached file
        """
        cache_key = self.get_cache_key(text, voice, language, provider, emotion)
        cache_file = self.cache_dir / f"{cache_key}.wav"
        
        # Copy file to cache
        import shutil
        shutil.copy2(audio_file, cache_file)
        
        # Update index
        self.cache_index[cache_key] = {
            'text': text[:100],  # Store first 100 chars
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
        
        return str(cache_file)
    
    def _cleanup_cache(self):
        """Cleanup old cache entries using LRU"""
        # Sort by last accessed time
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
                    cache_file.unlink()
                
                # Remove from index
                del self.cache_index[cache_key]
            
            logger.info(f"Cleaned up {entries_to_remove} old cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
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






