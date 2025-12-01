"""
TTS Streaming Queue

2-slot pipeline for parallel synthesis: synthesize N+1 while playing N.
Extracted from leibniz_pro.py (lines 1255-1760) and adapted for microservice.
"""

import asyncio
import hashlib
import logging
import tempfile
from collections import deque
from typing import Optional, Dict, Any, Callable, Awaitable
from pathlib import Path

import soundfile as sf

from .lemonfox_provider import LemonFoxProvider
from .audio_cache import AudioCache

logger = logging.getLogger(__name__)


class TTSStreamingQueue:
    """
    TTS streaming queue with 2-slot pipeline for parallel synthesis.
    
    Features:
    - 2-slot pipeline: synthesize N+1 while playing N
    - Sentence deduplication with MD5 hashes
    - Backpressure handling (merge sentences when full)
    - Cache integration
    """
    
    def __init__(
        self,
        provider: LemonFoxProvider,
        cache: Optional[AudioCache],
        config,
        audio_callback: Optional[Callable[[bytes, int, Dict[str, Any]], Awaitable[None]]] = None
    ):
        """
        Initialize TTS streaming queue.
        
        Args:
            provider: LemonFox TTS provider
            cache: Audio cache instance (optional)
            config: TTSStreamingConfig instance
            audio_callback: Callback function for audio chunks (audio_bytes, sample_rate, metadata)
        """
        self.provider = provider
        self.cache = cache
        self.config = config
        self.audio_callback = audio_callback
        
        self.queue = asyncio.Queue(maxsize=config.queue_max_size)
        self.synthesis_in_flight = set()  # Track in-flight synthesis by hash
        self.cancelled = asyncio.Event()
        
        # Statistics
        self.sentences_queued = 0
        self.sentences_synthesized = 0
        self.sentences_failed = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def enqueue_sentences(
        self,
        sentences: list,
        emotion: str = "helpful",
        voice: Optional[str] = None,
        language: Optional[str] = None
    ):
        """
        Enqueue sentences for synthesis.
        
        Args:
            sentences: List of sentence strings
            emotion: Emotion type for synthesis
            voice: Voice name (optional)
            language: Language code (optional)
        """
        for sentence in sentences:
            try:
                self.queue.put_nowait((sentence, emotion, voice, language))
                self.sentences_queued += 1
            except asyncio.QueueFull:
                # Backpressure: try to merge with last item
                logger.warning("Queue full, attempting to merge sentences")
                try:
                    last_item = self.queue.get_nowait()
                    last_sentence, last_emotion, last_voice, last_lang = last_item
                    merged = f"{last_sentence} {sentence}"
                    self.queue.put_nowait((merged, emotion or last_emotion, voice or last_voice, language or last_lang))
                except asyncio.QueueEmpty:
                    # Can't merge, drop this sentence
                    logger.warning(f"Dropped sentence due to queue pressure: {sentence[:50]}...")
                    self.sentences_failed += 1
    
    def send_sentinel(self):
        """Send sentinel to signal end of stream"""
        try:
            self.queue.put_nowait(None)
        except asyncio.QueueFull:
            # Force add sentinel
            asyncio.create_task(self._force_sentinel())
    
    async def _force_sentinel(self):
        """Force add sentinel by waiting"""
        await self.queue.put(None)
    
    def cancel(self):
        """Cancel current synthesis"""
        self.cancelled.set()
    
    def reset(self):
        """Reset cancellation flag"""
        self.cancelled.clear()
    
    async def consume_queue(self) -> Dict[str, Any]:
        """
        Consume queue with 2-slot pipeline.
        
        Returns:
            Dict with statistics: sentences_played, sentences_failed, total_duration_ms
        """
        stage = deque()  # Staging buffer for non-destructive peek
        current_result = None  # Audio result for current sentence
        next_future = None  # Background synthesis task for N+1
        
        sentences_played = 0
        total_duration_ms = 0.0
        
        try:
            while True:
                # Check for cancellation
                if self.cancelled.is_set():
                    logger.debug("TTS queue cancelled")
                    break
                
                # Fill staging buffer from queue (ensure at least 1 item)
                while len(stage) < 2:
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=0.5)
                        stage.append(item)
                        
                        if item is None:  # Sentinel
                            break
                    except asyncio.TimeoutError:
                        break  # No more items yet
                
                if not stage:
                    if self.cancelled.is_set():
                        break
                    continue
                
                # Get current sentence from staging buffer
                current_item = stage.popleft()
                
                if current_item is None:  # Sentinel - end of stream
                    break
                
                sentence, emotion, voice, language = current_item
                
                # If we have prefetched future from previous iteration, await it now
                if next_future:
                    try:
                        current_result = await next_future
                        logger.debug(f"Using prefetched audio for: '{sentence[:30]}...'")
                    except Exception as synth_error:
                        logger.error(f"Prefetch synthesis error: {synth_error}")
                        current_result = None
                    next_future = None
                else:
                    current_result = None
                
                # Peek next sentence from stage (non-destructive) and start synthesis
                if stage:  # Has next sentence in staging buffer
                    next_item = stage[0]  # Peek without removing
                    
                    if next_item is not None:  # Not sentinel
                        next_sentence, next_emotion, next_voice, next_lang = next_item
                        
                        # Check cache for next sentence
                        cached_path = None
                        if self.cache:
                            cached_path = self.cache.get_cached_audio(
                                next_sentence,
                                next_voice or self.provider.voice,
                                next_lang or self.provider.language,
                                "lemonfox",
                                next_emotion or "helpful"
                            )
                        
                        if cached_path:
                            # Found cached file - create synthetic result
                            async def _return_cached(path):
                                return {'audio_bytes': None, 'audio_path': path, 'cached': True}
                            next_future = asyncio.create_task(_return_cached(cached_path))
                            logger.debug(f"Prefetching N+1 (cached): '{next_sentence[:30]}...'")
                        else:
                            # Check deduplication
                            sentence_hash = self._normalize_and_hash(next_sentence)
                            
                            if sentence_hash not in self.synthesis_in_flight:
                                self.synthesis_in_flight.add(sentence_hash)
                                
                                # Start N+1 synthesis in background (don't await)
                                async def _synthesize_and_track(text, hash_key, emo, v, lang):
                                    try:
                                        logger.debug(f"Background synthesis STARTED for N+1: '{text[:30]}...'")
                                        result = await self._synthesize_sentence(text, emo, v, lang)
                                        logger.debug(f"Background synthesis COMPLETED for N+1: '{text[:30]}...'")
                                        return result
                                    finally:
                                        self.synthesis_in_flight.discard(hash_key)
                                
                                next_future = asyncio.create_task(
                                    _synthesize_and_track(next_sentence, sentence_hash, next_emotion, next_voice, next_lang)
                                )
                                logger.debug(f"Prefetching N+1: '{next_sentence[:30]}...'")
                            else:
                                logger.debug(f"Skipping duplicate in-flight synthesis: '{next_sentence[:30]}...'")
                                next_future = None
                    else:
                        next_future = None  # Next is sentinel
                
                # Process current sentence
                try:
                    if current_result:
                        result = current_result
                        logger.debug(f"Playing prefetched audio: '{sentence[:30]}...'")
                    else:
                        # First sentence or prefetch failed - synthesize now
                        logger.debug(f"Synthesizing current (no prefetch): '{sentence[:30]}...'")
                        
                        # Check cache
                        cached_path = None
                        if self.cache:
                            cached_path = self.cache.get_cached_audio(
                                sentence,
                                voice or self.provider.voice,
                                language or self.provider.language,
                                "lemonfox",
                                emotion or "helpful"
                            )
                        
                        if cached_path:
                            result = {'audio_bytes': None, 'audio_path': cached_path, 'cached': True}
                            self.cache_hits += 1
                        else:
                            # Check deduplication
                            sentence_hash = self._normalize_and_hash(sentence)
                            
                            if sentence_hash not in self.synthesis_in_flight:
                                self.synthesis_in_flight.add(sentence_hash)
                                try:
                                    result = await self._synthesize_sentence(sentence, emotion, voice, language)
                                finally:
                                    self.synthesis_in_flight.discard(sentence_hash)
                            else:
                                logger.warning(f"Duplicate synthesis blocked: '{sentence[:30]}...'")
                                result = None
                            self.cache_misses += 1
                    
                    if result:
                        # Load audio data
                        if result.get('audio_bytes'):
                            audio_bytes = result['audio_bytes']
                            sample_rate = self.config.sample_rate
                        elif result.get('audio_path'):
                            audio_data, sample_rate = await asyncio.to_thread(sf.read, result['audio_path'])
                            audio_bytes = audio_data.tobytes()
                        else:
                            logger.error("No audio data in result")
                            self.sentences_failed += 1
                            continue
                        
                        # Calculate duration
                        duration_ms = len(audio_bytes) / (sample_rate * 2) * 1000  # Approximate for int16
                        
                        # Call audio callback if provided
                        if self.audio_callback:
                            metadata = {
                                'sentence': sentence,
                                'sentence_index': sentences_played,
                                'emotion': emotion,
                                'cached': result.get('cached', False),
                                'duration_ms': duration_ms
                            }
                            await self.audio_callback(audio_bytes, sample_rate, metadata)
                        
                        sentences_played += 1
                        total_duration_ms += duration_ms
                        self.sentences_synthesized += 1
                        
                        logger.debug(f"Sentence {sentences_played} processed: {duration_ms:.0f}ms")
                    else:
                        self.sentences_failed += 1
                
                except Exception as e:
                    logger.error(f"Error processing sentence: {e}")
                    self.sentences_failed += 1
                
                # Check for cancellation after each sentence
                if self.cancelled.is_set():
                    break
        
        except asyncio.CancelledError:
            logger.debug("TTS queue consumer cancelled")
            raise
        except Exception as e:
            logger.error(f"TTS queue consumer error: {e}")
        
        return {
            'sentences_played': sentences_played,
            'sentences_failed': self.sentences_failed,
            'total_duration_ms': total_duration_ms,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }
    
    async def _synthesize_sentence(
        self,
        sentence: str,
        emotion: str = "helpful",
        voice: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesize a single sentence.
        
        Returns:
            Dict with 'audio_bytes' or 'audio_path' and 'cached' flag
        """
        try:
            # Synthesize with LemonFox
            audio_bytes = await self.provider.synthesize(
                text=sentence,
                voice=voice,
                language=language,
                emotion=emotion
            )
            
            # Save to temp file for caching
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.write(audio_bytes)
            temp_file.close()
            
            # Cache if enabled
            if self.cache:
                try:
                    cached_path = self.cache.cache_audio(
                        sentence,
                        voice or self.provider.voice,
                        language or self.provider.language,
                        "lemonfox",
                        emotion,
                        temp_path
                    )
                    # Use cached path instead of temp
                    import os
                    os.unlink(temp_path)  # Remove temp file
                    return {'audio_bytes': None, 'audio_path': cached_path, 'cached': False}
                except Exception as cache_error:
                    logger.warning(f"Cache error: {cache_error}")
                    return {'audio_bytes': audio_bytes, 'audio_path': temp_path, 'cached': False}
            else:
                return {'audio_bytes': audio_bytes, 'audio_path': temp_path, 'cached': False}
        
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            raise
    
    def _normalize_and_hash(self, text: str) -> str:
        """Normalize text and generate MD5 hash"""
        normalized = text.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            'queue_size': self.queue.qsize(),
            'sentences_queued': self.sentences_queued,
            'sentences_synthesized': self.sentences_synthesized,
            'sentences_failed': self.sentences_failed,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'in_flight_synthesis': len(self.synthesis_in_flight)
        }

