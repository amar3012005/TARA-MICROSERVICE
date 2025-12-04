"""
TTS Streaming Queue

True parallel synthesis pipeline:
- Synthesize multiple sentences in parallel (pre-fetch)
- Play sentences sequentially (one at a time, complete before next)
- Ultra-fast first sentence playback

Reference: Designed for human-like TTS streaming with minimal latency.

Optimizations:
- Direct memory streaming (no temp file I/O)
- Increased prefetch count for longer texts
- Detailed synthesis timing logs
"""

import asyncio
import hashlib
import io
import logging
import time
from typing import Optional, Dict, Any, Callable, Awaitable, List

import soundfile as sf
import numpy as np

from .lemonfox_provider import LemonFoxProvider
from .audio_cache import AudioCache

logger = logging.getLogger(__name__)


class TTSStreamingQueue:
    """
    TTS streaming queue with TRUE parallel synthesis pipeline.
    
    Architecture:
    - Producer: Enqueues sentences for synthesis
    - Synthesizer Pool: Synthesizes up to N sentences in parallel (pre-fetch)
    - Player: Plays sentences sequentially (waits for each to complete)
    
    Flow:
    1. Sentence 1 arrives → synthesize immediately → PLAY
    2. While playing → pre-synthesize sentences 2, 3, 4 in parallel
    3. Sentence 1 finishes → play sentence 2 (already synthesized!)
    4. Continue pre-fetching ahead
    
    This gives ultra-fast first-sentence playback and zero-gap between sentences.
    """
    
    # Number of sentences to pre-synthesize in parallel (increased for longer texts)
    PREFETCH_COUNT = 5
    
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
        
        # Input queue for sentences
        self.sentence_queue = asyncio.Queue(maxsize=config.queue_max_size)
        
        # Ready queue for synthesized audio (order preserved)
        self.audio_ready_queue = asyncio.Queue()
        
        # Track synthesis tasks
        self.synthesis_tasks: Dict[str, asyncio.Task] = {}
        self.synthesis_results: Dict[str, Any] = {}  # hash -> result
        self.sentence_order: List[str] = []  # Preserve order by hash
        
        self.cancelled = asyncio.Event()
        self.stream_complete = asyncio.Event()
        
        # Statistics
        self.sentences_queued = 0
        self.sentences_synthesized = 0
        self.sentences_played = 0
        self.sentences_failed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_synthesis_time_ms = 0
        self.total_play_time_ms = 0
    
    async def enqueue_sentences(
        self,
        sentences: list,
        emotion: str = "helpful",
        voice: Optional[str] = None,
        language: Optional[str] = None
    ):
        """
        Enqueue sentences for synthesis.
        Immediately starts parallel synthesis of first PREFETCH_COUNT sentences.
        """
        logger.info(f"📥 Enqueuing {len(sentences)} sentences for parallel synthesis")
        
        for i, sentence in enumerate(sentences):
            sentence_hash = self._normalize_and_hash(sentence)
            
            # Add to order tracking
            self.sentence_order.append(sentence_hash)
            self.sentences_queued += 1
            
            # Start synthesis immediately for first PREFETCH_COUNT sentences
            if i < self.PREFETCH_COUNT:
                logger.info(f"⚡ Pre-synthesizing sentence {i+1}: '{sentence[:40]}...'")
                task = asyncio.create_task(
                    self._synthesize_and_store(sentence, sentence_hash, emotion, voice, language, i)
                )
                self.synthesis_tasks[sentence_hash] = task
            else:
                # Queue remaining sentences for later synthesis
                await self.sentence_queue.put((sentence, sentence_hash, emotion, voice, language, i))
        
        # Signal end of sentences
        await self.sentence_queue.put(None)
    
    async def _synthesize_and_store(
        self,
        sentence: str,
        sentence_hash: str,
        emotion: str,
        voice: Optional[str],
        language: Optional[str],
        index: int
    ):
        """
        Synthesize a sentence and store the result in memory.
        
        Optimized: No temp file I/O - audio bytes stay in memory.
        """
        start_time = time.time()
        
        try:
            # Check cache first
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
                logger.info(f"✅ Cache HIT for sentence {index+1}: '{sentence[:30]}...'")
                self.cache_hits += 1
                
                # Load cached audio into memory
                audio_data, sample_rate = await asyncio.to_thread(sf.read, cached_path)
                
                # Convert to int16 bytes
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                if audio_data.dtype != np.int16:
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    audio_data = (audio_data * 32767).astype(np.int16)
                
                result = {
                    'audio_bytes': audio_data.tobytes(),
                    'sample_rate': sample_rate,
                    'cached': True,
                    'sentence': sentence,
                    'index': index,
                    'emotion': emotion
                }
            else:
                logger.info(f"🔄 Synthesizing sentence {index+1}: '{sentence[:30]}...'")
                self.cache_misses += 1
                
                synth_start = time.time()
                
                # Synthesize with LemonFox - returns WAV bytes directly
                wav_bytes = await self.provider.synthesize(
                    text=sentence,
                    voice=voice,
                    language=language,
                    emotion=emotion
                )
                
                synth_time = (time.time() - synth_start) * 1000
                logger.info(f"📡 LemonFox API returned {len(wav_bytes)/1024:.1f}KB in {synth_time:.0f}ms")
                
                # Parse WAV from memory (no temp file!)
                parse_start = time.time()
                wav_buffer = io.BytesIO(wav_bytes)
                audio_data, sample_rate = sf.read(wav_buffer)
                parse_time = (time.time() - parse_start) * 1000
                
                # Convert to int16 bytes
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                if audio_data.dtype != np.int16:
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    audio_data = (audio_data * 32767).astype(np.int16)
                
                audio_bytes_final = audio_data.tobytes()
                
                # Cache in background (non-blocking) if enabled
                if self.cache:
                    asyncio.create_task(self._cache_audio_async(
                        sentence, voice, language, emotion, wav_bytes
                    ))
                
                result = {
                    'audio_bytes': audio_bytes_final,
                    'sample_rate': sample_rate,
                    'cached': False,
                    'sentence': sentence,
                    'index': index,
                    'emotion': emotion,
                    'api_time_ms': synth_time,
                    'parse_time_ms': parse_time
                }
            
            synthesis_time = (time.time() - start_time) * 1000
            self.total_synthesis_time_ms += synthesis_time
            self.sentences_synthesized += 1
            
            result['synthesis_time_ms'] = synthesis_time
            
            # Calculate audio duration
            audio_duration_ms = len(result['audio_bytes']) / (result['sample_rate'] * 2) * 1000
            result['duration_ms'] = audio_duration_ms
            
            logger.info(
                f"✅ Sentence {index+1} ready: {synthesis_time:.0f}ms total, "
                f"{audio_duration_ms:.0f}ms audio (cached={result['cached']})"
            )
            
            # Store result
            self.synthesis_results[sentence_hash] = result
            
            # Signal that this sentence is ready
            await self.audio_ready_queue.put(sentence_hash)
            
        except Exception as e:
            logger.error(f"❌ Synthesis error for sentence {index+1}: {e}", exc_info=True)
            self.sentences_failed += 1
            self.synthesis_results[sentence_hash] = {'error': str(e), 'index': index}
            await self.audio_ready_queue.put(sentence_hash)
    
    async def _cache_audio_async(
        self,
        sentence: str,
        voice: Optional[str],
        language: Optional[str],
        emotion: str,
        wav_bytes: bytes
    ):
        """Cache audio in background without blocking playback."""
        try:
            import tempfile
            import os
            
            # Write to temp file for caching
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.write(wav_bytes)
            temp_file.close()
            
            self.cache.cache_audio(
                sentence,
                voice or self.provider.voice,
                language or self.provider.language,
                "lemonfox",
                emotion,
                temp_path
            )
            os.unlink(temp_path)
            logger.debug(f"📦 Cached audio for: '{sentence[:20]}...'")
        except Exception as e:
            logger.warning(f"Background cache error: {e}")
    
    async def _prefetch_worker(self):
        """
        Background worker that pre-synthesizes upcoming sentences.
        Runs while playback is happening.
        """
        try:
            while not self.cancelled.is_set():
                try:
                    item = await asyncio.wait_for(self.sentence_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if self.stream_complete.is_set():
                        break
                    continue
                
                if item is None:
                    break
                
                sentence, sentence_hash, emotion, voice, language, index = item
                
                # Check if already being synthesized or done
                if sentence_hash in self.synthesis_tasks or sentence_hash in self.synthesis_results:
                    continue
                
                # Start synthesis
                logger.debug(f"🔄 Pre-fetching sentence {index+1}")
                task = asyncio.create_task(
                    self._synthesize_and_store(sentence, sentence_hash, emotion, voice, language, index)
                )
                self.synthesis_tasks[sentence_hash] = task
                
        except asyncio.CancelledError:
            logger.debug("Prefetch worker cancelled")
        except Exception as e:
            logger.error(f"Prefetch worker error: {e}")
    
    async def consume_queue(self) -> Dict[str, Any]:
        """
        Play synthesized audio in order.
        Waits for each sentence to complete before playing the next.
        Pre-fetches upcoming sentences in parallel.
        """
        logger.info("▶️ Starting parallel TTS playback pipeline")
        
        # Start prefetch worker
        prefetch_task = asyncio.create_task(self._prefetch_worker())
        
        sentences_played = 0
        total_duration_ms = 0.0
        
        try:
            # Play sentences in order
            for expected_hash in self.sentence_order:
                if self.cancelled.is_set():
                    break
                
                # Wait for this sentence to be synthesized
                while expected_hash not in self.synthesis_results:
                    if self.cancelled.is_set():
                        break
                    try:
                        ready_hash = await asyncio.wait_for(self.audio_ready_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                
                if self.cancelled.is_set():
                    break
                
                # Get the result
                result = self.synthesis_results.get(expected_hash)
                if not result:
                    continue
                
                if 'error' in result:
                    logger.warning(f"⚠️ Skipping failed sentence {result.get('index', '?')}")
                    continue
                
                # Play audio directly from memory (no file I/O!)
                try:
                    play_start = time.time()
                    
                    # Get pre-processed audio bytes from memory
                    audio_bytes = result.get('audio_bytes')
                    sample_rate = result.get('sample_rate', 24000)
                    
                    if not audio_bytes:
                        logger.warning(f"⚠️ No audio bytes for sentence {result.get('index', '?')}")
                        continue
                    
                    # Convert back to numpy for gap addition
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    # Add natural pause between sentences (configurable, default 100ms)
                    gap_ms = getattr(self.config, "inter_sentence_gap_ms", 100)
                    if gap_ms > 0:
                        pause_samples = int(sample_rate * (gap_ms / 1000.0))
                        if pause_samples > 0:
                            silence = np.zeros(pause_samples, dtype=np.int16)
                            audio_data = np.concatenate((audio_data, silence))
                    
                    audio_bytes_final = audio_data.tobytes()
                    
                    # Calculate duration
                    duration_ms = len(audio_bytes_final) / (sample_rate * 2) * 1000
                    
                    # Send audio via callback
                    if self.audio_callback:
                        metadata = {
                            'sentence': result.get('sentence', ''),
                            'sentence_index': result.get('index', sentences_played),
                            'emotion': result.get('emotion', 'helpful'),
                            'cached': result.get('cached', False),
                            'duration_ms': duration_ms,
                            'synthesis_time_ms': result.get('synthesis_time_ms', 0),
                            'api_time_ms': result.get('api_time_ms', 0),
                            'parse_time_ms': result.get('parse_time_ms', 0)
                        }
                        
                        logger.info(
                            f"🔊 Playing sentence {result.get('index', 0)+1}: "
                            f"'{result.get('sentence', '')[:40]}...' "
                            f"({duration_ms:.0f}ms, synth={result.get('synthesis_time_ms', 0):.0f}ms)"
                        )
                        
                        # Send audio - this streams to the client
                        await self.audio_callback(audio_bytes_final, sample_rate, metadata)
                    
                    play_time = (time.time() - play_start) * 1000
                    self.total_play_time_ms += play_time
                    
                    sentences_played += 1
                    total_duration_ms += duration_ms
                    self.sentences_played += 1
                    
                    logger.debug(
                        f"✅ Sentence {result.get('index', 0)+1} sent to client "
                        f"({duration_ms:.0f}ms audio, {play_time:.0f}ms processing)"
                    )
                    
                except Exception as e:
                    logger.error(f"❌ Playback error: {e}", exc_info=True)
                    self.sentences_failed += 1
            
            self.stream_complete.set()
            
        except asyncio.CancelledError:
            logger.debug("TTS consumer cancelled")
            raise
        except Exception as e:
            logger.error(f"TTS consumer error: {e}")
        finally:
            # Cleanup
            self.stream_complete.set()
            prefetch_task.cancel()
            try:
                await prefetch_task
            except asyncio.CancelledError:
                pass
            
            # Cancel any remaining synthesis tasks
            for task in self.synthesis_tasks.values():
                if not task.done():
                    task.cancel()
        
        logger.info(f"✅ TTS pipeline complete: {sentences_played} sentences, {total_duration_ms:.0f}ms total audio")
        
        return {
            'sentences_played': sentences_played,
            'sentences_failed': self.sentences_failed,
            'total_duration_ms': total_duration_ms,
            'total_synthesis_time_ms': self.total_synthesis_time_ms,
            'total_play_time_ms': self.total_play_time_ms,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }
    
    def cancel(self):
        """Cancel current synthesis and playback"""
        logger.info("🛑 TTS queue cancelled")
        self.cancelled.set()
    
    def reset(self):
        """Reset for new stream"""
        self.cancelled.clear()
        self.stream_complete.clear()
        self.synthesis_tasks.clear()
        self.synthesis_results.clear()
        self.sentence_order.clear()
        
        # Clear queues
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
            except:
                break
        while not self.audio_ready_queue.empty():
            try:
                self.audio_ready_queue.get_nowait()
            except:
                break
    
    def _normalize_and_hash(self, text: str) -> str:
        """Normalize text and generate MD5 hash"""
        normalized = text.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            'sentence_queue_size': self.sentence_queue.qsize(),
            'audio_ready_queue_size': self.audio_ready_queue.qsize(),
            'sentences_queued': self.sentences_queued,
            'sentences_synthesized': self.sentences_synthesized,
            'sentences_played': self.sentences_played,
            'sentences_failed': self.sentences_failed,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'active_synthesis_tasks': len([t for t in self.synthesis_tasks.values() if not t.done()]),
            'avg_synthesis_time_ms': self.total_synthesis_time_ms / max(1, self.sentences_synthesized)
        }


# Legacy compatibility - keep old method names working
TTSStreamingQueue.send_sentinel = lambda self: None  # No longer needed
