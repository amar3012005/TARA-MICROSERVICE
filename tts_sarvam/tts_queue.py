"""
TTS Streaming Queue

True parallel synthesis pipeline:
- Synthesize multiple sentences in parallel (pre-fetch)
- Play sentences sequentially (one at a time, complete before next)
- Ultra-fast first sentence playback

Reference: Designed for human-like TTS streaming with minimal latency.
"""

import asyncio
import hashlib
import logging
import tempfile
import time
from collections import deque
from typing import Optional, Dict, Any, Callable, Awaitable, List
from pathlib import Path

import soundfile as sf
import numpy as np

from .sarvam_provider import SarvamProvider
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
    
    # Number of sentences to pre-synthesize in parallel
    PREFETCH_COUNT = 3
    
    def __init__(
        self,
        provider: SarvamProvider,
        cache: Optional[AudioCache],
        config,
        audio_callback: Optional[Callable[[bytes, int, Dict[str, Any]], Awaitable[None]]] = None,
        request_start_time: float = 0.0
    ):
        """
        Initialize TTS streaming queue.
        
        Args:
            provider: TTS provider instance
            cache: Audio cache instance
            config: Configuration object
            audio_callback: Async callback for audio data (bytes, sample_rate, metadata)
            request_start_time: Timestamp when the request was received (for latency tracking)
        """
        self.provider = provider
        self.cache = cache
        self.config = config
        self.audio_callback = audio_callback
        self.request_start_time = request_start_time or time.time()
        
        # Queues
        self.sentence_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        
        # State
        self.synthesis_tasks: Dict[int, asyncio.Task] = {}
        self.sentences_played = 0
        self.sentences_failed = 0
        self.total_play_time_ms = 0
        self.total_synthesis_time_ms = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.time_to_first_audio_ms = 0
        
        # Control
        self.stream_complete = asyncio.Event()
        self.cancelled = asyncio.Event()
        """
        Initialize TTS streaming queue.
        
        Args:
            provider: Sarvam TTS provider
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
        """Synthesize a sentence and store the result."""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_path = None
            if self.cache:
                cached_path = self.cache.get_cached_audio(
                    sentence,
                    voice or self.provider.speaker,
                    language or self.provider.language,
                    "sarvam",
                    emotion or "neutral"
                )
            
            if cached_path:
                self.cache_hits += 1
                result = {
                    'audio_path': cached_path,
                    'cached': True,
                    'sentence': sentence,
                    'index': index,
                    'emotion': emotion
                }
            else:
                self.cache_misses += 1
                
                # Synthesize with Sarvam
                audio_bytes = await self.provider.synthesize(
                    text=sentence,
                    speaker=voice,
                    language=language
                )
                
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_path = temp_file.name
                temp_file.write(audio_bytes)
                temp_file.close()
                
                # Cache if enabled
                final_path = temp_path
                if self.cache:
                    try:
                        cached_path = self.cache.cache_audio(
                            sentence,
                            voice or self.provider.speaker,
                            language or self.provider.language,
                            "sarvam",
                            emotion or "neutral",
                            temp_path
                        )
                        import os
                        os.unlink(temp_path)
                        final_path = cached_path
                    except Exception as cache_error:
                        logger.warning(f"Cache error: {cache_error}")
                
                result = {
                    'audio_path': final_path,
                    'cached': False,
                    'sentence': sentence,
                    'index': index,
                    'emotion': emotion
                }
            
            synthesis_time = (time.time() - start_time) * 1000
            self.total_synthesis_time_ms += synthesis_time
            self.sentences_synthesized += 1
            
            result['synthesis_time_ms'] = synthesis_time
            logger.info(f"✅ Sentence {index+1} ready in {synthesis_time:.0f}ms (cached={result['cached']})")
            
            # Store result
            self.synthesis_results[sentence_hash] = result
            
            # Signal that this sentence is ready
            await self.audio_ready_queue.put(sentence_hash)
            
        except Exception as e:
            logger.error(f"❌ Synthesis error for sentence {index+1}: {e}")
            self.sentences_failed += 1
            self.synthesis_results[sentence_hash] = {'error': str(e), 'index': index}
            await self.audio_ready_queue.put(sentence_hash)
    
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
                
                # Load and play audio
                try:
                    play_start = time.time()
                    
                    audio_path = result.get('audio_path')
                    if not audio_path:
                        continue
                    
                    # Load audio
                    audio_data, sample_rate = await asyncio.to_thread(sf.read, audio_path)
                    
                    # Ensure mono
                    if audio_data.ndim > 1:
                        audio_data = audio_data.mean(axis=1)
                    
                    # Convert to int16
                    if audio_data.dtype != np.int16:
                        audio_data = np.clip(audio_data, -1.0, 1.0)
                        audio_data = (audio_data * 32767).astype(np.int16)
                    
                    # Add natural pause between sentences (configurable, default 100ms)
                    gap_ms = getattr(self.config, "inter_sentence_gap_ms", 100)
                    if gap_ms > 0:
                        pause_samples = int(sample_rate * (gap_ms / 1000.0))
                        if pause_samples > 0:
                            silence = np.zeros(pause_samples, dtype=np.int16)
                            audio_data = np.concatenate((audio_data, silence))
                    
                    audio_bytes = audio_data.tobytes()
                    
                    # Calculate duration
                    duration_ms = len(audio_bytes) / (sample_rate * 2) * 1000
                    
                    # Calculate Time to First Audio (TTFA)
                    if sentences_played == 0:
                        self.time_to_first_audio_ms = (time.time() - self.request_start_time) * 1000
                        logger.info(f"⚡ TIME TO FIRST AUDIO: {self.time_to_first_audio_ms:.0f}ms")
                        logger.info(f"   📊 Breakdown: synthesis={result.get('synthesis_time_ms', 0):.0f}ms, cached={result.get('cached', False)}")
                    
                    # Send audio via callback
                    if self.audio_callback:
                        metadata = {
                            'sentence': result.get('sentence', ''),
                            'sentence_index': result.get('index', sentences_played),
                            'emotion': result.get('emotion', 'helpful'),
                            'cached': result.get('cached', False),
                            'duration_ms': duration_ms,
                            'synthesis_time_ms': result.get('synthesis_time_ms', 0),
                            'ttfa_ms': self.time_to_first_audio_ms if sentences_played == 0 else 0
                        }
                        
                        logger.info(f"🔊 Playing sentence {result.get('index', 0)+1}: '{result.get('sentence', '')[:40]}...' ({duration_ms:.0f}ms)")
                        
                        # Send audio - this streams to the client
                        await self.audio_callback(audio_bytes, sample_rate, metadata)
                    
                    play_time = (time.time() - play_start) * 1000
                    self.total_play_time_ms += play_time
                    
                    sentences_played += 1
                    total_duration_ms += duration_ms
                    self.sentences_played += 1
                    
                    logger.debug(f"✅ Sentence {result.get('index', 0)+1} played ({duration_ms:.0f}ms audio, {play_time:.0f}ms processing)")
                    
                except Exception as e:
                    logger.error(f"❌ Playback error: {e}")
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
            'cache_misses': self.cache_misses,
            'time_to_first_audio_ms': self.time_to_first_audio_ms
        }
    
    async def consume_queue_parallel(self, batch_size: int = 3) -> Dict[str, Any]:
        """
        Consume queue with parallel synthesis for lower latency.
        
        Synthesizes sentences in parallel batches while delivering audio
        in the correct order to maintain speech flow.
        
        Args:
            batch_size: Number of sentences to synthesize in parallel (default 3)
        
        Returns:
            Statistics dict with timing information
        """
        logger.info(f"▶️ Starting PARALLEL TTS pipeline (batch_size={batch_size})")
        
        start_time = time.time()
        request_start = self.request_start_time or start_time
        
        stats = {
            "sentences_queued": len(self.sentence_order),
            "sentences_synthesized": 0,
            "sentences_failed": 0,
            "sentences_played": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_duration_ms": 0,
            "time_to_first_audio_ms": 0,
            "parallel_batches": 0,
            "total_synthesis_time_ms": 0,
            "total_play_time_ms": 0
        }
        
        first_audio_sent = False
        
        # Collect all sentences with their metadata
        sentences_with_metadata = []
        for sentence_hash in self.sentence_order:
            # Find the sentence data - it's stored during enqueue
            # We need to reconstruct from the queue or synthesis results
            pass
        
        # Actually, we need to collect from the sentence_queue directly
        # Let's build a list from what was enqueued
        sentences_to_process = []
        
        # First, process the pre-synthesized sentences (already in synthesis_tasks)
        pre_synthesized_hashes = list(self.synthesis_tasks.keys())
        
        # Drain the sentence_queue to get remaining sentences
        remaining_sentences = []
        while not self.sentence_queue.empty():
            try:
                item = self.sentence_queue.get_nowait()
                if item is not None:
                    remaining_sentences.append(item)
            except asyncio.QueueEmpty:
                break
        
        # Combine: use sentence_order to maintain order
        # For parallel mode, we'll skip the prefetch approach and do batch synthesis
        
        # Get all sentence texts and metadata
        all_sentences = []
        sentence_metadata = {}
        
        # Extract from synthesis results (pre-synthesized) and remaining queue
        for sentence_hash in self.sentence_order:
            if sentence_hash in self.synthesis_results:
                result = self.synthesis_results[sentence_hash]
                all_sentences.append({
                    'hash': sentence_hash,
                    'sentence': result.get('sentence', ''),
                    'index': result.get('index', 0),
                    'emotion': result.get('emotion', 'helpful'),
                    'voice': None,
                    'language': None,
                    'pre_synthesized': True,
                    'audio_path': result.get('audio_path')
                })
            else:
                # Find in remaining sentences
                for item in remaining_sentences:
                    sentence, item_hash, emotion, voice, language, index = item
                    if item_hash == sentence_hash:
                        all_sentences.append({
                            'hash': item_hash,
                            'sentence': sentence,
                            'index': index,
                            'emotion': emotion,
                            'voice': voice,
                            'language': language,
                            'pre_synthesized': False,
                            'audio_path': None
                        })
                        break
        
        if not all_sentences:
            logger.warning("⚠️ No sentences to process in parallel mode")
            return stats
        
        stats["sentences_queued"] = len(all_sentences)
        
        # Process in batches
        for batch_start in range(0, len(all_sentences), batch_size):
            if self.cancelled.is_set():
                logger.info("🛑 Parallel synthesis cancelled")
                break
                
            batch = all_sentences[batch_start:batch_start + batch_size]
            stats["parallel_batches"] += 1
            
            logger.info(f"🔄 Processing parallel batch {stats['parallel_batches']}: {len(batch)} sentences")
            
            # Separate pre-synthesized from ones needing synthesis
            need_synthesis = []
            already_synthesized = []
            
            for item in batch:
                if item['pre_synthesized'] and item['audio_path']:
                    already_synthesized.append(item)
                else:
                    need_synthesis.append(item)
            
            # Synthesize batch in parallel if needed
            batch_results = []
            
            if need_synthesis:
                batch_texts = [item["sentence"] for item in need_synthesis]
                
                try:
                    synthesis_start = time.time()
                    parallel_results = await self.provider.synthesize_parallel(
                        texts=batch_texts,
                        max_concurrent=batch_size,
                        speaker=need_synthesis[0].get("voice") or self.provider.speaker,
                        language=need_synthesis[0].get("language") or self.provider.language
                    )
                    synthesis_time = (time.time() - synthesis_start) * 1000
                    stats["total_synthesis_time_ms"] += synthesis_time
                    
                    # Map results back to items
                    for idx, (audio_bytes, synth_time_ms) in enumerate(parallel_results):
                        item = need_synthesis[idx]
                        batch_results.append({
                            'item': item,
                            'audio_bytes': audio_bytes,
                            'synthesis_time_ms': synth_time_ms,
                            'from_parallel': True
                        })
                        stats["cache_misses"] += 1
                        
                except Exception as e:
                    logger.error(f"❌ Parallel synthesis error: {e}")
                    stats["sentences_failed"] += len(need_synthesis)
                    continue
            
            # Load already synthesized audio
            for item in already_synthesized:
                try:
                    audio_path = item['audio_path']
                    audio_data, sample_rate = await asyncio.to_thread(sf.read, audio_path)
                    
                    # Ensure mono
                    if audio_data.ndim > 1:
                        audio_data = audio_data.mean(axis=1)
                    
                    # Convert to int16
                    if audio_data.dtype != np.int16:
                        audio_data = np.clip(audio_data, -1.0, 1.0)
                        audio_data = (audio_data * 32767).astype(np.int16)
                    
                    audio_bytes = audio_data.tobytes()
                    
                    batch_results.append({
                        'item': item,
                        'audio_bytes': audio_bytes,
                        'synthesis_time_ms': 0,  # Already synthesized
                        'from_parallel': False
                    })
                    stats["cache_hits"] += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error loading pre-synthesized audio: {e}")
                    stats["sentences_failed"] += 1
            
            # Sort batch results by original index to maintain order
            batch_results.sort(key=lambda x: x['item']['index'])
            
            # Deliver audio in order
            for result in batch_results:
                if self.cancelled.is_set():
                    break
                    
                item = result['item']
                audio_bytes = result['audio_bytes']
                synth_time_ms = result['synthesis_time_ms']
                sentence_index = item['index']
                
                # Track first audio
                if not first_audio_sent:
                    first_audio_time = time.time()
                    ttfa_ms = (first_audio_time - request_start) * 1000
                    stats["time_to_first_audio_ms"] = ttfa_ms
                    self.time_to_first_audio_ms = ttfa_ms
                    first_audio_sent = True
                    logger.info(f"⚡ TIME TO FIRST AUDIO (parallel): {ttfa_ms:.0f}ms")
                
                # Add inter-sentence gap
                gap_ms = getattr(self.config, "inter_sentence_gap_ms", 100)
                if gap_ms > 0 and sentence_index > 0:
                    sample_rate = self.config.sample_rate
                    pause_samples = int(sample_rate * (gap_ms / 1000.0))
                    if pause_samples > 0:
                        silence = np.zeros(pause_samples, dtype=np.int16)
                        audio_with_gap = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_with_gap = np.concatenate((audio_with_gap, silence))
                        audio_bytes = audio_with_gap.tobytes()
                
                # Calculate duration
                sample_rate = self.config.sample_rate
                duration_ms = len(audio_bytes) / (sample_rate * 2) * 1000
                stats["total_duration_ms"] += duration_ms
                stats["sentences_synthesized"] += 1
                stats["sentences_played"] += 1
                
                # Deliver via callback
                if self.audio_callback:
                    play_start = time.time()
                    
                    await self.audio_callback(
                        audio_bytes,
                        sample_rate,
                        {
                            "sentence_index": sentence_index,
                            "sentence": item["sentence"],
                            "duration_ms": duration_ms,
                            "cached": not result['from_parallel'],
                            "parallel_batch": stats["parallel_batches"],
                            "synthesis_time_ms": synth_time_ms,
                            "emotion": item.get("emotion", "helpful")
                        }
                    )
                    
                    play_time = (time.time() - play_start) * 1000
                    stats["total_play_time_ms"] += play_time
                    
                    logger.info(
                        f"🔊 Sentence {sentence_index+1} delivered: "
                        f"'{item['sentence'][:40]}...' ({duration_ms:.0f}ms)"
                    )
        
        stats["total_time_ms"] = (time.time() - start_time) * 1000
        
        # Update instance stats
        self.sentences_played = stats["sentences_played"]
        self.sentences_failed = stats["sentences_failed"]
        self.cache_hits = stats["cache_hits"]
        self.cache_misses = stats["cache_misses"]
        self.total_synthesis_time_ms = stats["total_synthesis_time_ms"]
        self.total_play_time_ms = stats["total_play_time_ms"]
        
        logger.info(
            f"✅ Parallel TTS complete: {stats['sentences_played']} sentences, "
            f"{stats['parallel_batches']} batches, {stats['total_duration_ms']:.0f}ms total audio, "
            f"TTFA: {stats['time_to_first_audio_ms']:.0f}ms"
        )
        
        return stats
    
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
