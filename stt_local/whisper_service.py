"""
Faster Whisper Service for STT Local Microservice

Provides Faster Whisper model loading and inference for speech transcription.
Supports both partial (streaming) and final transcript generation.

Reference:
    faster-whisper library documentation
"""

import logging
import time
from typing import Optional, Tuple, List
import numpy as np
from faster_whisper import WhisperModel

from config import STTLocalConfig

logger = logging.getLogger(__name__)


class WhisperService:
    """
    Faster Whisper service wrapper for speech transcription.
    
    Manages model loading and provides transcription with streaming support.
    """
    
    def __init__(self, config: STTLocalConfig):
        """
        Initialize Whisper service.
        
        Args:
            config: STTLocalConfig instance
        """
        self.config = config
        self.model = None
        self._load_model()
        
        logger.info(
            f"✅ Whisper Service initialized | "
            f"Model: {config.whisper_model_size} | "
            f"Device: {config.whisper_device} | "
            f"Compute: {config.whisper_compute_type}"
        )
    
    def _load_model(self):
        """Load Faster Whisper model."""
        try:
            logger.info(f"📥 Loading Faster Whisper model ({self.config.whisper_model_size})...")
            
            # Check CUDA availability before attempting GPU load
            import torch
            cuda_available = torch.cuda.is_available() if self.config.use_gpu else False
            
            if self.config.use_gpu and not cuda_available:
                logger.warning("⚠️ GPU requested but CUDA not available - falling back to CPU")
                self.config.whisper_device = "cpu"
                self.config.whisper_compute_type = "float32"
                self.config.use_gpu = False
            
            # Load model with specified settings
            try:
                self.model = WhisperModel(
                    model_size_or_path=self.config.whisper_model_size,
                    device=self.config.whisper_device,
                    compute_type=self.config.whisper_compute_type,
                    num_workers=self.config.num_workers,
                    download_root=None  # Use default cache location
                )
                logger.info("✅ Faster Whisper model loaded successfully")
            except (SystemError, OSError, RuntimeError) as e:
                # CUDA/CUDNN errors - fallback to CPU
                if self.config.use_gpu and ("cuda" in str(e).lower() or "cudnn" in str(e).lower()):
                    logger.warning(f"⚠️ GPU load failed ({e}) - falling back to CPU")
                    self.config.whisper_device = "cpu"
                    self.config.whisper_compute_type = "float32"
                    self.config.use_gpu = False
                    # Retry with CPU
                    self.model = WhisperModel(
                        model_size_or_path=self.config.whisper_model_size,
                        device="cpu",
                        compute_type="float32",
                        num_workers=self.config.num_workers,
                        download_root=None
                    )
                    logger.info("✅ Faster Whisper model loaded with CPU fallback")
                else:
                    raise
            
            # Warmup: Only if GPU is actually available and working
            if self.config.use_gpu and cuda_available:
                logger.info("🔥 Warming up GPU kernels...")
                dummy_audio = np.zeros((16000,), dtype=np.float32)  # 1 second of silence
                try:
                    segments, info = self.model.transcribe(
                        dummy_audio,
                        language=self.config.whisper_language,
                        beam_size=1,  # Use minimal beam size for faster warmup
                        vad_filter=False  # We use Silero VAD, so disable Whisper VAD
                    )
                    # Consume generator (but don't wait too long)
                    list(segments)
                    logger.info("✅ GPU warmup complete")
                except (Exception, SystemError, OSError, RuntimeError) as e:
                    logger.warning(f"⚠️ GPU warmup failed: {e}")
                    logger.info("   Continuing with GPU (warmup skipped - first request may be slower)")
                    # Don't reload - just continue, model is already loaded
            else:
                logger.info("ℹ️ Skipping GPU warmup (CPU mode or CUDA unavailable)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Faster Whisper model: {e}")
            raise
    
    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        is_partial: bool = False
    ) -> Tuple[str, float]:
        """
        Transcribe audio using Faster Whisper.
        
        Args:
            audio: Audio samples as numpy array (float32, mono, 16kHz)
            language: Language code (optional, defaults to config)
            is_partial: Whether this is a partial transcription (for streaming)
            
        Returns:
            Tuple[str, float]: (transcript_text, confidence)
                - transcript_text: Transcribed text
                - confidence: Average confidence score (0.0-1.0)
        """
        if self.model is None:
            logger.error("Whisper model not loaded")
            return "", 0.0
        
        try:
            # Use config language if not specified
            lang = language or self.config.whisper_language
            
            # Check minimum audio length
            audio_duration = len(audio) / self.config.sample_rate
            if audio_duration < self.config.min_audio_length_for_stt:
                logger.debug(f"Audio too short for STT: {audio_duration:.2f}s")
                return "", 0.0
            
            # Transcribe
            start_time = time.time()
            
            segments, info = self.model.transcribe(
                audio,
                language=lang,
                beam_size=self.config.whisper_beam_size,
                vad_filter=False,  # We use Silero VAD, so disable Whisper VAD
                condition_on_previous_text=not is_partial  # Don't condition on previous for partials
            )
            
            # Collect segments
            transcript_parts = []
            total_confidence = 0.0
            segment_count = 0
            
            for segment in segments:
                text = segment.text.strip()
                if text:
                    transcript_parts.append(text)
                    total_confidence += segment.avg_logprob  # Use logprob as confidence proxy
                    segment_count += 1
            
            # Combine transcript
            transcript = " ".join(transcript_parts).strip()
            
            # Calculate average confidence (normalize logprob to 0-1 range)
            # Logprob is typically negative, so we normalize it
            avg_confidence = 0.0
            if segment_count > 0:
                # Normalize logprob: typically ranges from -1 to 0, map to 0-1
                normalized_confidence = (total_confidence / segment_count + 1.0) / 1.0
                avg_confidence = max(0.0, min(1.0, normalized_confidence))
            
            inference_time = time.time() - start_time
            
            if self.config.verbose:
                logger.debug(
                    f"Transcription: '{transcript[:50]}...' | "
                    f"Confidence: {avg_confidence:.2f} | "
                    f"Time: {inference_time:.3f}s | "
                    f"Partial: {is_partial}"
                )
            
            return transcript, avg_confidence
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", 0.0
    
    def transcribe_streaming(
        self,
        audio: np.ndarray,
        language: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Transcribe audio for streaming (partial) updates.
        
        Uses faster settings optimized for low latency.
        
        Args:
            audio: Audio samples as numpy array
            language: Language code (optional)
            
        Returns:
            Tuple[str, float]: (transcript_text, confidence)
        """
        return self.transcribe(audio, language=language, is_partial=True)
    
    def transcribe_final(
        self,
        audio: np.ndarray,
        language: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Transcribe audio for final (complete) transcript.
        
        Uses full beam search for accuracy.
        
        Args:
            audio: Audio samples as numpy array
            language: Language code (optional)
            
        Returns:
            Tuple[str, float]: (transcript_text, confidence)
        """
        return self.transcribe(audio, language=language, is_partial=False)
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            dict: Model information
        """
        return {
            "model_size": self.config.whisper_model_size,
            "device": self.config.whisper_device,
            "compute_type": self.config.whisper_compute_type,
            "language": self.config.whisper_language,
            "beam_size": self.config.whisper_beam_size
        }

