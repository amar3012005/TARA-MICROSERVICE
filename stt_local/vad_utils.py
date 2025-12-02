"""
Silero VAD Utilities for STT Local Microservice

Provides Silero VAD model loading and speech detection.
Uses torch.hub to load pre-trained Silero VAD models.

Reference:
    leibniz_agent/leibniz_silero_vad.py - Silero VAD implementation
"""

import logging
import time
from typing import Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Silero VAD wrapper for speech detection.
    
    Loads Silero VAD model from torch.hub and provides
    speech detection on audio chunks.
    """
    
    def __init__(self, threshold: float = 0.5, device: str = "cuda"):
        """
        Initialize Silero VAD.
        
        Args:
            threshold: Speech detection threshold (0.0-1.0)
            device: Device to run on ("cuda" or "cpu")
        """
        self.threshold = threshold
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = None
        self.utils = None
        self._load_model()
        
        logger.info(f"✅ Silero VAD initialized | Device: {self.device} | Threshold: {self.threshold}")
    
    def _load_model(self):
        """Load Silero VAD model from torch.hub."""
        try:
            logger.info("📥 Loading Silero VAD model from torch.hub...")
            
            # Load model and utils from torch.hub
            # Silero VAD v4 model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            # Move model to device
            if self.device == "cuda":
                model = model.to(self.device)
            
            self.model = model
            self.utils = utils
            
            logger.info("✅ Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Silero VAD model: {e}")
            raise
    
    def detect_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, float]:
        """
        Detect speech in audio chunk.
        
        Args:
            audio_chunk: Audio samples as numpy array (float32, mono, 16kHz)
            sample_rate: Sample rate of audio (must be 16000 for Silero VAD)
            
        Returns:
            Tuple[bool, float]: (is_speech, confidence)
                - is_speech: True if speech detected above threshold
                - confidence: VAD probability (0.0-1.0)
        """
        if self.model is None:
            logger.error("VAD model not loaded")
            return False, 0.0
        
        try:
            # Silero VAD requires exactly 512 samples (32ms at 16kHz)
            if len(audio_chunk) != 512:
                logger.warning(f"Invalid chunk size: {len(audio_chunk)}, expected 512")
                return False, 0.0
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_chunk).float()
            
            # Move to device
            if self.device == "cuda":
                audio_tensor = audio_tensor.to(self.device)
            
            # Run VAD inference
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, sample_rate).item()
            
            # Check against threshold
            is_speech = speech_prob > self.threshold
            
            return is_speech, speech_prob
            
        except Exception as e:
            logger.error(f"VAD detection error: {e}")
            return False, 0.0
    
    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Simple speech detection (returns boolean only).
        
        Args:
            audio_chunk: Audio samples as numpy array
            sample_rate: Sample rate (must be 16000)
            
        Returns:
            bool: True if speech detected
        """
        is_speech, _ = self.detect_speech(audio_chunk, sample_rate)
        return is_speech


class VADStateMachine:
    """
    State machine for VAD-based speech detection.
    
    Tracks speech start/end events and manages speech buffer.
    """
    
    def __init__(
        self,
        vad: SileroVAD,
        min_speech_duration_ms: int = 250,
        silence_timeout_ms: int = 800
    ):
        """
        Initialize VAD state machine.
        
        Args:
            vad: SileroVAD instance
            min_speech_duration_ms: Minimum speech duration to trigger
            silence_timeout_ms: Silence duration before ending speech
        """
        self.vad = vad
        self.min_speech_duration_ms = min_speech_duration_ms
        self.silence_timeout_ms = silence_timeout_ms
        
        # State
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.speech_chunks = []
        
    def process_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Process audio chunk and return state update.
        
        Args:
            audio_chunk: Audio samples (must be 512 samples for Silero VAD)
            sample_rate: Sample rate (must be 16000)
            
        Returns:
            dict: {
                "event": str,  # "speech_start", "speech_end", "speaking", "silent"
                "is_speaking": bool,
                "speech_duration_ms": float,
                "confidence": float
            }
        """
        # Detect speech
        is_speech, confidence = self.vad.detect_speech(audio_chunk, sample_rate)
        
        current_time = time.time()
        
        if is_speech:
            # Speech detected
            if not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start_time = current_time
                self.silence_start_time = None
                self.speech_chunks = [audio_chunk]
                
                return {
                    "event": "speech_start",
                    "is_speaking": True,
                    "speech_duration_ms": 0.0,
                    "confidence": confidence
                }
            else:
                # Continue speaking
                self.speech_chunks.append(audio_chunk)
                self.silence_start_time = None
                
                speech_duration_ms = (current_time - self.speech_start_time) * 1000
                
                return {
                    "event": "speaking",
                    "is_speaking": True,
                    "speech_duration_ms": speech_duration_ms,
                    "confidence": confidence
                }
        else:
            # No speech detected
            if self.is_speaking:
                # Check silence timeout
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                
                silence_duration_ms = (current_time - self.silence_start_time) * 1000
                
                if silence_duration_ms >= self.silence_timeout_ms:
                    # Speech ended
                    speech_duration_ms = (self.silence_start_time - self.speech_start_time) * 1000
                    
                    # Check minimum duration
                    if speech_duration_ms >= self.min_speech_duration_ms:
                        # Valid speech segment
                        result = {
                            "event": "speech_end",
                            "is_speaking": False,
                            "speech_duration_ms": speech_duration_ms,
                            "confidence": confidence,
                            "audio_segment": np.concatenate(self.speech_chunks)
                        }
                        
                        # Reset state
                        self.is_speaking = False
                        self.speech_start_time = None
                        self.silence_start_time = None
                        self.speech_chunks = []
                        
                        return result
                    else:
                        # Too short, ignore
                        self.is_speaking = False
                        self.speech_start_time = None
                        self.silence_start_time = None
                        self.speech_chunks = []
                        
                        return {
                            "event": "speech_too_short",
                            "is_speaking": False,
                            "speech_duration_ms": speech_duration_ms,
                            "confidence": confidence
                        }
                else:
                    # Still in silence period
                    speech_duration_ms = (self.silence_start_time - self.speech_start_time) * 1000
                    
                    return {
                        "event": "silence",
                        "is_speaking": True,  # Still considered speaking during silence timeout
                        "speech_duration_ms": speech_duration_ms,
                        "confidence": confidence
                    }
            else:
                # Already silent
                return {
                    "event": "silent",
                    "is_speaking": False,
                    "speech_duration_ms": 0.0,
                    "confidence": confidence
                }
    
    def get_current_speech_audio(self) -> Optional[np.ndarray]:
        """
        Get accumulated speech audio for current speech segment.
        
        Returns:
            numpy array of concatenated speech chunks, or None if not speaking
        """
        if not self.is_speaking or not self.speech_chunks:
            return None
        
        return np.concatenate(self.speech_chunks)
    
    def reset(self):
        """Reset state machine."""
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.speech_chunks = []

