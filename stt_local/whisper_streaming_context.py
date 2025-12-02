"""
Whisper Streaming Context Manager

Maintains state across streaming transcription calls to prevent
re-transcription artifacts and provide accurate incremental transcripts.
"""

import logging
import time
from typing import Optional, Tuple, Dict, List
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperStreamingContext:
    """
    Maintains state across streaming transcription calls.
    Prevents re-transcription artifacts.
    """
    
    def __init__(self, model: WhisperModel, config):
        self.model = model
        self.config = config
        self.accumulated_audio = np.array([], dtype=np.float32)
        self.previous_transcript = ""
        self.transcribed_until_samples = 0
        self.segment_history: List[Dict] = []
        logger.debug("✅ WhisperStreamingContext initialized")
    
    def add_chunk_and_transcribe(
        self,
        new_audio: np.ndarray,
        sample_rate: int = 16000,
        is_final: bool = False
    ) -> Tuple[str, float]:
        """Add new audio chunk and return incremental transcript."""
        
        self.accumulated_audio = np.concatenate([self.accumulated_audio, new_audio])
        duration_seconds = len(self.accumulated_audio) / sample_rate
        min_length_seconds = self.config.min_audio_length_for_stt
        
        if duration_seconds < min_length_seconds:
            return "", 0.0
        
        try:
            segments, info = self.model.transcribe(
                self.accumulated_audio,
                language=self.config.whisper_language,
                beam_size=5,  # CRITICAL FOR ACCURACY
                best_of=5,
                vad_filter=False,
                condition_on_previous_text=bool(self.previous_transcript),
                initial_prompt=self.previous_transcript if self.previous_transcript else None
            )
            
            new_text_parts = []
            total_confidence = 0.0
            segment_count = 0
            
            for segment in segments:
                segment_start_samples = int(segment.start * sample_rate)
                
                if segment_start_samples < self.transcribed_until_samples:
                    continue
                
                text = segment.text.strip()
                if text:
                    new_text_parts.append(text)
                    total_confidence += segment.avg_logprob
                    segment_count += 1
                    self.segment_history.append({
                        'start': segment.start,
                        'end': segment.end,
                        'text': text,
                        'confidence': segment.avg_logprob
                    })
                    self.transcribed_until_samples = int(segment.end * sample_rate)
            
            new_text = " ".join(new_text_parts).strip()
            
            if new_text:
                if self.previous_transcript:
                    self.previous_transcript += " " + new_text
                else:
                    self.previous_transcript = new_text
            
            avg_confidence = 0.0
            if segment_count > 0:
                normalized_logprob = (total_confidence / segment_count + 1.0)
                avg_confidence = max(0.0, min(1.0, normalized_logprob))
            
            return new_text, avg_confidence
            
        except Exception as e:
            logger.error(f"Streaming transcription error: {e}")
            return "", 0.0
    
    def get_final_transcript(self) -> str:
        """Get final complete transcript."""
        return self.previous_transcript.strip()
    
    def reset(self):
        """Reset state for new speech segment."""
        self.accumulated_audio = np.array([], dtype=np.float32)
        self.previous_transcript = ""
        self.transcribed_until_samples = 0
        self.segment_history = []
        logger.debug("🔄 WhisperStreamingContext reset")



