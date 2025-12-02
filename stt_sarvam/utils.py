"""
Utility Functions for STT/VAD Microservice

Provides shared utilities for transcript normalization and audio validation.

Functions:
    normalize_english_transcript: Comprehensive English transcript normalization
    validate_audio_chunk: Validate incoming PCM audio chunks
    format_transcript_fragment: Format transcript fragment for WebSocket response
    TranscriptBuffer: Smart transcript fragment accumulator with word boundary detection

Reference:
    leibniz_agent/leibniz_stt.py (lines 454-566) - normalize_english_transcript
    leibniz_agent/leibniz_vad.py (lines 255-380) - TranscriptBuffer
"""

import re
import time
import logging
from typing import Dict, Any, List
from collections import deque

logger = logging.getLogger(__name__)

# ============================================================================
# Transcript Normalization
# ============================================================================

def normalize_english_transcript(text: str) -> str:
    """
    Enhanced English transcript normalization with comprehensive cleanup.
    
    Ported from leibniz_stt.py (lines 454-566).
    
    Performs the following normalization steps in order:
    1. Strip whitespace and convert to lowercase
    2. Remove transcription artifacts ([inaudible], [unclear], [silence])
    3. Remove stuttering (repeated word fragments: "I I I want" → "I want")
    4. Remove repeated consecutive words ("the the book" → "the book")
    5. Remove expanded set of English filler words (um, uh, like, well, etc.)
    6. Clean up excessive punctuation (multiple periods, question marks)
    7. Strip leading/trailing punctuation
    8. Preserve patterns: phone numbers (10+ digits), emails (@), URLs (http/www)
    
    Args:
        text: Raw transcript text from STT engine
        
    Returns:
        Normalized, cleaned transcript
        
    Examples:
        >>> normalize_english_transcript("Um, I I I want the the book.")
        "I want the book"
        
        >>> normalize_english_transcript("So, like, th-th-thank you very much!")
        "thank you very much"
        
        >>> normalize_english_transcript("[unclear] My number is 555-123-4567")
        "My number is 555-123-4567"
    """
    if not text:
        return ""
    
    # Basic cleanup: strip whitespace (preserve original capitalization)
    text = text.strip()
    
    # Step 1: Remove transcription artifacts
    artifact_patterns = [
        r'\[inaudible\]', r'\[unclear\]', r'\[silence\]',
        r'\[noise\]', r'\[background\]', r'\[music\]',
        r'\[crosstalk\]', r'\[overlapping\]'
    ]
    for pattern in artifact_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Step 2: Remove stuttering (repeated word fragments with hyphens)
    # Matches patterns like "th-th-thank", "I-I-I", "w-w-wait"
    text = re.sub(r'\b(\w+)-(\1-)*\1\b', r'\1', text, flags=re.IGNORECASE)
    
    # Step 3: Remove consecutive duplicate words (case-insensitive matching)
    # Matches "the the", "I I I", "and and", etc.
    # Preserve first occurrence's capitalization
    def dedup_words(match):
        return match.group(1)
    text = re.sub(r'\b(\w+)(\s+\1)+\b', dedup_words, text, flags=re.IGNORECASE)
    
    # Step 4: Preserve important patterns before filler removal
    # Phone numbers: preserve sequences of 10+ digits (with optional separators)
    phone_pattern = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phones = re.findall(phone_pattern, text)
    phone_placeholders = {f"__PHONE{i}__": phone for i, phone in enumerate(phones)}
    for placeholder, phone in phone_placeholders.items():
        text = text.replace(phone, placeholder)
    
    # Email addresses: preserve anything with @
    email_pattern = r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    email_placeholders = {f"__EMAIL{i}__": email for i, email in enumerate(emails)}
    for placeholder, email in email_placeholders.items():
        text = text.replace(email, placeholder)
    
    # URLs: preserve http/https/www patterns
    url_pattern = r'\b(?:https?://|www\.)\S+\b'
    urls = re.findall(url_pattern, text)
    url_placeholders = {f"__URL{i}__": url for i, url in enumerate(urls)}
    for placeholder, url in url_placeholders.items():
        text = text.replace(url, placeholder)
    
    # Step 5: Remove expanded set of English filler words
    filler_words = [
        r'\bum+\b', r'\buh+\b', r'\buhm+\b', r'\bhmm+\b',
        r'\blike\b', r'\byou know\b', r'\bi mean\b',
        r'\bwell\b', r'\bkind of\b', r'\bsort of\b',
        r'\bbasically\b', r'\bactually\b', r'\bliterally\b',
        r'\byou see\b', r'\bright\b', r'\bokay\b', r'\balright\b',
        r'\bso+\b', r'\banyway\b', r'\banyhow\b'
    ]
    for filler in filler_words:
        text = re.sub(filler, '', text, flags=re.IGNORECASE)
    
    # Step 6: Clean up excessive punctuation
    # Multiple periods → single period
    text = re.sub(r'\.{2,}', '.', text)
    # Multiple question marks → single
    text = re.sub(r'\?{2,}', '?', text)
    # Multiple exclamation marks → single
    text = re.sub(r'!{2,}', '!', text)
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Step 7: Restore preserved patterns
    for placeholder, phone in phone_placeholders.items():
        text = text.replace(placeholder, phone)
    for placeholder, email in email_placeholders.items():
        text = text.replace(placeholder, email)
    for placeholder, url in url_placeholders.items():
        text = text.replace(placeholder, url)
    
    # Step 8: Final cleanup
    # Remove extra spaces created by removals
    text = " ".join(text.split())
    
    # Strip leading/trailing punctuation (but keep internal punctuation)
    text = text.strip('.,!?;: ')
    
    return text.strip()


# ============================================================================
# Audio Validation
# ============================================================================

def validate_audio_chunk(audio_data: bytes) -> Dict[str, Any]:
    """
    Validate incoming PCM audio chunks.
    
    Checks:
    - Size constraints (min 100 bytes, max 100KB)
    - Format verification (raw bytes)
    - Data integrity
    
    Args:
        audio_data: Raw PCM audio bytes
        
    Returns:
        dict: {
            "valid": bool,
            "errors": List[str],
            "warnings": List[str],
            "size_bytes": int
        }
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    # Check if data exists
    if not audio_data:
        errors.append("Audio data is empty")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "size_bytes": 0
        }
    
    # Check size constraints
    size_bytes = len(audio_data)
    min_size = 100  # 100 bytes minimum
    max_size = 100 * 1024  # 100KB maximum per chunk
    
    if size_bytes < min_size:
        errors.append(f"Audio chunk too small: {size_bytes} bytes (min: {min_size})")
    
    if size_bytes > max_size:
        errors.append(f"Audio chunk too large: {size_bytes} bytes (max: {max_size})")
    
    # Check if it's bytes type
    if not isinstance(audio_data, bytes):
        errors.append(f"Audio data must be bytes, got {type(audio_data).__name__}")
    
    # Warnings for unusual sizes
    if size_bytes < 1000:
        warnings.append("Unusually small audio chunk (< 1KB)")
    
    if size_bytes > 50 * 1024:
        warnings.append("Large audio chunk (> 50KB), consider smaller chunks for smoother streaming")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "size_bytes": size_bytes
    }


# ============================================================================
# Response Formatting
# ============================================================================

# ============================================================================
# Transcript Buffering (Fix Fragmentation)
# ============================================================================

class TranscriptBuffer:
    """
    Smart transcript fragment accumulator with word boundary detection (Critical Fix #2)
    
    Prevents incomplete words like "Al ice" by buffering partial words until complete.
    Implements deduplication to handle repeated fragments from Gemini Live.
    
    Ported from leibniz_vad.py (lines 255-380).
    """
    
    def __init__(self):
        self.fragments: List[str] = []
        self.pending_partial: str = ""  # Buffered incomplete word from last fragment
        self.last_fragment: str = ""    # For deduplication
        self.total_fragments_received = 0
        self.duplicates_skipped = 0
        self.buffered_words_count = 0
    
    def add_fragment(self, text: str) -> str:
        """
        Add a transcript fragment with smart word boundary detection
        
        Args:
            text: Raw fragment from Gemini Live
            
        Returns:
            Complete text ready for display (buffered partial NOT included)
        """
        if not text or not text.strip():
            return ""
        
        self.total_fragments_received += 1
        
        # Deduplication: Skip exact duplicates
        if text == self.last_fragment:
            self.duplicates_skipped += 1
            logger.debug(f" Skipped duplicate fragment: '{text[:30]}...'")
            return ""
        
        self.last_fragment = text
        
        # Combine with any pending partial from previous fragment (NO space for word continuation)
        combined_text = (self.pending_partial + text).strip() if self.pending_partial else text
        
        # Check if fragment ends with complete word
        if self._ends_complete_word(combined_text):
            # Complete fragment - add to buffer and return
            self.fragments.append(combined_text)
            self.pending_partial = ""  # Clear buffered partial
            logger.debug(f" Complete fragment added: '{combined_text[:50]}...'")
            return combined_text
        else:
            # Incomplete word at end - buffer last word
            words = combined_text.rsplit(maxsplit=1)
            if len(words) == 2:
                complete_portion, partial_word = words
                self.fragments.append(complete_portion)
                self.pending_partial = partial_word
                self.buffered_words_count += 1
                logger.debug(
                    f" Buffered partial word: '{partial_word}' "
                    f"(complete portion: '{complete_portion[:40]}...')"
                )
                return complete_portion
            else:
                # Single incomplete word - buffer entirely
                self.pending_partial = combined_text
                self.buffered_words_count += 1
                logger.debug(f" Buffered single incomplete word: '{combined_text}'")
                return ""
    
    def _ends_complete_word(self, text: str) -> bool:
        """
        Check if text ends with a complete word (not mid-word fragment)
        
        Word boundary indicators:
        - Ends with punctuation (. , ! ? ; :)
        - Ends with whitespace
        - Last word is complete (not single letter, no trailing hyphen, etc.)
        """
        if not text:
            return False
        
        # Check if ends with punctuation or whitespace
        if text[-1] in ".,!?;: \t\n":
            return True
        
        # Get last word
        words = text.split()
        if not words:
            return False
        
        last_word = words[-1].strip()
        
        # Incomplete word patterns
        if len(last_word) == 1 and last_word.isalpha():
            # Single letter likely incomplete (except "I" or "a")
            return last_word.lower() in ["i", "a"]
        
        if last_word.endswith("-"):
            # Trailing hyphen indicates incomplete word
            return False
        
        # Check for common incomplete patterns
        incomplete_patterns = [
            r"^[A-Z]$",  # Single capital letter
            r"[a-z]-$",  # Word ending with hyphen
            r"^[a-z]{1,2}$"  # Very short words (likely partial)
        ]
        
        for pattern in incomplete_patterns:
            if re.match(pattern, last_word):
                return False
        
        # Assume complete if passed all checks
        return True
    
    def get_final_transcript(self) -> str:
        """
        Get complete final transcript including any buffered partial word
        
        Returns:
            Final transcript with all fragments joined, including pending partial
        """
        all_fragments = self.fragments.copy()
        if self.pending_partial:
            all_fragments.append(self.pending_partial)
        
        final = " ".join(all_fragments).strip()
        
        # Clear state after finalization
        self.fragments = []
        self.pending_partial = ""
        self.last_fragment = ""
        
        return final
    
    def reset(self):
        """Reset buffer state (clear all fragments and pending partial)"""
        self.fragments = []
        self.pending_partial = ""
        self.last_fragment = ""
        self.total_fragments_received = 0
        self.duplicates_skipped = 0
        self.buffered_words_count = 0


# ============================================================================
# Response Formatting
# ============================================================================

def format_transcript_fragment(text: str, is_final: bool, confidence: float = 1.0) -> Dict[str, Any]:
    """
    Format transcript fragment for WebSocket response.
    
    Args:
        text: Transcript text (raw or normalized)
        is_final: Whether this is the final transcript
        confidence: Confidence score (0.0-1.0)
        
    Returns:
        dict: {
            "transcript": str,
            "is_final": bool,
            "confidence": float,
            "timestamp_ms": int,
            "normalized": bool
        }
    """
    # Apply normalization only if final
    normalized_text = normalize_english_transcript(text) if is_final else text
    
    return {
        "transcript": normalized_text,
        "is_final": is_final,
        "confidence": confidence,
        "timestamp_ms": int(time.time() * 1000),
        "normalized": is_final
    }
