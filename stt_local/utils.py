"""
Utility Functions for STT Local Microservice

Provides shared utilities for transcript normalization and audio validation.
Reuses utilities from stt_vad service.

Reference:
    services/stt_vad/utils.py
"""

import re
import time
import logging
from typing import Dict, Any, List
from collections import deque

logger = logging.getLogger(__name__)


def normalize_english_transcript(text: str) -> str:
    """
    Enhanced English transcript normalization with comprehensive cleanup.
    
    Ported from services/stt_vad/utils.py.
    """
    if not text:
        return ""
    
    # Basic cleanup: strip whitespace
    text = text.strip()
    
    # Step 1: Remove transcription artifacts
    artifact_patterns = [
        r'\[inaudible\]', r'\[unclear\]', r'\[silence\]',
        r'\[noise\]', r'\[background\]', r'\[music\]',
        r'\[crosstalk\]', r'\[overlapping\]'
    ]
    for pattern in artifact_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Step 2: Remove stuttering
    text = re.sub(r'\b(\w+)-(\1-)*\1\b', r'\1', text, flags=re.IGNORECASE)
    
    # Step 3: Remove consecutive duplicate words
    def dedup_words(match):
        return match.group(1)
    text = re.sub(r'\b(\w+)(\s+\1)+\b', dedup_words, text, flags=re.IGNORECASE)
    
    # Step 4: Preserve important patterns
    phone_pattern = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phones = re.findall(phone_pattern, text)
    phone_placeholders = {f"__PHONE{i}__": phone for i, phone in enumerate(phones)}
    for placeholder, phone in phone_placeholders.items():
        text = text.replace(phone, placeholder)
    
    email_pattern = r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    email_placeholders = {f"__EMAIL{i}__": email for i, email in enumerate(emails)}
    for placeholder, email in email_placeholders.items():
        text = text.replace(email, placeholder)
    
    url_pattern = r'\b(?:https?://|www\.)\S+\b'
    urls = re.findall(url_pattern, text)
    url_placeholders = {f"__URL{i}__": url for i, url in enumerate(urls)}
    for placeholder, url in url_placeholders.items():
        text = text.replace(url, placeholder)
    
    # Step 5: Remove filler words
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
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Step 7: Restore preserved patterns
    for placeholder, phone in phone_placeholders.items():
        text = text.replace(placeholder, phone)
    for placeholder, email in email_placeholders.items():
        text = text.replace(placeholder, email)
    for placeholder, url in url_placeholders.items():
        text = text.replace(placeholder, url)
    
    # Step 8: Final cleanup
    text = " ".join(text.split())
    text = text.strip('.,!?;: ')
    
    return text.strip()


class TranscriptBuffer:
    """
    Smart transcript fragment accumulator with word boundary detection.
    """
    
    def __init__(self):
        self.fragments: List[str] = []
        self.pending_partial: str = ""
        self.last_fragment: str = ""
        self.total_fragments_received = 0
    
    def add_fragment(self, text: str) -> str:
        """Add transcript fragment with smart word boundary detection."""
        if not text or not text.strip():
            return ""
        
        self.total_fragments_received += 1
        
        # Deduplication
        if text == self.last_fragment:
            return ""
        
        self.last_fragment = text
        
        # Combine with pending partial
        combined_text = (self.pending_partial + text).strip() if self.pending_partial else text
        
        # Check if ends with complete word
        if self._ends_complete_word(combined_text):
            self.fragments.append(combined_text)
            self.pending_partial = ""
            return combined_text
        else:
            # Incomplete word at end
            words = combined_text.rsplit(maxsplit=1)
            if len(words) == 2:
                complete_portion, partial_word = words
                self.fragments.append(complete_portion)
                self.pending_partial = partial_word
                return complete_portion
            else:
                self.pending_partial = combined_text
                return ""
    
    def _ends_complete_word(self, text: str) -> bool:
        """Check if text ends with complete word."""
        if not text:
            return False
        
        if text[-1] in ".,!?;: \t\n":
            return True
        
        words = text.split()
        if not words:
            return False
        
        last_word = words[-1].strip()
        
        if len(last_word) == 1 and last_word.isalpha():
            return last_word.lower() in ["i", "a"]
        
        if last_word.endswith("-"):
            return False
        
        return True
    
    def get_final_transcript(self) -> str:
        """Get complete final transcript."""
        all_fragments = self.fragments.copy()
        if self.pending_partial:
            all_fragments.append(self.pending_partial)
        
        final = " ".join(all_fragments).strip()
        
        # Clear state
        self.fragments = []
        self.pending_partial = ""
        self.last_fragment = ""
        
        return final
    
    def reset(self):
        """Reset buffer state."""
        self.fragments = []
        self.pending_partial = ""
        self.last_fragment = ""
        self.total_fragments_received = 0




