"""
Sentence Splitter

Extracted from leibniz_pro.py (lines 960-1022) for sentence-level chunking.
Handles abbreviation protection, punctuation preservation, and fallback chunking.
"""

import re
from typing import List


def split_into_sentences(text: str, min_words: int = 10) -> List[str]:
    """
    Split text into sentences for streaming TTS (English-focused with TARA pattern)
    
    CRITICAL: Short texts (< min_words total) are returned as a single sentence.
    This prevents breaking up short greetings like "Hello! How can I help you?"
    
    Args:
        text: Input text
        min_words: Minimum word count before splitting is allowed (default: 10)
        
    Returns:
        List of sentences with preserved punctuation
    """
    text = text.strip()
    if not text:
        return []
    
    # CRITICAL: If the entire text has fewer than min_words, return as-is (single sentence)
    total_words = len(text.split())
    if total_words < min_words:
        return [text]
    
    # Preserve abbreviations by protecting periods
    abbrev_map = {
        "Dr.": "Dr<PERIOD>",
        "Mr.": "Mr<PERIOD>",
        "Mrs.": "Mrs<PERIOD>",
        "Ms.": "Ms<PERIOD>",
        "Prof.": "Prof<PERIOD>",
        "Sr.": "Sr<PERIOD>",
        "Jr.": "Jr<PERIOD>",
        "etc.": "etc<PERIOD>",
        "vs.": "vs<PERIOD>",
        "e.g.": "e<PERIOD>g<PERIOD>",
        "i.e.": "i<PERIOD>e<PERIOD>",
    }
    
    # Replace abbreviations
    for abbrev, placeholder in abbrev_map.items():
        text = text.replace(abbrev, placeholder)
    
    # Split on sentence delimiters - TARA pattern
    # Only split on strong terminators (.?!) - NOT on comma/semicolon
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    
    # Filter empty and very short fragments
    valid_sentences = []
    current_fragment = ""

    def should_flush(fragment: str) -> bool:
        """
        Decide whether to emit the current fragment as a sentence.
        A fragment is emitted when it contains at least min_words words.
        """
        if not fragment:
            return False
        return len(fragment.split()) >= min_words
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Restore abbreviations
        for abbrev, placeholder in abbrev_map.items():
            sentence = sentence.replace(placeholder, abbrev)
            
        if not sentence:
            continue

        # Accumulate fragment until we have enough words
        if current_fragment:
            current_fragment = (current_fragment + " " + sentence).strip()
        else:
            current_fragment = sentence

        if should_flush(current_fragment):
            valid_sentences.append(current_fragment)
            current_fragment = ""
            
    # Add any remaining fragment
    if current_fragment:
        valid_sentences.append(current_fragment.strip())
    
    # If no valid sentences found (no punctuation), split on length
    if not valid_sentences and text.strip():
        # Fallback: split long text into ~50 char chunks at word boundaries
        words = text.split()
        chunk = []
        chunk_len = 0
        for word in words:
            chunk.append(word)
            chunk_len += len(word) + 1
            if chunk_len >= 50:
                valid_sentences.append(' '.join(chunk))
                chunk = []
                chunk_len = 0
        if chunk:
            valid_sentences.append(' '.join(chunk))
    
    return valid_sentences
