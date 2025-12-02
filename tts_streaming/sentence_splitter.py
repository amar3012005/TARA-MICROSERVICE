"""
Sentence Splitter

Extracted from leibniz_pro.py (lines 960-1022) for sentence-level chunking.
Handles abbreviation protection, punctuation preservation, and fallback chunking.
"""

import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences for streaming TTS (English-focused with TARA pattern)
    
    Args:
        text: Input text
        
    Returns:
        List of sentences with preserved punctuation
    """
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
    # Keep punctuation with sentences
    # Added comma (,) and semicolon (;) to split long sentences for lower latency
    sentences = re.split(r'(?<=[.?!,;])\s+', text.strip())
    
    # Filter empty and very short fragments
    valid_sentences = []
    current_fragment = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Restore abbreviations
        for abbrev, placeholder in abbrev_map.items():
            sentence = sentence.replace(placeholder, abbrev)
            
        if not sentence:
            continue
            
        # If fragment is too short (e.g. just a comma split like "However,"), buffer it
        # Unless it ends with a strong terminator (.?!)
        is_strong_end = sentence[-1] in '.?!'
        
        if len(current_fragment) + len(sentence) < 20 and not is_strong_end:
            current_fragment += " " + sentence
        else:
            if current_fragment:
                full_sentence = (current_fragment + " " + sentence).strip()
                current_fragment = ""
            else:
                full_sentence = sentence
            
            valid_sentences.append(full_sentence)
            
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


