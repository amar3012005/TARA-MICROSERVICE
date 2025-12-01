"""
Intelligent Chunking Strategies for RAG System

Extracted from leibniz_rag.py for microservice deployment.

Reference:
    - leibniz_rag.py (lines 391-624) - Original chunking logic

Strategies:
    1. FAQ Q&A pairs: Split by question-answer patterns
    2. Markdown sections: Split by headers with size constraints
    3. Semantic paragraphs: Split by paragraphs with overlap
    4. Sentence-based: Simple sentence chunking with overlap
"""

import re
from typing import List
from leibniz_agent.services.rag.config import RAGConfig


def intelligent_chunk_text(content: str, filename: str, config: RAGConfig) -> List[str]:
    """
    Apply different chunking strategies based on file type.
    
    Args:
        content: Text content to chunk
        filename: Source filename for strategy selection
        config: RAG configuration with chunk size constraints
        
    Returns:
        List of text chunks
    """
    # FAQ files - split by Q&A pairs
    if 'faq' in filename.lower() or 'frequently' in filename.lower():
        return split_qa_content(content, config)
    
    # Guide/procedure files - split by sections
    if any(term in filename.lower() for term in ['guide', 'process', 'procedure', 'enrollment', 'admission']):
        return split_by_sections(content, config)
    
    # Default - semantic chunking
    return split_text_semantically(content, config)


def split_qa_content(content: str, config: RAGConfig) -> List[str]:
    """
    Split FAQ content by question-answer pairs.
    
    Args:
        content: FAQ text content
        config: RAG configuration
        
    Returns:
        List of Q&A pair chunks
    """
    chunks = []
    
    # English Q&A patterns (start-anchored)
    qa_patterns = [
        r'^\s*[Qq]\d*[\.\):]\s+',
        r'^\s*Question\s*[\d\.\):]*\s*',
        r'^\s*FAQ\s*[\d\.\):]*\s*'
    ]
    
    # Split by Q&A patterns
    current_chunk = ""
    lines = content.split('\n')
    
    for line in lines:
        is_question = any(re.match(pattern, line) for pattern in qa_patterns)
        
        if is_question and current_chunk:
            # Save previous Q&A pair
            if 50 < len(current_chunk.strip()) < config.chunk_size_max:
                chunks.append(current_chunk.strip())
            elif len(current_chunk.strip()) >= config.chunk_size_max:
                # Split oversized chunk with overlap
                sub_chunks = split_into_chunks(current_chunk, config.chunk_size_max, config.chunk_overlap)
                chunks.extend(sub_chunks)
            else:
                # Too short, continue accumulating
                pass
            
            current_chunk = line + '\n'
        else:
            current_chunk += line + '\n'
    
    # Add final chunk
    if current_chunk.strip():
        if 50 < len(current_chunk.strip()) < config.chunk_size_max:
            chunks.append(current_chunk.strip())
        elif len(current_chunk.strip()) >= config.chunk_size_max:
            sub_chunks = split_into_chunks(current_chunk, config.chunk_size_max, config.chunk_overlap)
            chunks.extend(sub_chunks)
    
    # If no chunks created, fallback to semantic splitting
    if not chunks:
        return split_text_semantically(content, config)
    
    return chunks


def split_by_sections(content: str, config: RAGConfig) -> List[str]:
    """
    Split content by markdown headers.
    
    Args:
        content: Markdown text content
        config: RAG configuration
        
    Returns:
        List of section chunks
    """
    chunks = []
    
    # Split by headers (## or ###)
    sections = re.split(r'\n\s*#{1,6}\s+', content)
    
    current_chunk = ""
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # If adding this section would exceed max size, save current chunk
        if len(current_chunk) + len(section) > config.chunk_size_max and current_chunk:
            if len(current_chunk) >= config.chunk_size_min:
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                # Current chunk too small, continue accumulating
                current_chunk += "\n\n" + section
        else:
            if current_chunk:
                current_chunk += "\n\n" + section
            else:
                current_chunk = section
    
    # Add final chunk
    if current_chunk.strip():
        if len(current_chunk) >= config.chunk_size_min:
            chunks.append(current_chunk.strip())
        elif chunks:
            # Add to last chunk if too small
            chunks[-1] += "\n\n" + current_chunk.strip()
    
    # If no chunks created or only one small chunk, fallback
    if not chunks or (len(chunks) == 1 and len(chunks[0]) < config.chunk_size_min):
        return split_text_semantically(content, config)
    
    return chunks


def split_text_semantically(content: str, config: RAGConfig) -> List[str]:
    """
    Split text by paragraphs with size constraints.
    
    Args:
        content: Plain text content
        config: RAG configuration
        
    Returns:
        List of paragraph chunks
    """
    chunks = []
    
    # Split by paragraphs
    paragraphs = content.split('\n\n')
    
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If paragraph is very large, split it
        if len(para) > config.chunk_size_max:
            # Save current chunk if any
            if current_chunk and len(current_chunk) >= config.chunk_size_min:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split large paragraph
            sub_chunks = split_large_paragraph(para, config)
            chunks.extend(sub_chunks)
        
        # If adding this paragraph would exceed max size
        elif len(current_chunk) + len(para) > config.chunk_size_max:
            if len(current_chunk) >= config.chunk_size_min:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                # Current chunk too small, continue accumulating
                current_chunk += "\n\n" + para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Add final chunk
    if current_chunk.strip() and len(current_chunk) >= config.chunk_size_min:
        chunks.append(current_chunk.strip())
    elif current_chunk.strip() and chunks:
        # Add to last chunk
        chunks[-1] += "\n\n" + current_chunk.strip()
    
    # Fallback to split_into_chunks if no valid chunks
    if not chunks:
        return split_into_chunks(content, config.chunk_size_max, config.chunk_overlap)
    
    return chunks


def split_large_paragraph(paragraph: str, config: RAGConfig) -> List[str]:
    """
    Split large paragraph by sentences.
    
    Args:
        paragraph: Large paragraph text (>800 chars)
        config: RAG configuration
        
    Returns:
        List of sentence chunks with overlap
    """
    chunks = []
    
    # Split by sentences (English period)
    sentences = paragraph.split('. ')
    
    current_chunk = ""
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Add period back (except for last sentence which may have it)
        if i < len(sentences) - 1:
            sentence += '.'
        
        if len(current_chunk) + len(sentence) > config.chunk_size_max:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap (last sentence)
                current_chunk = sentence
            else:
                # Single sentence too long, force split
                chunks.append(sentence[:config.chunk_size_max])
                current_chunk = sentence[config.chunk_size_max - config.chunk_overlap:]  # overlap
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [paragraph]


def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Simple sentence-based chunking with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between consecutive chunks in characters
        
    Returns:
        List of chunks with overlap
    """
    chunks = []
    
    # Split by sentences (English period)
    sentences = text.split('. ')
    
    current_chunk = ""
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Add period back
        if i < len(sentences) - 1:
            sentence += '.'
        
        if len(current_chunk) + len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                
                # Create overlap by including last part of current chunk
                if len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Handle final chunk
    if current_chunk.strip():
        # Avoid very small final chunk by merging with previous
        if len(current_chunk) < 200 and chunks:
            chunks[-1] += " " + current_chunk.strip()
        else:
            chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]
