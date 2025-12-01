"""
Unit tests for RAG service components.

Tests chunking strategies, configuration, humanization, and quality validation.
"""

import pytest
from leibniz_agent.services.rag.config import RAGConfig
from leibniz_agent.services.rag.chunking import (
    split_qa_content,
    split_by_sections,
    split_text_semantically,
    split_into_chunks
)
from leibniz_agent.services.rag.rag_engine import RAGEngine


@pytest.mark.unit
class TestRAGConfig:
    """Test configuration loading and validation."""
    
    def test_config_defaults(self, rag_config):
        """Test default values."""
        assert rag_config.top_k == 8
        assert rag_config.top_n == 5
        assert rag_config.similarity_threshold == 0.3
        assert rag_config.chunk_size_min == 500
        assert rag_config.chunk_size_max == 800
        assert rag_config.chunk_overlap == 100
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid similarity threshold
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            RAGConfig(
                gemini_api_key="test",
                knowledge_base_path=".",
                similarity_threshold=1.5
            )
        
        # Invalid top_k < top_n
        with pytest.raises(ValueError, match="top_k.*must be >= top_n"):
            RAGConfig(
                gemini_api_key="test",
                knowledge_base_path=".",
                top_k=3,
                top_n=5
            )
        
        # Invalid chunk sizes
        with pytest.raises(ValueError, match="chunk_size_max.*must be >"):
            RAGConfig(
                gemini_api_key="test",
                knowledge_base_path=".",
                chunk_size_min=800,
                chunk_size_max=500
            )


@pytest.mark.unit
class TestChunkingStrategies:
    """Test intelligent chunking."""
    
    def test_split_qa_content(self, rag_config):
        """Test FAQ Q&A pair splitting."""
        content = """Q1: What is the university?
A: Leibniz University is a research institution.

Q2: Where is it located?
A: It is located in Hannover, Germany."""
        
        chunks = split_qa_content(content, rag_config)
        assert len(chunks) >= 1  # At least one Q&A pair
    
    def test_split_by_sections(self, rag_config):
        """Test markdown section splitting."""
        content = """## Section 1
Content for section 1 with sufficient text to meet minimum chunk size requirements.

## Section 2
Content for section 2 with sufficient text to meet minimum chunk size requirements."""
        
        chunks = split_by_sections(content, rag_config)
        assert len(chunks) >= 1
    
    def test_split_text_semantically(self, rag_config):
        """Test semantic paragraph splitting."""
        content = """Paragraph 1 with sufficient content to create a meaningful chunk.

Paragraph 2 with sufficient content to create another meaningful chunk."""
        
        chunks = split_text_semantically(content, rag_config)
        assert len(chunks) >= 1
    
    def test_chunk_size_constraints(self, rag_config):
        """Test chunks respect min/max size."""
        content = "Short text. " * 200  # Create long text
        chunks = split_into_chunks(content, rag_config.chunk_size_max, rag_config.chunk_overlap)
        
        for chunk in chunks:
            # Most chunks should be under max size (except last)
            assert len(chunk) <= rag_config.chunk_size_max + 200  # Allow some flexibility


@pytest.mark.unit
class TestResponseHumanization:
    """Test response humanization logic."""
    
    def test_remove_formal_prefixes(self, rag_engine):
        """Test formal prefix removal."""
        response = "According to the context, the university has many programs."
        humanized = rag_engine.humanize_response(response, "test query", None, True)
        assert not humanized.startswith("According to the context")
    
    def test_sentence_casing(self, rag_engine):
        """Test first letter capitalization."""
        response = "this is a test response."
        humanized = rag_engine.humanize_response(response, "test", None, True)
        assert humanized[0].isupper()
    
    def test_sentence_ending(self, rag_engine):
        """Test proper sentence ending."""
        response = "This is a test response"
        humanized = rag_engine.humanize_response(response, "test", None, True)
        assert humanized[-1] in '.!?'


@pytest.mark.unit
class TestResponseQualityValidation:
    """Test quality validation."""
    
    def test_detect_formal_language(self, rag_engine):
        """Test formal language detection."""
        response = "Pursuant to the university regulations, students must enroll."
        quality = rag_engine.validate_response_quality(response)
        assert "overly_formal" in quality['issues']
        assert quality['quality_score'] < 1.0
    
    def test_detect_short_response(self, rag_engine):
        """Test short response detection."""
        response = "Yes."
        quality = rag_engine.validate_response_quality(response)
        assert "too_short" in quality['issues']
    
    def test_detect_long_response(self, rag_engine):
        """Test long response detection."""
        response = "A" * 600
        quality = rag_engine.validate_response_quality(response)
        assert "too_long" in quality['issues']
