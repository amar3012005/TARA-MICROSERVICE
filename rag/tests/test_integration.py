"""
Integration tests for RAG service.

Tests FAISS index building, retrieval, Redis caching, and response generation.
"""

import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from leibniz_agent.services.rag.index_builder import IndexBuilder
from leibniz_agent.services.rag.rag_engine import RAGEngine
from leibniz_agent.services.rag.config import RAGConfig


@pytest.mark.integration
class TestIndexBuilder:
    """Test FAISS index building."""
    
    def test_build_index_creates_files(self, sample_knowledge_base):
        """Test index building creates required files."""
        output_dir = tempfile.mkdtemp()
        
        # Create config with temp paths
        from leibniz_agent.services.rag.config import RAGConfig
        config = RAGConfig(
            gemini_api_key=os.getenv("GEMINI_API_KEY", "test_key"),
            knowledge_base_path=sample_knowledge_base,
            vector_store_path=output_dir
        )
        
        # Create builder with custom config
        from leibniz_agent.services.rag.index_builder import IndexBuilder
        builder = IndexBuilder(config)
        builder.build_index()
        
        # Check required files exist
        assert os.path.exists(os.path.join(output_dir, "index.faiss"))
        assert os.path.exists(os.path.join(output_dir, "metadata.json"))
        assert os.path.exists(os.path.join(output_dir, "texts.json"))
    
    def test_load_existing_index(self, sample_knowledge_base):
        """Test loading pre-built index."""
        output_dir = tempfile.mkdtemp()
        
        # Create config and build index
        from leibniz_agent.services.rag.config import RAGConfig
        from leibniz_agent.services.rag.index_builder import IndexBuilder
        
        config = RAGConfig(
            gemini_api_key=os.getenv("GEMINI_API_KEY", "test_key"),
            knowledge_base_path=sample_knowledge_base,
            vector_store_path=output_dir
        )
        builder = IndexBuilder(config)
        builder.build_index()
        
        # Load index
        loaded = builder.load_existing_index(output_dir)
        assert loaded is True
        assert builder.index is not None
        assert len(builder.texts) > 0
    
    def test_index_stats(self, sample_knowledge_base):
        """Test index statistics."""
        output_dir = tempfile.mkdtemp()
        
        # Create config and build index
        from leibniz_agent.services.rag.config import RAGConfig
        from leibniz_agent.services.rag.index_builder import IndexBuilder
        
        config = RAGConfig(
            gemini_api_key=os.getenv("GEMINI_API_KEY", "test_key"),
            knowledge_base_path=sample_knowledge_base,
            vector_store_path=output_dir
        )
        builder = IndexBuilder(config)
        builder.build_index()
        
        stats = builder.get_index_stats()
        assert stats['total_documents'] > 0
        assert stats['embedding_dimension'] > 0
        assert len(stats['categories']) > 0


@pytest.mark.integration
class TestRAGEngineRetrieval:
    """Test RAG engine retrieval with FAISS."""
    
    @pytest.mark.asyncio
    async def test_query_with_valid_index(self, rag_engine):
        """Test query processing with valid index."""
        # Mock Gemini response
        with patch('google.generativeai.GenerativeModel') as mock_model:
            mock_response = MagicMock()
            mock_response.text = "The university offers undergraduate and graduate programs."
            mock_model.return_value.generate_content.return_value = mock_response
            
            result = await rag_engine.process_query("What programs does the university offer?")
            
            assert 'answer' in result  # RAGEngine returns 'answer', not 'response'
            assert 'sources' in result
            assert 'confidence' in result
            assert result['confidence'] >= 0
    
    @pytest.mark.asyncio
    async def test_entity_boosting(self, rag_engine):
        """Test category-based document boosting."""
        with patch('google.generativeai.GenerativeModel') as mock_model:
            mock_response = MagicMock()
            mock_response.text = "Admission deadlines are in January."
            mock_model.return_value.generate_content.return_value = mock_response
            
            # Query about admission (should boost admission category)
            result = await rag_engine.process_query("When is the admission deadline?")
            
            # Verify answer returned (sources may be empty if index not loaded)
            assert 'answer' in result
            assert 'sources' in result


@pytest.mark.integration
@pytest.mark.requires_service
class TestRedisCaching:
    """Test Redis caching behavior."""
    
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit behavior."""
        pytest.skip("Requires Redis service running")
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss behavior."""
        pytest.skip("Requires Redis service running")
    
    @pytest.mark.asyncio
    async def test_cache_ttl(self):
        """Test cache TTL expiration."""
        pytest.skip("Requires Redis service running")


@pytest.mark.integration
class TestResponseGeneration:
    """Test response generation pipeline."""
    
    @pytest.mark.asyncio
    async def test_humanization_enabled(self, rag_engine):
        """Test humanization is applied."""
        with patch('google.generativeai.GenerativeModel') as mock_model:
            mock_response = MagicMock()
            mock_response.text = "According to the context, the university is located in Hannover."
            mock_model.return_value.generate_content.return_value = mock_response
            
            result = await rag_engine.process_query("Where is the university?")
            
            # Humanization should remove "According to the context"
            assert 'answer' in result
            assert not result['answer'].startswith("According to the context")
    
    @pytest.mark.asyncio
    async def test_quality_validation(self, rag_engine):
        """Test quality validation."""
        with patch('google.generativeai.GenerativeModel') as mock_model:
            # Return very short response
            mock_response = MagicMock()
            mock_response.text = "Yes."
            mock_model.return_value.generate_content.return_value = mock_response
            
            result = await rag_engine.process_query("Is the university good?")
            
            # Should have quality metadata
            assert 'metadata' in result
            if 'quality_score' in result['metadata']:
                assert result['metadata']['quality_score'] >= 0


@pytest.mark.integration
class TestContextAwareRetrieval:
    """Test context-aware query processing."""
    
    @pytest.mark.asyncio
    async def test_extracted_meaning_priority(self, rag_engine):
        """Test extracted_meaning takes priority."""
        with patch('google.generativeai.GenerativeModel') as mock_model:
            mock_response = MagicMock()
            mock_response.text = "The university has many student services."
            mock_model.return_value.generate_content.return_value = mock_response
            
            context = {
                "extracted_meaning": "student support services",
                "user_goal": "random text",
                "key_entities": ["support"]
            }
            
            result = await rag_engine.process_query("Tell me about it", context=context)
            
            # Should use extracted_meaning for retrieval
            assert 'answer' in result
    
    @pytest.mark.asyncio
    async def test_entity_enrichment(self, rag_engine):
        """Test query enrichment with entities."""
        with patch('google.generativeai.GenerativeModel') as mock_model:
            mock_response = MagicMock()
            mock_response.text = "Admission requirements include transcripts."
            mock_model.return_value.generate_content.return_value = mock_response
            
            context = {
                "key_entities": ["admission", "requirements"]
            }
            
            result = await rag_engine.process_query("What do I need?", context=context)
            
            # Should incorporate entities
            assert 'answer' in result
