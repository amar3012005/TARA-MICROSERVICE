"""
Test configuration and fixtures for RAG service tests.

Reference:
    - services/intent/tests/conftest.py - Test fixture pattern
"""

import pytest
import asyncio
import tempfile
import os
from typing import Generator

from leibniz_agent.services.rag.config import RAGConfig
from leibniz_agent.services.rag.rag_engine import RAGEngine
from leibniz_agent.services.rag.index_builder import IndexBuilder


@pytest.fixture
def rag_config() -> RAGConfig:
    """RAG configuration for testing."""
    return RAGConfig(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "test_key"),
        knowledge_base_path="leibniz_knowledge_base",
        vector_store_path="leibniz_agent/services/rag/index",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        gemini_model="gemini-2.0-flash-lite",
        top_k=8,
        top_n=5,
        similarity_threshold=0.3,
        chunk_size_min=500,
        chunk_size_max=800,
        chunk_overlap=100,
        response_style="friendly_casual",
        max_response_length=500,
        enable_humanization=True,
        min_quality_score=0.5,
        timeout=30.0,
        enable_streaming=True,
        cache_ttl=3600,
        log_queries=False,
        verbose=False
    )


@pytest.fixture
def rag_engine(rag_config) -> RAGEngine:
    """RAG engine instance for testing."""
    return RAGEngine(rag_config)


@pytest.fixture
def index_builder(rag_config) -> IndexBuilder:
    """Index builder instance for testing."""
    return IndexBuilder(rag_config)


@pytest.fixture
def sample_knowledge_base() -> Generator[str, None, None]:
    """Create temporary knowledge base for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample category directory
        category_dir = os.path.join(tmpdir, "01_test_category")
        os.makedirs(category_dir)
        
        # Create sample markdown file
        test_file = os.path.join(category_dir, "test_doc.md")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("""# Test Document

This is a test document for RAG testing.

## Section 1

Information about section 1 with sufficient content to create a meaningful chunk.

## Section 2

Information about section 2 with sufficient content to create another meaningful chunk.
""")
        
        yield tmpdir


# Configure pytest
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (with Redis, mocked Gemini)")
    config.addinivalue_line("markers", "requires_service: Tests requiring running service")
    config.addinivalue_line("markers", "slow: Tests taking >5 seconds")
