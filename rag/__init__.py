"""
RAG (Retrieval-Augmented Generation) Microservice for Leibniz Agent

This package provides context-aware knowledge base queries via HTTP REST API,
using FAISS vector search for document retrieval and Gemini 2.0 Flash for
response generation.

Components:
    - RAGEngine: Core RAG processing with FAISS retrieval and Gemini generation
    - RAGConfig: Configuration dataclass with environment variable loading
    - IndexBuilder: FAISS index builder for knowledge base ingestion
    - Chunking strategies: Intelligent text chunking (FAQ, sections, semantic)
    - FastAPI app: HTTP REST endpoints at port 8003

Reference:
    - Cloud Transformation doc (lines 474-641) - RAG service specifications
    - leibniz_rag.py (1418 lines) - Monolithic RAG implementation
    - Main entry point: leibniz_agent/services/rag/app.py

Architecture:
    Client → FastAPI → Redis Cache → RAGEngine → FAISS + Gemini → Response

Endpoints:
    - POST /api/v1/query - Process knowledge base query with context
    - GET /health - Service health check with index status
    - GET /metrics - Performance metrics
    - POST /api/v1/admin/rebuild_index - Rebuild FAISS index
"""

from leibniz_agent.services.rag.config import RAGConfig
from leibniz_agent.services.rag.rag_engine import RAGEngine
from leibniz_agent.services.rag.index_builder import IndexBuilder

__all__ = [
    "RAGConfig",
    "RAGEngine",
    "IndexBuilder",
]

__version__ = "1.0.0"
