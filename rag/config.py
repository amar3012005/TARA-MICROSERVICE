"""
RAG Service Configuration

Extracted from leibniz_rag.py for microservice deployment.
Supports TARA mode for Telugu TASK organization customer service.

Reference:
    - leibniz_rag.py (lines 65-160) - Original configuration logic
    - services/intent/config.py - Configuration pattern
    - .env.leibniz (lines 64-103) - RAG environment variables
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Supported languages for TARA mode
SUPPORTED_LANGUAGES = {
    "en": "English",
    "te": "Telugu (pure)",
    "te-mixed": "Telugu mixed with English (Tenglish)",
    "hi": "Hindi",
    "hi-mixed": "Hindi mixed with English (Hinglish)"
}


@dataclass
class RAGConfig:
    """
    Configuration for RAG microservice.
    
    Attributes:
        knowledge_base_path: Knowledge base directory path (required)
        vector_store_path: Vector store directory (default: /app/index)
        embedding_model_name: HuggingFace model (default: sentence-transformers/all-MiniLM-L6-v2)
        gemini_api_key: Gemini API key for response generation
        gemini_model: Gemini model name (default: gemini-2.0-flash-lite)
        top_k: Top-K candidates to retrieve (default: 8)
        top_n: Top-N documents after filtering (default: 5)
        similarity_threshold: Minimum similarity (default: 0.3)
        chunk_size_min: Minimum chunk size (default: 500)
        chunk_size_max: Maximum chunk size (default: 800)
        chunk_overlap: Overlap between chunks (default: 100)
        response_style: Response style (default: friendly_casual)
        max_response_length: Max response length (default: 500)
        enable_humanization: Enable humanization (default: true)
        min_quality_score: Min quality score (default: 0.5)
        timeout: Query timeout (default: 30.0s)
        enable_streaming: Enable streaming responses (default: true)
        cache_ttl: Redis cache TTL in seconds (default: 3600)
        enable_hybrid_search: Enable hybrid search with pattern detection (default: true)
    """
    
    # Required settings
    gemini_api_key: str
    knowledge_base_path: str
    
    # Vector store settings
    vector_store_path: str = "/app/index"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Gemini settings
    gemini_model: str = "models/gemini-2.5-flash-lite"
    
    # Retrieval settings
    top_k: int = 8
    top_n: int = 5
    similarity_threshold: float = 0.3
    
    # Chunking settings
    chunk_size_min: int = 500
    chunk_size_max: int = 800
    chunk_overlap: int = 100
    
    # Response settings
    response_style: str = "friendly_casual"
    max_response_length: int = 500
    enable_humanization: bool = True
    min_quality_score: float = 0.5
    timeout: float = 30.0
    enable_streaming: bool = True
    
    # Cache settings
    cache_ttl: int = 3600
    
    # Prewarming settings
    enable_prewarming: bool = field(default_factory=lambda: os.getenv("ENABLE_PREWARMING", "true").lower() == "true")
    warmup_queries_count: int = field(default_factory=lambda: int(os.getenv("WARMUP_QUERIES_COUNT", "10")))
    enable_model_persistence: bool = field(default_factory=lambda: os.getenv("ENABLE_MODEL_PERSISTENCE", "true").lower() == "true")
    prepopulate_cache: bool = field(default_factory=lambda: os.getenv("PREPOPULATE_CACHE", "true").lower() == "true")

    # Performance tuning
    gemini_timeout_ms: int = field(default_factory=lambda: int(os.getenv("GEMINI_TIMEOUT_MS", "5000")))
    embedding_batch_size: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))
    
    # Hybrid search settings
    enable_hybrid_search: bool = True
    
    # TARA Mode Configuration (Telugu TASK Customer Service)
    tara_mode: bool = True  # Default: TARA enabled for TASK
    response_language: str = "te-mixed"  # Default: Telugu mixed with English
    organization_name: str = "T.A.S.K"  # Default: Telangana Academy for Skill and Knowledge
    agent_name: str = "TARA"  # Default: TARA agent (T.A.S.K AI Response Assistant)
    
    # Logging
    log_queries: bool = False
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        
        # Validate knowledge base path
        if not os.path.exists(self.knowledge_base_path):
            logger.warning(
                f"️ Knowledge base path does not exist: {self.knowledge_base_path}"
            )
        
        # Validate similarity threshold
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
            )
        
        # Validate top_k >= top_n
        if self.top_k < self.top_n:
            raise ValueError(
                f"top_k ({self.top_k}) must be >= top_n ({self.top_n})"
            )
        
        # Validate chunk sizes
        if self.chunk_size_max <= self.chunk_size_min:
            raise ValueError(
                f"chunk_size_max ({self.chunk_size_max}) must be > chunk_size_min ({self.chunk_size_min})"
            )
        
        # Validate chunk overlap
        if self.chunk_overlap >= self.chunk_size_min:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size_min ({self.chunk_size_min})"
            )
        
        # Validate timeout
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        
        # Validate quality score
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError(
                f"min_quality_score must be between 0.0 and 1.0, got {self.min_quality_score}"
            )
        
        # Warn if Gemini API key missing
        if not self.gemini_api_key:
            logger.warning(
                "️ GEMINI_API_KEY not set. Response generation will fail."
            )
        
        # Validate response language
        if self.response_language not in SUPPORTED_LANGUAGES:
            logger.warning(
                f"⚠️ Unsupported response_language '{self.response_language}'. "
                f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        
        # Log TARA mode configuration
        if self.tara_mode:
            logger.info("=" * 70)
            logger.info("🇮🇳 TARA MODE ENABLED - RAG Service")
            logger.info(f"   Agent: {self.agent_name}")
            logger.info(f"   Organization: {self.organization_name}")
            logger.info(f"   Language: {self.response_language} ({SUPPORTED_LANGUAGES.get(self.response_language, 'Unknown')})")
            logger.info("=" * 70)
        
        # Log configuration if verbose
        if self.verbose:
            logger.info(
                f" RAGConfig loaded: model={self.gemini_model}, "
                f"top_k={self.top_k}, top_n={self.top_n}, "
                f"similarity_threshold={self.similarity_threshold}, "
                f"chunk_size={self.chunk_size_min}-{self.chunk_size_max}, "
                f"timeout={self.timeout}s, cache_ttl={self.cache_ttl}s"
            )
    
    @staticmethod
    def from_env() -> "RAGConfig":
        """
        Load configuration from environment variables.
        
        Returns:
            RAGConfig instance with values from environment
        
        Environment Variables:
            GEMINI_API_KEY: Gemini API key (required)
            GEMINI_MODEL: Gemini model name
            LEIBNIZ_RAG_KNOWLEDGE_BASE_PATH: Knowledge base directory
            LEIBNIZ_RAG_VECTOR_STORE_PATH: Vector store directory
            LEIBNIZ_RAG_EMBEDDING_MODEL: Embedding model name
            LEIBNIZ_RAG_TOP_K: Top-K candidates
            LEIBNIZ_RAG_TOP_N: Top-N documents
            LEIBNIZ_RAG_SIMILARITY_THRESHOLD: Minimum similarity
            LEIBNIZ_RAG_CHUNK_SIZE_MIN: Minimum chunk size
            LEIBNIZ_RAG_CHUNK_SIZE_MAX: Maximum chunk size
            LEIBNIZ_RAG_CHUNK_OVERLAP: Chunk overlap
            LEIBNIZ_RAG_RESPONSE_STYLE: Response style
            LEIBNIZ_RAG_MAX_RESPONSE_LENGTH: Max response length
            LEIBNIZ_RAG_ENABLE_HUMANIZATION: Enable humanization
            LEIBNIZ_RAG_MIN_QUALITY_SCORE: Min quality score
            LEIBNIZ_RAG_TIMEOUT: Query timeout
            LEIBNIZ_RAG_ENABLE_STREAMING: Enable streaming
            LEIBNIZ_RAG_CACHE_TTL: Cache TTL
            LEIBNIZ_RAG_ENABLE_HYBRID_SEARCH: Enable hybrid search (pattern-based optimization)
            LOG_LEVEL: Logging level
        """
        # Get API key - use default if not set or empty
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not gemini_api_key:
            gemini_api_key = "AIzaSyBiKXVYsQ0UcxRicctFI1U5dEpQct2ieOA"
        
        return RAGConfig(
            # Required settings
            gemini_api_key=gemini_api_key,
            knowledge_base_path=os.getenv(
                "TARA_RAG_KNOWLEDGE_BASE_PATH",
                os.getenv("LEIBNIZ_RAG_KNOWLEDGE_BASE_PATH", "/app/task_knowledge_base")
            ),
            
            # Vector store settings
            vector_store_path=os.getenv(
                "TARA_RAG_VECTOR_STORE_PATH",
                os.getenv("LEIBNIZ_RAG_VECTOR_STORE_PATH", "/app/index")
            ),
            embedding_model_name=os.getenv(
                "TARA_RAG_EMBEDDING_MODEL",
                os.getenv("LEIBNIZ_RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            ),
            
            # Gemini settings
            gemini_model=os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash-lite"),
            
            # Retrieval settings
            top_k=int(os.getenv("LEIBNIZ_RAG_TOP_K", "8")),
            top_n=int(os.getenv("LEIBNIZ_RAG_TOP_N", "5")),
            similarity_threshold=float(os.getenv("LEIBNIZ_RAG_SIMILARITY_THRESHOLD", "0.3")),
            
            # Chunking settings
            chunk_size_min=int(os.getenv("LEIBNIZ_RAG_CHUNK_SIZE_MIN", "500")),
            chunk_size_max=int(os.getenv("LEIBNIZ_RAG_CHUNK_SIZE_MAX", "800")),
            chunk_overlap=int(os.getenv("LEIBNIZ_RAG_CHUNK_OVERLAP", "100")),
            
            # Response settings
            response_style=os.getenv("LEIBNIZ_RAG_RESPONSE_STYLE", "friendly_casual"),
            max_response_length=int(os.getenv("LEIBNIZ_RAG_MAX_RESPONSE_LENGTH", "500")),
            enable_humanization=os.getenv("LEIBNIZ_RAG_ENABLE_HUMANIZATION", "true").lower() == "true",
            min_quality_score=float(os.getenv("LEIBNIZ_RAG_MIN_QUALITY_SCORE", "0.5")),
            timeout=float(os.getenv("LEIBNIZ_RAG_TIMEOUT", "30.0")),
            enable_streaming=os.getenv("LEIBNIZ_RAG_ENABLE_STREAMING", "true").lower() == "true",
            
            # Cache settings
            cache_ttl=int(os.getenv("LEIBNIZ_RAG_CACHE_TTL", "3600")),
            
            # Hybrid search settings
            enable_hybrid_search=os.getenv("LEIBNIZ_RAG_ENABLE_HYBRID_SEARCH", "true").lower() == "true",
            
            # TARA Mode Configuration (Telugu TASK Customer Service)
            tara_mode=os.getenv("TARA_MODE", "true").lower() == "true",  # Default: TARA enabled
            response_language=os.getenv("TARA_RAG_RESPONSE_LANGUAGE", os.getenv("LEIBNIZ_RAG_RESPONSE_LANGUAGE", "te-mixed")),  # Default: Telugu mixed
            organization_name=os.getenv("TARA_RAG_ORGANIZATION", os.getenv("LEIBNIZ_RAG_ORGANIZATION", "TASK")),  # Default: TASK
            agent_name=os.getenv("TARA_RAG_AGENT_NAME", os.getenv("LEIBNIZ_RAG_AGENT_NAME", "TARA")),  # Default: TARA
            
            # Logging
            log_queries=os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG",
            verbose=os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG",
        )
