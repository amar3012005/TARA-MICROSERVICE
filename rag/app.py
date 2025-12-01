"""
RAG Service FastAPI Application

HTTP REST API for knowledge base queries with Redis caching.

Reference:
    - Cloud Transformation doc (lines 474-641) - RAG service specifications
    - services/intent/app.py - FastAPI pattern
"""

import os
import sys
import time
import logging
import json
import hashlib
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from leibniz_agent.services.shared.redis_client import get_redis_client, close_redis_client, ping_redis
from leibniz_agent.services.rag.config import RAGConfig
from leibniz_agent.services.rag.rag_engine import RAGEngine
from leibniz_agent.services.rag.index_builder import IndexBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    context: Optional[Dict[str, Any]] = Field(None, description="Context from intent service")
    enable_streaming: Optional[bool] = Field(None, description="Enable streaming response")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(..., description="Source document filenames")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Retrieval confidence")
    timing_breakdown: Dict[str, float] = Field(..., description="Timing metrics")
    cached: bool = Field(..., description="Whether result was served from cache")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    index_size: int
    cache_hit_rate: float
    redis_connected: bool
    gemini_available: bool
    uptime_seconds: float


class RebuildIndexRequest(BaseModel):
    knowledge_base_path: Optional[str] = Field(None, description="Override knowledge base path")


class RebuildIndexResponse(BaseModel):
    status: str
    documents_indexed: int
    categories: int
    build_time_seconds: float


# Global state (initialized in lifespan)
rag_engine: Optional[RAGEngine] = None
redis_client: Optional[redis.Redis] = None
cache_hits = 0
cache_misses = 0
app_start_time = 0.0


# Redis client utilities
# Removed custom implementation in favor of shared client
# async def get_redis_client() -> redis.Redis: ...
# async def close_redis_client(client: redis.Redis): ...
# async def ping_redis(client: redis.Redis) -> bool: ...


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup/shutdown."""
    global rag_engine, redis_client, cache_hits, cache_misses, app_start_time
    
    # Startup
    logger.info(" Starting RAG service...")
    app_start_time = time.time()
    
    try:
        # Load config
        config = RAGConfig.from_env()
        logger.info(" Configuration loaded")
        
        # Create RAG engine
        rag_engine = RAGEngine(config)
        
        # If index not loaded, try to build it
        if not rag_engine.vector_store or not rag_engine.documents:
            logger.warning(" FAISS index not found, attempting to build...")
            from leibniz_agent.services.rag.index_builder import IndexBuilder
            builder = IndexBuilder(config)
            if builder.build_index():
                rag_engine.load_index()  # Reload after build
                logger.info(f" Index built successfully: {len(rag_engine.documents)} documents")
            else:
                logger.error(" Index build failed")
        
        # If still no index, log warning but continue (degraded mode)
        if not rag_engine.vector_store or not rag_engine.documents:
            logger.error(" FAISS index not available - service in degraded mode")
            # Don't raise - allow service to start in degraded mode
        else:
            logger.info(f" RAG engine initialized: {len(rag_engine.documents)} documents")
        
        # Connect to Redis (optional - service can run without it)
        try:
            # Use shared client which handles config from env vars
            redis_client = await asyncio.wait_for(get_redis_client(), timeout=15.0)
            await asyncio.wait_for(ping_redis(redis_client), timeout=5.0)
            logger.info(f" Redis connected successfully")
        except asyncio.TimeoutError:
            logger.warning(f" Redis connection timeout - service will run in degraded mode")
            redis_client = None
        except Exception as redis_error:
            logger.warning(f" Redis connection failed: {redis_error} - caching disabled")
            redis_client = None
        
        # Initialize counters
        cache_hits = 0
        cache_misses = 0
        
        # Store in app state
        app.state.rag_engine = rag_engine
        app.state.redis = redis_client
        app.state.cache_hits = cache_hits
        app.state.cache_misses = cache_misses
        app.state.start_time = app_start_time
        
        logger.info(" RAG service ready")
        
        yield
        
        # Shutdown
        logger.info(" Shutting down RAG service...")
        
        # Log performance stats
        if rag_engine:
            stats = rag_engine.get_performance_stats()
            logger.info(f" Performance stats: {stats}")
        
        # Close Redis
        if redis_client:
            # Shared client manages its own lifecycle, but we can close our reference
            # Actually, shared client is singleton, so we shouldn't close it here if other services use it
            # But since this is microservice, we are the only user in this process.
            # However, close_redis_client in shared lib closes the global client.
            await close_redis_client() 
            logger.info(" Redis connection closed")
        
        logger.info(" RAG service shutdown complete")
    
    except Exception as e:
        logger.error(f" Startup error: {e}", exc_info=True)
        raise


# Create FastAPI app
app = FastAPI(
    title="Leibniz RAG Service",
    description="Context-aware knowledge base queries with FAISS retrieval and Gemini generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Process knowledge base query with context-aware retrieval.
    
    Uses Redis caching with 1-hour TTL. Returns answer, sources, confidence, and timing.
    """
    try:
        # Generate cache key
        cache_key = f"rag:{hashlib.md5(request.query.encode()).hexdigest()}"
        
        # Check Redis cache (only if connected)
        cached = None
        if app.state.redis:
            try:
                cached = await app.state.redis.get(cache_key)
            except Exception as cache_read_error:
                logger.warning(f" Cache read failed: {cache_read_error}")
                cached = None
        
        if cached:
            # Cache hit
            app.state.cache_hits += 1
            result = json.loads(cached)
            result['cached'] = True
            
            logger.info(f" CACHE HIT: {request.query[:50]}...")
            return QueryResponse(**result)
        
        # Cache miss
        app.state.cache_misses += 1
        
        # Process query
        result = await app.state.rag_engine.process_query(
            request.query,
            request.context,
            streaming_callback=None  # Streaming handled separately if needed
        )
        
        # Add cached flag
        result['cached'] = False
        
        # Cache result (only if Redis is available)
        if app.state.redis:
            try:
                await app.state.redis.setex(
                    cache_key,
                    app.state.rag_engine.config.cache_ttl,
                    json.dumps({
                        'answer': result['answer'],
                        'sources': result['sources'],
                        'confidence': result['confidence'],
                        'timing_breakdown': result['timing_breakdown'],
                        'metadata': result['metadata']
                    })
                )
            except Exception as cache_error:
                logger.warning(f"️ Cache write failed: {cache_error}")
        
        # Log query
        if app.state.rag_engine.config.log_queries:
            logger.info(
                f" QUERY: {request.query[:50]}... → "
                f"{result['confidence']:.2f} confidence, "
                f"{result['timing_breakdown']['total_ms']:.1f}ms"
            )
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f" Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Service health check.
    
    Returns index status, cache hit rate, Redis/Gemini availability, and uptime.
    """
    try:
        # Calculate cache hit rate
        total_requests = app.state.cache_hits + app.state.cache_misses
        cache_hit_rate = app.state.cache_hits / total_requests if total_requests > 0 else 0.0
        
        # Check Redis health (if available)
        redis_connected = False
        if app.state.redis:
            redis_connected = await ping_redis(app.state.redis)
        
        # Get RAG engine stats
        index_loaded = app.state.rag_engine.vector_store is not None
        index_size = len(app.state.rag_engine.documents)
        gemini_available = app.state.rag_engine.gemini_model is not None
        
        # Calculate uptime
        uptime_seconds = time.time() - app.state.start_time
        
        # Determine status
        if not index_loaded:
            status = "unhealthy"
            status_code = 503
        elif not redis_connected:
            status = "degraded"
            status_code = 200
        else:
            status = "healthy"
            status_code = 200
        
        return HealthResponse(
            status=status,
            index_loaded=index_loaded,
            index_size=index_size,
            cache_hit_rate=cache_hit_rate,
            redis_connected=redis_connected,
            gemini_available=gemini_available,
            uptime_seconds=uptime_seconds
        )
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/metrics")
async def get_metrics():
    """
    Detailed performance metrics.
    
    Returns RAG engine stats, cache stats, and index stats.
    """
    try:
        # RAG engine stats
        rag_stats = app.state.rag_engine.get_performance_stats()
        
        # Cache stats
        total_requests = app.state.cache_hits + app.state.cache_misses
        cache_stats = {
            'cache_hits': app.state.cache_hits,
            'cache_misses': app.state.cache_misses,
            'cache_hit_rate': app.state.cache_hits / total_requests if total_requests > 0 else 0.0
        }
        
        # Index stats
        index_stats = {
            'total_documents': len(app.state.rag_engine.documents),
            'categories': len(set(m.get('category', '') for m in app.state.rag_engine.doc_metadata)),
            'embedding_dimension': app.state.rag_engine.vector_store.d if app.state.rag_engine.vector_store else 0
        }
        
        # Uptime
        uptime_seconds = time.time() - app.state.start_time
        
        return {
            'rag_engine': rag_stats,
            'cache': cache_stats,
            'index': index_stats,
            'uptime_seconds': uptime_seconds
        }
    
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@app.post("/api/v1/admin/rebuild_index", response_model=RebuildIndexResponse)
async def rebuild_index(request: RebuildIndexRequest):
    """
    Rebuild FAISS index from knowledge base (admin endpoint).
    
    Useful for knowledge base updates. Clears cache after rebuild.
    """
    try:
        build_start = time.time()
        
        # Create config (override path if provided)
        config = RAGConfig.from_env()
        if request.knowledge_base_path:
            config.knowledge_base_path = request.knowledge_base_path
        
        # Build index
        builder = IndexBuilder(config)
        success = builder.build_index()
        
        if not success:
            raise HTTPException(status_code=500, detail="Index build failed")
        
        # Reload index in RAG engine
        app.state.rag_engine.load_index()
        
        # Clear cache (only if Redis is available)
        if app.state.redis:
            try:
                # Delete all rag:* keys
                keys = await app.state.redis.keys("rag:*")
                if keys:
                    await app.state.redis.delete(*keys)
                    logger.info(f"️ Cleared {len(keys)} cached queries")
            except Exception as cache_error:
                logger.warning(f"️ Cache clear failed: {cache_error}")
        else:
            logger.info("️ Redis not available - skipping cache clear")
        
        # Get stats
        stats = builder.get_index_stats()
        build_time = time.time() - build_start
        
        logger.info(f" Index rebuilt: {stats['total_documents']} documents in {build_time:.2f}s")
        
        return RebuildIndexResponse(
            status="success",
            documents_indexed=stats['total_documents'],
            categories=stats['categories'],
            build_time_seconds=build_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rebuild error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Docker vs local: single worker to avoid FAISS index duplication
    # Rely on async concurrency for throughput
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8003")),
        workers=1,
        log_level="info"
    )
