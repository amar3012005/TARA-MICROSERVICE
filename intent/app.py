"""
Intent Classification Microservice - FastAPI Application

Provides HTTP REST API for intent classification with Redis caching.

Reference:
    leibniz_agent/services/stt_vad/app.py - Service application pattern
    Cloud Transformation document - Phase 3 specifications
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import redis.asyncio as redis

from leibniz_agent.services.intent.config import IntentConfig
from leibniz_agent.services.intent.intent_classifier import IntentClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state
redis_client: Optional[redis.Redis] = None
classifier: Optional[IntentClassifier] = None
config: Optional[IntentConfig] = None

# Redis cache configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = int(os.getenv("INTENT_CACHE_TTL", "1800"))  # 30 minutes default
CACHE_PREFIX = "leibniz:intent:"


# ============================================================================
# Pydantic Models
# ============================================================================

class ClassifyRequest(BaseModel):
    """Request model for intent classification"""
    text: str = Field(..., min_length=1, description="User input text to classify")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional conversation context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I want to schedule an appointment with admissions",
                "context": {"previous_intent": "RAG_QUERY"}
            }
        }


class ClassifyResponse(BaseModel):
    """Response model for intent classification (3-Layer Architecture)"""
    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    context: Dict[str, Any] = Field(..., description="Extracted context (user_goal, key_entities, extracted_meaning)")
    reasoning: str = Field(..., description="Classification explanation")
    layer_type: str = Field(..., description="Layer used: L1 (Regex), L2 (Semantic), L3 (LLM), CACHE")
    decision_path: str = Field(..., description="Description of classification path taken")
    response_time: float = Field(..., ge=0.0, description="Classification time in seconds")
    cached: bool = Field(default=False, description="True if result was served from cache")
    
    # Backward compatibility fields
    fast_route: bool = Field(default=True, description="True if Layer 1 or 2, false if Layer 3 (deprecated, use layer_type)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "intent": "APPOINTMENT_SCHEDULING",
                "confidence": 0.95,
                "context": {
                    "user_goal": "wants to schedule appointment with admissions",
                    "key_entities": {"department": "admissions"},
                    "extracted_meaning": "schedule appointment admissions"
                },
                "reasoning": "Appointment keywords detected: ['appointment', 'schedule']",
                "layer_type": "L1",
                "decision_path": "L1: Regex match (conf=0.95)",
                "response_time": 0.012,
                "cached": False,
                "fast_route": True
            }
        }


class HealthResponse(BaseModel):
    """Health check response
    
    Status values:
        - healthy: All components operational
        - degraded: Service functional but cache unavailable (reduced performance)
        - unhealthy: Core classifier not initialized
    """
    status: str  # One of: healthy, degraded, unhealthy
    redis: str
    classifier: str
    config: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Performance metrics response (3-Layer Architecture)"""
    total_requests: int
    layer1_count: int = Field(..., description="Layer 1 (Regex) classifications")
    layer2_count: int = Field(..., description="Layer 2 (Semantic) classifications")
    layer3_count: int = Field(..., description="Layer 3 (LLM) classifications")
    cache_hits: int = Field(..., description="Cache hit count")
    layer1_percentage: float = Field(..., description="Percentage of Layer 1 hits")
    layer2_percentage: float = Field(..., description="Percentage of Layer 2 hits")
    layer3_percentage: float = Field(..., description="Percentage of Layer 3 hits")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    average_confidence: float = Field(..., description="Average classification confidence")
    cache_enabled: bool = Field(..., description="Whether caching is enabled")
    cache_size: int = Field(..., description="Current cache size")
    
    # Backward compatibility fields
    fast_route_count: int = Field(default=0, description="Deprecated: sum of layer1_count + layer2_count")
    gemini_route_count: int = Field(default=0, description="Deprecated: same as layer3_count")
    fast_route_percentage: float = Field(default=0.0, description="Deprecated: sum of layer1 + layer2 percentages")


class ClearCacheResponse(BaseModel):
    """Cache clearing response"""
    message: str
    keys_deleted: int


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage service lifecycle (startup/shutdown)
    """
    global redis_client, classifier, config
    
    logger.info(" Starting Intent Classification service...")
    
    # Load configuration
    try:
        config = IntentConfig.from_env()
        logger.info(" Configuration loaded")
    except Exception as e:
        logger.error(f" Failed to load configuration: {e}")
        raise
    
    # Initialize Redis client
    try:
        redis_client = await asyncio.wait_for(
            redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True),
            timeout=5.0
        )
        await asyncio.wait_for(redis_client.ping(), timeout=2.0)
        logger.info(f" Redis connected: {REDIS_URL}")
    except asyncio.TimeoutError:
        logger.warning("️ Redis connection timeout - continuing without cache")
        redis_client = None
    except Exception as e:
        logger.warning(f"️ Redis connection failed: {e} - continuing without cache")
        redis_client = None
    
    # Initialize intent classifier
    try:
        classifier = IntentClassifier(config)
        logger.info(" Intent classifier initialized")
    except Exception as e:
        logger.error(f" Failed to initialize classifier: {e}")
        raise
    
    logger.info(" Intent Classification service ready")
    
    yield
    
    # Shutdown
    logger.info(" Shutting down Intent Classification service...")
    
    if redis_client:
        await redis_client.close()
        logger.info(" Redis connection closed")
    
    logger.info(" Service stopped")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Leibniz Intent Classification Service",
    description="3-Layer intent classification: Regex → Semantic → LLM with LRU caching",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/v1/classify", response_model=ClassifyResponse)
async def classify_intent_endpoint(request: ClassifyRequest):
    """
    Classify user intent with caching.
    
    Process:
        1. Check Redis cache for existing classification
        2. If cache miss, perform classification (fast pattern → Gemini fallback)
        3. Cache result for 30 minutes
        4. Return classification result
    
    Returns:
        Classification result with intent, confidence, context, and metadata
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    # Generate cache key
    cache_key = f"{CACHE_PREFIX}{request.text.strip().lower()}"
    
    # Try cache first (if Redis available)
    if redis_client:
        try:
            cached_result = await redis_client.get(cache_key)
            if cached_result:
                import json
                result = json.loads(cached_result)
                result["cached"] = True
                result["layer_type"] = result.get("layer_type", "CACHE")
                result["decision_path"] = result.get("decision_path", f"CACHE: {result.get('layer_type', 'L1')}")
                result["fast_route"] = result.get("layer_type", "L3") in ("L1", "L2", "CACHE")
                logger.info(f" CACHE HIT: {request.text[:50]} → {result.get('intent', 'UNKNOWN')} ({result.get('layer_type', 'CACHE')})")
                return ClassifyResponse(**result)
        except Exception as e:
            logger.warning(f"️ Cache read failed: {e}")
    
    # Cache miss - perform classification
    try:
        result = await classifier.classify_intent(request.text, request.context)
        result["cached"] = False
        
        # Backward compatibility: add fast_route field
        result["fast_route"] = result.get("layer_type", "L3") in ("L1", "L2", "CACHE")
        
        # Cache result (if Redis available)
        if redis_client:
            try:
                import json
                await redis_client.setex(
                    cache_key,
                    CACHE_TTL,
                    json.dumps({
                        "intent": result["intent"],
                        "confidence": result["confidence"],
                        "context": result["context"],
                        "reasoning": result["reasoning"],
                        "layer_type": result.get("layer_type", "L1"),
                        "decision_path": result.get("decision_path", ""),
                        "fast_route": result["fast_route"],
                        "response_time": result["response_time"]
                    })
                )
                logger.info(f" CACHED: {request.text[:50]} → {result['intent']} ({result.get('layer_type', 'L1')})")
            except Exception as e:
                logger.warning(f"️ Cache write failed: {e}")
        
        return ClassifyResponse(**result)
    
    except Exception as e:
        logger.error(f" Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Checks:
        - Service status (always up if responding)
        - Redis connection status
        - Classifier initialization status
        - Configuration summary
    
    Returns:
        Health status for all components
    """
    redis_status = "disconnected"
    if redis_client:
        try:
            await asyncio.wait_for(redis_client.ping(), timeout=1.0)
            redis_status = "connected"
        except:
            redis_status = "error"
    
    classifier_status = "initialized" if classifier else "not_initialized"
    
    config_summary = {}
    if config:
        config_summary = {
            "gemini_model": config.gemini_model,
            "confidence_threshold": config.confidence_threshold,
            "gemini_timeout": config.gemini_timeout,
            "fast_route_target": config.fast_route_target,
            "has_gemini_key": bool(config.gemini_api_key)
        }
    
    # Determine overall health status
    if redis_status != "connected":
        overall_status = "degraded"  # Cache unavailable - reduced performance
    elif classifier_status != "initialized":
        overall_status = "unhealthy"  # Core functionality broken
    else:
        overall_status = "healthy"
    
    return HealthResponse(
        status=overall_status,
        redis=redis_status,
        classifier=classifier_status,
        config=config_summary
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get performance metrics.
    
    Metrics:
        - Total classification requests
        - Fast route count (pattern matches)
        - Gemini route count (LLM fallbacks)
        - Fast route percentage
        - Average confidence
    
    Returns:
        Performance statistics
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    stats = classifier.get_performance_stats()
    
    # Add backward compatibility fields
    stats["fast_route_count"] = stats.get("layer1_count", 0) + stats.get("layer2_count", 0)
    stats["gemini_route_count"] = stats.get("layer3_count", 0)
    stats["fast_route_percentage"] = stats.get("layer1_percentage", 0.0) + stats.get("layer2_percentage", 0.0)
    
    return MetricsResponse(**stats)


@app.post("/admin/clear_cache", response_model=ClearCacheResponse)
async def clear_cache():
    """
    Clear all cached intent classifications.
    
    Deletes all Redis keys with the intent cache prefix.
    
    Returns:
        Number of keys deleted
    """
    if not redis_client:
        return ClearCacheResponse(message="Redis not connected", keys_deleted=0)
    
    try:
        # Find all intent cache keys
        cursor = 0
        keys_deleted = 0
        
        while True:
            cursor, keys = await redis_client.scan(cursor, match=f"{CACHE_PREFIX}*", count=100)
            if keys:
                deleted = await redis_client.delete(*keys)
                keys_deleted += deleted
            
            if cursor == 0:
                break
        
        logger.info(f" Cache cleared: {keys_deleted} keys deleted")
        return ClearCacheResponse(message="Cache cleared successfully", keys_deleted=keys_deleted)
    
    except Exception as e:
        logger.error(f" Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """
    Service information endpoint
    """
    return {
        "service": "Leibniz Intent Classification Service",
        "version": "2.0.0",
        "architecture": "3-Layer (Regex → Semantic → LLM)",
        "status": "operational",
        "endpoints": {
            "classify": "POST /api/v1/classify",
            "health": "GET /health",
            "metrics": "GET /metrics",
            "clear_cache": "POST /admin/clear_cache"
        }
    }


if __name__ == "__main__":
    # NOTE: This block is for local development only.
    # Docker deployment uses the CMD in Dockerfile which bypasses this.
    # Configuration: Modify Dockerfile CMD or docker-compose.yml environment variables.
    import uvicorn
    port = int(os.getenv("PORT", "8002"))
    workers = int(os.getenv("WORKERS", "1"))  # Changed to 1 for consistency
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level="info"
    )
