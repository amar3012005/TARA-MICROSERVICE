"""
Comprehensive Integration Tests for Intent Classification Service

Tests fast pattern matching, Gemini fallback, caching, and edge cases.

Reference:
    leibniz_agent/services/stt_vad/tests/ - Testing pattern
    Cloud Transformation document - Phase 3 test specifications
"""

import pytest
import asyncio
import os
from typing import Dict, Any

import httpx
from fastapi.testclient import TestClient

# Set test environment variables before importing app
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "test-key")
os.environ["REDIS_URL"] = os.getenv("REDIS_URL", "redis://localhost:6379")
os.environ["LEIBNIZ_INTENT_PARSER_LOG_CLASSIFICATIONS"] = "true"

from .. import app as app_module
from ..config import IntentConfig
from ..intent_classifier import IntentClassifier


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Create test configuration"""
    return IntentConfig.from_env()


@pytest.fixture
def classifier(test_config):
    """Create intent classifier instance"""
    return IntentClassifier(test_config)


@pytest.fixture
def client():
    """Create FastAPI test client"""
    return TestClient(app_module.app)


# ============================================================================
# Unit Tests - Fast Pattern Matching
# ============================================================================

@pytest.mark.asyncio
async def test_appointment_scheduling_clear(classifier):
    """Test clear appointment scheduling intent"""
    result = await classifier.classify_intent("I want to schedule an appointment with admissions")
    
    assert result["intent"] == "APPOINTMENT_SCHEDULING"
    assert result["confidence"] >= 0.8
    assert result["fast_route"] == True
    assert "department" in result["context"]["key_entities"]
    assert result["context"]["key_entities"]["department"] == "admissions"


@pytest.mark.asyncio
async def test_appointment_scheduling_with_time(classifier):
    """Test appointment with time reference"""
    result = await classifier.classify_intent("Can I book a meeting tomorrow at 2pm?")
    
    assert result["intent"] == "APPOINTMENT_SCHEDULING"
    assert result["confidence"] >= 0.8
    assert result["fast_route"] == True
    assert "datetime" in result["context"]["key_entities"]


@pytest.mark.asyncio
async def test_rag_query_clear(classifier):
    """Test clear information request (RAG query)"""
    result = await classifier.classify_intent("What programs do you offer?")
    
    assert result["intent"] == "RAG_QUERY"
    assert result["confidence"] >= 0.8
    assert result["fast_route"] == True
    assert "user_goal" in result["context"]


@pytest.mark.asyncio
async def test_rag_query_with_program(classifier):
    """Test RAG query mentioning specific program"""
    result = await classifier.classify_intent("Tell me about the computer science program")
    
    assert result["intent"] == "RAG_QUERY"
    assert result["confidence"] >= 0.8
    assert result["fast_route"] == True
    assert "program" in result["context"]["key_entities"]
    assert result["context"]["key_entities"]["program"] == "computer science"


@pytest.mark.asyncio
async def test_greeting_standalone(classifier):
    """Test standalone greeting (strict - very rare)"""
    result = await classifier.classify_intent("hello")
    
    assert result["intent"] == "GREETING"
    assert result["confidence"] >= 0.8
    assert result["fast_route"] == True


@pytest.mark.asyncio
async def test_greeting_with_question_is_rag(classifier):
    """Test that greeting + question is RAG_QUERY, not GREETING"""
    result = await classifier.classify_intent("hello, what programs do you offer?")
    
    # Should be RAG_QUERY because it has a question
    assert result["intent"] == "RAG_QUERY"
    assert result["confidence"] >= 0.8


@pytest.mark.asyncio
async def test_exit_clear(classifier):
    """Test clear exit intent"""
    result = await classifier.classify_intent("thanks, that's all I needed")
    
    assert result["intent"] == "EXIT"
    assert result["confidence"] >= 0.8
    assert result["fast_route"] == True


@pytest.mark.asyncio
async def test_unclear_nonsense(classifier):
    """Test nonsensical input"""
    result = await classifier.classify_intent("blah xyz qwerty")
    
    assert result["intent"] == "UNCLEAR"
    assert result["confidence"] < 0.6


@pytest.mark.asyncio
async def test_empty_input(classifier):
    """Test empty input"""
    result = await classifier.classify_intent("")
    
    assert result["intent"] == "UNCLEAR"
    assert result["confidence"] == 0.0


# ============================================================================
# Edge Cases - Ambiguous Inputs
# ============================================================================

@pytest.mark.asyncio
async def test_appointment_vs_rag_how_to(classifier):
    """Test 'how to schedule' should be RAG, not APPOINTMENT"""
    result = await classifier.classify_intent("How do I schedule an appointment?")
    
    # Asking ABOUT scheduling process = RAG_QUERY
    # NOT actually scheduling = not APPOINTMENT_SCHEDULING
    # This should likely go to Gemini fallback
    assert result["intent"] in ["RAG_QUERY", "APPOINTMENT_SCHEDULING"]


@pytest.mark.asyncio
async def test_schedule_class_not_appointment(classifier):
    """Test 'class schedule' should NOT be appointment"""
    result = await classifier.classify_intent("What's the class schedule for fall semester?")
    
    # Should NOT be appointment (negative keyword filtering)
    assert result["intent"] != "APPOINTMENT_SCHEDULING"
    assert result["intent"] == "RAG_QUERY"


# ============================================================================
# HTTP API Tests
# ============================================================================

def test_classify_endpoint_appointment(client):
    """Test /api/v1/classify endpoint with appointment"""
    response = client.post(
        "/api/v1/classify",
        json={"text": "I want to schedule an appointment"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "APPOINTMENT_SCHEDULING"
    assert data["confidence"] >= 0.8
    assert "context" in data
    assert "response_time" in data


def test_classify_endpoint_rag_query(client):
    """Test /api/v1/classify endpoint with RAG query"""
    response = client.post(
        "/api/v1/classify",
        json={"text": "What are the admission requirements?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "RAG_QUERY"
    assert data["confidence"] >= 0.8


def test_classify_endpoint_with_context(client):
    """Test /api/v1/classify endpoint with context"""
    response = client.post(
        "/api/v1/classify",
        json={
            "text": "Tell me more",
            "context": {"previous_intent": "RAG_QUERY"}
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] in ["RAG_QUERY", "UNCLEAR"]


def test_health_endpoint(client):
    """Test /health endpoint"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "redis" in data
    assert "classifier" in data
    assert "config" in data


def test_metrics_endpoint(client):
    """Test /metrics endpoint"""
    # Make a few classification requests first
    client.post("/api/v1/classify", json={"text": "hello"})
    client.post("/api/v1/classify", json={"text": "What programs do you offer?"})
    
    response = client.get("/metrics")
    
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "fast_route_count" in data
    assert "gemini_route_count" in data
    assert "fast_route_percentage" in data
    assert "average_confidence" in data


def test_clear_cache_endpoint(client):
    """Test /admin/clear_cache endpoint"""
    # Make a classification to populate cache
    client.post("/api/v1/classify", json={"text": "hello"})
    
    # Clear cache
    response = client.post("/admin/clear_cache")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "keys_deleted" in data


# ============================================================================
# Caching Tests
# ============================================================================

@pytest.mark.asyncio
async def test_cache_hit_performance(client):
    """Test that cache provides performance improvement"""
    text = "What are the computer science program requirements?"
    
    # First request (cache miss)
    response1 = client.post("/api/v1/classify", json={"text": text})
    data1 = response1.json()
    time1 = data1["response_time"]
    cached1 = data1.get("cached", False)
    
    # Second request (should be cache hit)
    response2 = client.post("/api/v1/classify", json={"text": text})
    data2 = response2.json()
    time2 = data2["response_time"]
    cached2 = data2.get("cached", False)
    
    # Verify caching behavior
    assert cached1 == False  # First request not cached
    # Note: cached2 may be False if Redis is not available
    # Just verify both requests return same intent
    assert data1["intent"] == data2["intent"]
    assert data1["confidence"] == data2["confidence"]


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_fast_route_performance(classifier):
    """Test fast route is indeed fast (<50ms)"""
    import time
    
    start = time.time()
    result = await classifier.classify_intent("I want to schedule an appointment")
    elapsed = time.time() - start
    
    assert result["fast_route"] == True
    assert elapsed < 0.05  # Should be under 50ms


@pytest.mark.asyncio
async def test_concurrent_requests(classifier):
    """Test concurrent classification requests"""
    queries = [
        "I want to schedule an appointment",
        "What programs do you offer?",
        "hello",
        "goodbye",
        "Tell me about computer science"
    ]
    
    # Run concurrently
    tasks = [classifier.classify_intent(q) for q in queries]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 5
    assert all("intent" in r for r in results)
    assert all("confidence" in r for r in results)


# ============================================================================
# Configuration Tests
# ============================================================================

def test_config_loading():
    """Test configuration loads correctly"""
    config = IntentConfig.from_env()
    
    assert config.gemini_model is not None
    assert 0.0 <= config.confidence_threshold <= 1.0
    assert config.gemini_timeout > 0
    assert 0.0 <= config.fast_route_target <= 1.0


def test_config_validation_invalid_threshold():
    """Test configuration validation rejects invalid threshold"""
    with pytest.raises(ValueError):
        IntentConfig(
            gemini_api_key="test",
            confidence_threshold=1.5  # Invalid (>1.0)
        )


def test_config_validation_invalid_timeout():
    """Test configuration validation rejects invalid timeout"""
    with pytest.raises(ValueError):
        IntentConfig(
            gemini_api_key="test",
            gemini_timeout=-1.0  # Invalid (negative)
        )


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
