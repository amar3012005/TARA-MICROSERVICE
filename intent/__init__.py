"""
Intent Classification Microservice for Leibniz Agent

Provides two-tier intent classification (fast pattern matching + Gemini fallback)
via HTTP REST API with Redis caching and context extraction.

Reference:
    leibniz_agent/docs/Cloud Transformation.md (Phase 3, lines 325-433)
    
Main Entry Point:
    app.py - FastAPI application with POST /api/v1/classify endpoint
    
Components:
    - IntentClassifier: Core classification logic with fast patterns + Gemini LLM
    - IntentConfig: Configuration dataclass with environment variable loading
    - classify_intent: Convenience async function for classification
"""

from .intent_classifier import IntentClassifier
from .config import IntentConfig

__all__ = [
    "IntentClassifier",
    "IntentConfig",
]
