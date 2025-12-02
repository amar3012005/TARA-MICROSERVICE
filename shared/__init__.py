"""
Shared Utilities Module for Leibniz Microservices

This module provides common utilities used across all Leibniz microservices:
- Redis client factory and connection pooling
- Health check utilities for Redis and HTTP endpoints
- Service validation and monitoring

All microservices should import from this module to ensure consistent patterns
for caching, state management, and health monitoring.

Reference: leibniz_agent/docs/Cloud Transformation.md

Usage:
    from leibniz_agent.services.shared import get_redis_client, check_redis_health
    
    redis = await get_redis_client()
    await redis.set("key", "value", ex=3600)
    
    health = await check_redis_health()
    print(f"Redis status: {health.status}")
"""

from .redis_client import (
    get_redis_client,
    get_redis_pool,
    close_redis_client,
    ping_redis,
    get_redis_info,
    RedisConfig,
)

from .health_check import (
    check_redis_health,
    check_service_health,
    check_all_services,
    HealthCheckResult,
)

__all__ = [
    # Redis client utilities
    "get_redis_client",
    "get_redis_pool",
    "close_redis_client",
    "ping_redis",
    "get_redis_info",
    "RedisConfig",
    # Health check utilities
    "check_redis_health",
    "check_service_health",
    "check_all_services",
    "HealthCheckResult",
]
