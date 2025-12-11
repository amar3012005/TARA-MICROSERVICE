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
    get_stream_length,
    get_redis_stream_client,
    RedisConfig,
)

from .health_check import (
    check_redis_health,
    check_service_health,
    check_all_services,
    HealthCheckResult,
)

from .events import (
    VoiceEvent,
    EventTypes,
)

from .event_broker import (
    EventBroker,
)

from .event_consumer import (
    EventConsumer,
    MultiStreamConsumer,
    ConsumerConfig,
    ConsumerMetrics,
    ProcessingResult,
    create_consumer,
)

from .observability import (
    setup_tracing,
    get_tracer,
    trace_span,
    traced,
    inject_trace_context,
    extract_trace_context,
    setup_metrics,
    get_metrics_response,
    record_event_processed,
    record_event_error,
    record_stt_latency,
    record_intent_latency,
    record_rag_latency,
    record_tts_first_chunk_latency,
    record_e2e_latency,
    set_active_sessions,
    set_stream_queue_size,
    time_event_processing,
    MetricTimer,
    EVENTS_TOTAL,
    EVENT_PROCESSING_DURATION,
    E2E_LATENCY,
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
    # Event utilities
    "VoiceEvent",
    "EventTypes",
    "EventBroker",
    # Event consumer utilities
    "EventConsumer",
    "MultiStreamConsumer",
    "ConsumerConfig",
    "ConsumerMetrics",
    "ProcessingResult",
    "create_consumer",
    # Observability utilities
    "setup_tracing",
    "get_tracer",
    "trace_span",
    "traced",
    "inject_trace_context",
    "extract_trace_context",
    "setup_metrics",
    "get_metrics_response",
    "record_event_processed",
    "record_event_error",
    "record_stt_latency",
    "record_intent_latency",
    "record_rag_latency",
    "record_tts_first_chunk_latency",
    "record_e2e_latency",
    "set_active_sessions",
    "set_stream_queue_size",
    "time_event_processing",
    "MetricTimer",
    "EVENTS_TOTAL",
    "EVENT_PROCESSING_DURATION",
    "E2E_LATENCY",
]
