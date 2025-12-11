"""
Observability Module for Leibniz TARA Microservices

Provides:
- OpenTelemetry distributed tracing setup
- Prometheus metrics for monitoring
- Utility decorators for easy instrumentation
- Event-driven metrics tracking

Reference:
    docs/EVENT_DRIVEN_TRANSFORMATION_GUIDE.md - Architecture guide
"""

import os
import time
import logging
import functools
from typing import Dict, Any, Optional, Callable
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# =============================================================================
# OpenTelemetry Tracing
# =============================================================================

# Try to import OpenTelemetry (optional dependency)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    
    # Try to import OTLP exporter (for production)
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        OTLP_AVAILABLE = True
    except ImportError:
        OTLP_AVAILABLE = False
        OTLPSpanExporter = None
    
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
    TracerProvider = None
    BatchSpanProcessor = None
    ConsoleSpanExporter = None
    OTLPSpanExporter = None
    Resource = None
    SERVICE_NAME = None
    Status = None
    StatusCode = None
    TraceContextTextMapPropagator = None
    OTLP_AVAILABLE = False

# =============================================================================
# Prometheus Metrics
# =============================================================================

# Try to import Prometheus client (optional dependency)
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None
    Gauge = None
    Info = None
    generate_latest = None
    CONTENT_TYPE_LATEST = None
    CollectorRegistry = None
    REGISTRY = None


# Global tracer (initialized lazily)
_tracer: Optional["trace.Tracer"] = None
_propagator = None

# Global metrics registry
_metrics_initialized = False


def setup_tracing(
    service_name: str,
    otlp_endpoint: Optional[str] = None,
    enable_console_export: bool = False
) -> Optional["trace.Tracer"]:
    """
    Setup OpenTelemetry distributed tracing.
    
    Args:
        service_name: Name of the service for tracing
        otlp_endpoint: OTLP collector endpoint (e.g., "http://jaeger:4317")
        enable_console_export: Enable console span export for debugging
        
    Returns:
        Tracer instance or None if OpenTelemetry not available
    """
    global _tracer, _propagator
    
    if not OPENTELEMETRY_AVAILABLE:
        logger.warning("OpenTelemetry not available - tracing disabled")
        return None
    
    try:
        # Create resource
        resource = Resource.create({
            SERVICE_NAME: service_name,
            "service.version": os.getenv("SERVICE_VERSION", "1.0.0"),
            "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development")
        })
        
        # Create provider
        provider = TracerProvider(resource=resource)
        
        # Add exporters
        if otlp_endpoint and OTLP_AVAILABLE:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"✅ OTLP trace exporter enabled: {otlp_endpoint}")
        
        if enable_console_export:
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))
            logger.info("✅ Console trace exporter enabled")
        
        # Set global provider
        trace.set_tracer_provider(provider)
        
        # Get tracer
        _tracer = trace.get_tracer(service_name)
        _propagator = TraceContextTextMapPropagator()
        
        logger.info(f"✅ OpenTelemetry tracing initialized for {service_name}")
        return _tracer
        
    except Exception as e:
        logger.error(f"Failed to setup tracing: {e}")
        return None


def get_tracer() -> Optional["trace.Tracer"]:
    """Get the global tracer instance."""
    return _tracer


@asynccontextmanager
async def trace_span(
    name: str,
    attributes: Dict[str, Any] = None,
    kind: Optional[int] = None
):
    """
    Async context manager for creating a trace span.
    
    Usage:
        async with trace_span("process_event", {"session_id": sid}):
            # Do work
            pass
    """
    if not _tracer:
        yield None
        return
    
    try:
        from opentelemetry.trace import SpanKind
        span_kind = SpanKind(kind) if kind else SpanKind.INTERNAL
    except:
        span_kind = None
    
    with _tracer.start_as_current_span(name, kind=span_kind) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        try:
            yield span
        except Exception as e:
            if span and OPENTELEMETRY_AVAILABLE:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            raise


def traced(
    name: Optional[str] = None,
    attributes_fn: Optional[Callable] = None
):
    """
    Decorator to trace a function or async function.
    
    Usage:
        @traced("my_operation")
        async def my_function(session_id: str):
            pass
        
        @traced(attributes_fn=lambda args: {"session_id": args[0]})
        async def process(session_id: str):
            pass
    """
    def decorator(func):
        span_name = name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not _tracer:
                return await func(*args, **kwargs)
            
            attrs = {}
            if attributes_fn:
                try:
                    attrs = attributes_fn(args, kwargs)
                except:
                    pass
            
            async with trace_span(span_name, attrs):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not _tracer:
                return func(*args, **kwargs)
            
            attrs = {}
            if attributes_fn:
                try:
                    attrs = attributes_fn(args, kwargs)
                except:
                    pass
            
            with _tracer.start_as_current_span(span_name) as span:
                if attrs:
                    for key, value in attrs.items():
                        span.set_attribute(key, str(value))
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def inject_trace_context(carrier: Dict[str, str]):
    """Inject trace context into a carrier dict (for propagation)."""
    if _propagator and OPENTELEMETRY_AVAILABLE:
        _propagator.inject(carrier)


def extract_trace_context(carrier: Dict[str, str]):
    """Extract trace context from a carrier dict."""
    if _propagator and OPENTELEMETRY_AVAILABLE:
        return _propagator.extract(carrier)
    return None


# =============================================================================
# Prometheus Metrics Definitions
# =============================================================================

# Event processing metrics
if PROMETHEUS_AVAILABLE:
    # Counters
    EVENTS_TOTAL = Counter(
        'leibniz_events_total',
        'Total number of events processed',
        ['service', 'event_type', 'status']
    )
    
    EVENTS_ERRORS = Counter(
        'leibniz_events_errors_total',
        'Total number of event processing errors',
        ['service', 'event_type', 'error_type']
    )
    
    # Histograms
    EVENT_PROCESSING_DURATION = Histogram(
        'leibniz_event_processing_duration_seconds',
        'Time spent processing events',
        ['service', 'event_type'],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
    )
    
    # Latency metrics for voice pipeline stages
    STT_LATENCY = Histogram(
        'leibniz_stt_latency_seconds',
        'STT processing latency',
        ['service'],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    )
    
    INTENT_LATENCY = Histogram(
        'leibniz_intent_latency_seconds',
        'Intent detection latency',
        ['service'],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
    )
    
    RAG_LATENCY = Histogram(
        'leibniz_rag_latency_seconds',
        'RAG processing latency',
        ['service'],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
    )
    
    TTS_FIRST_CHUNK_LATENCY = Histogram(
        'leibniz_tts_first_chunk_latency_seconds',
        'Time to first TTS audio chunk',
        ['service', 'provider'],
        buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
    )
    
    E2E_LATENCY = Histogram(
        'leibniz_e2e_latency_seconds',
        'End-to-end latency (STT final to TTS first chunk)',
        ['service'],
        buckets=(0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
    )
    
    # Gauges
    ACTIVE_SESSIONS = Gauge(
        'leibniz_active_sessions',
        'Number of active voice sessions',
        ['service']
    )
    
    STREAM_QUEUE_SIZE = Gauge(
        'leibniz_stream_queue_size',
        'Size of Redis Stream pending queue',
        ['service', 'stream']
    )
    
    # Service info
    SERVICE_INFO = Info(
        'leibniz_service',
        'Service information'
    )
else:
    # Stub classes when Prometheus not available
    class StubMetric:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def dec(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def info(self, *args, **kwargs):
            pass
        def time(self):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    EVENTS_TOTAL = StubMetric()
    EVENTS_ERRORS = StubMetric()
    EVENT_PROCESSING_DURATION = StubMetric()
    STT_LATENCY = StubMetric()
    INTENT_LATENCY = StubMetric()
    RAG_LATENCY = StubMetric()
    TTS_FIRST_CHUNK_LATENCY = StubMetric()
    E2E_LATENCY = StubMetric()
    ACTIVE_SESSIONS = StubMetric()
    STREAM_QUEUE_SIZE = StubMetric()
    SERVICE_INFO = StubMetric()


def setup_metrics(service_name: str, service_version: str = "1.0.0"):
    """
    Setup Prometheus metrics for a service.
    
    Args:
        service_name: Name of the service
        service_version: Version string
    """
    global _metrics_initialized
    
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus client not available - metrics disabled")
        return
    
    if _metrics_initialized:
        return
    
    try:
        SERVICE_INFO.info({
            'service': service_name,
            'version': service_version,
            'environment': os.getenv('DEPLOYMENT_ENV', 'development')
        })
        _metrics_initialized = True
        logger.info(f"✅ Prometheus metrics initialized for {service_name}")
    except Exception as e:
        logger.error(f"Failed to setup metrics: {e}")


def get_metrics_response():
    """
    Get Prometheus metrics as HTTP response content.
    
    Returns:
        Tuple of (content_bytes, content_type) for HTTP response
    """
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus not available\n", "text/plain"
    
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


# =============================================================================
# Metric Recording Utilities
# =============================================================================

def record_event_processed(
    service: str,
    event_type: str,
    status: str = "success",
    duration_seconds: float = None
):
    """Record an event processing metric."""
    EVENTS_TOTAL.labels(service=service, event_type=event_type, status=status).inc()
    
    if duration_seconds is not None:
        EVENT_PROCESSING_DURATION.labels(
            service=service, 
            event_type=event_type
        ).observe(duration_seconds)


def record_event_error(
    service: str,
    event_type: str,
    error_type: str
):
    """Record an event processing error."""
    EVENTS_ERRORS.labels(
        service=service,
        event_type=event_type,
        error_type=error_type
    ).inc()


def record_stt_latency(service: str, duration_seconds: float):
    """Record STT processing latency."""
    STT_LATENCY.labels(service=service).observe(duration_seconds)


def record_intent_latency(service: str, duration_seconds: float):
    """Record intent detection latency."""
    INTENT_LATENCY.labels(service=service).observe(duration_seconds)


def record_rag_latency(service: str, duration_seconds: float):
    """Record RAG processing latency."""
    RAG_LATENCY.labels(service=service).observe(duration_seconds)


def record_tts_first_chunk_latency(
    service: str,
    provider: str,
    duration_seconds: float
):
    """Record time to first TTS audio chunk."""
    TTS_FIRST_CHUNK_LATENCY.labels(
        service=service,
        provider=provider
    ).observe(duration_seconds)


def record_e2e_latency(service: str, duration_seconds: float):
    """Record end-to-end latency."""
    E2E_LATENCY.labels(service=service).observe(duration_seconds)


def set_active_sessions(service: str, count: int):
    """Set the number of active sessions."""
    ACTIVE_SESSIONS.labels(service=service).set(count)


def set_stream_queue_size(service: str, stream: str, size: int):
    """Set the Redis Stream queue size."""
    STREAM_QUEUE_SIZE.labels(service=service, stream=stream).set(size)


# =============================================================================
# Context Manager for Timing
# =============================================================================

class MetricTimer:
    """Context manager for timing operations and recording to metrics."""
    
    def __init__(
        self,
        histogram,
        labels: Dict[str, str] = None
    ):
        self.histogram = histogram
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if self.start_time:
            duration = time.time() - self.start_time
            if self.labels:
                self.histogram.labels(**self.labels).observe(duration)
            else:
                self.histogram.observe(duration)
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, *args):
        if self.start_time:
            duration = time.time() - self.start_time
            if self.labels:
                self.histogram.labels(**self.labels).observe(duration)
            else:
                self.histogram.observe(duration)


def time_event_processing(service: str, event_type: str):
    """Create a timer context manager for event processing."""
    return MetricTimer(
        EVENT_PROCESSING_DURATION,
        {"service": service, "event_type": event_type}
    )


# Import asyncio at module level
import asyncio


