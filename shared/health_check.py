"""
Health Check Utilities for Leibniz Microservices

Provides reusable health check functions for validating:
- Redis connectivity and performance
- HTTP service endpoints
- Overall system health

All health checks return HealthCheckResult with standardized status codes:
- "healthy": Service is fully operational
- "degraded": Service is running but with issues
- "unhealthy": Service is not functional

Usage:
    from leibniz_agent.services.shared.health_check import check_redis_health
    
    result = await check_redis_health()
    if result.is_healthy():
        print(f"Redis is healthy (latency: {result.latency_ms}ms)")
    else:
        print(f"Redis is {result.status}: {result.details}")

CLI Usage:
    python -m leibniz_agent.services.shared.health_check --redis
    python -m leibniz_agent.services.shared.health_check --service stt http://localhost:8001/health
    python -m leibniz_agent.services.shared.health_check --all

Reference: leibniz_agent/leibniz_config.py, tara_web_server.py
"""

import asyncio
import logging
import time
import sys
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    import httpx
except ImportError:
    httpx = None  # Will check before using

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """
    Standardized health check result.
    
    Attributes:
        service_name: Name of the service being checked
        status: "healthy", "unhealthy", or "degraded"
        latency_ms: Response time in milliseconds
        details: Additional information (error messages, metrics, etc.)
        timestamp: Unix timestamp when check was performed
    """
    service_name: str
    status: str  # "healthy", "unhealthy", "degraded"
    latency_ms: float
    details: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return self.status == "healthy"


async def check_redis_health(redis_client: Optional[Any] = None) -> HealthCheckResult:
    """
    Check Redis connectivity and performance.
    
    Args:
        redis_client: Redis client instance (optional, creates temporary if not provided)
    
    Returns:
        HealthCheckResult: Health check result with Redis metrics
    """
    start_time = time.time()
    service_name = "redis"
    
    try:
        # Import here to avoid circular dependency
        from redis_client import (
            get_redis_client,
            ping_redis,
            get_redis_info,
        )
        
        # Get or use provided client
        if redis_client is None:
            redis_client = await get_redis_client()
        
        # Test PING
        ping_success = await ping_redis(redis_client)
        if not ping_success:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=service_name,
                status="unhealthy",
                latency_ms=latency_ms,
                details={"error": "PING command failed"},
                timestamp=time.time(),
            )
        
        # Get server info
        info = await get_redis_info(redis_client)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Determine status based on metrics
        status = "healthy"
        if "error" in info:
            status = "degraded"
        
        return HealthCheckResult(
            service_name=service_name,
            status=status,
            latency_ms=latency_ms,
            details={
                "version": info.get("redis_version", "unknown"),
                "uptime_seconds": info.get("uptime_seconds", 0),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
            },
            timestamp=time.time(),
        )
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"Redis health check failed: {e}")
        return HealthCheckResult(
            service_name=service_name,
            status="unhealthy",
            latency_ms=latency_ms,
            details={"error": str(e)},
            timestamp=time.time(),
        )


async def check_service_health(
    service_name: str,
    url: str,
    timeout: float = 5.0
) -> HealthCheckResult:
    """
    Check HTTP service health endpoint.
    
    Args:
        service_name: Name of the service (for display)
        url: Health check endpoint URL
        timeout: Request timeout in seconds (default: 5.0)
    
    Returns:
        HealthCheckResult: Health check result with HTTP metrics
    """
    start_time = time.time()
    
    try:
        if httpx is None:
            raise ImportError("httpx library not installed. Install with: pip install httpx")
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            latency_ms = (time.time() - start_time) * 1000
            
            # Determine status from HTTP code
            if response.status_code == 200:
                status = "healthy"
            elif response.status_code == 503:
                status = "degraded"
            else:
                status = "unhealthy"
            
            # Parse JSON response if available
            details = {}
            try:
                details = response.json()
            except Exception:
                details = {"raw_response": response.text[:200]}
            
            details["status_code"] = response.status_code
            
            return HealthCheckResult(
                service_name=service_name,
                status=status,
                latency_ms=latency_ms,
                details=details,
                timestamp=time.time(),
            )
    
    except httpx.TimeoutException:
        latency_ms = timeout * 1000
        logger.error(f"{service_name} health check timed out after {timeout}s")
        return HealthCheckResult(
            service_name=service_name,
            status="unhealthy",
            latency_ms=latency_ms,
            details={"error": f"Request timed out after {timeout}s"},
            timestamp=time.time(),
        )
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"{service_name} health check failed: {e}")
        return HealthCheckResult(
            service_name=service_name,
            status="unhealthy",
            latency_ms=latency_ms,
            details={"error": str(e)},
            timestamp=time.time(),
        )


async def check_all_services(services: Dict[str, str]) -> Dict[str, HealthCheckResult]:
    """
    Check health of multiple services concurrently.
    
    Args:
        services: Dict mapping service_name -> health_url
                  Special key "redis" will use Redis health check
    
    Returns:
        dict: Mapping of service_name -> HealthCheckResult
    
    Example:
        services = {
            "redis": "redis",  # Special key for Redis check
            "stt-service": "http://localhost:8001/health",
            "intent-service": "http://localhost:8002/health",
        }
        results = await check_all_services(services)
    """
    tasks = []
    service_names = []
    
    for service_name, url in services.items():
        if service_name.lower() == "redis" or url.lower() == "redis":
            # Special case: Redis health check
            tasks.append(check_redis_health())
        else:
            # HTTP health check
            tasks.append(check_service_health(service_name, url))
        
        service_names.append(service_name)
    
    # Run all checks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Build result dict
    result_dict = {}
    for service_name, result in zip(service_names, results):
        if isinstance(result, Exception):
            # Convert exception to HealthCheckResult
            result_dict[service_name] = HealthCheckResult(
                service_name=service_name,
                status="unhealthy",
                latency_ms=0.0,
                details={"error": str(result)},
                timestamp=time.time(),
            )
        else:
            result_dict[service_name] = result
    
    return result_dict


# ============================================================================
# CLI Interface
# ============================================================================

def _print_result(result: HealthCheckResult):
    """Pretty print health check result"""
    status_emoji = {
        "healthy": "",
        "degraded": "️",
        "unhealthy": "",
    }
    
    emoji = status_emoji.get(result.status, "")
    print(f"\n{emoji} {result.service_name.upper()}")
    print(f"   Status: {result.status}")
    print(f"   Latency: {result.latency_ms:.2f}ms")
    
    if result.details:
        print(f"   Details:")
        for key, value in result.details.items():
            print(f"      {key}: {value}")


async def _cli_main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Health check utility for Leibniz microservices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m leibniz_agent.services.shared.health_check --redis
  python -m leibniz_agent.services.shared.health_check --service stt http://localhost:8001/health
  python -m leibniz_agent.services.shared.health_check --all
        """
    )
    
    parser.add_argument(
        "--redis",
        action="store_true",
        help="Check Redis health"
    )
    
    parser.add_argument(
        "--service",
        nargs=2,
        metavar=("NAME", "URL"),
        action="append",
        help="Check HTTP service health (can be used multiple times)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all configured services (Redis + all HTTP services)"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP request timeout in seconds (default: 5.0)"
    )
    
    args = parser.parse_args()
    
    # Build service list
    services = {}
    
    if args.redis or args.all:
        services["redis"] = "redis"
    
    if args.service:
        for name, url in args.service:
            services[name] = url
    
    if args.all:
        # Add default services (can be extended based on environment)
        default_services = {
            "stt-vad": "http://localhost:8001/health",
            "intent": "http://localhost:8002/health",
            "rag": "http://localhost:8003/health",
            "tts": "http://localhost:8004/health",
            "appointment": "http://localhost:8005/health",
            "orchestrator": "http://localhost:8000/health",
        }
        # Only add if not already specified
        for name, url in default_services.items():
            if name not in services:
                services[name] = url
    
    if not services:
        parser.print_help()
        print("\n Error: No services specified. Use --redis, --service, or --all")
        sys.exit(1)
    
    # Run health checks
    print(" Running health checks...")
    results = await check_all_services(services)
    
    # Print results
    for service_name, result in results.items():
        _print_result(result)
    
    # Summary
    healthy_count = sum(1 for r in results.values() if r.is_healthy())
    total_count = len(results)
    
    print(f"\n Summary: {healthy_count}/{total_count} services healthy")
    
    # Exit code: 0 if all healthy, 1 if any unhealthy
    if healthy_count == total_count:
        print(" All services are healthy!")
        sys.exit(0)
    else:
        print("️  Some services are unhealthy")
        sys.exit(1)


if __name__ == "__main__":
    # Configure logging for CLI
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s"
    )
    
    # Run CLI
    asyncio.run(_cli_main())
