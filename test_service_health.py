#!/usr/bin/env python3
"""
Service Health Check Script for Leibniz Microservices

Tests health endpoints of all Leibniz microservices and provides
comprehensive status report with troubleshooting suggestions.

Usage:
    python leibniz_agent/services/test_service_health.py

Requirements:
    pip install httpx redis
"""

import asyncio
import json
import sys
from typing import Dict, List, Tuple
import time

try:
    import httpx
except ImportError:
    print(" httpx not installed. Install with: pip install httpx")
    sys.exit(1)

try:
    import redis.asyncio as redis
except ImportError:
    print(" redis not installed. Install with: pip install redis[asyncio]")
    sys.exit(1)


# Service configuration
SERVICES = {
    "intent": {
        "name": "Intent Classification",
        "port": 8002,
        "health_url": "http://localhost:8002/health",
        "description": "Classifies user intents with pattern matching and LLM fallback"
    },
    "appointment": {
        "name": "Appointment FSM",
        "port": 8005,
        "health_url": "http://localhost:8005/health",
        "description": "Manages appointment booking sessions with Redis persistence"
    },
    "tts": {
        "name": "Text-to-Speech",
        "port": 8004,
        "health_url": "http://localhost:8004/health",
        "description": "Synthesizes speech from text with multi-provider support"
    },
    "stt_vad": {
        "name": "STT/VAD",
        "port": 8001,
        "health_url": "http://localhost:8001/health",
        "description": "Speech-to-text transcription with voice activity detection"
    },
    "rag": {
        "name": "RAG",
        "port": 8003,
        "health_url": "http://localhost:8003/health",
        "description": "Retrieval-augmented generation for knowledge base queries"
    }
}


class ServiceHealthChecker:
    """Checks health of Leibniz microservices"""

    def __init__(self, timeout: float = 5.0, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.results = {}

    async def check_redis_connectivity(self) -> Tuple[bool, str]:
        """Check Redis connectivity"""
        try:
            client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
            await asyncio.wait_for(client.ping(), timeout=2.0)
            await client.close()
            return True, "connected"
        except asyncio.TimeoutError:
            return False, "timeout"
        except redis.exceptions.ConnectionError:
            return False, "connection_refused"
        except Exception as e:
            return False, "error", f"Unexpected error: {e}"

    async def check_service_health(self, service_name: str, service_config: Dict) -> Dict:
        """Check health of a single service"""
        health_url = service_config["health_url"]
        result = {
            "name": service_config["name"],
            "port": service_config["port"],
            "status": "unknown",
            "response_time": None,
            "error": None,
            "details": {}
        }

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(health_url)
                    response_time = time.time() - start_time

                    result["response_time"] = round(response_time, 3)

                    if response.status_code == 200:
                        try:
                            health_data = response.json()
                            result["status"] = health_data.get("status", "unknown")
                            result["details"] = health_data
                        except json.JSONDecodeError:
                            result["status"] = "healthy"  # Assume healthy if 200 but no JSON
                            result["error"] = "Invalid JSON response"
                    else:
                        result["status"] = "unhealthy"
                        result["error"] = f"HTTP {response.status_code}"

                    break  # Success, exit retry loop

            except httpx.TimeoutException:
                result["error"] = f"Timeout (attempt {attempt + 1}/{self.max_retries})"
                if attempt == self.max_retries - 1:
                    result["status"] = "unhealthy"
                await asyncio.sleep(0.5)  # Brief delay before retry

            except httpx.ConnectError:
                result["error"] = f"Connection refused (attempt {attempt + 1}/{self.max_retries})"
                if attempt == self.max_retries - 1:
                    result["status"] = "unhealthy"
                await asyncio.sleep(0.5)

            except Exception as e:
                result["error"] = f"Unexpected error: {str(e)}"
                if attempt == self.max_retries - 1:
                    result["status"] = "unhealthy"
                await asyncio.sleep(0.5)

        return result

    async def check_all_services(self) -> Dict:
        """Check health of all services"""
        print(" Checking Leibniz microservices health...")
        print("=" * 60)

        # Check Redis first
        print(" Checking Redis connectivity...")
        redis_healthy, redis_status = await self.check_redis_connectivity()
        if redis_healthy:
            print(" Redis: Connected")
        else:
            print(f" Redis: {redis_status}")
        print()

        # Check each service
        tasks = []
        for service_name, service_config in SERVICES.items():
            task = self.check_service_health(service_name, service_config)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Organize results
        self.results = {service_name: result for service_name, result in zip(SERVICES.keys(), results)}

        return self.results

    def print_report(self):
        """Print comprehensive health report"""
        print(" HEALTH REPORT")
        print("=" * 60)

        healthy_services = []
        unhealthy_services = []
        degraded_services = []

        for service_name, result in self.results.items():
            status = result["status"]
            name = result["name"]
            port = result["port"]
            response_time = result.get("response_time")
            error = result.get("error")

            if status == "healthy":
                healthy_services.append(service_name)
                status_icon = ""
                status_color = "green"
            elif status == "degraded":
                degraded_services.append(service_name)
                status_icon = "️"
                status_color = "yellow"
            else:
                unhealthy_services.append(service_name)
                status_icon = ""
                status_color = "red"

            print(f"{status_icon} {name} (port {port}): {status.upper()}")
            if response_time:
                print(f"   Response time: {response_time:.3f}s")
            if error:
                print(f"   Error: {error}")
            print()

        # Summary
        total_services = len(SERVICES)
        healthy_count = len(healthy_services)
        degraded_count = len(degraded_services)
        unhealthy_count = len(unhealthy_services)

        print(" SUMMARY")
        print("=" * 60)
        print(f"Total Services: {total_services}")
        print(f"Healthy: {healthy_count}")
        print(f"Degraded: {degraded_count}")
        print(f"Unhealthy: {unhealthy_count}")
        print()

        # Overall status
        if unhealthy_count > 0:
            overall_status = " SYSTEM UNHEALTHY"
            overall_message = f"{unhealthy_count} service(s) are unhealthy"
        elif degraded_count > 0:
            overall_status = "️  SYSTEM DEGRADED"
            overall_message = f"{degraded_count} service(s) are degraded"
        else:
            overall_status = " SYSTEM HEALTHY"
            overall_message = "All services are operating normally"

        print(f"{overall_status}")
        print(f"{overall_message}")
        print()

        # Troubleshooting suggestions
        if unhealthy_services or degraded_services:
            print(" TROUBLESHOOTING SUGGESTIONS")
            print("=" * 60)

            for service_name in unhealthy_services + degraded_services:
                result = self.results[service_name]
                name = result["name"]
                error = result.get("error", "")
                details = result.get("details", {})

                print(f"• {name}:")

                if "Connection refused" in str(error):
                    print("  - Service not running or wrong port")
                    print(f"  - Start with: docker run -d --name leibniz-{service_name} -p {result['port']}:{result['port']} leibniz-{service_name}:latest")
                elif "Timeout" in str(error):
                    print("  - Service is slow to respond")
                    print("  - Check service logs: docker logs leibniz-{service_name}")
                elif "redis" in str(error).lower() or "redis" in str(details).lower():
                    print("  - Redis connection issue")
                    print("  - Ensure Redis is running: docker ps | grep redis")
                    print("  - Check Redis connectivity: docker exec leibniz-redis redis-cli ping")
                elif "api key" in str(error).lower() or "gemini" in str(error).lower():
                    print("  - Missing or invalid API key")
                    print("  - Set GEMINI_API_KEY environment variable")
                elif "cache" in str(error).lower():
                    print("  - Cache directory issue")
                    print("  - Ensure cache directory exists and is writable")
                else:
                    print("  - Check service logs for detailed error information")
                    print("  - Verify environment variables are set correctly")

                print()

    def get_exit_code(self) -> int:
        """Get appropriate exit code based on health results"""
        unhealthy_count = sum(1 for result in self.results.values() if result["status"] not in ["healthy", "degraded"])
        degraded_count = sum(1 for result in self.results.values() if result["status"] == "degraded")

        if unhealthy_count > 0:
            return 2  # Critical
        elif degraded_count > 0:
            return 1  # Warning
        else:
            return 0  # Success


async def main():
    """Main entry point"""
    print(" Leibniz Microservices Health Checker")
    print("Checking health of all services...")
    print()

    checker = ServiceHealthChecker()

    try:
        await checker.check_all_services()
        checker.print_report()

        exit_code = checker.get_exit_code()
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n Health check interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f" Health check failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())