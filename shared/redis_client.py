"""
Async Redis Client Wrapper for Leibniz Microservices

Provides connection pooling, helper functions, and singleton pattern for
Redis connectivity across all Leibniz microservices.

Configuration is read from environment variables (no load_dotenv() calls):
- LEIBNIZ_REDIS_HOST: Redis server host (default: localhost)
- LEIBNIZ_REDIS_PORT: Redis server port (default: 6379)
- LEIBNIZ_REDIS_DB: Redis database number (default: 0)
- LEIBNIZ_REDIS_PASSWORD: Redis password (optional, default: None)
- LEIBNIZ_REDIS_MAX_CONNECTIONS: Connection pool size (default: 50)
- LEIBNIZ_REDIS_SOCKET_TIMEOUT: Socket timeout in seconds (default: 5.0)
- LEIBNIZ_REDIS_SOCKET_CONNECT_TIMEOUT: Connect timeout in seconds (default: 5.0)
- LEIBNIZ_REDIS_URL: Alternative connection string format (overrides individual settings)

Usage:
    from leibniz_agent.services.shared.redis_client import get_redis_client
    
    redis = await get_redis_client()
    await redis.set("key", "value", ex=3600)
    value = await redis.get("key")
    await close_redis_client()

Reference: leibniz_agent/leibniz_config.py, leibniz_agent/leibniz_persistent_services.py
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import redis.asyncio as redis

# Module-level logger
logger = logging.getLogger(__name__)

# Singleton instances (similar to leibniz_persistent_services.py pattern)
_redis_client: Optional[redis.Redis] = None
_redis_pool: Optional[redis.ConnectionPool] = None
_lock = asyncio.Lock()


@dataclass
class RedisConfig:
    """Redis configuration dataclass (similar to VoiceConfig in leibniz_config.py)"""
    
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    
    @staticmethod
    def from_env() -> 'RedisConfig':
        """Load configuration from environment variables"""
        # Check for REDIS_URL first (overrides individual settings)
        redis_url = os.getenv("LEIBNIZ_REDIS_URL") or os.getenv("REDIS_URL")
        if redis_url:
            # Parse URL manually for config display purposes
            # Actual connection will use the URL directly
            logger.info(f"Loading Redis config from REDIS_URL")
        
        return RedisConfig(
            host=os.getenv("LEIBNIZ_REDIS_HOST", "localhost"),
            port=int(os.getenv("LEIBNIZ_REDIS_PORT", "6379")),
            db=int(os.getenv("LEIBNIZ_REDIS_DB", "0")),
            password=os.getenv("LEIBNIZ_REDIS_PASSWORD") or None,
            max_connections=int(os.getenv("LEIBNIZ_REDIS_MAX_CONNECTIONS", "50")),
            socket_timeout=float(os.getenv("LEIBNIZ_REDIS_SOCKET_TIMEOUT", "10.0")),
            socket_connect_timeout=float(os.getenv("LEIBNIZ_REDIS_SOCKET_CONNECT_TIMEOUT", "10.0")),
        )
    
    def get_redis_url(self) -> str:
        """Generate Redis connection URL"""
        # Check for explicit URL first
        env_url = os.getenv("LEIBNIZ_REDIS_URL") or os.getenv("REDIS_URL")
        if env_url:
            return env_url
        
        # Build URL from individual settings
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


async def get_redis_pool() -> redis.ConnectionPool:
    """
    Get or create Redis connection pool.
    
    Uses singleton pattern to reuse pool across calls.
    Connection pool provides better performance for high-throughput scenarios.
    
    Returns:
        redis.ConnectionPool: Connection pool instance
    """
    global _redis_pool
    
    async with _lock:
        if _redis_pool is None:
            config = RedisConfig.from_env()
            
            logger.info(
                f"Creating Redis connection pool: {config.host}:{config.port}/{config.db} "
                f"(max_connections={config.max_connections})"
            )
            
            # Create connection pool
            # NOTE: decode_responses=True automatically converts bytes to strings.
            # This is suitable for most caching use cases (JSON, session data, text).
            # For binary data (audio chunks, images), store as hex/base64 strings
            # and decode manually: bytes.fromhex(await redis.get(key))
            _redis_pool = redis.ConnectionPool.from_url(
                config.get_redis_url(),
                max_connections=config.max_connections,
                socket_timeout=config.socket_timeout,
                socket_connect_timeout=config.socket_connect_timeout,
                decode_responses=True,  # Auto-decode bytes to strings
            )
        
        return _redis_pool


async def get_redis_client() -> redis.Redis:
    """
    Get or create async Redis client instance.
    
    Uses singleton pattern to reuse connection across calls.
    Automatically reconnects on connection loss.
    
    Returns:
        redis.Redis: Async Redis client instance
    
    Raises:
        redis.exceptions.ConnectionError: If connection fails after retries
    """
    global _redis_client
    
    async with _lock:
        if _redis_client is None:
            pool = await get_redis_pool()
            _redis_client = redis.Redis(connection_pool=pool)
            
            # Test connection with retry logic
            max_retries = 3
            retry_delays = [1.0, 2.0, 4.0]  # Exponential backoff
            
            for attempt in range(max_retries):
                try:
                    await _redis_client.ping()
                    logger.info(" Redis client connected successfully")
                    break
                except redis.exceptions.ConnectionError as e:
                    if attempt < max_retries - 1:
                        delay = retry_delays[attempt]
                        logger.warning(
                            f"Redis connection attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f" Redis connection failed after {max_retries} attempts: {e}")
                        _redis_client = None
                        raise
        
        return _redis_client


async def ping_redis(client: Optional[redis.Redis] = None) -> bool:
    """
    Test Redis connectivity with simple PING command.
    
    Args:
        client: Redis client instance (optional, uses singleton if not provided)
    
    Returns:
        bool: True if PING successful, False otherwise
    """
    try:
        if client is None:
            client = await get_redis_client()
        
        result = await client.ping()
        return result is True
    except Exception as e:
        logger.error(f"Redis PING failed: {e}")
        return False


async def get_redis_info(client: Optional[redis.Redis] = None) -> Dict[str, Any]:
    """
    Get Redis server information for monitoring.
    
    Args:
        client: Redis client instance (optional, uses singleton if not provided)
    
    Returns:
        dict: Redis server info (version, uptime, memory, clients, etc.)
    """
    try:
        if client is None:
            client = await get_redis_client()
        
        info = await client.info()
        
        # Extract key metrics
        return {
            "redis_version": info.get("redis_version", "unknown"),
            "uptime_seconds": info.get("uptime_in_seconds", 0),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "used_memory_peak_human": info.get("used_memory_peak_human", "unknown"),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
            "keyspace": info.get("db0", {}),
        }
    except Exception as e:
        logger.error(f"Failed to get Redis info: {e}")
        return {"error": str(e)}


async def close_redis_client():
    """
    Gracefully close Redis client and connection pool.
    
    Should be called during application shutdown to ensure proper cleanup.
    """
    global _redis_client, _redis_pool
    
    async with _lock:
        if _redis_client is not None:
            try:
                await _redis_client.close()
                logger.info("Redis client closed")
            except Exception as e:
                logger.error(f"Error closing Redis client: {e}")
            finally:
                _redis_client = None
        
        if _redis_pool is not None:
            try:
                await _redis_pool.disconnect()
                logger.info("Redis connection pool disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting Redis pool: {e}")
            finally:
                _redis_pool = None


# Context manager support for proper cleanup
class RedisClientContext:
    """Async context manager for Redis client with automatic cleanup"""
    
    def __init__(self):
        self.client = None
    
    async def __aenter__(self) -> redis.Redis:
        self.client = await get_redis_client()
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Don't close singleton client, just ensure connection is healthy
        if exc_type is not None:
            logger.error(f"Redis context error: {exc_type.__name__}: {exc_val}")
        return False  # Don't suppress exceptions


# CLI test interface
async def _test_connection():
    """Test Redis connection (for debugging)"""
    print("Testing Redis connection...")
    
    try:
        client = await get_redis_client()
        
        # Test PING
        if await ping_redis(client):
            print(" PING successful")
        else:
            print(" PING failed")
            return
        
        # Get server info
        info = await get_redis_info(client)
        print(f"\n Redis Server Info:")
        print(f"   Version: {info.get('redis_version')}")
        print(f"   Uptime: {info.get('uptime_seconds')}s")
        print(f"   Connected clients: {info.get('connected_clients')}")
        print(f"   Memory usage: {info.get('used_memory_human')}")
        print(f"   Ops/sec: {info.get('instantaneous_ops_per_sec')}")
        
        # Test set/get
        test_key = "test_leibniz_connection"
        test_value = "Hello from Leibniz microservices!"
        
        await client.set(test_key, test_value, ex=60)
        print(f"\n SET {test_key}={test_value}")
        
        retrieved = await client.get(test_key)
        print(f" GET {test_key}={retrieved}")
        
        await client.delete(test_key)
        print(f" DEL {test_key}")
        
        print("\n All tests passed!")
        
    except Exception as e:
        print(f"\n Connection test failed: {e}")
    finally:
        await close_redis_client()


if __name__ == "__main__":
    # Run connection test
    asyncio.run(_test_connection())
