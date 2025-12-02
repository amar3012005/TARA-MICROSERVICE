import asyncio
import redis.asyncio as redis
import os

async def test_redis():
    host = os.getenv("LEIBNIZ_REDIS_HOST", "redis")
    port = int(os.getenv("LEIBNIZ_REDIS_PORT", 6379))
    print(f"Connecting to {host}:{port}...")
    
    try:
        client = redis.Redis(host=host, port=port, socket_connect_timeout=5.0, socket_timeout=5.0)
        print("Ping...")
        await client.ping()
        print("Success!")
        await client.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_redis())

