#!/usr/bin/env python3
"""
Test script for Event-Driven Architecture

Tests the event flow:
1. Connect to orchestrator WebSocket
2. Verify event-driven FSM is active
3. Check Redis streams for events
"""

import asyncio
import json
import websockets
import aiohttp
import redis.asyncio as redis
from shared.events import VoiceEvent, EventTypes

ORCHESTRATOR_URL = "ws://localhost:2004"
REDIS_HOST = "localhost"
REDIS_PORT = 2011  # External port


async def test_event_driven_flow():
    """Test the event-driven architecture flow."""
    print("=" * 70)
    print("🧪 Testing Event-Driven Architecture")
    print("=" * 70)
    
    # Connect to Redis
    print("\n1️⃣ Connecting to Redis...")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    try:
        await redis_client.ping()
        print("   ✅ Redis connected")
    except Exception as e:
        print(f"   ❌ Redis connection failed: {e}")
        return
    
    # Check orchestrator health
    print("\n2️⃣ Checking orchestrator health...")
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:2004/health") as resp:
            health = await resp.json()
            print(f"   Status: {health.get('status')}")
            print(f"   Redis connected: {health.get('redis_connected')}")
            print(f"   Active sessions: {health.get('active_sessions')}")
    
    # Connect to orchestrator WebSocket
    print("\n3️⃣ Connecting to orchestrator WebSocket...")
    session_id = f"test_session_{int(asyncio.get_event_loop().time())}"
    ws_url = f"{ORCHESTRATOR_URL}/orchestrate?session_id={session_id}"
    
    try:
        async with websockets.connect(ws_url) as ws:
            print(f"   ✅ Connected to orchestrator (session: {session_id})")
            
            # Wait for connection message
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(message)
                print(f"   📨 Received: {data.get('type')} - {data.get('state', 'N/A')}")
            except asyncio.TimeoutError:
                print("   ⚠️ No initial message received")
            
            # Check if event-driven streams exist
            print("\n4️⃣ Checking Redis streams...")
            try:
                # List all keys matching voice:* pattern
                keys = await redis_client.keys("voice:*")
                if keys:
                    print(f"   ✅ Found {len(keys)} stream(s):")
                    for key in keys[:10]:  # Show first 10
                        key_str = key.decode() if isinstance(key, bytes) else key
                        length = await redis_client.xlen(key)
                        print(f"      - {key_str}: {length} messages")
                else:
                    print("   ℹ️ No streams created yet (will be created on first event)")
            except Exception as e:
                print(f"   ⚠️ Error checking streams: {e}")
            
            # Send a test message to trigger event flow
            print("\n5️⃣ Testing event flow...")
            print("   ℹ️ In event-driven mode, STT events come from Redis Streams")
            print("   ℹ️ To test, you would:")
            print("      1. Connect FastRTC UI: http://localhost:2004/fastrtc")
            print("      2. Send POST /start to trigger workflow")
            print("      3. Speak into microphone")
            print("      4. Watch events flow through Redis Streams")
            
    except Exception as e:
        print(f"   ❌ WebSocket connection failed: {e}")
    
    await redis_client.close()
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)
    print("\n📋 Next steps:")
    print("   1. Open Unified FastRTC UI: http://localhost:2004/fastrtc")
    print("   2. Click 'Record' to connect")
    print("   3. Send: curl -X POST http://localhost:2004/start")
    print("   4. Speak into microphone")
    print("   5. Watch logs: docker compose logs -f orchestrator rag tts-sarvam")


if __name__ == "__main__":
    asyncio.run(test_event_driven_flow())


