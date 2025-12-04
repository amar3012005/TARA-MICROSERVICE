#!/usr/bin/env python3
"""Quick smoke tests for the XTTS FastRTC endpoint."""

import asyncio
import httpx
import json

BASE_URL = "http://localhost:8005"


async def test_fastrtc_synthesis():
    """Test FastRTC synthesis endpoint"""
    print("=" * 70)
    print("Testing XTTS FastRTC synthesis")
    print("=" * 70)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test synthesis
        response = await client.post(
            f"{BASE_URL}/api/v1/fastrtc/synthesize",
            json={
                "text": "Hello! This is the XTTS native streaming microservice speaking in real time.",
                "emotion": "helpful"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Synthesis successful!")
            print(f"   Duration: {result.get('duration_ms', 0):.0f}ms")
        else:
            print(f"❌ Synthesis failed: {response.status_code}")
            print(f"   Response: {response.text}")


async def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 70)
    print("Testing health check")
    print("=" * 70)
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Service is {health.get('status', 'unknown')}")
            print(f"   Provider: {health.get('provider', 'unknown')}")
            print(f"   Cache: {health.get('cache', {})}")
            print(f"   Active sessions: {health.get('active_sessions', 0)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")


if __name__ == "__main__":
    print("\n🚀 XTTS-v2 Streaming Smoke Test")
    print("=" * 70)
    print("Make sure the service is running:")
    print("  docker-compose up -d tts-xtts-v2-service")
    print("=" * 70 + "\n")
    
    asyncio.run(test_health())
    asyncio.run(test_fastrtc_synthesis())
    
    print("\n" + "=" * 70)
    print("📱 To hear audio:")
    print("   1. Open http://localhost:8005/fastrtc in your browser")
    print("   2. Use the HTTPS URL provided by Gradio (required for audio)")
    print("   3. Send text via POST /api/v1/fastrtc/synthesize")
    print("   4. Audio will stream to your browser speakers")
    print("=" * 70)






