#!/usr/bin/env python3
"""
Test script for TTS Streaming Service with FastRTC

Tests the FastRTC audio output functionality.
"""

import asyncio
import httpx
import json

BASE_URL = "http://localhost:8005"


async def test_fastrtc_synthesis():
    """Test FastRTC synthesis endpoint"""
    print("=" * 70)
    print("Testing FastRTC TTS Synthesis")
    print("=" * 70)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test synthesis
        response = await client.post(
            f"{BASE_URL}/api/v1/fastrtc/synthesize",
            json={
                "text": "Hello! This is a test of the TTS streaming service. How does it sound?",
                "emotion": "helpful"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Synthesis successful!")
            print(f"   Sentences: {result.get('sentences', 0)}")
            print(f"   Duration: {result.get('total_duration_ms', 0):.0f}ms")
            print(f"   Sentences played: {result.get('sentences_played', 0)}")
        else:
            print(f"❌ Synthesis failed: {response.status_code}")
            print(f"   Response: {response.text}")


async def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 70)
    print("Testing Health Check")
    print("=" * 70)
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Service is {health.get('status', 'unknown')}")
            print(f"   Provider: {health.get('provider', 'unknown')}")
            print(f"   Cache: {health.get('cache', 'unknown')}")
            print(f"   Active sessions: {health.get('active_sessions', 0)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")


if __name__ == "__main__":
    print("\n🚀 TTS Streaming Service Test")
    print("=" * 70)
    print("Make sure the service is running:")
    print("  docker-compose up -d tts-streaming-service")
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


