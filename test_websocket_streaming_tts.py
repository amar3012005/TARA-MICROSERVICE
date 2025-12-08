#!/usr/bin/env python3
"""
Test script for Sarvam WebSocket Streaming TTS Integration

Tests the new sarvamai-based streaming implementation for <500ms first audio chunk delivery.
"""

import asyncio
import base64
import json
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import websockets

# Test configuration
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "ws://localhost:8025")
TEST_TEXTS = [
    "This is a longer test sentence to verify that the streaming system is actually sending multiple chunks of audio instead of waiting for the entire text to be synthesized. We hope to see partial audio arriving very quickly.",
]


async def test_websocket_streaming():
    """Test the WebSocket streaming endpoint with streaming enabled"""
    print("=" * 70)
    print("🧪 Testing WebSocket Streaming TTS")
    print("=" * 70)
    
    session_id = f"test_session_{int(time.time())}"
    ws_url = f"{TTS_SERVICE_URL}/api/v1/stream?session_id={session_id}"
    
    print(f"📡 Connecting to: {ws_url}")
    
    try:
        async with websockets.connect(ws_url, ping_interval=30) as ws:
            # Wait for connected message
            connected_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(connected_msg)
            print(f"✅ Connected: {data}")
            
            for text in TEST_TEXTS:
                print(f"\n{'='*70}")
                print(f"📝 Testing: {text[:50]}...")
                
                start_time = time.time()
                first_chunk_time = None
                total_chunks = 0
                total_bytes = 0
                
                # Send synthesis request with streaming enabled
                await ws.send(json.dumps({
                    "type": "synthesize",
                    "text": text,
                    "emotion": "helpful",
                    "streaming": True  # Enable streaming mode
                }))
                
                # Receive responses
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        data = json.loads(msg)
                        msg_type = data.get("type")
                        
                        if msg_type == "streaming_started":
                            ultra_fast = data.get("ultra_fast", False)
                            print(f"🚀 Streaming started (ultra_fast={ultra_fast})")
                        
                        elif msg_type == "audio":
                            total_chunks += 1
                            audio_b64 = data.get("data", "")
                            if audio_b64:
                                audio_bytes = base64.b64decode(audio_b64)
                                total_bytes += len(audio_bytes)
                                
                                if first_chunk_time is None:
                                    first_chunk_time = time.time() - start_time
                                    latency_ms = first_chunk_time * 1000
                                    
                                    if latency_ms < 500:
                                        print(f"⚡ FIRST CHUNK: {latency_ms:.0f}ms ✓ TARGET MET (<500ms)")
                                    else:
                                        print(f"⚠️ FIRST CHUNK: {latency_ms:.0f}ms ✗ TARGET MISSED")
                                    
                                    print(f"   Chunk size: {len(audio_bytes)} bytes")
                                    print(f"   Ultra-fast: {data.get('ultra_fast', False)}")
                        
                        elif msg_type == "complete":
                            total_time = time.time() - start_time
                            print(f"✅ Complete in {total_time*1000:.0f}ms")
                            print(f"   Total chunks: {total_chunks}")
                            print(f"   Total bytes: {total_bytes}")
                            print(f"   First chunk latency: {first_chunk_time*1000:.0f}ms" if first_chunk_time else "   No audio received")
                            
                            # Check metrics
                            if data.get("first_chunk_latency_ms"):
                                print(f"   Reported first chunk: {data['first_chunk_latency_ms']:.0f}ms")
                            break
                        
                        elif msg_type == "error":
                            print(f"❌ Error: {data.get('message', 'Unknown')}")
                            break
                        
                        elif msg_type == "timeout":
                            print(f"⏰ Timeout: {data.get('message', 'Unknown')}")
                            break
                    
                    except asyncio.TimeoutError:
                        print("⏰ Response timeout")
                        break
                
                # Small delay between tests
                await asyncio.sleep(1.0)
        
        print(f"\n{'='*70}")
        print("✅ WebSocket streaming test complete!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_health_endpoint():
    """Test the health endpoint for streaming status"""
    import aiohttp
    
    print("\n" + "=" * 70)
    print("🏥 Testing Health Endpoint")
    print("=" * 70)
    
    health_url = TTS_SERVICE_URL.replace("ws://", "http://").replace("wss://", "https://") + "/health"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(health_url, timeout=5.0) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check passed")
                    print(f"   Provider: {data.get('provider', 'unknown')}")
                    print(f"   Streaming Provider: {data.get('streaming_provider', 'unknown')}")
                    print(f"   WebSocket TTS: {data.get('websocket_tts', 'unknown')}")
                    
                    metrics = data.get("websocket_tts_metrics", {})
                    if metrics:
                        print(f"   Connected: {metrics.get('connected', False)}")
                        print(f"   Avg Latency: {metrics.get('avg_latency', 0):.3f}s")
                        print(f"   Fallback Count: {metrics.get('fallback_count', 0)}")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


async def test_websocket_tts_metrics():
    """Test the WebSocket TTS metrics endpoint"""
    import aiohttp
    
    print("\n" + "=" * 70)
    print("📊 Testing WebSocket TTS Metrics")
    print("=" * 70)
    
    metrics_url = TTS_SERVICE_URL.replace("ws://", "http://").replace("wss://", "https://") + "/websocket-tts/metrics"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(metrics_url, timeout=5.0) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Metrics retrieved")
                    print(f"   Status: {data.get('status', 'unknown')}")
                    print(f"   WebSocket Connected: {data.get('websocket_connected', False)}")
                    print(f"   Total Chunks: {data.get('total_chunks', 0)}")
                    print(f"   Avg Latency: {data.get('avg_latency_seconds', 0):.3f}s")
                    print(f"   Target Latency: {data.get('target_latency_seconds', 0.5):.3f}s")
                    print(f"   Fallback Count: {data.get('fallback_count', 0)}")
                    return True
                else:
                    print(f"❌ Metrics failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Metrics error: {e}")
        return False


async def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("🚀 SARVAM WEBSOCKET STREAMING TTS TEST SUITE")
    print("=" * 70)
    print(f"Target: {TTS_SERVICE_URL}")
    print(f"Goal: <500ms first audio chunk")
    print("=" * 70)
    
    results = {
        "health": await test_health_endpoint(),
        "metrics": await test_websocket_tts_metrics(),
        "streaming": await test_websocket_streaming()
    }
    
    print("\n" + "=" * 70)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️ SOME TESTS FAILED")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
