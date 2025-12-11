#!/usr/bin/env python3
"""
Ultra-Fast WebSocket TTS Streaming Test

Tests <500ms first audio chunk delivery via Sarvam WebSocket TTS.
"""

import asyncio
import websockets
import json
import time
import base64
import requests

async def test_ultra_fast_websocket_tts():
    """Test ultra-fast WebSocket TTS streaming"""
    uri = "ws://localhost:2005/api/v1/stream?session_id=ultra_fast_test"

    print("🎯 Testing Ultra-Fast WebSocket TTS Streaming (<500ms target)")
    print("=" * 60)

    try:
        async with websockets.connect(uri) as ws:
            print("🔌 Connected to TTS WebSocket")

            # Wait for connection confirmation
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"📨 Connection: {data}")

            # Test WebSocket TTS metrics
            print("\n📊 Checking WebSocket TTS Status...")
            response = requests.get("http://localhost:2005/websocket-tts/metrics", timeout=5)
            if response.status_code == 200:
                metrics = response.json()
                print(f"   Status: {metrics.get('status', 'unknown')}")
                print(f"   Connected: {metrics.get('websocket_connected', False)}")
                print(".3f"                print(f"   Total Chunks: {metrics.get('total_chunks', 0)}")
                print(f"   Fallbacks: {metrics.get('fallback_count', 0)}")

            # Send ultra-fast streaming request
            test_text = "Hello! This should deliver the first audio chunk in under 500ms with ultra-fast WebSocket streaming."
            request = {
                "type": "synthesize",
                "text": test_text,
                "emotion": "helpful",
                "streaming": True
            }

            start_time = time.time()
            await ws.send(json.dumps(request))
            print(f"\n📤 Sent streaming request: '{test_text[:50]}...'")
            print(".3f"
            # Wait for responses
            first_audio_time = None
            chunks_received = 0
            streaming_started = False

            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    data = json.loads(msg)

                    if data.get('type') == 'streaming_started':
                        streaming_started = True
                        ultra_fast = data.get('ultra_fast', False)
                        print(f"✅ Streaming started (ultra_fast: {ultra_fast})")

                    elif data.get('type') == 'audio':
                        chunks_received += 1

                        if chunks_received == 1:
                            first_audio_time = time.time() - start_time
                            latency_ms = first_audio_time * 1000
                            audio_size = len(base64.b64decode(data['data']))
                            ultra_fast = data.get('ultra_fast', False)

                            print("🎵 FIRST AUDIO CHUNK RECEIVED!")
                            print(f"   Latency: {first_audio_time:.3f}s")
                            print(f"   Size: {audio_size} bytes")                            print(f"   Chunk Size: {audio_size} bytes")
                            print(f"   Ultra Fast: {ultra_fast}")

                            # Check if we met the target
                            if first_audio_time < 0.5:
                                print("✅ SUCCESS: Ultra-fast streaming achieved! (<500ms)")
                            elif first_audio_time < 1.0:
                                print("⚠️ GOOD: Fast streaming achieved (<1s)")
                            else:
                                print("❌ SLOW: First chunk took too long (>1s)")

                    elif data.get('type') == 'complete':
                        break

                    elif data.get('type') == 'error':
                        print(f"❌ Error: {data.get('message', 'Unknown error')}")
                        break

                except asyncio.TimeoutError:
                    print("⏰ Timeout waiting for response")
                    break

            total_time = time.time() - start_time
            print("\n🏁 Test Results:")
            print(f"   Total Time: {total_time:.3f}s")
            print(f"   Chunks Received: {chunks_received}")
            print(f"   Streaming Started: {streaming_started}")

            if first_audio_time:
                print(f"   First Chunk Latency: {first_audio_time:.3f}s")
                print(f"   Target Met: {first_audio_time < 0.5}")                if first_audio_time < 0.5:
                    print("🎉 EXCELLENT: Achieved ultra-low latency target!")
                elif first_audio_time < 1.0:
                    print("👍 GOOD: Fast enough for real-time experience")
                else:
                    print("⚠️ SLOW: May need optimization")
            else:
                print("❌ FAILED: No audio chunks received")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    print("\n" + "=" * 60)
    return first_audio_time < 0.5 if first_audio_time else False

async def test_fastrtc_ultra_fast():
    """Test FastRTC with ultra-fast streaming"""
    print("\n🚀 Testing FastRTC Ultra-Fast Streaming...")

    try:
        start_time = time.time()

        response = requests.post(
            "http://localhost:2005/api/v1/fastrtc/synthesize",
            json={
                "text": "FastRTC ultra-fast streaming test.",
                "emotion": "helpful"
            },
            timeout=10
        )

        request_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print(".3f"            print(f"   Streaming: {data.get('streaming', False)}")
            print(f"   Message: {data.get('message', 'No message')}")
        else:
            print(f"❌ FastRTC failed: {response.status_code}")

    except Exception as e:
        print(f"❌ FastRTC test failed: {e}")

def run_all_tests():
    """Run all ultra-fast TTS tests"""
    print("🎤 ULTRA-FAST WEBSOCKET TTS PERFORMANCE TEST")
    print("Target: <500ms first audio chunk")
    print("=" * 60)

    # Test WebSocket streaming
    success = asyncio.run(test_ultra_fast_websocket_tts())

    # Test FastRTC integration
    asyncio.run(test_fastrtc_ultra_fast())

    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: Ultra-fast WebSocket TTS streaming is working!")
        print("   First audio chunks are delivered in <500ms")
    else:
        print("⚠️ PARTIAL: WebSocket TTS is running but may need optimization")
        print("   Check metrics at: http://localhost:2005/websocket-tts/metrics")

if __name__ == "__main__":
    run_all_tests()



