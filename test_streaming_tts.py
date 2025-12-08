#!/usr/bin/env python3
"""
Test script for TTS streaming WebSocket API
"""

import asyncio
import websockets
import json
import time

async def test_streaming_tts():
    """Test the WebSocket streaming TTS endpoint"""
    uri = "ws://localhost:2005/api/v1/stream?session_id=test_streaming"

    try:
        async with websockets.connect(uri) as websocket:
            print("🔌 Connected to TTS streaming WebSocket")

            # Wait for connection confirmation
            response = await websocket.recv()
            data = json.loads(response)
            print(f"📨 Connection response: {data}")

            # Send streaming synthesis request
            request = {
                "type": "synthesize",
                "text": "Hello! This is a test of the ultra-low latency streaming TTS API.",
                "emotion": "helpful",
                "streaming": True
            }

            print(f"📤 Sending request: {request}")
            await websocket.send(json.dumps(request))

            start_time = time.time()
            audio_chunks = 0

            # Listen for responses
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(response)

                    msg_type = data.get("type")
                    print(f"📨 Received {msg_type}: {data.get('type', 'unknown')}")

                    if msg_type == "audio":
                        audio_chunks += 1
                        if audio_chunks == 1:
                            first_audio_time = time.time() - start_time
                            print(f"🎵 First audio chunk received in {first_audio_time:.2f}s")
                    elif msg_type == "sentence_start":
                        print(f"🎵 Sentence started: {data.get('text', '')[:50]}...")

                    elif msg_type == "complete":
                        total_time = time.time() - start_time
                        print(f"✅ Synthesis complete in {total_time:.2f}s")
                        print(f"   Audio chunks received: {audio_chunks}")
                        break

                    elif msg_type == "error":
                        print(f"❌ Error: {data}")
                        break

                except asyncio.TimeoutError:
                    print("⏰ Timeout waiting for response")
                    break

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_streaming_tts())
