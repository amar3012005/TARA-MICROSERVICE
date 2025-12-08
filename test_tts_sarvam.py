import asyncio
import httpx
import websockets
import json
import time

TTS_BASE_URL = "http://localhost:2005"
TTS_WS_URL = "ws://localhost:2005/api/v1/stream?session_id=test_session"

async def test_http_synthesize():
    """Test the current HTTP /api/v1/synthesize endpoint (non-streaming)."""
    print("🧪 Testing HTTP /api/v1/synthesize (Non-Streaming)...")
    start_time = time.time()
    
    payload = {
        "text": "Namaskaram, ela unnaru?",  # Telugu greeting
        "language": "te-IN",
        "voice": "meera"  # Example voice
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:2005/health")  # Quick health check first
            if response.status_code != 200:
                print("   ❌ Service not healthy")
                return
            
            response = await client.post(f"{TTS_BASE_URL}/api/v1/synthesize", json=payload)
            print(f"   Status: {response.status_code}")
            if response.status_code != 200:
                print(f"   ❌ Bad status: {response.text}")
                return
            data = response.json()
            
            latency = time.time() - start_time
            print(f"Latency: {latency:.2f}s")
            print(f"   Response Keys: {list(data.keys())}")
            print(f"   Audio Length (base64): {len(data.get('audio', ''))} chars")
            
            # Check if it's base64 audio (non-streaming)
            if "audio" in data and data.get("success"):
                print("   ✅ Full audio generated and returned (non-streaming).")
            else:
                print("   ❌ No audio or error in response.")
                
    except Exception as e:
        print(f"   ❌ HTTP Test Failed: {e}")

async def test_websocket_stream():
    """Test the WebSocket /api/v1/stream endpoint (intended for streaming)."""
    print("\n🧪 Testing WebSocket /api/v1/stream (Streaming)...")
    start_time = time.time()
    
    try:
        async with websockets.connect(TTS_WS_URL) as websocket:
            # Send synthesis request
            await websocket.send(json.dumps({
                "type": "synthesize",
                "text": "Namaskaram, ela unnaru?"
            }))
            
            chunks_received = 0
            total_bytes = 0
            
            # Receive messages (should be JSON with audio data)
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")
                print(f"   📦 Received {msg_type}: {data}")
                
                if msg_type == "audio":
                    chunks_received += 1
                    audio_b64 = data.get("data", "")
                    total_bytes += len(audio_b64)
                    
                    # Stop after a few chunks for testing
                    if chunks_received >= 3:
                        break
                elif msg_type == "error":
                    print(f"   ❌ Error: {data.get('message')}")
                    break
                elif msg_type == "complete":
                    print("   ✅ Synthesis complete")
                    break
                
    except Exception as e:
        print(f"   ❌ WebSocket Test Failed: {e}")

async def main():
    """Run both tests."""
    print("🚀 Starting TTS Sarvam Test Suite\n")
    
    await test_http_synthesize()
    await test_websocket_stream()
    
    print("\n📊 Summary:")
    print("   - HTTP: Should return full base64 audio (current behavior).")
    print("   - WebSocket: Should stream binary audio chunks (per Sarvam API).")
    print("   - If WebSocket fails or returns no chunks, implement Sarvam streaming.")

if __name__ == "__main__":
    asyncio.run(main())