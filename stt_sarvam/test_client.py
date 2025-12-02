"""
Test Client for STT/VAD Microservice

Tests the WebSocket endpoint with PCM audio streaming.
"""

import asyncio
import json
import wave
import struct
import httpx
from websockets import connect

SERVICE_URL = "http://localhost:8001"
WS_URL = "ws://localhost:8001/api/v1/transcribe/stream"


async def test_health_check():
    """Test health endpoint"""
    print("\n=== Testing Health Check ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{SERVICE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")


async def test_metrics():
    """Test metrics endpoint"""
    print("\n=== Testing Metrics ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{SERVICE_URL}/metrics")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")


async def test_websocket_with_synthetic_audio():
    """Test WebSocket with synthetic PCM audio (sine wave)"""
    print("\n=== Testing WebSocket with Synthetic Audio ===")
    
    # Generate 3 seconds of synthetic audio (sine wave at 440Hz)
    sample_rate = 16000
    duration = 3.0
    frequency = 440.0
    
    import math
    import numpy as np
    
    num_samples = int(sample_rate * duration)
    audio_samples = np.array([
        int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t / sample_rate))
        for t in range(num_samples)
    ], dtype=np.int16)
    
    # Convert to bytes
    audio_bytes = audio_samples.tobytes()
    
    # Split into chunks (50ms = 800 samples = 1600 bytes)
    chunk_size = 1600
    chunks = [audio_bytes[i:i+chunk_size] for i in range(0, len(audio_bytes), chunk_size)]
    
    print(f"Generated {len(chunks)} chunks ({duration}s of audio)")
    
    async with connect(WS_URL) as websocket:
        print(" WebSocket connected")
        
        # Send audio chunks
        for i, chunk in enumerate(chunks):
            await websocket.send(chunk)
            if i % 10 == 0:
                print(f" Sent chunk {i+1}/{len(chunks)}")
            await asyncio.sleep(0.05)  # 50ms delay between chunks
        
        print(" All chunks sent, waiting for responses...")
        
        # Receive responses
        try:
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                print(f" Received: {data}")
                
                if data["type"] in ["final", "timeout", "error"]:
                    break
        except asyncio.TimeoutError:
            print("⏱️ Response timeout")


async def test_websocket_with_silence():
    """Test WebSocket with silence (should timeout)"""
    print("\n=== Testing WebSocket with Silence (Timeout Expected) ===")
    
    # Generate 2 seconds of silence
    sample_rate = 16000
    duration = 2.0
    
    import numpy as np
    num_samples = int(sample_rate * duration)
    audio_samples = np.zeros(num_samples, dtype=np.int16)
    audio_bytes = audio_samples.tobytes()
    
    # Split into chunks
    chunk_size = 1600
    chunks = [audio_bytes[i:i+chunk_size] for i in range(0, len(audio_bytes), chunk_size)]
    
    print(f"Generated {len(chunks)} silent chunks")
    
    async with connect(WS_URL) as websocket:
        print(" WebSocket connected")
        
        # Send silence chunks
        for i, chunk in enumerate(chunks):
            await websocket.send(chunk)
            await asyncio.sleep(0.05)
        
        print(" Sent silence, waiting for timeout...")
        
        # Receive responses
        try:
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                data = json.loads(message)
                print(f" Received: {data}")
                
                if data["type"] in ["final", "timeout", "error"]:
                    break
        except asyncio.TimeoutError:
            print("⏱️ No response within 15s")


async def main():
    """Run all tests"""
    try:
        # Test health
        await test_health_check()
        
        # Test metrics
        await test_metrics()
        
        # Test WebSocket with synthetic audio
        await test_websocket_with_synthetic_audio()
        
        # Test WebSocket with silence
        await test_websocket_with_silence()
        
        # Final metrics
        await test_metrics()
        
        print("\n All tests complete!")
        
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
