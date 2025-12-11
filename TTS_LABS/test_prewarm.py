#!/usr/bin/env python3
"""
Test script for TTS_LABS WebSocket prewarm functionality.

This script verifies that pre-warming the ElevenLabs connection
reduces the time-to-first-audio to <200ms (ideally ~150ms).

Usage:
    python test_prewarm.py [--no-prewarm]
"""

import asyncio
import json
import time
import argparse
import websockets

TTS_LABS_WS_URL = "ws://localhost:2007/api/v1/stream"
TEST_TEXT = "Hello! This is a test of the ultra-low latency streaming."


async def test_with_prewarm():
    """Test with pre-warming enabled."""
    print("=" * 60)
    print("TEST: With Pre-warming")
    print("=" * 60)
    
    session_id = f"test_prewarm_{int(time.time())}"
    url = f"{TTS_LABS_WS_URL}?session_id={session_id}"
    
    async with websockets.connect(url) as ws:
        # Wait for connection confirmation
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"[1] Connected: {data.get('type')}")
        
        # Step 1: Send prewarm
        prewarm_start = time.time()
        await ws.send(json.dumps({"type": "prewarm"}))
        print(f"[2] Sent prewarm request...")
        
        # Wait for prewarm confirmation
        msg = await ws.recv()
        data = json.loads(msg)
        prewarm_duration = (time.time() - prewarm_start) * 1000
        print(f"[3] Prewarm response: {data.get('type')} - {data.get('status')}")
        server_prewarm_ms = data.get('prewarm_duration_ms')
        server_str = f"{server_prewarm_ms:.0f}ms" if server_prewarm_ms else "N/A"
        print(f"    Prewarm took: {prewarm_duration:.0f}ms (server reported: {server_str})")
        
        # Step 2: Simulate user speaking (wait 2 seconds)
        print(f"[4] Simulating user speaking for 2 seconds...")
        await asyncio.sleep(2.0)
        
        # Step 3: Send text chunk and measure time to first audio
        chunk_send_time = time.time()
        await ws.send(json.dumps({"type": "stream_chunk", "text": TEST_TEXT}))
        print(f"[5] Sent text chunk: '{TEST_TEXT[:40]}...'")
        
        # Step 4: Immediately send stream_end to trigger audio generation
        await ws.send(json.dumps({"type": "stream_end"}))
        print(f"[6] Sent stream_end to trigger audio generation")
        
        # Wait for first audio
        first_audio_time = None
        audio_chunks = 0
        
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(msg)
                
                if data.get("type") == "audio":
                    if first_audio_time is None:
                        first_audio_time = time.time()
                        latency_ms = (first_audio_time - chunk_send_time) * 1000
                        print(f"[7] FIRST AUDIO RECEIVED!")
                        print(f"    ⚡ Time to First Audio: {latency_ms:.0f}ms")
                        if latency_ms < 200:
                            print(f"    ✅ SUCCESS: Latency is under 200ms target!")
                        else:
                            print(f"    ⚠️  WARNING: Latency exceeded 200ms target")
                    
                    audio_chunks += 1
                    
                    # Check if final
                    if data.get("is_final"):
                        print(f"[8] Stream complete. Total chunks: {audio_chunks}")
                        break
                
                elif data.get("type") == "stream_complete":
                    print(f"[8] Stream complete signal received. Total chunks: {audio_chunks}")
                    break
                        
            except asyncio.TimeoutError:
                print("[!] Timeout waiting for audio")
                break
        
        return first_audio_time - chunk_send_time if first_audio_time else None


async def test_without_prewarm():
    """Test without pre-warming (cold start)."""
    print("=" * 60)
    print("TEST: Without Pre-warming (Cold Start)")
    print("=" * 60)
    
    session_id = f"test_cold_{int(time.time())}"
    url = f"{TTS_LABS_WS_URL}?session_id={session_id}"
    
    async with websockets.connect(url) as ws:
        # Wait for connection confirmation
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"[1] Connected: {data.get('type')}")
        
        # Step 1: Send text chunk directly (no prewarm)
        chunk_send_time = time.time()
        await ws.send(json.dumps({"type": "stream_chunk", "text": TEST_TEXT}))
        print(f"[2] Sent text chunk (cold): '{TEST_TEXT[:40]}...'")
        
        # Step 2: Immediately send stream_end to signal we're done
        await ws.send(json.dumps({"type": "stream_end"}))
        print(f"[3] Sent stream_end to trigger audio generation")
        
        # Wait for first audio
        first_audio_time = None
        audio_chunks = 0
        
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
                data = json.loads(msg)
                
                if data.get("type") == "audio":
                    if first_audio_time is None:
                        first_audio_time = time.time()
                        latency_ms = (first_audio_time - chunk_send_time) * 1000
                        print(f"[4] FIRST AUDIO RECEIVED!")
                        print(f"    ⏱️  Time to First Audio (cold): {latency_ms:.0f}ms")
                    
                    audio_chunks += 1
                    
                    if data.get("is_final"):
                        print(f"[5] Stream complete. Total chunks: {audio_chunks}")
                        break
                
                elif data.get("type") == "stream_complete":
                    print(f"[5] Stream complete signal received. Total chunks: {audio_chunks}")
                    break
                        
            except asyncio.TimeoutError:
                print("[!] Timeout waiting for audio")
                break
        
        return first_audio_time - chunk_send_time if first_audio_time else None


async def main():
    parser = argparse.ArgumentParser(description="Test TTS_LABS prewarm functionality")
    parser.add_argument("--no-prewarm", action="store_true", help="Test without pre-warming")
    parser.add_argument("--compare", action="store_true", help="Run both tests and compare")
    args = parser.parse_args()
    
    if args.compare:
        # Run both tests
        cold_latency = await test_without_prewarm()
        print("\n")
        await asyncio.sleep(1.0)  # Brief pause between tests
        warm_latency = await test_with_prewarm()
        
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        if cold_latency and warm_latency:
            cold_ms = cold_latency * 1000
            warm_ms = warm_latency * 1000
            improvement = cold_ms - warm_ms
            print(f"Cold Start Latency:  {cold_ms:.0f}ms")
            print(f"Pre-warmed Latency:  {warm_ms:.0f}ms")
            print(f"Improvement:         {improvement:.0f}ms ({improvement/cold_ms*100:.1f}% faster)")
        
    elif args.no_prewarm:
        await test_without_prewarm()
    else:
        await test_with_prewarm()


if __name__ == "__main__":
    asyncio.run(main())



