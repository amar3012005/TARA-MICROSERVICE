"""
Test script for ElevenLabs TTS Direct Streaming from Orchestrator

Tests the end-to-end latency of the ElevenLabs TTS client when used
from the orchestrator, including prewarm and continuous streaming scenarios.

Usage:
    python test_eleven_tts_direct.py [--tts-url URL] [--text TEXT]

Example:
    python test_eleven_tts_direct.py --tts-url http://localhost:8006
"""

import asyncio
import argparse
import base64
import json
import time
import sys
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("aiohttp not installed. Run: pip install aiohttp")
    sys.exit(1)


# Test text samples
GERMAN_TEST_TEXT = "Willkommen bei unserem Kundenservice! Wie kann ich Ihnen heute helfen?"
TELUGU_TEST_TEXT = "నమస్కారం! నేను మీకు ఎలా సహాయం చేయగలను?"
ENGLISH_TEST_TEXT = "Hello! How can I help you today?"

# Simulated RAG streaming chunks (as if coming from LLM)
RAG_CHUNKS = [
    "Thank you for ",
    "your question. ",
    "I'd be happy to ",
    "help you with ",
    "that inquiry. ",
    "Let me check ",
    "our system and ",
    "provide you with ",
    "the most accurate ",
    "information available."
]


async def test_with_prewarm(tts_url: str, session_id: str):
    """Test with prewarmed connection (simulating VAD-triggered prewarm)."""
    print("\n" + "=" * 70)
    print("TEST: ElevenLabs Direct Streaming WITH PREWARM")
    print("=" * 70)
    
    ws_url = f"{tts_url.replace('http://', 'ws://').replace('https://', 'wss://')}/api/v1/stream?session_id={session_id}"
    
    metrics = {
        "prewarm_latency_ms": None,
        "first_audio_latency_ms": None,
        "total_audio_bytes": 0,
        "chunk_count": 0,
        "total_time_ms": None
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, timeout=aiohttp.ClientTimeout(total=30)) as ws:
                # Wait for connection confirmation
                msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "connected":
                        print(f"✅ Connected to TTS service: {data.get('provider', 'unknown')}")
                
                # Step 1: Send prewarm message (simulating VAD detection)
                print("\n📡 Sending prewarm message...")
                prewarm_start = time.time()
                
                await ws.send_json({"type": "prewarm"})
                
                # Wait for prewarm confirmation
                msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "prewarmed":
                        prewarm_latency = (time.time() - prewarm_start) * 1000
                        metrics["prewarm_latency_ms"] = prewarm_latency
                        server_prewarm_ms = data.get("prewarm_duration_ms", 0)
                        print(f"⚡ Pre-warmed in {prewarm_latency:.0f}ms (server: {server_prewarm_ms:.0f}ms)")
                    elif data.get("type") == "error":
                        print(f"❌ Prewarm error: {data.get('message')}")
                        return metrics
                
                # Step 2: Send text chunks (simulating RAG streaming)
                print("\n📤 Sending text chunks (simulating RAG streaming)...")
                stream_start = time.time()
                first_audio_time = None
                
                # Start background task to receive audio
                audio_receive_task = asyncio.create_task(
                    receive_audio(ws, metrics, stream_start, lambda t: first_audio_time if first_audio_time else t)
                )
                
                for i, chunk in enumerate(RAG_CHUNKS):
                    await ws.send_json({
                        "type": "stream_chunk",
                        "text": chunk,
                        "emotion": "helpful"
                    })
                    print(f"  Chunk {i+1}/{len(RAG_CHUNKS)}: {len(chunk)} chars")
                    await asyncio.sleep(0.05)  # Small delay to simulate LLM token generation
                
                # Send stream end
                await ws.send_json({"type": "stream_end"})
                print("\n📤 Stream end sent")
                
                # Wait for audio to complete
                try:
                    first_audio_time = await asyncio.wait_for(audio_receive_task, timeout=15.0)
                    if first_audio_time:
                        metrics["first_audio_latency_ms"] = (first_audio_time - stream_start) * 1000
                except asyncio.TimeoutError:
                    print("⚠️ Timeout waiting for audio")
                
                metrics["total_time_ms"] = (time.time() - stream_start) * 1000
                
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    # Print results
    print("\n" + "-" * 40)
    print("RESULTS (with prewarm):")
    print(f"  Prewarm latency:    {metrics['prewarm_latency_ms']:.0f}ms" if metrics['prewarm_latency_ms'] else "  Prewarm: FAILED")
    print(f"  First audio:        {metrics['first_audio_latency_ms']:.0f}ms" if metrics['first_audio_latency_ms'] else "  First audio: N/A")
    print(f"  Total audio:        {metrics['total_audio_bytes']} bytes ({metrics['chunk_count']} chunks)")
    print(f"  Total time:         {metrics['total_time_ms']:.0f}ms" if metrics['total_time_ms'] else "  Total time: N/A")
    print("-" * 40)
    
    return metrics


async def test_without_prewarm(tts_url: str, session_id: str):
    """Test without prewarmed connection (cold start)."""
    print("\n" + "=" * 70)
    print("TEST: ElevenLabs Direct Streaming WITHOUT PREWARM (Cold Start)")
    print("=" * 70)
    
    ws_url = f"{tts_url.replace('http://', 'ws://').replace('https://', 'wss://')}/api/v1/stream?session_id={session_id}"
    
    metrics = {
        "first_audio_latency_ms": None,
        "total_audio_bytes": 0,
        "chunk_count": 0,
        "total_time_ms": None
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, timeout=aiohttp.ClientTimeout(total=30)) as ws:
                # Wait for connection confirmation
                msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "connected":
                        print(f"✅ Connected to TTS service: {data.get('provider', 'unknown')}")
                
                # Send text chunks immediately (no prewarm)
                print("\n📤 Sending text chunks (cold start - no prewarm)...")
                stream_start = time.time()
                
                # Start background task to receive audio
                audio_receive_task = asyncio.create_task(
                    receive_audio(ws, metrics, stream_start, lambda t: t)
                )
                
                for i, chunk in enumerate(RAG_CHUNKS):
                    await ws.send_json({
                        "type": "stream_chunk",
                        "text": chunk,
                        "emotion": "helpful"
                    })
                    print(f"  Chunk {i+1}/{len(RAG_CHUNKS)}: {len(chunk)} chars")
                    await asyncio.sleep(0.05)
                
                # Send stream end
                await ws.send_json({"type": "stream_end"})
                print("\n📤 Stream end sent")
                
                # Wait for audio to complete
                try:
                    first_audio_time = await asyncio.wait_for(audio_receive_task, timeout=15.0)
                    if first_audio_time:
                        metrics["first_audio_latency_ms"] = (first_audio_time - stream_start) * 1000
                except asyncio.TimeoutError:
                    print("⚠️ Timeout waiting for audio")
                
                metrics["total_time_ms"] = (time.time() - stream_start) * 1000
                
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    # Print results
    print("\n" + "-" * 40)
    print("RESULTS (cold start):")
    print(f"  First audio:        {metrics['first_audio_latency_ms']:.0f}ms" if metrics['first_audio_latency_ms'] else "  First audio: N/A")
    print(f"  Total audio:        {metrics['total_audio_bytes']} bytes ({metrics['chunk_count']} chunks)")
    print(f"  Total time:         {metrics['total_time_ms']:.0f}ms" if metrics['total_time_ms'] else "  Total time: N/A")
    print("-" * 40)
    
    return metrics


async def receive_audio(ws, metrics: dict, stream_start: float, get_first_time):
    """Background task to receive and process audio chunks."""
    first_audio_time = None
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get("type")
                
                if msg_type == "audio":
                    audio_b64 = data.get("data", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        metrics["total_audio_bytes"] += len(audio_bytes)
                        metrics["chunk_count"] += 1
                        
                        if first_audio_time is None:
                            first_audio_time = time.time()
                            latency_ms = (first_audio_time - stream_start) * 1000
                            
                            if latency_ms < 150:
                                print(f"  ⚡ ULTRA-FAST first audio: {latency_ms:.0f}ms")
                            elif latency_ms < 300:
                                print(f"  ✅ Fast first audio: {latency_ms:.0f}ms")
                            else:
                                print(f"  ⚠️ Slow first audio: {latency_ms:.0f}ms")
                
                elif msg_type == "stream_complete":
                    print("  📥 Stream complete")
                    break
                
                elif msg_type == "error":
                    print(f"  ❌ Error: {data.get('message')}")
                    break
                    
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
                
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"  ❌ Receive error: {e}")
    
    return first_audio_time


async def test_single_synthesis(tts_url: str, session_id: str, text: str = ENGLISH_TEST_TEXT):
    """Test single text synthesis (for intro greeting scenarios)."""
    print("\n" + "=" * 70)
    print("TEST: Single Text Synthesis")
    print(f"TEXT: {text[:50]}...")
    print("=" * 70)
    
    ws_url = f"{tts_url.replace('http://', 'ws://').replace('https://', 'wss://')}/api/v1/stream?session_id={session_id}"
    
    metrics = {
        "first_audio_latency_ms": None,
        "total_audio_bytes": 0,
        "chunk_count": 0,
        "total_time_ms": None
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, timeout=aiohttp.ClientTimeout(total=30)) as ws:
                # Wait for connection
                msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "connected":
                        print(f"✅ Connected")
                
                # Send synthesize request
                print("\n📤 Sending synthesize request...")
                stream_start = time.time()
                
                await ws.send_json({
                    "type": "synthesize",
                    "text": text,
                    "emotion": "helpful"
                })
                
                # Receive audio
                first_audio_time = None
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        
                        if msg_type == "audio":
                            audio_b64 = data.get("data", "")
                            if audio_b64:
                                audio_bytes = base64.b64decode(audio_b64)
                                metrics["total_audio_bytes"] += len(audio_bytes)
                                metrics["chunk_count"] += 1
                                
                                if first_audio_time is None:
                                    first_audio_time = time.time()
                                    latency_ms = (first_audio_time - stream_start) * 1000
                                    metrics["first_audio_latency_ms"] = latency_ms
                                    print(f"  First audio: {latency_ms:.0f}ms")
                        
                        elif msg_type == "complete":
                            metrics["total_time_ms"] = (time.time() - stream_start) * 1000
                            print(f"  ✅ Complete")
                            break
                        
                        elif msg_type == "error":
                            print(f"  ❌ Error: {data.get('message')}")
                            break
                            
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    # Print results
    print("\n" + "-" * 40)
    print("RESULTS (single synthesis):")
    print(f"  First audio:        {metrics['first_audio_latency_ms']:.0f}ms" if metrics['first_audio_latency_ms'] else "  First audio: N/A")
    print(f"  Total audio:        {metrics['total_audio_bytes']} bytes ({metrics['chunk_count']} chunks)")
    print(f"  Total time:         {metrics['total_time_ms']:.0f}ms" if metrics['total_time_ms'] else "  Total time: N/A")
    print("-" * 40)
    
    return metrics


async def main():
    parser = argparse.ArgumentParser(description="Test ElevenLabs TTS Direct Streaming")
    parser.add_argument("--tts-url", default="http://localhost:8006", help="TTS service URL")
    parser.add_argument("--text", default=ENGLISH_TEST_TEXT, help="Test text")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ElevenLabs TTS Direct Streaming Test")
    print("=" * 70)
    print(f"TTS URL: {args.tts_url}")
    print("=" * 70)
    
    # Test 1: Single text synthesis
    await test_single_synthesis(args.tts_url, f"test_single_{int(time.time())}", args.text)
    
    await asyncio.sleep(1.0)  # Brief pause between tests
    
    # Test 2: Streaming with prewarm
    prewarm_metrics = await test_with_prewarm(args.tts_url, f"test_prewarm_{int(time.time())}")
    
    await asyncio.sleep(1.0)
    
    # Test 3: Streaming without prewarm (cold start)
    cold_metrics = await test_without_prewarm(args.tts_url, f"test_cold_{int(time.time())}")
    
    # Final comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    if prewarm_metrics.get("first_audio_latency_ms") and cold_metrics.get("first_audio_latency_ms"):
        prewarm_latency = prewarm_metrics["first_audio_latency_ms"]
        cold_latency = cold_metrics["first_audio_latency_ms"]
        improvement = cold_latency - prewarm_latency
        improvement_pct = (improvement / cold_latency) * 100 if cold_latency > 0 else 0
        
        print(f"  Pre-warmed first audio: {prewarm_latency:.0f}ms")
        print(f"  Cold start first audio: {cold_latency:.0f}ms")
        print(f"  Improvement:            {improvement:.0f}ms ({improvement_pct:.1f}%)")
        
        if prewarm_latency < 150:
            print("\n  ⚡ ULTRA-LOW LATENCY TARGET MET (<150ms)")
        elif prewarm_latency < 300:
            print("\n  ✅ Good latency (<300ms)")
        else:
            print("\n  ⚠️ Latency above target (>300ms)")
    else:
        print("  Unable to compare - one or both tests failed")
    
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
