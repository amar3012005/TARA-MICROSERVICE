#!/usr/bin/env python3
"""
Test script for German TTS streaming with prewarm and first audio chunk measurement.

Tests the WebSocket streaming endpoint with a large German phrase and measures
the time to first audio chunk with and without pre-warming.
"""

import asyncio
import json
import time
import websockets
import base64

TTS_LABS_WS_URL = "ws://localhost:2007/api/v1/stream"

# Large German phrase for testing
GERMAN_TEXT = """Guten Tag! Ich hoffe, dass Sie einen wunderbaren Tag haben. 
Mein Name ist TARA, und ich bin ein fortgeschrittener KI-Assistent, der speziell 
dafür entwickelt wurde, Ihnen bei verschiedenen Aufgaben zu helfen. Ich kann 
Informationen bereitstellen, Fragen beantworten, Texte analysieren und vieles mehr. 
Heute möchte ich Ihnen zeigen, wie effizient und schnell die Sprachsynthese-Technologie 
funktioniert, insbesondere wenn wir die Vorwärmung der Verbindung nutzen. 
Dies ermöglicht es uns, Audio nahezu in Echtzeit zu generieren, was für eine 
natürliche und flüssige Konversation von entscheidender Bedeutung ist. 
Die Technologie hinter dieser Implementierung nutzt modernste WebSocket-Verbindungen 
und optimierte Streaming-Protokolle, um die Latenz zu minimieren und die Qualität 
der generierten Sprache zu maximieren."""


async def test_german_streaming_with_prewarm():
    """Test German streaming with pre-warming enabled."""
    print("=" * 70)
    print("🇩🇪 GERMAN STREAMING TEST - WITH PREWARM")
    print("=" * 70)
    
    session_id = f"german_prewarm_{int(time.time())}"
    url = f"{TTS_LABS_WS_URL}?session_id={session_id}"
    
    async with websockets.connect(url) as ws:
        # Step 1: Connection confirmation
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"[1] ✅ Connected: {data.get('type')}")
        
        # Step 2: Prewarm connection
        print(f"[2] 🔥 Sending prewarm request...")
        prewarm_start = time.time()
        await ws.send(json.dumps({"type": "prewarm"}))
        
        msg = await ws.recv()
        data = json.loads(msg)
        prewarm_duration = (time.time() - prewarm_start) * 1000
        print(f"[3] ✅ Prewarm complete: {data.get('status')}")
        print(f"    ⏱️  Prewarm duration: {prewarm_duration:.0f}ms")
        
        # Step 3: Simulate user speaking time (2 seconds)
        print(f"[4] ⏳ Simulating 2 seconds of user speaking time...")
        await asyncio.sleep(2.0)
        
        # Step 4: Send German text chunk
        print(f"[5] 📤 Sending German text chunk ({len(GERMAN_TEXT)} chars)...")
        print(f"    Text preview: '{GERMAN_TEXT[:80]}...'")
        
        chunk_send_time = time.time()
        await ws.send(json.dumps({"type": "stream_chunk", "text": GERMAN_TEXT}))
        
        # Step 5: Send stream_end to trigger generation
        await ws.send(json.dumps({"type": "stream_end"}))
        print(f"[6] ✅ Sent stream_end to trigger audio generation")
        
        # Step 6: Measure time to first audio
        first_audio_time = None
        audio_chunks = 0
        total_audio_bytes = 0
        
        print(f"[7] 🎧 Waiting for audio chunks...")
        
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=20.0)
                data = json.loads(msg)
                
                if data.get("type") == "audio":
                    if first_audio_time is None:
                        first_audio_time = time.time()
                        latency_ms = (first_audio_time - chunk_send_time) * 1000
                        print(f"\n{'='*70}")
                        print(f"🎯 FIRST AUDIO CHUNK RECEIVED!")
                        print(f"{'='*70}")
                        print(f"    ⚡ Time to First Audio: {latency_ms:.0f}ms")
                        
                        if latency_ms < 150:
                            print(f"    ✅ EXCELLENT: Ultra-low latency (<150ms)")
                        elif latency_ms < 200:
                            print(f"    ✅ GOOD: Low latency (<200ms)")
                        elif latency_ms < 300:
                            print(f"    ⚠️  ACCEPTABLE: Moderate latency (<300ms)")
                        else:
                            print(f"    ❌ SLOW: High latency (>300ms)")
                    
                    # Decode audio to get size
                    audio_b64 = data.get("data", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        total_audio_bytes += len(audio_bytes)
                    
                    audio_chunks += 1
                    
                    if data.get("is_final"):
                        print(f"\n[8] ✅ Stream complete!")
                        print(f"    📊 Total audio chunks: {audio_chunks}")
                        print(f"    📦 Total audio size: {total_audio_bytes / 1024:.1f} KB")
                        break
                
                elif data.get("type") == "stream_complete":
                    print(f"\n[8] ✅ Stream complete signal received!")
                    print(f"    📊 Total audio chunks: {audio_chunks}")
                    print(f"    📦 Total audio size: {total_audio_bytes / 1024:.1f} KB")
                    break
                    
            except asyncio.TimeoutError:
                print(f"\n[!] ⚠️  Timeout waiting for audio")
                break
        
        if first_audio_time:
            return (first_audio_time - chunk_send_time) * 1000
        return None


async def test_german_streaming_cold():
    """Test German streaming without pre-warming (cold start)."""
    print("\n" + "=" * 70)
    print("🇩🇪 GERMAN STREAMING TEST - COLD START")
    print("=" * 70)
    
    session_id = f"german_cold_{int(time.time())}"
    url = f"{TTS_LABS_WS_URL}?session_id={session_id}"
    
    async with websockets.connect(url) as ws:
        # Step 1: Connection confirmation
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"[1] ✅ Connected: {data.get('type')}")
        
        # Step 2: Send German text chunk (cold start)
        print(f"[2] 📤 Sending German text chunk ({len(GERMAN_TEXT)} chars)...")
        print(f"    Text preview: '{GERMAN_TEXT[:80]}...'")
        
        chunk_send_time = time.time()
        await ws.send(json.dumps({"type": "stream_chunk", "text": GERMAN_TEXT}))
        
        # Step 3: Send stream_end
        await ws.send(json.dumps({"type": "stream_end"}))
        print(f"[3] ✅ Sent stream_end")
        
        # Step 4: Measure time to first audio
        first_audio_time = None
        audio_chunks = 0
        total_audio_bytes = 0
        
        print(f"[4] 🎧 Waiting for audio chunks...")
        
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=20.0)
                data = json.loads(msg)
                
                if data.get("type") == "audio":
                    if first_audio_time is None:
                        first_audio_time = time.time()
                        latency_ms = (first_audio_time - chunk_send_time) * 1000
                        print(f"\n{'='*70}")
                        print(f"🎯 FIRST AUDIO CHUNK RECEIVED!")
                        print(f"{'='*70}")
                        print(f"    ⏱️  Time to First Audio (cold): {latency_ms:.0f}ms")
                    
                    audio_b64 = data.get("data", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        total_audio_bytes += len(audio_bytes)
                    
                    audio_chunks += 1
                    
                    if data.get("is_final"):
                        print(f"\n[5] ✅ Stream complete!")
                        print(f"    📊 Total audio chunks: {audio_chunks}")
                        print(f"    📦 Total audio size: {total_audio_bytes / 1024:.1f} KB")
                        break
                
                elif data.get("type") == "stream_complete":
                    print(f"\n[5] ✅ Stream complete signal received!")
                    print(f"    📊 Total audio chunks: {audio_chunks}")
                    print(f"    📦 Total audio size: {total_audio_bytes / 1024:.1f} KB")
                    break
                    
            except asyncio.TimeoutError:
                print(f"\n[!] ⚠️  Timeout waiting for audio")
                break
        
        if first_audio_time:
            return (first_audio_time - chunk_send_time) * 1000
        return None


async def main():
    """Run both tests and compare results."""
    print("\n" + "🚀 " * 35)
    print("GERMAN TTS STREAMING PERFORMANCE TEST")
    print("🚀 " * 35 + "\n")
    
    # Test with prewarm
    warm_latency = await test_german_streaming_with_prewarm()
    
    # Brief pause between tests
    print("\n" + "-" * 70)
    await asyncio.sleep(2.0)
    
    # Test without prewarm
    cold_latency = await test_german_streaming_cold()
    
    # Comparison
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 70)
    
    if cold_latency and warm_latency:
        improvement = cold_latency - warm_latency
        improvement_pct = (improvement / cold_latency) * 100
        
        print(f"❄️  Cold Start Latency:  {cold_latency:.0f}ms")
        print(f"🔥 Pre-warmed Latency:  {warm_latency:.0f}ms")
        print(f"⚡ Improvement:         {improvement:.0f}ms ({improvement_pct:.1f}% faster)")
        
        if warm_latency < 200:
            print(f"\n✅ SUCCESS: Pre-warmed latency is under 200ms target!")
        else:
            print(f"\n⚠️  Pre-warmed latency exceeds 200ms target")
    else:
        print("⚠️  Could not complete comparison (one or both tests failed)")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
