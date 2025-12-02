#!/usr/bin/env python3
"""
Simple WebSocket client to connect to the Orchestrator.

Usage:
    python test_orchestrator.py

Then in another terminal:
    curl -X POST http://localhost:8004/start
"""

import asyncio
import json
import websockets

async def connect_orchestrator():
    uri = "ws://localhost:8004/orchestrate?session_id=test-session-001"
    
    print("=" * 60)
    print("🔌 Connecting to Orchestrator...")
    print(f"   URI: {uri}")
    print("=" * 60)
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ Connected to Orchestrator!")
            print("")
            print("📋 Now run this in another terminal:")
            print("   curl -X POST http://localhost:8004/start")
            print("")
            print("🎧 Waiting for messages... (Ctrl+C to exit)")
            print("=" * 60)
            
            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=60.0)
                    
                    # Try to parse as JSON
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "unknown")
                        
                        if msg_type == "connected":
                            print(f"✅ Session connected: {data.get('session_id')}")
                            print(f"   State: {data.get('state')}")
                        
                        elif msg_type == "service_status":
                            print(f"📡 Service Status: {data.get('message')}")
                            print(f"   STT: {data.get('stt_ready')}, TTS: {data.get('tts_ready')}")
                        
                        elif msg_type == "intro_complete":
                            print(f"🎤 {data.get('message')}")
                            print(f"   State: {data.get('state')}")
                        
                        elif msg_type == "state_update":
                            print(f"📊 State: {data.get('state')}")
                            if data.get('text_buffer'):
                                print(f"   Buffer: {data.get('text_buffer')}")
                        
                        elif msg_type == "response_ready":
                            print(f"💬 Response: {data.get('text')[:100]}...")
                            print(f"   Thinking time: {data.get('thinking_ms'):.0f}ms")
                        
                        elif msg_type == "turn_complete":
                            print(f"✅ Turn {data.get('turn_number')} complete")
                            print(f"   State: {data.get('state')}")
                        
                        elif msg_type == "tts_error":
                            print(f"❌ TTS Error: {data.get('message')}")
                        
                        else:
                            print(f"📨 {msg_type}: {json.dumps(data, indent=2)[:200]}")
                    
                    except json.JSONDecodeError:
                        # Binary audio data
                        print(f"🔊 Received audio chunk: {len(message)} bytes")
                
                except asyncio.TimeoutError:
                    print("⏳ Still waiting for messages...")
                    continue
                    
    except websockets.exceptions.ConnectionClosed as e:
        print(f"🔌 Connection closed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(connect_orchestrator())
    except KeyboardInterrupt:
        print("\n👋 Disconnected")

