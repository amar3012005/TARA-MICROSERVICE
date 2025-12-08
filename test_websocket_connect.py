#!/usr/bin/env python3
"""
Simple WebSocket client to connect to TARA Orchestrator
Creates an active session so /start endpoint works
"""
import asyncio
import websockets
import json
import sys

ORCHESTRATOR_URL = "ws://localhost:5204/orchestrate"
SESSION_ID = f"test_session_{int(asyncio.get_event_loop().time())}"

async def connect_and_wait():
    uri = f"{ORCHESTRATOR_URL}?session_id={SESSION_ID}"
    print(f"🔌 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected! Session ID: {SESSION_ID}")
            print("📡 Waiting for messages (Ctrl+C to exit)...")
            
            # Send a message to keep connection alive
            await websocket.send(json.dumps({"type": "ping"}))
            
            try:
                while True:
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "unknown")
                        print(f"📩 Received: {msg_type}")
                        if msg_type == "connected":
                            print(f"   ✅ Session: {data.get('session_id')}")
                            print(f"   📊 State: {data.get('state')}")
                        elif msg_type == "intro_greeting":
                            print(f"   🎤 Intro: {data.get('text', '')[:100]}...")
                        elif msg_type == "state_update":
                            print(f"   🔄 State: {data.get('state')}")
                        else:
                            print(f"   📄 Data: {json.dumps(data, indent=2)[:200]}")
                    except json.JSONDecodeError:
                        print(f"📦 Binary data received ({len(message)} bytes)")
            except asyncio.TimeoutError:
                print("⏱️ No messages received for 30s, keeping connection alive...")
                print(f"✅ WebSocket session active! You can now:")
                print(f"   curl -X POST http://localhost:5204/start")
                print(f"\nPress Ctrl+C to disconnect...")
                # Keep connection alive
                await asyncio.sleep(3600)  # Wait 1 hour
    except KeyboardInterrupt:
        print("\n👋 Disconnecting...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 70)
    print("🎻 TARA Orchestrator WebSocket Test Client")
    print("=" * 70)
    print(f"Session ID: {SESSION_ID}")
    print("=" * 70)
    
    try:
        asyncio.run(connect_and_wait())
    except KeyboardInterrupt:
        print("\n✅ Test complete")





