import asyncio
import json
import websockets
import redis.asyncio as redis
import time
import os
import aiohttp

async def test_streaming():
    session_id = f"test_session_{int(time.time())}"
    # Connect to localhost on HOST ports
    # Orchestrator: 2004
    uri = f"ws://localhost:2004/orchestrate?session_id={session_id}"
    
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # Wait for initial messages
            # We might receive 'connected', 'service_status'
            while True:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(msg)
                    print(f"Received: {data.get('type')}")
                    if data.get('type') == 'service_status':
                        print(f"Status: {data.get('status')}")
                        # Break after we see some status, assuming connection is established
                        break
                except asyncio.TimeoutError:
                    print("Timeout waiting for initial messages, proceeding...")
                    break
            
            # Simulate STT event via Redis
            # Redis: 2006
            redis_host = "localhost"
            redis_port = 2006
            print(f"Connecting to Redis at {redis_host}:{redis_port}...")
            r = redis.Redis(host=redis_host, port=redis_port, db=0)
            
            # Simulate Service Connections to start the workflow
            print("Simulating Service Connections...")
            await r.publish("leibniz:events:stt:connected", json.dumps({
                "session_id": "stt_session_1"
            }))
            await r.publish("leibniz:events:tts:connected", json.dumps({
                "session_id": "tts_session_1"
            }))
            
            # Wait a bit for state transition to LISTENING
            print("Waiting for workflow to start...")
            await asyncio.sleep(2)
            
            # Call /start endpoint
            print("Calling /start endpoint...")
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://localhost:2004/start?session_id={session_id}") as resp:
                    print(f"Start response: {resp.status}")
                    print(await resp.text())
            
            print("Waiting for intro to complete...")
            while True:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    if isinstance(msg, bytes):
                        # print(f"Received intro audio chunk: {len(msg)} bytes")
                        continue 
                    data = json.loads(msg)
                    print(f"Received: {data.get('type')}")
                    if data.get('type') == 'intro_complete':
                        print("Intro complete!")
                        break
                except asyncio.TimeoutError:
                    print("Timeout waiting for intro complete")
                    break
            
            unique_text = f"Testing streaming flow {int(time.time())}"
            print(f"Publishing STT event: {unique_text}")
            await r.publish("leibniz:events:stt", json.dumps({
                "text": unique_text,
                "session_id": "stt_session_1",
                "is_final": True
            }))
            # We need to ensure the orchestrator has subscribed.
            # It subscribes on startup, so it should be ready.
            
            await r.publish("leibniz:events:stt", json.dumps({
                "text": "Tell me a story about a brave knight.",
                "session_id": "stt_session_1",
                "is_final": True
            }))
            
            print("Waiting for response...")
            start_time = time.time()
            first_audio = False
            
            try:
                while True:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    if isinstance(msg, bytes):
                        if not first_audio:
                            print(f"✅ First audio received in {(time.time() - start_time)*1000:.0f}ms")
                            first_audio = True
                        # print(f"Received audio chunk: {len(msg)} bytes")
                    else:
                        data = json.loads(msg)
                        print(f"Received: {data.get('type')} - {data.get('text', '')[:50]}")
                        if data.get('type') == 'response_ready':
                            print(f"Response ready in {data.get('thinking_ms'):.0f}ms")
            except asyncio.TimeoutError:
                print("Timeout waiting for response (or stream ended)")
            except Exception as e:
                print(f"Error receiving: {e}")
                
            await r.close()
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_streaming())
