"""
Integration test for STT-VAD + Intent + Orchestrator flow

Tests the basic conversation flow without RAG.
"""

import asyncio
import json
import logging
import websockets
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_service_health():
    """Test all services are healthy"""
    services = {
        "Redis": None,  # No HTTP endpoint
        "STT-VAD": "http://localhost:8009/health",
        "Intent": "http://localhost:8010/health",
        "Orchestrator": "http://localhost:8011/health"
    }
    
    async with aiohttp.ClientSession() as session:
        for name, url in services.items():
            if url:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            logger.info(f"✅ {name}: {data.get('status', 'healthy')}")
                        else:
                            logger.error(f"❌ {name}: HTTP {resp.status}")
                except Exception as e:
                    logger.error(f"❌ {name}: {e}")
            else:
                logger.info(f"⏭️  {name}: Skipped (no HTTP endpoint)")


async def test_orchestrator_websocket():
    """Test orchestrator WebSocket with Intent-only flow"""
    session_id = f"test_{int(asyncio.get_event_loop().time())}"
    uri = f"ws://localhost:8011/orchestrate?session_id={session_id}"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info(f"✅ Connected to orchestrator: {session_id}")
            
            # Receive connection message
            msg = await websocket.recv()
            data = json.loads(msg)
            logger.info(f"📨 Connection: {data}")
            assert data.get("type") == "connected", "Expected 'connected' message"
            
            # Send STT fragment
            await websocket.send(json.dumps({
                "type": "stt_fragment",
                "session_id": session_id,
                "text": "I want to schedule an appointment",
                "is_final": False,
                "timestamp": asyncio.get_event_loop().time()
            }))
            
            # Receive state update
            msg = await websocket.recv()
            data = json.loads(msg)
            logger.info(f"📨 State update: {data.get('state')}")
            
            # Send VAD end (end of turn)
            await websocket.send(json.dumps({
                "type": "vad_end",
                "session_id": session_id,
                "confidence": 0.95
            }))
            
            # Receive response ready
            msg = await websocket.recv()
            data = json.loads(msg)
            logger.info(f"📨 Response ready: {json.dumps(data, indent=2)}")
            
            assert data.get("type") == "response_ready", "Expected 'response_ready' message"
            assert "text" in data, "Response should contain text"
            assert "intent" in data, "Response should contain intent"
            
            logger.info(f"✅ Intent classified: {data.get('intent')}")
            logger.info(f"✅ Response text: {data.get('text', '')[:100]}...")
            logger.info(f"✅ Latency: {data.get('thinking_ms', 0):.0f}ms")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ WebSocket test failed: {e}", exc_info=True)
        return False


async def test_intent_service_directly():
    """Test Intent service directly"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "http://localhost:8010/api/v1/classify",
                json={"text": "I want to schedule an appointment"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ Intent service response: {json.dumps(data, indent=2)}")
                    return True
                else:
                    logger.error(f"❌ Intent service: HTTP {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Intent service error: {e}")
            return False


async def main():
    """Run all integration tests"""
    logger.info("=" * 70)
    logger.info("🧪 Testing STT-VAD + Intent + Orchestrator Integration")
    logger.info("=" * 70)
    
    # Test 1: Service health
    logger.info("\n1️⃣ Testing service health...")
    await test_service_health()
    
    # Test 2: Intent service directly
    logger.info("\n2️⃣ Testing Intent service directly...")
    intent_ok = await test_intent_service_directly()
    
    if not intent_ok:
        logger.error("❌ Intent service not responding. Is it running?")
        logger.error("   Start with: docker-compose up -d intent-service")
        return
    
    # Test 3: Orchestrator WebSocket
    logger.info("\n3️⃣ Testing Orchestrator WebSocket flow...")
    ws_ok = await test_orchestrator_websocket()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 Test Summary")
    logger.info("=" * 70)
    logger.info(f"Intent Service: {'✅ PASS' if intent_ok else '❌ FAIL'}")
    logger.info(f"Orchestrator WebSocket: {'✅ PASS' if ws_ok else '❌ FAIL'}")
    logger.info("=" * 70)
    
    if intent_ok and ws_ok:
        logger.info("\n🎉 Integration test passed! STT-VAD + Intent + Orchestrator is working.")
    else:
        logger.error("\n❌ Some tests failed. Check logs above.")


if __name__ == "__main__":
    asyncio.run(main())


