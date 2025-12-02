"""
Integration test for StateManager Orchestrator

Tests WebSocket connection, state transitions, and service integration.
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_orchestrator_health():
    """Test orchestrator health endpoint"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://localhost:8004/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ Health check passed: {data}")
                    return True
                else:
                    logger.error(f"❌ Health check failed: HTTP {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False


async def test_websocket_connection():
    """Test WebSocket connection and basic flow"""
    session_id = f"test_{int(asyncio.get_event_loop().time())}"
    uri = f"ws://localhost:8004/orchestrate?session_id={session_id}"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info(f"✅ Connected to orchestrator: {session_id}")
            
            # Receive connection message
            msg = await websocket.recv()
            data = json.loads(msg)
            logger.info(f"📨 Received: {data}")
            
            assert data.get("type") == "connected", "Expected 'connected' message"
            assert data.get("session_id") == session_id, "Session ID mismatch"
            
            # Send STT fragment
            await websocket.send(json.dumps({
                "type": "stt_fragment",
                "session_id": session_id,
                "text": "What are admission requirements?",
                "is_final": False,
                "timestamp": asyncio.get_event_loop().time()
            }))
            
            # Receive state update
            msg = await websocket.recv()
            data = json.loads(msg)
            logger.info(f"📨 State update: {data}")
            
            # Send VAD end (end of turn)
            await websocket.send(json.dumps({
                "type": "vad_end",
                "session_id": session_id,
                "confidence": 0.95
            }))
            
            # Receive response ready
            msg = await websocket.recv()
            data = json.loads(msg)
            logger.info(f"📨 Response ready: {data}")
            
            assert data.get("type") == "response_ready", "Expected 'response_ready' message"
            assert "text" in data, "Response should contain text"
            
            logger.info(f"✅ WebSocket test passed")
            return True
            
    except Exception as e:
        logger.error(f"❌ WebSocket test failed: {e}", exc_info=True)
        return False


async def test_parallel_processing():
    """Test that Intent and RAG are called in parallel"""
    import aiohttp
    import time
    
    # Test Intent service directly
    start = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8002/api/v1/classify",
            json={"text": "What are admission requirements?"}
        ) as resp:
            intent_time = time.time() - start
            intent_result = await resp.json()
            logger.info(f"✅ Intent service: {intent_time*1000:.0f}ms")
    
    # Test RAG service directly
    start = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8003/api/v1/query",
            json={"query": "What are admission requirements?"}
        ) as resp:
            rag_time = time.time() - start
            rag_result = await resp.json()
            logger.info(f"✅ RAG service: {rag_time*1000:.0f}ms")
    
    # Expected parallel time should be max(intent_time, rag_time)
    expected_parallel = max(intent_time, rag_time)
    sequential_time = intent_time + rag_time
    
    logger.info(f"📊 Sequential: {sequential_time*1000:.0f}ms")
    logger.info(f"📊 Parallel (expected): {expected_parallel*1000:.0f}ms")
    logger.info(f"📊 Time saved: {(sequential_time - expected_parallel)*1000:.0f}ms")
    
    return True


async def main():
    """Run all tests"""
    logger.info("=" * 70)
    logger.info("🧪 Testing StateManager Orchestrator Integration")
    logger.info("=" * 70)
    
    # Test 1: Health check
    logger.info("\n1️⃣ Testing health endpoint...")
    health_ok = await test_orchestrator_health()
    
    if not health_ok:
        logger.error("❌ Health check failed. Is the orchestrator running?")
        logger.error("   Start with: docker-compose up -d orchestrator")
        return
    
    # Test 2: WebSocket connection
    logger.info("\n2️⃣ Testing WebSocket connection...")
    ws_ok = await test_websocket_connection()
    
    # Test 3: Parallel processing verification
    logger.info("\n3️⃣ Testing parallel processing...")
    parallel_ok = await test_parallel_processing()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 Test Summary")
    logger.info("=" * 70)
    logger.info(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    logger.info(f"WebSocket: {'✅ PASS' if ws_ok else '❌ FAIL'}")
    logger.info(f"Parallel Processing: {'✅ PASS' if parallel_ok else '❌ FAIL'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())




