import asyncio
import websockets
import json
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ORCHESTRATOR_URL = "ws://localhost:5204/orchestrate"
SESSION_ID = f"test_session_{int(time.time())}"

async def test_intro():
    uri = f"{ORCHESTRATOR_URL}?session_id={SESSION_ID}"
    logger.info(f"🔌 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected!")
            
            # Wait for initial messages
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    logger.info(f"📩 Received: {msg_type}")
                    logger.debug(json.dumps(data, indent=2))
                    
                    if msg_type == "connected":
                        logger.info(f"   Session: {data.get('session_id')}")
                        logger.info(f"   State: {data.get('state')}")
                        
                    elif msg_type == "service_status":
                        logger.info(f"   Status: {data.get('status')}")
                        
                    elif msg_type == "intro_greeting":
                        text = data.get("text", "")
                        logger.info("🎉 INTRO RECEIVED!")
                        logger.info(f"   Text: {text}")
                        
                        # Verify Telugu greeting
                        if "నమస్కారం" in text or "TARA" in text:
                            logger.info("✅ TARA Telugu greeting verified!")
                            return True
                        else:
                            logger.error(f"❌ Unexpected greeting: {text}")
                            return False
                            
                except asyncio.TimeoutError:
                    logger.warning("⏱️ Timeout waiting for intro")
                    break
                    
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    # Install websockets if missing
    # pip install websockets
    try:
        asyncio.run(test_intro())
    except KeyboardInterrupt:
        pass





