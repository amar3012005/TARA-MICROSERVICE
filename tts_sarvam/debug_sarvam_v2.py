import asyncio
import os
import logging
import json
from sarvamai import AsyncSarvamAI, AudioOutput, ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_sarvam")

API_KEY = os.getenv("SARVAM_API_KEY", "sk_2d8w6udi_gaPItmPcCEsf3CoON7RBzqPr")

async def test_streaming():
    client = AsyncSarvamAI(api_subscription_key=API_KEY)
    
    logger.info("Connecting...")
    ws_context = client.text_to_speech_streaming.connect(
        model="bulbul:v2",
        send_completion_event=True
    )
    
    ws = await ws_context.__aenter__()
    logger.info("Connected.")
    
    # Test 17: Config with min_buffer_size=50 + Flush
    logger.info("--- Test 17: Config with min_buffer_size=50 + Flush ---")
    
    try:
        await ws.configure(
            target_language_code="te-IN", 
            speaker="anushka", 
            output_audio_codec="linear16",
            min_buffer_size=50,
            max_chunk_length=150
        )
        logger.info("Configure sent.")
    except Exception as e:
        logger.error(f"Configure failed: {e}")

    logger.info("Sending short text...")
    await ws.convert("Hello")
    await ws.flush()
    
    logger.info("Waiting for messages (Test 17)...")
    try:
        async for message in ws:
            if isinstance(message, ErrorResponse):
                logger.error(f"Received ErrorResponse: {message}")
                break
            elif isinstance(message, AudioOutput):
                logger.info(f"Received AudioOutput: {len(message.data.audio)} bytes (base64)")
                break
            else:
                logger.info(f"Received other message: {type(message)}")
    except Exception as e:
        logger.error(f"Error during iteration: {e}")
        
    await ws_context.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(test_streaming())
