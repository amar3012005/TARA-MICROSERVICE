import asyncio
import os
import logging
import json
from sarvamai import AsyncSarvamAI, AudioOutput, ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_sarvam")

API_KEY = os.getenv("SARVAM_API_KEY", "")

async def test_streaming():
    client = AsyncSarvamAI(api_subscription_key=API_KEY)
    
    logger.info("Connecting...")
    ws_context = client.text_to_speech_streaming.connect(
        model="bulbul:v2",
        send_completion_event=True
    )
    
    ws = await ws_context.__aenter__()
    logger.info("Connected.")
    
    import inspect
    sig = inspect.signature(ws.configure)
    logger.info(f"Signature of configure: {sig}")

        # Test 13: Telugu + Anushka + Linear16 + Convert
    logger.info("--- Test 13: Telugu + Anushka + Linear16 + Convert ---")
    
    try:
        await ws.configure(target_language_code="te-IN", speaker="anushka", output_audio_codec="linear16")
        logger.info("Configure sent.")
    except Exception as e:
        logger.error(f"Configure failed: {e}")

    logger.info("Sending text...")
    await ws.convert("Hello testing convert")
    await ws.flush()

    logger.info("Waiting for messages (Test 13)...")
    try:
        async for message in ws:
            if isinstance(message, ErrorResponse):
                logger.error(f"Received ErrorResponse: {message}")
                break
            elif isinstance(message, AudioOutput):
                logger.info(f"Received AudioOutput: {len(message.audio)} bytes")
                break
            else:
                logger.info(f"Received other message: {type(message)}")
    except Exception as e:
        logger.error(f"Error during iteration: {e}")

        
    # We need to reconnect for a clean test if the previous one failed or closed?
    # The SDK might not support multiple configures?
    # Let's just exit.
    await ws_context.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(test_streaming())
