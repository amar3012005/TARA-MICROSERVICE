"""
Test script for FastRTC STT/VAD integration

Tests the basic audio streaming from FastRTC to Docker service.
Run this to verify the integration works before full deployment.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_audio_handler(audio: Tuple[np.ndarray, int]):
    """Mock audio handler for testing FastRTC stream."""
    logger.info(f"Received audio chunk: shape={audio[0].shape}, dtype={audio[0].dtype}, rate={audio[1]}")

    # Simulate processing delay
    await asyncio.sleep(0.01)

    # Log some audio stats
    audio_array = audio[0]
    if len(audio_array) > 0:
        rms = np.sqrt(np.mean(audio_array**2))
        logger.info(f"Audio stats: RMS={rms:.4f}, max={np.max(np.abs(audio_array)):.4f}")

def test_fastrtc_import():
    """Test that FastRTC components can be imported."""
    try:
        from fastrtc import Stream
        logger.info("✓ FastRTC Stream import successful")
        return True
    except ImportError as e:
        logger.error(f"✗ FastRTC import failed: {e}")
        return False

def test_numpy_audio():
    """Test numpy audio array creation and conversion."""
    try:
        # Create test audio (1 second of 16kHz silence)
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)

        # Create float32 audio (-1 to 1 range)
        audio_float = np.random.normal(0, 0.1, samples).astype(np.float32)

        # Convert to int16 as the handler expects
        audio_int16 = (audio_float * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        logger.info(f"✓ Audio conversion successful: {len(audio_bytes)} bytes")

        # Test the audio handler (fix asyncio issue)
        # Create a new event loop for this test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_audio_handler((audio_float, sample_rate)))
        finally:
            loop.close()

        return True
    except Exception as e:
        logger.error(f"✗ Audio processing failed: {e}")
        return False

def test_websocket_import():
    """Test WebSocket imports for Docker communication."""
    try:
        import websockets
        import json
        logger.info("✓ WebSocket imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ WebSocket import failed: {e}")
        return False

async def test_docker_connection():
    """Test connection to Docker STT/VAD service."""
    try:
        import websockets
        import os

        host = os.getenv("STT_VAD_HOST", "localhost")
        port = int(os.getenv("STT_VAD_PORT", "8001"))
        url = f"ws://{host}:{port}/api/v1/transcribe/stream"

        logger.info(f"Testing connection to {url}")

        # Try to connect with timeout
        try:
            async with websockets.connect(url) as ws:
                logger.info("✓ Docker WebSocket connection successful")

                # Send a test message
                test_msg = {"type": "ping", "session_id": "test_session", "timestamp": time.time()}
                await ws.send_json(test_msg)
                logger.info("✓ Test message sent")

                # Try to receive response
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    logger.info(f"✓ Response received: {response[:100]}...")
                except asyncio.TimeoutError:
                    logger.warning("⚠ No response received (service may not be running)")

                return True

        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"⚠ Connection closed: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Connection failed: {e}")
            return False

    except ImportError:
        logger.error("✗ WebSocket library not available")
        return False

async def run_all_tests():
    """Run all integration tests."""
    logger.info("🚀 Starting FastRTC STT/VAD Integration Tests")
    logger.info("=" * 50)

    results = []

    # Test 1: FastRTC imports
    logger.info("Test 1: FastRTC Imports")
    results.append(("FastRTC Imports", test_fastrtc_import()))

    # Test 2: Audio processing
    logger.info("\nTest 2: Audio Processing")
    results.append(("Audio Processing", test_numpy_audio()))

    # Test 3: WebSocket imports
    logger.info("\nTest 3: WebSocket Imports")
    results.append(("WebSocket Imports", test_websocket_import()))

    # Test 4: Docker connection
    logger.info("\nTest 4: Docker Service Connection")
    results.append(("Docker Connection", await test_docker_connection()))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS SUMMARY")

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status} {test_name}")
        if success:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Ready for integration.")
        return True
    else:
        logger.warning("⚠️ Some tests failed. Check Docker service and dependencies.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)