#!/usr/bin/env python3
"""
Run STT Local Service Locally with FastRTC

Quick test script to run the service locally and test FastRTC integration.
Transcripts will appear in the terminal in real-time.

Usage:
    python3 run_local_fastrtc.py
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging for terminal visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run the STT Local service locally"""
    import uvicorn
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Set environment variables for local testing
    os.environ.setdefault("STT_LOCAL_HOST", "0.0.0.0")
    os.environ.setdefault("STT_LOCAL_PORT", "8006")
    os.environ.setdefault("FAST_RTC_PORT", "7861")
    os.environ.setdefault("LEIBNIZ_STT_LOCAL_WHISPER_DEVICE", "cpu")  # Use CPU for local testing
    os.environ.setdefault("LEIBNIZ_STT_LOCAL_USE_GPU", "false")
    
    logger.info("=" * 70)
    logger.info("🚀 Starting STT Local Service (Local Mode)")
    logger.info("=" * 70)
    logger.info("📊 FastRTC UI will be available at: http://localhost:7861/fastrtc")
    logger.info("📊 API endpoint: http://localhost:8006")
    logger.info("📊 Health check: http://localhost:8006/health")
    logger.info("")
    logger.info("🎤 Instructions:")
    logger.info("   1. Open http://localhost:7861/fastrtc in your browser")
    logger.info("   2. Click 'Start' to begin audio streaming")
    logger.info("   3. Speak into your microphone")
    logger.info("   4. Watch terminal for real-time transcripts!")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8006,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        logger.info("\n🛑 Service stopped by user")
    except Exception as e:
        logger.error(f"❌ Service error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()



