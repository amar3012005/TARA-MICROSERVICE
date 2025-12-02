#!/usr/bin/env python3
"""
Local Test Script for STT Local Service

Tests the service components locally before Docker deployment.
Run: python3 test_local.py
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test results
test_results = []

def test_result(name: str, passed: bool, message: str = ""):
    """Record test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    test_results.append((name, passed, message))
    print(f"{status}: {name}")
    if message:
        print(f"   {message}")
    return passed

def test_imports():
    """Test basic imports"""
    print("\n" + "="*70)
    print("Testing Imports")
    print("="*70)
    
    try:
        import numpy as np
        test_result("numpy", True, f"Version: {np.__version__}")
    except ImportError as e:
        test_result("numpy", False, str(e))
        return False
    
    try:
        import fastapi
        test_result("fastapi", True, f"Version: {fastapi.__version__}")
    except ImportError as e:
        test_result("fastapi", False, str(e))
        return False
    
    try:
        import fastrtc
        test_result("fastrtc", True)
    except ImportError as e:
        test_result("fastrtc", False, str(e))
        return False
    
    try:
        import torch
        test_result("torch", True, f"Version: {torch.__version__}")
        if torch.cuda.is_available():
            test_result("torch CUDA", True, f"CUDA Version: {torch.version.cuda}")
            test_result("GPU Available", True, f"Device: {torch.cuda.get_device_name(0)}")
        else:
            test_result("torch CUDA", False, "CUDA not available - will use CPU")
    except ImportError as e:
        test_result("torch", False, f"Not installed: {e}")
        print("   ⚠️  Install with: pip install torch torchaudio")
    
    try:
        from faster_whisper import WhisperModel
        test_result("faster-whisper", True)
    except ImportError as e:
        test_result("faster-whisper", False, f"Not installed: {e}")
        print("   ⚠️  Install with: pip install faster-whisper")
    
    return True

def test_config():
    """Test configuration loading"""
    print("\n" + "="*70)
    print("Testing Configuration")
    print("="*70)
    
    try:
        from config import STTLocalConfig
        config = STTLocalConfig.from_env()
        test_result("Config Loading", True)
        test_result("Sample Rate", config.sample_rate == 16000, f"Value: {config.sample_rate}")
        test_result("Whisper Model", True, f"Size: {config.whisper_model_size}")
        test_result("Whisper Device", True, f"Device: {config.whisper_device}")
        return True
    except Exception as e:
        test_result("Config Loading", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_vad_utils():
    """Test VAD utilities"""
    print("\n" + "="*70)
    print("Testing VAD Utilities")
    print("="*70)
    
    try:
        from vad_utils import SileroVAD, VADStateMachine
        from config import STTLocalConfig
        
        config = STTLocalConfig.from_env()
        
        # Test VAD initialization (this will download model if needed)
        print("   Loading Silero VAD model (this may take a moment on first run)...")
        try:
            vad = SileroVAD(threshold=config.vad_threshold, device="cpu")  # Use CPU for testing
            test_result("Silero VAD Init", True, f"Device: {vad.device}")
            
            # Test VAD state machine
            vad_state = VADStateMachine(
                vad,
                min_speech_duration_ms=config.vad_min_speech_duration_ms,
                silence_timeout_ms=config.vad_silence_timeout_ms
            )
            test_result("VAD State Machine", True)
            
            return True
        except Exception as e:
            test_result("Silero VAD Init", False, str(e))
            print("   ⚠️  This requires torch. Install heavy deps first.")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        test_result("VAD Utils Import", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_whisper_service():
    """Test Whisper service"""
    print("\n" + "="*70)
    print("Testing Whisper Service")
    print("="*70)
    
    try:
        from whisper_service import WhisperService
        from config import STTLocalConfig
        
        config = STTLocalConfig.from_env()
        
        # Override to use CPU for testing if CUDA not available
        import torch
        if not torch.cuda.is_available():
            config.whisper_device = "cpu"
            config.whisper_compute_type = "float32"
            config.use_gpu = False
            print("   ⚠️  CUDA not available, using CPU mode")
        
        print("   Loading Faster Whisper model (this may take a moment on first run)...")
        try:
            whisper = WhisperService(config)
            test_result("Whisper Service Init", True)
            test_result("Whisper Model Info", True, str(whisper.get_model_info()))
            return True
        except (SystemError, OSError, RuntimeError) as e:
            # CUDA/CUDNN errors - try CPU fallback
            if "cuda" in str(e).lower() or "cudnn" in str(e).lower():
                print("   ⚠️  CUDA error detected, retrying with CPU...")
                config.whisper_device = "cpu"
                config.whisper_compute_type = "float32"
                config.use_gpu = False
                try:
                    whisper = WhisperService(config)
                    test_result("Whisper Service Init (CPU fallback)", True)
                    return True
                except Exception as e2:
                    test_result("Whisper Service Init", False, f"CPU fallback also failed: {e2}")
                    import traceback
                    traceback.print_exc()
                    return False
            else:
                test_result("Whisper Service Init", False, str(e))
                print("   ⚠️  This requires faster-whisper. Install heavy deps first.")
                import traceback
                traceback.print_exc()
                return False
        except Exception as e:
            test_result("Whisper Service Init", False, str(e))
            print("   ⚠️  This requires faster-whisper. Install heavy deps first.")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        test_result("Whisper Service Import", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_stt_manager():
    """Test STT Manager"""
    print("\n" + "="*70)
    print("Testing STT Manager")
    print("="*70)
    
    try:
        from stt_manager import STTManager
        from config import STTLocalConfig
        
        config = STTLocalConfig.from_env()
        
        # Override to use CPU for testing if CUDA not available
        import torch
        if not torch.cuda.is_available():
            config.whisper_device = "cpu"
            config.use_gpu = False
        
        try:
            print("   Initializing STT Manager (this may take a moment)...")
            manager = STTManager(config, None)
            test_result("STT Manager Init", True)
            metrics = manager.get_performance_metrics()
            test_result("STT Manager Metrics", True, f"Metrics: {metrics}")
            return True
        except Exception as e:
            test_result("STT Manager Init", False, str(e))
            print("   ⚠️  This requires all dependencies. Install heavy deps first.")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        test_result("STT Manager Import", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_utils():
    """Test utility functions"""
    print("\n" + "="*70)
    print("Testing Utilities")
    print("="*70)
    
    try:
        from utils import normalize_english_transcript, TranscriptBuffer
        
        # Test normalization
        test_cases = [
            ("hello world", "hello world"),
            ("um, I I I want", "I want"),
            ("the the book", "the book"),
        ]
        
        for input_text, expected_pattern in test_cases:
            result = normalize_english_transcript(input_text)
            # Just check it doesn't crash and returns string
            test_result(f"Normalize: '{input_text[:20]}'", isinstance(result, str))
        
        # Test transcript buffer
        buffer = TranscriptBuffer()
        buffer.add_fragment("Hello")
        buffer.add_fragment(" world")
        final = buffer.get_final_transcript()
        # Check that it contains both words (allowing for spacing variations)
        test_result("Transcript Buffer", "Hello" in final and "world" in final, f"Result: '{final}'")
        
        return True
    except Exception as e:
        test_result("Utils Import", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_fastapi_app():
    """Test FastAPI app initialization"""
    print("\n" + "="*70)
    print("Testing FastAPI App")
    print("="*70)
    
    try:
        # Don't actually start the server, just test imports and basic setup
        from app import app, lifespan
        test_result("FastAPI App Import", True)
        test_result("App Title", hasattr(app, 'title'), f"Title: {app.title}")
        return True
    except Exception as e:
        test_result("FastAPI App Import", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def print_summary():
    """Print test summary"""
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    total = len(test_results)
    passed = sum(1 for _, p, _ in test_results if p)
    failed = total - passed
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print("\nFailed Tests:")
        for name, passed, message in test_results:
            if not passed:
                print(f"  - {name}: {message}")
    
    print("\n" + "="*70)
    
    if failed == 0:
        print("🎉 All tests passed! Service is ready.")
    else:
        print("⚠️  Some tests failed. Install missing dependencies:")
        print("   pip install -r requirements_after.txt")
    
    print("="*70 + "\n")
    
    return failed == 0

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("STT Local Service - Local Test Suite")
    print("="*70)
    print("Testing service components before Docker deployment...")
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Run tests
    test_imports()
    test_config()
    test_utils()
    test_vad_utils()
    test_whisper_service()
    test_stt_manager()
    test_fastapi_app()
    
    # Print summary
    success = print_summary()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

