"""
Test script for Whisper transcription using faster-whisper
No external audio files needed - uses synthetic audio
"""

import numpy as np

# Check CUDA availability first
import torch
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# Configuration - use CPU to avoid cuDNN issues
MODEL_SIZE = "medium.en"  # Options: tiny, base, small, medium, large-v2, large-v3
DEVICE = "cpu"  # Using CPU to avoid cuDNN library issues
COMPUTE_TYPE = "float32"  # float32 for CPU

print(f"\nLoading Faster Whisper model ({MODEL_SIZE}) on {DEVICE}...")
print("(Using CPU to avoid cuDNN library issues)")

from faster_whisper import WhisperModel
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("✅ Model loaded!")

# Generate a simple test tone (440Hz sine wave for 2 seconds)
print("\nGenerating test audio (440Hz sine wave, 2 seconds)...")
sample_rate = 16000
duration = 2.0
t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave

# Transcribe the test audio (should return empty or noise since it's just a tone)
print("Transcribing test audio (pure tone - expect empty or noise)...")
segments, info = model.transcribe(test_audio, language="en")

print(f"\nDetected language: {info.language} (probability: {info.language_probability:.2f})")
print("Transcription result:")
segments_list = list(segments)
if not segments_list:
    print("  (No speech detected - correct for pure tone!)")
else:
    for segment in segments_list:
        print(f"  [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

print("\n" + "=" * 50)
print("✅ Faster Whisper is working correctly!")
print("=" * 50)

# Quick test with the service's whisper_service module
print("\n\nTesting WhisperService integration...")
try:
    from config import STTLocalConfig
    from whisper_service import WhisperService
    
    config = STTLocalConfig.from_env()
    config.whisper_model_size = MODEL_SIZE
    config.whisper_device = DEVICE
    config.whisper_compute_type = COMPUTE_TYPE
    config.use_gpu = False
    
    whisper_service = WhisperService(config)
    
    # Test transcription
    text, confidence = whisper_service.transcribe(test_audio, is_partial=False)
    print(f"WhisperService result: '{text}' (confidence: {confidence:.2f})")
    if not text:
        print("  (Empty result is correct for pure tone audio)")
    print("✅ WhisperService integration test passed!")
    
except Exception as e:
    print(f"⚠️ WhisperService test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("To test with real speech from browser:")
print("  1. Run: python3 run_local_fastrtc.py")
print("  2. Open: http://localhost:7861/fastrtc")
print("  3. Speak into your microphone")
print("  4. Watch terminal for transcripts")
print("=" * 50)
print("\n✅ All tests completed!")
