#!/usr/bin/env python3
"""
Real-time Whisper Transcription Test Script

Uses whisper-tiny for fast, real-time transcription from microphone.
Shows transcript fragments in the terminal as you speak.

Based on: https://huggingface.co/openai/whisper-large-v3

Usage:
    python3 test_whisper_realtime.py
"""

import sys
import time
import threading
import queue
import numpy as np

# Check dependencies
try:
    import sounddevice as sd
except ImportError:
    print("❌ sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("❌ faster-whisper not installed. Run: pip install faster-whisper")
    sys.exit(1)

# Configuration
SAMPLE_RATE = 16000  # Whisper requires 16kHz
CHUNK_DURATION = 0.5  # Seconds per chunk
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
MODEL_SIZE = "tiny"  # tiny, base, small, medium, large-v2, large-v3
DEVICE = "cpu"  # cpu or cuda
COMPUTE_TYPE = "float32"  # float32 for CPU, float16 for GPU

# Audio buffer
audio_queue = queue.Queue()
is_recording = True


def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio chunk."""
    if status:
        print(f"⚠️ Audio status: {status}")
    # Convert to float32 and add to queue
    audio_queue.put(indata.copy().flatten())


def transcribe_worker(model):
    """Worker thread that transcribes audio from the queue."""
    global is_recording
    
    audio_buffer = np.array([], dtype=np.float32)
    min_audio_length = SAMPLE_RATE * 1.0  # Minimum 1 second for transcription
    
    print("\n" + "=" * 70)
    print("🎤 LISTENING... Speak into your microphone!")
    print("=" * 70)
    print("   Press Ctrl+C to stop\n")
    
    while is_recording:
        try:
            # Get audio from queue (non-blocking with timeout)
            try:
                chunk = audio_queue.get(timeout=0.1)
                audio_buffer = np.concatenate([audio_buffer, chunk])
            except queue.Empty:
                continue
            
            # Check if we have enough audio
            if len(audio_buffer) < min_audio_length:
                continue
            
            # Calculate RMS for audio level display
            rms = np.sqrt(np.mean(audio_buffer ** 2))
            
            # Skip if too quiet (likely silence)
            if rms < 0.01:
                # Keep last 0.5 seconds for context
                audio_buffer = audio_buffer[-CHUNK_SIZE:]
                continue
            
            # Transcribe
            start_time = time.time()
            
            segments, info = model.transcribe(
                audio_buffer,
                language="en",
                beam_size=1,  # Greedy for speed
                best_of=1,
                vad_filter=True,  # Use Whisper's built-in VAD
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
            )
            
            # Process segments
            for segment in segments:
                text = segment.text.strip()
                if text:
                    inference_time = time.time() - start_time
                    no_speech = getattr(segment, 'no_speech_prob', 0.0)
                    
                    # Skip if high no-speech probability
                    if no_speech > 0.8:
                        continue
                    
                    # Print transcript fragment
                    print(f"\n{'=' * 70}")
                    print(f"📝 TRANSCRIPT: {text}")
                    print(f"   ⏱️  Latency: {inference_time:.2f}s | RMS: {rms:.4f} | NoSpeech: {no_speech:.2f}")
                    print(f"{'=' * 70}")
            
            # Keep last 0.5 seconds for context (overlap)
            audio_buffer = audio_buffer[-CHUNK_SIZE:]
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            import traceback
            traceback.print_exc()


def main():
    global is_recording
    
    print("\n" + "=" * 70)
    print("🚀 WHISPER REAL-TIME TRANSCRIPTION TEST")
    print("=" * 70)
    print(f"   Model: whisper-{MODEL_SIZE}")
    print(f"   Device: {DEVICE}")
    print(f"   Sample Rate: {SAMPLE_RATE}Hz")
    print("=" * 70)
    
    # Load model
    print(f"\n📥 Loading Whisper {MODEL_SIZE} model...")
    start = time.time()
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    print(f"✅ Model loaded in {time.time() - start:.1f}s")
    
    # Warmup
    print("🔥 Warming up model...")
    dummy_audio = np.zeros((SAMPLE_RATE,), dtype=np.float32)
    list(model.transcribe(dummy_audio, language="en", beam_size=1))
    print("✅ Warmup complete")
    
    # Start transcription worker thread
    worker_thread = threading.Thread(target=transcribe_worker, args=(model,))
    worker_thread.daemon = True
    worker_thread.start()
    
    # Start audio stream
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
        ):
            print("✅ Microphone stream started")
            
            # Wait for Ctrl+C
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")
        is_recording = False
        worker_thread.join(timeout=2.0)
        print("✅ Done!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        is_recording = False


if __name__ == "__main__":
    main()



