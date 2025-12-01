"""
Real Audio Test Client for STT/VAD Microservice

Captures microphone audio and streams it to the WebSocket endpoint.
Press SPACE to start recording, release to stop.
"""

import asyncio
import json
import numpy as np
import sounddevice as sd
from websockets import connect
import sys

SERVICE_URL = "ws://localhost:8001/api/v1/transcribe/stream"

# Audio configuration (must match service expectations)
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.int16


class RealAudioTester:
    """Real-time audio capture and streaming to STT/VAD service"""
    
    def __init__(self):
        self.audio_queue = asyncio.Queue()
        self.is_recording = False
        self.websocket = None
        self.loop = None
        
    def audio_callback(self, indata, frames, time_info, status):
        """Sounddevice callback - captures audio chunks"""
        if status:
            print(f"️ Audio status: {status}")
        
        if self.is_recording and self.loop:
            # Convert float32 to int16 PCM
            audio_data = (indata.copy() * 32767).astype(np.int16).tobytes()
            
            # Queue for async transmission (thread-safe)
            self.loop.call_soon_threadsafe(
                self.audio_queue.put_nowait, audio_data
            )
    
    async def send_audio_task(self):
        """Background task - sends audio from queue to WebSocket"""
        try:
            while self.is_recording:
                try:
                    audio_chunk = await asyncio.wait_for(
                        self.audio_queue.get(),
                        timeout=0.5
                    )
                    
                    if self.websocket:
                        await self.websocket.send(audio_chunk)
                        
                except asyncio.TimeoutError:
                    continue
                    
        except Exception as e:
            print(f" Send error: {e}")
    
    async def receive_responses_task(self):
        """Background task - receives transcript fragments from WebSocket"""
        try:
            while self.websocket:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                msg_type = data.get("type", "unknown")
                text = data.get("text", "")
                
                if msg_type == "partial":
                    print(f"\r Partial: {text}", end="", flush=True)
                elif msg_type == "final":
                    print(f"\n Final: {text}")
                    break
                elif msg_type == "timeout":
                    print("\n⏱️ Timeout - no speech detected")
                    break
                elif msg_type == "error":
                    print(f"\n Error: {text}")
                    break
                    
        except Exception as e:
            print(f"\n Receive error: {e}")
    
    async def test_with_microphone(self):
        """Main test flow with microphone input"""
        print(" Real Audio STT/VAD Test")
        print("=" * 60)
        print("\nMicrophone will capture when you start speaking...")
        print("Service timeout: ~20 seconds of silence")
        print("\nPress ENTER to start recording, then speak clearly.")
        print("Press CTRL+C to stop.\n")
        
        # Get current event loop
        self.loop = asyncio.get_event_loop()
        
        try:
            # Wait for user to start
            input("Press ENTER when ready...")
            
            print("\n Recording started - speak now!")
            print("-" * 60)
            
            # Connect to WebSocket
            async with connect(SERVICE_URL) as websocket:
                self.websocket = websocket
                print(" Connected to STT/VAD service")
                
                # Start recording
                self.is_recording = True
                
                # Start audio stream
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype='float32',
                    blocksize=800,  # 50ms chunks
                    callback=self.audio_callback
                )
                
                with stream:
                    # Start background tasks
                    send_task = asyncio.create_task(self.send_audio_task())
                    receive_task = asyncio.create_task(self.receive_responses_task())
                    
                    # Wait for completion or user interrupt
                    try:
                        await receive_task
                    except KeyboardInterrupt:
                        print("\n\n Stopped by user")
                    finally:
                        self.is_recording = False
                        send_task.cancel()
                        
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass
                
                print("\n" + "=" * 60)
                print(" Recording session complete")
                
        except KeyboardInterrupt:
            print("\n\n Test cancelled")
        except Exception as e:
            print(f"\n Test failed: {e}")
            import traceback
            traceback.print_exc()


async def list_audio_devices():
    """List available audio input devices"""
    print("\n Available Audio Devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default = " (DEFAULT)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default}")
            print(f"      Channels: {device['max_input_channels']}, "
                  f"Sample Rate: {device['default_samplerate']:.0f} Hz")
    print("-" * 60)


async def main():
    """Run real audio test"""
    
    # Show available devices
    await list_audio_devices()
    
    # Create tester
    tester = RealAudioTester()
    
    # Run test
    await tester.test_with_microphone()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n Goodbye!")
        sys.exit(0)
