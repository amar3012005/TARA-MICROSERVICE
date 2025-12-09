# Google Cloud TTS Streaming Implementation
# Requires: pip install google-cloud-texttospeech

import time
from google.cloud import texttospeech

def test_google_cloud_tts_streaming():
    """Test Google Cloud TTS streaming with latency measurement."""
    client = texttospeech.TextToSpeechClient()
    
    # Chirp 3: HD voice (required for streaming)
    streaming_config = texttospeech.StreamingSynthesizeConfig(
        voice=texttospeech.VoiceSelectionParams(
            name="en-US-Chirp3-HD-Charon",
            language_code="en-US",
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,  # Optimal for streaming
        ),
    )
    
    # Send config first, then stream text
    config_request = texttospeech.StreamingSynthesizeRequest(
        streaming_config=streaming_config
    )
    
    # Split text into chunks for incremental sending
    text_chunks = [
        "Hello, ",
        "how can ",
        "I help ",
        "you today?"
    ]
    
    def request_generator():
        """Generator to stream requests."""
        yield config_request
        for chunk in text_chunks:
            yield texttospeech.StreamingSynthesizeRequest(
                input=texttospeech.StreamingSynthesisInput(text=chunk)
            )
    
    # Measure first audio chunk latency
    start_time = time.time()
    first_chunk_received = False
    
    try:
        streaming_responses = client.streaming_synthesize(request_generator())
        
        for response in streaming_responses:
            if response.audio_content and not first_chunk_received:
                first_chunk_latency = (time.time() - start_time) * 1000
                print(f"✓ First Audio Chunk: {first_chunk_latency:.0f}ms")
                first_chunk_received = True
            
            print(f"  Chunk size: {len(response.audio_content)} bytes")
    
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_google_cloud_tts_streaming()
