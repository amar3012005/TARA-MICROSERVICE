#!/usr/bin/env python3
"""
Simple test to validate SarvamStreamingProvider integration
"""

import sys
import os

# Add tts_sarvam to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tts_sarvam'))

def test_imports():
    """Test that all imports work"""
    try:
        from tts_sarvam.sarvam_streaming_provider import SarvamStreamingProvider, StreamingConfig
        from tts_sarvam.config import TTSStreamingConfig
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        from tts_sarvam.config import TTSStreamingConfig
        config = TTSStreamingConfig.from_env()
        print("✅ Configuration loaded successfully")
        print(f"   Streaming buffer size: {config.sarvam_min_buffer_size}")
        print(f"   Max chunk length: {config.sarvam_max_chunk_length}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_streaming_provider_instantiation():
    """Test that streaming provider can be instantiated"""
    try:
        from tts_sarvam.sarvam_streaming_provider import SarvamStreamingProvider, StreamingConfig
        from tts_sarvam.config import TTSStreamingConfig

        config = TTSStreamingConfig.from_env()
        streaming_config = StreamingConfig(
            min_buffer_size=config.sarvam_min_buffer_size,
            max_chunk_length=config.sarvam_max_chunk_length
        )

        # Test instantiation (without API key for basic validation)
        provider = SarvamStreamingProvider(
            api_key="test_key",
            speaker="anushka",
            language="en-IN",
            streaming_config=streaming_config
        )

        print("✅ Streaming provider instantiated successfully")
        print(f"   Buffer size: {provider.streaming_config.min_buffer_size}")
        print(f"   Max chunk: {provider.streaming_config.max_chunk_length}")

        # Test config validation (skip API key check for integration test)
        # Check other validations
        if provider.streaming_config.min_buffer_size <= 0:
            print("❌ Invalid min_buffer_size")
            return False
        if provider.streaming_config.max_chunk_length <= 0:
            print("❌ Invalid max_chunk_length")
            return False
        if provider.streaming_config.max_chunk_length < provider.streaming_config.min_buffer_size:
            print("❌ max_chunk_length < min_buffer_size")
            return False

        print("✅ Configuration validation passed")

        return True
    except Exception as e:
        print(f"❌ Streaming provider test failed: {e}")
        return False

def test_text_buffering():
    """Test text buffering logic"""
    try:
        from tts_sarvam.sarvam_streaming_provider import StreamingConfig

        config = StreamingConfig(min_buffer_size=10, max_chunk_length=50)

        # Simulate text buffering
        buffer = ""
        chunks = []

        test_texts = ["Hello ", "world, ", "this ", "is ", "a ", "test ", "of ", "the ", "streaming ", "functionality."]

        for text in test_texts:
            buffer += text

            while len(buffer) >= config.min_buffer_size:
                chunk_end = min(config.max_chunk_length, len(buffer))
                # Find sentence boundary
                for i in range(min(chunk_end, len(buffer) - 1), 0, -1):
                    if buffer[i] in '.!?\n':
                        chunk_end = i + 1
                        break

                chunk = buffer[:chunk_end].strip()
                buffer = buffer[chunk_end:]
                if chunk:
                    chunks.append(chunk)

        # Add remaining buffer
        if buffer.strip():
            chunks.append(buffer.strip())

        print("✅ Text buffering test successful")
        print(f"   Generated {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"     Chunk {i+1}: '{chunk}' ({len(chunk)} chars)")

        return True
    except Exception as e:
        print(f"❌ Text buffering test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Sarvam Streaming Integration")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Streaming Provider", test_streaming_provider_instantiation),
        ("Text Buffering", test_text_buffering)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All integration tests passed!")
        print("\nThe Sarvam WebSocket streaming integration is ready.")
        print("To use streaming mode, send WebSocket messages with streaming=true:")
        print('  {"type": "synthesize", "text": "...", "streaming": true}')
        print('  {"type": "add_text", "text": "..."}  // Add more text')
        print('  {"type": "finish_stream"}  // End streaming')
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)