"""
Provider Tests for TTS Microservice

Tests provider implementations with mocked APIs:
    - Google Cloud TTS (SSML escaping, WAV wrapping)
    - ElevenLabs (SDK compatibility, PCM→WAV conversion)
    - Gemini Live (emotion prompts, streaming)
    - XTTS Local (voice cloning, WAV output)
    - Mock (silent audio generation)

Target Coverage: 70%+

Run with:
    pytest leibniz_agent/services/tts/tests/test_providers.py -v
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, Mock
import numpy as np
import io

from leibniz_agent.services.tts.config import TTSConfig


@pytest.mark.unit
class TestMockProvider:
    """Test MockTTSProvider (always available, no dependencies)."""
    
    @pytest.mark.asyncio
    async def test_mock_synthesize_returns_wav(self, tts_config):
        """Test mock provider returns valid WAV bytes."""
        from leibniz_agent.services.tts.providers.mock import MockTTSProvider
        
        config = TTSConfig(provider="mock")
        provider = MockTTSProvider(config)
        
        audio = await provider.synthesize("Hello world")
        
        # Should return bytes
        assert isinstance(audio, bytes)
        # Should be non-empty
        assert len(audio) > 0
        # Should start with RIFF header (WAV file)
        assert audio[:4] == b'RIFF'
    
    @pytest.mark.asyncio
    async def test_mock_duration_estimation(self, tts_config):
        """Test mock provider estimates duration from text length."""
        from leibniz_agent.services.tts.providers.mock import MockTTSProvider
        
        config = TTSConfig(provider="mock", sample_rate=24000)
        provider = MockTTSProvider(config)
        
        short_text = "Hi"
        long_text = "This is a much longer sentence with many more characters."
        
        short_audio = await provider.synthesize(short_text)
        long_audio = await provider.synthesize(long_text)
        
        # Longer text should produce longer audio
        assert len(long_audio) > len(short_audio)
    
    def test_mock_validate_config(self):
        """Test mock provider always validates successfully."""
        from leibniz_agent.services.tts.providers.mock import MockTTSProvider
        
        config = TTSConfig(provider="mock")
        provider = MockTTSProvider(config)
        
        is_valid, error = provider.validate_config()
        assert is_valid is True
        assert error is None


@pytest.mark.integration
class TestGoogleCloudProvider:
    """Test GoogleCloudTTSProvider with mocked API."""
    
    def test_ssml_escaping(self):
        """Test SSML special character escaping."""
        from leibniz_agent.services.tts.providers.google_cloud import GoogleCloudTTSProvider
        
        # Mock config to avoid credential requirement
        with patch.object(GoogleCloudTTSProvider, '__init__', lambda self, config: None):
            provider = GoogleCloudTTSProvider.__new__(GoogleCloudTTSProvider)
            
            # Test escaping
            text = '<tag> & "quotes" & \'apostrophe\''
            escaped = provider._escape_ssml(text)
            
            assert '&lt;' in escaped
            assert '&gt;' in escaped
            assert '&amp;' in escaped
            assert '&quot;' in escaped
            assert '&apos;' in escaped
            assert '<tag>' not in escaped
    
    def test_emotion_params_mapping(self):
        """Test emotion to pitch/rate mapping."""
        from leibniz_agent.services.tts.providers.google_cloud import GoogleCloudTTSProvider
        
        with patch.object(GoogleCloudTTSProvider, '__init__', lambda self, config: None):
            provider = GoogleCloudTTSProvider.__new__(GoogleCloudTTSProvider)
            provider.config = MagicMock(pitch=0.0, speaking_rate=1.0)
            
            # Test emotion mappings
            helpful_pitch, helpful_rate = provider._get_emotion_params("helpful")
            assert helpful_pitch > 0.0  # Slightly higher pitch
            assert helpful_rate == 1.0
            
            excited_pitch, excited_rate = provider._get_emotion_params("excited")
            assert excited_pitch > 0.0
            assert excited_rate > 1.0  # Faster rate
            
            calm_pitch, calm_rate = provider._get_emotion_params("calm")
            assert calm_pitch == 0.0
            assert calm_rate < 1.0  # Slower rate


@pytest.mark.integration
class TestElevenLabsProvider:
    """Test ElevenLabsTTSProvider with mocked API."""
    
    def test_pcm_to_wav_16bit(self):
        """Test 16-bit PCM to WAV conversion."""
        from leibniz_agent.services.tts.providers.elevenlabs import ElevenLabsTTSProvider
        
        # Create provider with mocked dependencies
        with patch.object(ElevenLabsTTSProvider, '__init__', lambda self, config: None):
            provider = ElevenLabsTTSProvider.__new__(ElevenLabsTTSProvider)
            
            # Mock soundfile
            import soundfile as sf
            provider.sf = sf
            
            # Generate 16-bit PCM (100 samples)
            pcm_data = np.random.randint(-32768, 32767, 100, dtype=np.int16).tobytes()
            
            wav_bytes = provider._convert_pcm_to_wav(pcm_data, 24000)
            
            # Should return WAV bytes
            assert wav_bytes[:4] == b'RIFF'
            assert b'WAVE' in wav_bytes[:12]
    
    def test_pcm_to_wav_24bit(self):
        """Test 24-bit PCM to WAV conversion."""
        from leibniz_agent.services.tts.providers.elevenlabs import ElevenLabsTTSProvider
        
        with patch.object(ElevenLabsTTSProvider, '__init__', lambda self, config: None):
            provider = ElevenLabsTTSProvider.__new__(ElevenLabsTTSProvider)
            
            import soundfile as sf
            provider.sf = sf
            
            # Generate 24-bit PCM (99 bytes = 33 samples * 3 bytes)
            pcm_data = bytes(99)  # All zeros
            
            wav_bytes = provider._convert_pcm_to_wav(pcm_data, 24000)
            
            # Should return WAV bytes
            assert wav_bytes[:4] == b'RIFF'
            assert b'WAVE' in wav_bytes[:12]
    
    def test_voice_settings_defaults(self):
        """Test ElevenLabs voice settings use config defaults."""
        from leibniz_agent.services.tts.providers.elevenlabs import ElevenLabsTTSProvider
        
        with patch.object(ElevenLabsTTSProvider, '__init__', lambda self, config: None):
            provider = ElevenLabsTTSProvider.__new__(ElevenLabsTTSProvider)
            
            # Mock VoiceSettings class
            MockVoiceSettings = MagicMock()
            provider.VoiceSettings = MockVoiceSettings
            provider.config = MagicMock(
                elevenlabs_stability=0.7,
                elevenlabs_similarity_boost=0.8
            )
            
            settings = provider._get_voice_settings("neutral")
            
            # Should call VoiceSettings with config values
            MockVoiceSettings.assert_called_once_with(
                stability=0.7,
                similarity_boost=0.8
            )


@pytest.mark.integration  
class TestGeminiProvider:
    """Test GeminiLiveTTSProvider with mocked API."""
    
    def test_emotion_system_prompts(self):
        """Test emotion maps to different system prompts."""
        from leibniz_agent.services.tts.providers.gemini_live import GeminiLiveTTSProvider
        
        with patch.object(GeminiLiveTTSProvider, '__init__', lambda self, config: None):
            provider = GeminiLiveTTSProvider.__new__(GeminiLiveTTSProvider)
            provider.config = MagicMock(gemini_emotion_support=True)
            
            # Test different emotions produce different prompts
            helpful_prompt = provider._get_emotion_system_prompt("helpful")
            excited_prompt = provider._get_emotion_system_prompt("excited")
            calm_prompt = provider._get_emotion_system_prompt("calm")
            
            assert helpful_prompt != excited_prompt
            assert helpful_prompt != calm_prompt
            assert "helpful" in helpful_prompt.lower() or "warm" in helpful_prompt.lower()
            assert "excited" in excited_prompt.lower() or "energetic" in excited_prompt.lower()
    
    def test_emotion_disabled_returns_neutral(self):
        """Test emotion support disabled returns neutral prompt."""
        from leibniz_agent.services.tts.providers.gemini_live import GeminiLiveTTSProvider
        
        with patch.object(GeminiLiveTTSProvider, '__init__', lambda self, config: None):
            provider = GeminiLiveTTSProvider.__new__(GeminiLiveTTSProvider)
            provider.config = MagicMock(gemini_emotion_support=False)
            
            prompt = provider._get_emotion_system_prompt("excited")
            
            # Should return neutral prompt when emotion disabled
            assert "professional" in prompt.lower() or "clear" in prompt.lower()


@pytest.mark.integration
class TestXTTSProvider:
    """Test XTTSLocalProvider with mocked TTS library."""
    
    @pytest.mark.asyncio
    async def test_xtts_returns_wav_not_pcm(self, tts_config):
        """Test XTTS returns WAV bytes (not raw PCM)."""
        from leibniz_agent.services.tts.providers.xtts_local import XTTSLocalProvider
        
        # Mock TTS library to avoid loading heavy model
        with patch('leibniz_agent.services.tts.providers.xtts_local.TTS') as MockTTS:
            mock_tts_instance = MagicMock()
            MockTTS.return_value = mock_tts_instance
            
            # Mock speaker sample file
            with patch('os.path.exists', return_value=True):
                config = TTSConfig(
                    provider="xtts_local",
                    xtts_speaker_sample="/fake/path.wav",
                    xtts_language="en",
                    xtts_device="cpu"
                )
                
                provider = XTTSLocalProvider(config)
                
                # Mock the synthesis to avoid actual TTS call
                with patch.object(provider.tts, 'tts_to_file'):
                    # Mock soundfile to return fake audio
                    with patch('soundfile.read', return_value=(np.zeros(24000, dtype=np.int16), 24000)):
                        audio = await provider.synthesize("Test text")
                        
                        # Should return WAV bytes
                        assert isinstance(audio, bytes)
                        assert audio[:4] == b'RIFF'


@pytest.mark.unit
def test_provider_availability_flags():
    """Test provider availability flags are set correctly."""
    from leibniz_agent.services.tts import providers
    
    # Mock provider should always be available
    assert providers.MOCK_AVAILABLE is True
    
    # Other providers depend on installed packages
    # Just verify flags exist
    assert hasattr(providers, 'GOOGLE_AVAILABLE')
    assert hasattr(providers, 'ELEVENLABS_AVAILABLE')
    assert hasattr(providers, 'GEMINI_AVAILABLE')
    assert hasattr(providers, 'XTTS_AVAILABLE')
