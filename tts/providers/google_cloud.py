"""
Google Cloud TTS Provider

Extracted from leibniz_tts.py (GoogleCloudTTSProvider, lines 473-618) for microservice deployment.

Implements Google Cloud Text-to-Speech API with:
    - SSML escaping for special characters
    - Pitch/speaking_rate modulation via SSML prosody tags
    - LINEAR16 PCM audio with WAV header wrapping
    - Emotion mapping to pitch/rate adjustments

Reference:
    leibniz_agent/leibniz_tts.py - Original GoogleCloudTTSProvider (lines 473-618)
"""

import os
import struct
import logging
from typing import AsyncIterator, Dict, Any, Optional, Tuple

from ..config import TTSConfig

logger = logging.getLogger(__name__)


class GoogleCloudTTSProvider:
    """
    Google Cloud TTS provider with SSML and emotion modulation.
    
    Features:
        - SSML escaping for special characters (<, >, &, ", ')
        - Pitch/speaking_rate modulation via <prosody> tags
        - LINEAR16 PCM audio (raw) with WAV header wrapping
        - Emotion → pitch/rate mapping
        - Voice selection from 200+ Neural2 voices
    
    Emotion Modulation (via SSML prosody):
        - helpful: pitch +0.05 semitones, rate 1.0
        - excited: pitch +0.15 semitones, rate 1.2
        - calm: pitch 0.0, rate 0.95
        - neutral: pitch 0.0, rate 1.0
    
    Audio Format:
        - Encoding: LINEAR16 (raw PCM)
        - Sample Rate: 24000 Hz (config.sample_rate)
        - Channels: 1 (mono)
        - Final Output: WAV wrapped PCM
    """
    
    def __init__(self, config: TTSConfig):
        """
        Initialize Google Cloud TTS provider.
        
        Args:
            config: TTSConfig with google_credentials_path, google_voice, sample_rate, etc.
        
        Raises:
            ValueError: If google_credentials_path not set or file doesn't exist
            ImportError: If Google Cloud TTS SDK not installed
        """
        # Lazy import Google Cloud SDK
        try:
            from google.cloud import texttospeech
            from google.oauth2 import service_account
            self.texttospeech = texttospeech
            self.service_account = service_account
        except ImportError as e:
            raise ImportError(
                "Google Cloud TTS SDK not installed. "
                "Install with: pip install google-cloud-texttospeech google-auth"
            ) from e
        
        self.config = config
        
        # Validate credentials
        is_valid, error_msg = self.validate_config()
        if not is_valid:
            raise ValueError(f"Invalid Google Cloud TTS config: {error_msg}")
        
        # Initialize client
        credentials = self.service_account.Credentials.from_service_account_file(
            self.config.google_credentials_path
        )
        self.client = self.texttospeech.TextToSpeechClient(credentials=credentials)
        
        logger.info(f" Google Cloud TTS initialized (voice: {self.config.google_voice})")
    
    def _escape_ssml(self, text: str) -> str:
        """
        Escape special characters for SSML.
        
        Replaces:
            < → &lt;
            > → &gt;
            & → &amp;
            " → &quot;
            ' → &apos;
        
        Args:
            text: Raw text
        
        Returns:
            SSML-escaped text
        """
        text = text.replace("&", "&amp;")  # Must be first
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text
    
    def _get_emotion_params(self, emotion: str) -> Tuple[float, float]:
        """
        Map emotion to pitch and speaking_rate adjustments.
        
        Emotion mappings:
            helpful: (+0.05 semitones, 1.0x rate)
            excited: (+0.15 semitones, 1.2x rate)
            calm: (0.0 semitones, 0.95x rate)
            neutral: (0.0 semitones, 1.0x rate)
            default: (0.0 semitones, 1.0x rate)
        
        Args:
            emotion: Emotion descriptor
        
        Returns:
            Tuple of (pitch_adjustment, speaking_rate_multiplier)
        """
        emotion_map = {
            "helpful": (0.05, 1.0),
            "excited": (0.15, 1.2),
            "calm": (0.0, 0.95),
            "neutral": (0.0, 1.0),
        }
        pitch_adj, rate_mult = emotion_map.get(emotion.lower(), (0.0, 1.0))
        
        # Apply config base values
        final_pitch = self.config.pitch + pitch_adj
        final_rate = self.config.speaking_rate * rate_mult
        
        # Clamp to valid ranges
        final_pitch = max(-20.0, min(20.0, final_pitch))
        final_rate = max(0.25, min(4.0, final_rate))
        
        return final_pitch, final_rate
    
    def _build_ssml(self, text: str, emotion: str) -> str:
        """
        Build SSML string with prosody tags for emotion modulation.
        
        Args:
            text: Raw text to synthesize
            emotion: Emotion descriptor
        
        Returns:
            SSML string with <speak> and <prosody> tags
        
        Example:
            Input: "Hello world", "excited"
            Output: <speak><prosody pitch="+0.15st" rate="1.2">Hello world</prosody></speak>
        """
        escaped_text = self._escape_ssml(text)
        pitch, rate = self._get_emotion_params(emotion)
        
        # Build prosody tag
        pitch_str = f"{pitch:+.2f}st"  # e.g., "+0.15st"
        rate_str = f"{rate:.2f}"       # e.g., "1.20"
        
        ssml = f'<speak><prosody pitch="{pitch_str}" rate="{rate_str}">{escaped_text}</prosody></speak>'
        
        return ssml
    
    def _wrap_pcm_in_wav(self, pcm_data: bytes, sample_rate: int) -> bytes:
        """
        Wrap raw PCM data in WAV header.
        
        Google returns LINEAR16 encoding (raw PCM), must wrap with WAV header
        for standard playback compatibility.
        
        WAV Header Format (44 bytes):
            - RIFF chunk descriptor (12 bytes)
            - fmt subchunk (24 bytes)
            - data subchunk (8 bytes + audio data)
        
        Args:
            pcm_data: Raw PCM bytes (LINEAR16, 16-bit signed)
            sample_rate: Audio sample rate (Hz)
        
        Returns:
            Complete WAV file bytes
        """
        num_channels = 1  # Mono
        bytes_per_sample = 2  # 16-bit = 2 bytes
        byte_rate = sample_rate * num_channels * bytes_per_sample
        block_align = num_channels * bytes_per_sample
        
        # Calculate sizes
        data_size = len(pcm_data)
        file_size = 36 + data_size  # Total file size - 8 bytes
        
        # Build WAV header (44 bytes)
        wav_header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',          # ChunkID
            file_size,        # ChunkSize
            b'WAVE',          # Format
            b'fmt ',          # Subchunk1ID
            16,               # Subchunk1Size (PCM)
            1,                # AudioFormat (PCM = 1)
            num_channels,     # NumChannels
            sample_rate,      # SampleRate
            byte_rate,        # ByteRate
            block_align,      # BlockAlign
            16,               # BitsPerSample
            b'data',          # Subchunk2ID
            data_size         # Subchunk2Size
        )
        
        return wav_header + pcm_data
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: str = "neutral",
        **kwargs
    ) -> bytes:
        """
        Synthesize text to WAV audio using Google Cloud TTS.
        
        Args:
            text: Text to synthesize
            voice: Google voice name (e.g., "en-US-Neural2-F"), uses config default if None
            language: Language code (e.g., "en-US"), uses config default if None
            emotion: Emotion descriptor (helpful, excited, calm, neutral)
            **kwargs: Additional parameters (pitch, speaking_rate override config)
        
        Returns:
            bytes: WAV audio data (PCM 16-bit, mono, 24kHz)
        
        Raises:
            Exception: On API error or synthesis failure
        """
        voice_name = voice or self.config.google_voice
        lang_code = language or self.config.language_code
        
        # Build SSML with emotion modulation
        ssml = self._build_ssml(text, emotion)
        
        # Configure synthesis request
        synthesis_input = self.texttospeech.SynthesisInput(ssml=ssml)
        
        voice_params = self.texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name
        )
        
        audio_config = self.texttospeech.AudioConfig(
            audio_encoding=self.texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.config.sample_rate,
            volume_gain_db=self.config.volume_gain_db
        )
        
        # Call Google Cloud TTS API
        try:
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )
            
            # Wrap LINEAR16 PCM in WAV header
            wav_audio = self._wrap_pcm_in_wav(response.audio_content, self.config.sample_rate)
            
            logger.debug(f" Google TTS synthesized {len(text)} chars → {len(wav_audio)} bytes")
            return wav_audio
        
        except Exception as e:
            logger.error(f" Google TTS synthesis failed: {e}")
            raise
    
    async def stream_synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: str = "neutral",
        **kwargs
    ) -> AsyncIterator[bytes]:
        """
        Synthesize text to streaming audio chunks.
        
        Note: Google Cloud TTS does NOT support true streaming - returns complete audio.
        This method wraps synthesize() for protocol compliance.
        
        Args:
            text: Text to synthesize
            voice: Google voice name
            language: Language code
            emotion: Emotion descriptor
            **kwargs: Additional parameters
        
        Yields:
            bytes: Complete audio as single chunk
        """
        audio = await self.synthesize(text, voice, language, emotion, **kwargs)
        yield audio
    
    def get_available_voices(self) -> Dict[str, Any]:
        """
        Get list of available Google Cloud TTS voices.
        
        Returns:
            Dict mapping voice names to metadata
        
        Example:
            {
                "en-US-Neural2-F": {
                    "name": "en-US-Neural2-F",
                    "language": "en-US",
                    "gender": "FEMALE",
                    "description": "English (US) Female Neural 2"
                },
                ...
            }
        """
        try:
            response = self.client.list_voices()
            
            voices = {}
            for voice in response.voices:
                for lang_code in voice.language_codes:
                    voice_id = voice.name
                    voices[voice_id] = {
                        "name": voice_id,
                        "language": lang_code,
                        "gender": self.texttospeech.SsmlVoiceGender(voice.ssml_gender).name,
                        "description": f"{lang_code} {self.texttospeech.SsmlVoiceGender(voice.ssml_gender).name.title()} {voice.name.split('-')[-1]}"
                    }
            
            logger.info(f" Google TTS: {len(voices)} voices available")
            return voices
        
        except Exception as e:
            logger.error(f" Failed to list Google voices: {e}")
            return {}
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """
        Validate Google Cloud TTS configuration.
        
        Checks:
            - google_credentials_path is set
            - Credentials file exists
            - File is valid JSON service account key
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.config.google_credentials_path:
            return False, "google_credentials_path not set (GOOGLE_APPLICATION_CREDENTIALS)"
        
        if not os.path.exists(self.config.google_credentials_path):
            return False, f"Credentials file not found: {self.config.google_credentials_path}"
        
        try:
            # Attempt to load credentials (use lazy-loaded module)
            self.service_account.Credentials.from_service_account_file(
                self.config.google_credentials_path
            )
            return True, None
        
        except Exception as e:
            return False, f"Invalid credentials file: {e}"
