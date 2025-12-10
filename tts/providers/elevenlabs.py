"""
ElevenLabs TTS Provider

Extracted from leibniz_tts.py (ElevenLabsTTSProvider, lines 620-762) for microservice deployment.

Implements ElevenLabs TTS API with:
    - SDK compatibility layer (3 different API methods across versions)
    - PCM_24000 audio format with WAV conversion
    - Stability/similarity_boost parameters (no direct emotion support)
    - Streaming synthesis support

Reference:
    leibniz_agent/leibniz_tts.py - Original ElevenLabsTTSProvider (lines 620-762)
"""

import io
import logging
from typing import AsyncIterator, Dict, Any, Optional, Tuple

# Import from parent directory (works in Docker)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TTSConfig

logger = logging.getLogger(__name__)


class ElevenLabsTTSProvider:
    """
    ElevenLabs TTS provider with streaming and voice cloning.
    
    Features:
        - SDK compatibility layer (supports 3 API versions)
        - PCM_24000 audio format (24kHz, 24-bit PCM)
        - WAV conversion via soundfile
        - Stability/similarity_boost tuning
        - Streaming synthesis support
        - 50+ premium voices
    
    Emotion Modulation:
        ElevenLabs does NOT support direct emotion parameters.
        Uses stability/similarity_boost for voice character control:
            - stability: Voice consistency (0.0-1.0, default 0.5)
            - similarity_boost: Voice likeness (0.0-1.0, default 0.75)
        
        Recommendation: Use different voices for different emotions
        (e.g., energetic voice for excited, calm voice for soothing)
    
    Audio Format:
        - Encoding: PCM_24000 (24kHz, 24-bit PCM)
        - Sample Rate: 24000 Hz
        - Channels: 1 (mono)
        - Final Output: WAV wrapped PCM
    """
    
    def __init__(self, config: TTSConfig):
        """
        Initialize ElevenLabs TTS provider.
        
        Args:
            config: TTSConfig with elevenlabs_api_key, elevenlabs_voice, etc.
        
        Raises:
            ValueError: If elevenlabs_api_key not set
            ImportError: If ElevenLabs SDK not installed
        """
        # Lazy import ElevenLabs SDK and soundfile
        try:
            import soundfile as sf
            from elevenlabs.client import ElevenLabs
            from elevenlabs import Voice, VoiceSettings
            self.sf = sf
            self.ElevenLabs = ElevenLabs
            self.Voice = Voice
            self.VoiceSettings = VoiceSettings
        except ImportError as e:
            raise ImportError(
                "ElevenLabs SDK not installed. "
                "Install with: pip install elevenlabs soundfile"
            ) from e
        
        self.config = config
        
        # Validate API key
        is_valid, error_msg = self.validate_config()
        if not is_valid:
            raise ValueError(f"Invalid ElevenLabs config: {error_msg}")
        
        # Initialize client
        self.client = self.ElevenLabs(api_key=self.config.elevenlabs_api_key)
        
        logger.info(f" ElevenLabs TTS initialized (voice: {self.config.elevenlabs_voice})")
    
    def _convert_pcm_to_wav(self, pcm_data: bytes, sample_rate: int) -> bytes:
        """
        Convert PCM bytes to WAV format using soundfile.
        
        ElevenLabs returns PCM_24000 (raw PCM), must convert to WAV for
        standard playback compatibility. Detects 16-bit vs 24-bit PCM automatically.
        
        Args:
            pcm_data: Raw PCM bytes (24-bit or 16-bit)
            sample_rate: Audio sample rate (Hz)
        
        Returns:
            WAV file bytes
        """
        try:
            import numpy as np
            
            # Detect bit depth by byte alignment
            # 24-bit PCM: 3 bytes per sample (len % 3 == 0)
            # 16-bit PCM: 2 bytes per sample (len % 2 == 0)
            pcm_len = len(pcm_data)
            
            if pcm_len % 3 == 0:
                # Likely 24-bit PCM
                # Convert to numpy array (24-bit → int32 for soundfile compatibility)
                # Read as uint8 first, then convert to int32
                audio_bytes = np.frombuffer(pcm_data, dtype=np.uint8)
                # Reshape to (num_samples, 3) and convert to int32
                num_samples = pcm_len // 3
                audio_array = np.zeros(num_samples, dtype=np.int32)
                for i in range(num_samples):
                    # Combine 3 bytes (little-endian) into int32
                    sample_bytes = audio_bytes[i*3:(i+1)*3]
                    # Sign-extend from 24-bit to 32-bit
                    sample = (sample_bytes[0] | 
                             (sample_bytes[1] << 8) | 
                             (sample_bytes[2] << 16))
                    if sample & 0x800000:  # Sign bit set
                        sample |= 0xFF000000  # Sign extend
                    audio_array[i] = sample
                
                # Convert int32 to float32 for soundfile (normalize to -1.0 to 1.0)
                audio_float = audio_array.astype(np.float32) / (2**31)
                
                # Write 24-bit WAV
                wav_buffer = io.BytesIO()
                self.sf.write(wav_buffer, audio_float, sample_rate, format='WAV', subtype='PCM_24')
                logger.debug(f" Converted 24-bit PCM ({pcm_len} bytes) → WAV")
            
            else:
                # Assume 16-bit PCM (most common)
                audio_array = np.frombuffer(pcm_data, dtype=np.int16)
                
                # Write 16-bit WAV
                wav_buffer = io.BytesIO()
                self.sf.write(wav_buffer, audio_array, sample_rate, format='WAV', subtype='PCM_16')
                logger.debug(f" Converted 16-bit PCM ({pcm_len} bytes) → WAV")
            
            wav_bytes = wav_buffer.getvalue()
            return wav_bytes
        
        except Exception as e:
            logger.error(f" PCM to WAV conversion failed: {e}")
            raise
    
    def _get_voice_settings(self, emotion: str = "neutral"):
        """
        Get ElevenLabs voice settings (stability and similarity_boost).
        
        Since ElevenLabs doesn't support emotion directly, we use voice settings
        to control character. For true emotion, use different voice IDs.
        
        Args:
            emotion: Ignored (ElevenLabs doesn't support emotion)
        
        Returns:
            VoiceSettings with stability and similarity_boost from config
        """
        return self.VoiceSettings(
            stability=self.config.elevenlabs_stability,
            similarity_boost=self.config.elevenlabs_similarity_boost
        )
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: str = "neutral",
        **kwargs
    ) -> bytes:
        """
        Synthesize text to WAV audio using ElevenLabs TTS.
        
        Args:
            text: Text to synthesize
            voice: ElevenLabs voice ID (e.g., "AnvlJBAqSLDzEevYr9Ap" for Rachel)
            language: Ignored (ElevenLabs auto-detects language)
            emotion: Ignored (ElevenLabs doesn't support emotion)
            **kwargs: Additional parameters (stability, similarity_boost override config)
        
        Returns:
            bytes: WAV audio data (PCM 16-bit, mono, 24kHz)
        
        Raises:
            Exception: On API error or synthesis failure
        """
        voice_id = voice or self.config.elevenlabs_voice
        voice_settings = self._get_voice_settings(emotion)
        
        # Override settings from kwargs if provided
        if "stability" in kwargs:
            voice_settings.stability = kwargs["stability"]
        if "similarity_boost" in kwargs:
            voice_settings.similarity_boost = kwargs["similarity_boost"]
        
        try:
            # SDK compatibility layer - try 3 different API methods
            pcm_audio = None
            
            # Method 1: generate() with Voice object (latest SDK)
            try:
                audio_generator = self.client.generate(
                    text=text,
                    voice=self.Voice(
                        voice_id=voice_id,
                        settings=voice_settings
                    ),
                    model=self.config.elevenlabs_model
                )
                
                # Collect all chunks
                pcm_chunks = []
                for chunk in audio_generator:
                    pcm_chunks.append(chunk)
                
                pcm_audio = b''.join(pcm_chunks)
            
            except AttributeError:
                # Method 2: text_to_speech.convert() (older SDK)
                try:
                    audio_generator = self.client.text_to_speech.convert(
                        text=text,
                        voice_id=voice_id,
                        model_id=self.config.elevenlabs_model,
                        voice_settings=voice_settings
                    )
                    
                    pcm_chunks = []
                    for chunk in audio_generator:
                        pcm_chunks.append(chunk)
                    
                    pcm_audio = b''.join(pcm_chunks)
                
                except AttributeError:
                    # Method 3: synthesize() (oldest SDK)
                    pcm_audio = self.client.synthesize(
                        text=text,
                        voice=voice_id,
                        model=self.config.elevenlabs_model,
                        settings=voice_settings
                    )
            
            if not pcm_audio:
                raise Exception("Failed to synthesize with ElevenLabs (no audio returned)")
            
            # Convert PCM to WAV
            wav_audio = self._convert_pcm_to_wav(pcm_audio, 24000)
            
            logger.debug(f" ElevenLabs TTS synthesized {len(text)} chars → {len(wav_audio)} bytes")
            return wav_audio
        
        except Exception as e:
            logger.error(f" ElevenLabs TTS synthesis failed: {e}")
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
        
        ElevenLabs supports true streaming - yields chunks as they arrive.
        
        Args:
            text: Text to synthesize
            voice: ElevenLabs voice ID
            language: Ignored (auto-detected)
            emotion: Ignored (not supported)
            **kwargs: Additional parameters
        
        Yields:
            bytes: WAV audio chunks (converted from PCM)
        """
        voice_id = voice or self.config.elevenlabs_voice
        voice_settings = self._get_voice_settings(emotion)
        
        try:
            # Stream from ElevenLabs
            audio_generator = self.client.generate(
                text=text,
                voice=self.Voice(
                    voice_id=voice_id,
                    settings=voice_settings
                ),
                model=self.config.elevenlabs_model,
                stream=True  # Enable streaming
            )
            
            # Yield chunks as they arrive
            for pcm_chunk in audio_generator:
                # Convert each chunk to WAV
                wav_chunk = self._convert_pcm_to_wav(pcm_chunk, 24000)
                yield wav_chunk
        
        except Exception as e:
            logger.error(f" ElevenLabs streaming synthesis failed: {e}")
            raise
    
    def get_available_voices(self) -> Dict[str, Any]:
        """
        Get list of available ElevenLabs voices.
        
        Returns:
            Dict mapping voice IDs to metadata
        
        Example:
            {
                "AnvlJBAqSLDzEevYr9Ap": {
                    "name": "Rachel",
                    "language": "en",
                    "gender": "female",
                    "description": "Calm, young adult female"
                },
                ...
            }
        """
        try:
            voices_response = self.client.voices.get_all()
            
            voices = {}
            for voice in voices_response.voices:
                voices[voice.voice_id] = {
                    "name": voice.name,
                    "language": getattr(voice, 'language', 'unknown'),
                    "gender": getattr(voice, 'gender', 'unknown'),
                    "description": getattr(voice, 'description', voice.name)
                }
            
            logger.info(f" ElevenLabs TTS: {len(voices)} voices available")
            return voices
        
        except Exception as e:
            logger.error(f" Failed to list ElevenLabs voices: {e}")
            return {}
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """
        Validate ElevenLabs TTS configuration.
        
        Checks:
            - elevenlabs_api_key is set
            - API key is non-empty
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.config.elevenlabs_api_key:
            return False, "elevenlabs_api_key not set (ELEVENLABS_API_KEY)"
        
        if len(self.config.elevenlabs_api_key) < 10:
            return False, "elevenlabs_api_key appears invalid (too short)"
        
        return True, None
