"""
TTS Provider Implementations

Extracted from leibniz_tts.py for microservice deployment.

Providers:
    - GoogleCloudTTSProvider: Google Cloud TTS (stable, production-ready)
    - ElevenLabsTTSProvider: ElevenLabs TTS (premium quality)
    - GeminiLiveTTSProvider: Gemini Live TTS (emotion-aware, preview)
    - XTTSLocalProvider: XTTS Local TTS (voice cloning, requires torch)
    - MockTTSProvider: Mock TTS (silent audio for testing)
    - LemonFoxTTSProvider: LemonFox TTS (premium quality)

Capability Flags:
    - GOOGLE_AVAILABLE: Google Cloud TTS SDK installed
    - ELEVENLABS_AVAILABLE: ElevenLabs SDK installed
    - GEMINI_AVAILABLE: Gemini SDK installed (google-genai)
    - XTTS_AVAILABLE: XTTS (TTS library) installed
    - MOCK_AVAILABLE: Mock provider (always True)
    - LEMONFOX_AVAILABLE: LemonFox TTS SDK installed
"""

from .base import TTSProvider

# Base provider protocol (always available)
__all__ = ["TTSProvider"]

# Google Cloud TTS Provider (optional)
try:
    from .google_cloud import GoogleCloudTTSProvider
    GOOGLE_AVAILABLE = True
    __all__.append("GoogleCloudTTSProvider")
except ImportError:
    GoogleCloudTTSProvider = None
    GOOGLE_AVAILABLE = False

# ElevenLabs TTS Provider (optional)
try:
    from .elevenlabs import ElevenLabsTTSProvider
    ELEVENLABS_AVAILABLE = True
    __all__.append("ElevenLabsTTSProvider")
except ImportError:
    ElevenLabsTTSProvider = None
    ELEVENLABS_AVAILABLE = False

# Gemini Live TTS Provider (optional)
try:
    from .gemini_live import GeminiLiveTTSProvider
    GEMINI_AVAILABLE = True
    __all__.append("GeminiLiveTTSProvider")
except ImportError:
    GeminiLiveTTSProvider = None
    GEMINI_AVAILABLE = False

# XTTS Local Provider (optional - requires torch ~2GB)
try:
    from .xtts_local import XTTSLocalProvider
    XTTS_AVAILABLE = True
    __all__.append("XTTSLocalProvider")
except ImportError:
    XTTSLocalProvider = None
    XTTS_AVAILABLE = False

# Mock TTS Provider (always available - no external deps)
try:
    from .mock import MockTTSProvider
    MOCK_AVAILABLE = True
    __all__.append("MockTTSProvider")
except ImportError:
    MockTTSProvider = None
    MOCK_AVAILABLE = False

# LemonFox TTS Provider (optional)
try:
    from .lemonfox import LemonFoxTTSProvider
    LEMONFOX_AVAILABLE = True
    __all__.append("LemonFoxTTSProvider")
except ImportError:
    LemonFoxTTSProvider = None
    LEMONFOX_AVAILABLE = False

# Capability flags
__all__.extend([
    "GOOGLE_AVAILABLE",
    "ELEVENLABS_AVAILABLE",
    "GEMINI_AVAILABLE",
    "XTTS_AVAILABLE",
    "MOCK_AVAILABLE",
    "LEMONFOX_AVAILABLE",
])
