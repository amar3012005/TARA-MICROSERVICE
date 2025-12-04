# EXACT CODE CHANGES REQUIRED
## Line-by-Line Migration from LemonFox to Sarvam AI TTS

---

## FILE 1: config.py

### SECTION A: Imports (No changes needed)

### SECTION B: TTSStreamingConfig dataclass - CHANGES REQUIRED

**FIND THIS:**
```python
@dataclass
class TTSStreamingConfig:
    """Configuration for TTS Streaming microservice with hardcoded LemonFox settings"""

    # LemonFox API (hardcoded endpoint)
    lemonfox_api_key: str = os.getenv("LEMONFOX_API_KEY", "")
    lemonfox_voice: str = os.getenv("LEIBNIZ_LEMONFOX_VOICE", "sarah")
    lemonfox_language: str = os.getenv("LEIBNIZ_LEMONFOX_LANGUAGE", "en-us")
    lemonfox_api_endpoint: str = "https://api.lemonfox.ai/v1/audio/speech" # Hardcoded
```

**REPLACE WITH:**
```python
@dataclass
class TTSStreamingConfig:
    """Configuration for TTS Streaming microservice with Sarvam AI settings"""

    # Sarvam AI API (hardcoded endpoint)
    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")
    sarvam_model: str = os.getenv("SARVAM_TTS_MODEL", "bulbul:v2")
    sarvam_speaker: str = os.getenv("SARVAM_TTS_SPEAKER", "anushka")
    sarvam_language: str = os.getenv("SARVAM_TTS_LANGUAGE", "en-IN")
    sarvam_api_endpoint: str = "https://api.sarvam.ai/text-to-speech"  # Hardcoded
    
    # Voice control parameters (NEW - Sarvam specific)
    sarvam_pitch: float = float(os.getenv("SARVAM_TTS_PITCH", "0.0"))        # -0.75 to 0.75
    sarvam_pace: float = float(os.getenv("SARVAM_TTS_PACE", "1.0"))          # 0.3 to 3.0
    sarvam_loudness: float = float(os.getenv("SARVAM_TTS_LOUDNESS", "1.0"))  # 0.1 to 3.0
    sarvam_preprocessing: bool = os.getenv("SARVAM_TTS_PREPROCESSING", "false").lower() == "true"
```

### SECTION C: Audio settings - CHANGE SAMPLE RATE

**FIND THIS:**
```python
    # Audio settings
    sample_rate: int = 24000
```

**REPLACE WITH:**
```python
    # Audio settings (16000 for Sarvam, was 24000 for LemonFox)
    sample_rate: int = 16000
```

### SECTION D: from_env() method - REPLACE RETURN STATEMENT

**FIND THIS:**
```python
    @staticmethod
    def from_env() -> "TTSStreamingConfig":
        """Load configuration from environment variables"""
        return TTSStreamingConfig(
            lemonfox_api_key=os.getenv("LEMONFOX_API_KEY", ""),
            lemonfox_voice=os.getenv("LEIBNIZ_LEMONFOX_VOICE", "sarah"),
            lemonfox_language=os.getenv("LEIBNIZ_LEMONFOX_LANGUAGE", "en-us"),
            sample_rate=int(os.getenv("LEIBNIZ_TTS_SAMPLE_RATE", "24000")),
            ...
```

**REPLACE WITH:**
```python
    @staticmethod
    def from_env() -> "TTSStreamingConfig":
        """Load configuration from environment variables"""
        return TTSStreamingConfig(
            sarvam_api_key=os.getenv("SARVAM_API_KEY", ""),
            sarvam_model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v2"),
            sarvam_speaker=os.getenv("SARVAM_TTS_SPEAKER", "anushka"),
            sarvam_language=os.getenv("SARVAM_TTS_LANGUAGE", "en-IN"),
            sarvam_pitch=float(os.getenv("SARVAM_TTS_PITCH", "0.0")),
            sarvam_pace=float(os.getenv("SARVAM_TTS_PACE", "1.0")),
            sarvam_loudness=float(os.getenv("SARVAM_TTS_LOUDNESS", "1.0")),
            sarvam_preprocessing=os.getenv("SARVAM_TTS_PREPROCESSING", "false").lower() == "true",
            sample_rate=int(os.getenv("LEIBNIZ_TTS_SAMPLE_RATE", "16000")),
            ...
```

### SECTION E: __post_init__() method - ADD VALIDATION

**FIND THIS:**
```python
    def __post_init__(self):
        """Validate configuration"""
        if not self.lemonfox_api_key:
            logger.warning("LEMONFOX_API_KEY not set - service may not function properly")
```

**REPLACE WITH:**
```python
    def __post_init__(self):
        """Validate configuration"""
        if not self.sarvam_api_key:
            logger.warning("SARVAM_API_KEY not set - service may not function properly")
        
        # Validate Sarvam-specific ranges
        if not (-0.75 <= self.sarvam_pitch <= 0.75):
            raise ValueError(f"Invalid sarvam_pitch: {self.sarvam_pitch} (must be -0.75 to 0.75)")
        
        if not (0.3 <= self.sarvam_pace <= 3.0):
            raise ValueError(f"Invalid sarvam_pace: {self.sarvam_pace} (must be 0.3 to 3.0)")
        
        if not (0.1 <= self.sarvam_loudness <= 3.0):
            raise ValueError(f"Invalid sarvam_loudness: {self.sarvam_loudness} (must be 0.1 to 3.0)")
```

### SECTION F: Logging - UPDATE INFO MESSAGE

**FIND THIS:**
```python
        logger.info(
            f"TTS Streaming Config: voice={self.lemonfox_voice}, "
            f"language={self.lemonfox_language}, sample_rate={self.sample_rate}Hz, "
            ...
```

**REPLACE WITH:**
```python
        logger.info(
            f"TTS Streaming Config: model={self.sarvam_model}, "
            f"speaker={self.sarvam_speaker}, language={self.sarvam_language}, "
            f"sample_rate={self.sample_rate}Hz, pitch={self.sarvam_pitch}, "
            f"pace={self.sarvam_pace}, loudness={self.sarvam_loudness}, "
            ...
```

---

## FILE 2: tts_queue.py

### SECTION A: Imports - UPDATE PROVIDER

**FIND THIS:**
```python
from .lemonfox_provider import LemonFoxProvider
```

**REPLACE WITH:**
```python
from .sarvam_provider import SarvamProvider
```

### SECTION B: __init__ - UPDATE TYPE HINT

**FIND THIS:**
```python
    def __init__(
        self,
        provider: LemonFoxProvider,
        cache: Optional[AudioCache],
```

**REPLACE WITH:**
```python
    def __init__(
        self,
        provider: SarvamProvider,
        cache: Optional[AudioCache],
```

### SECTION C: Cache call in _synthesize_and_store

**FIND THIS (line ~xx):**
```python
        if self.cache:
            cached_path = self.cache.get_cached_audio(
                sentence,
                voice or self.provider.voice,
                language or self.provider.language,
                "lemonfox",
                emotion or "helpful"
            )
```

**REPLACE WITH:**
```python
        if self.cache:
            cached_path = self.cache.get_cached_audio(
                sentence,
                voice or self.provider.speaker,      # Changed from .voice to .speaker
                language or self.provider.language,
                "sarvam",                             # Changed provider name
                emotion or "helpful"
            )
```

### SECTION D: Cache.cache_audio call

**FIND THIS:**
```python
                cached_path = self.cache.cache_audio(
                    sentence,
                    voice or self.provider.voice,
                    language or self.provider.language,
                    "lemonfox",
                    emotion,
                    temp_path
                )
```

**REPLACE WITH:**
```python
                cached_path = self.cache.cache_audio(
                    sentence,
                    voice or self.provider.speaker,   # Changed from .voice to .speaker
                    language or self.provider.language,
                    "sarvam",                          # Changed provider name
                    emotion,
                    temp_path
                )
```

---

## FILE 3: app.py

### SECTION A: Imports - UPDATE PROVIDER

**FIND THIS:**
```python
from .lemonfox_provider import LemonFoxProvider
```

**REPLACE WITH:**
```python
from .sarvam_provider import SarvamProvider
```

### SECTION B: lifespan function - Provider initialization

**FIND THIS:**
```python
    # Initialize LemonFox provider
    try:
        if not config.lemonfox_api_key:
            logger.warning("⚠️ LEMONFOX_API_KEY not set - service will not function properly")
            provider = None
        else:
            provider = LemonFoxProvider(
                api_key=config.lemonfox_api_key,
                voice=config.lemonfox_voice,
                language=config.lemonfox_language
            )
            logger.info("✅ LemonFox provider initialized")
```

**REPLACE WITH:**
```python
    # Initialize Sarvam AI provider
    try:
        if not config.sarvam_api_key:
            logger.warning("⚠️ SARVAM_API_KEY not set - service will not function properly")
            provider = None
        else:
            provider = SarvamProvider(
                api_key=config.sarvam_api_key,
                speaker=config.sarvam_speaker,
                language=config.sarvam_language,
                model=config.sarvam_model,
                pitch=config.sarvam_pitch,
                pace=config.sarvam_pace,
                loudness=config.sarvam_loudness,
                enable_preprocessing=config.sarvam_preprocessing
            )
            logger.info("✅ Sarvam AI provider initialized")
```

### SECTION C: Startup logging

**FIND THIS:**
```python
        config = TTSStreamingConfig.from_env()
        logger.info(f"📋 Configuration loaded | Voice: {config.lemonfox_voice} | Language: {config.lemonfox_language}")
```

**REPLACE WITH:**
```python
        config = TTSStreamingConfig.from_env()
        logger.info(
            f"📋 Configuration loaded | "
            f"Model: {config.sarvam_model} | "
            f"Speaker: {config.sarvam_speaker} | "
            f"Language: {config.sarvam_language}"
        )
```

### SECTION D: Shutdown logging

**FIND THIS:**
```python
        # Close provider session
        if provider:
            await provider.close()
            logger.info("✅ LemonFox provider closed")
```

**REPLACE WITH:**
```python
        # Close provider session
        if provider:
            await provider.close()
            logger.info("✅ Sarvam AI provider closed")
```

### SECTION E: WebSocket error messages

**FIND ALL:**
```python
"LemonFox provider not initialized. Check LEMONFOX_API_KEY."
```

**REPLACE WITH:**
```python
"Sarvam AI provider not initialized. Check SARVAM_API_KEY."
```

### SECTION F: FastRTC synthesize endpoint

**FIND THIS:**
```python
async def fastrtc_synthesize(request: SynthesizeRequest):
    """
    HTTP endpoint for FastRTC text synthesis.
    Synthesizes text and streams to FastRTC handler for browser playback.
    """
    if not provider:
        raise HTTPException(
            status_code=503,
            detail="LemonFox provider not initialized. Check LEMONFOX_API_KEY."
        )
```

**REPLACE WITH:**
```python
async def fastrtc_synthesize(request: SynthesizeRequest):
    """
    HTTP endpoint for FastRTC text synthesis.
    Synthesizes text and streams to FastRTC handler for browser playback.
    """
    if not provider:
        raise HTTPException(
            status_code=503,
            detail="Sarvam AI provider not initialized. Check SARVAM_API_KEY."
        )
```

---

## FILE 4: Dockerfile (No changes needed)

The Dockerfile imports remain the same since we're only replacing Python modules, not system dependencies.

---

## FILE 5: docker-compose.yml

### Update service environment

**FIND THIS:**
```yaml
    environment:
      - LEMONFOX_API_KEY=${LEMONFOX_API_KEY}
      - LEIBNIZ_LEMONFOX_VOICE=sarah
      - LEIBNIZ_LEMONFOX_LANGUAGE=en-us
```

**REPLACE WITH:**
```yaml
    environment:
      - SARVAM_API_KEY=${SARVAM_API_KEY}
      - SARVAM_TTS_MODEL=bulbul:v2
      - SARVAM_TTS_SPEAKER=anushka
      - SARVAM_TTS_LANGUAGE=en-IN
      - SARVAM_TTS_PITCH=0.0
      - SARVAM_TTS_PACE=1.0
      - SARVAM_TTS_LOUDNESS=1.0
      - SARVAM_TTS_PREPROCESSING=false
      
      # CRITICAL: Change sample rate from 24000 to 16000
      - LEIBNIZ_TTS_SAMPLE_RATE=16000
```

---

## FILE 6: requirements.txt

**NO CHANGES NEEDED** - Both LemonFox and Sarvam use `aiohttp`

Current requirements still work:
```
aiohttp>=3.9.0
```

---

## FILE 7: audio_cache.py

**NO CHANGES NEEDED** - Provider name stored as string

Cache key automatically includes "sarvam" instead of "lemonfox"

---

## SUMMARY OF CHANGES

| File | Changes | Type |
|------|---------|------|
| config.py | Replace lemonfox_* with sarvam_*, add voice control params, change sample_rate | MAJOR |
| tts_queue.py | Update imports, provider type hints, cache provider string | MINOR |
| app.py | Replace provider init, error messages, logging | MAJOR |
| Dockerfile | NO CHANGES | - |
| docker-compose.yml | Update env vars, change sample rate | MAJOR |
| requirements.txt | NO CHANGES | - |
| audio_cache.py | NO CHANGES | - |
| lemonfox_provider.py | DELETE FILE | DELETION |
| sarvam_provider.py | CREATE NEW FILE | NEW |

---

## DEPLOYMENT CHECKLIST

- [ ] Create sarvam_provider.py (new file)
- [ ] Update config.py (replace lemonfox_ params)
- [ ] Update tts_queue.py (import and type hints)
- [ ] Update app.py (provider initialization)
- [ ] Update docker-compose.yml (environment variables)
- [ ] Verify SARVAM_API_KEY is set
- [ ] Change LEIBNIZ_TTS_SAMPLE_RATE from 24000 to 16000
- [ ] Build Docker image: `docker build -t tts-streaming:sarvam .`
- [ ] Test API connectivity
- [ ] Verify audio latency (<380ms)
- [ ] Delete lemonfox_provider.py (old file)
