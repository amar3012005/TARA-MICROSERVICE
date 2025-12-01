"""
TTS Microservice Test Suite

Test coverage for:
    - Configuration (TTSConfig validation)
    - Audio caching (MD5 keys, LRU, hit/miss)
    - Providers (Google, ElevenLabs, Gemini, XTTS, Mock)
    - Synthesizer (multi-provider orchestration, fallback, retry)
    - FastAPI service (REST endpoints)

Run with:
    pytest leibniz_agent/services/tts/tests/ -v
    pytest leibniz_agent/services/tts/tests/ --cov=leibniz_agent.services.tts
"""
