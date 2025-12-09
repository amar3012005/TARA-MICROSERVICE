"""
STT/VAD Microservice for Leibniz Agent

This package provides real-time speech-to-text transcription with voice activity
detection via Gemini Live API. The service accepts PCM audio streams over WebSocket
and returns normalized transcripts in real-time.

Architecture:
    - FastAPI application with WebSocket streaming endpoint
    - Gemini Live session management with connection pooling
    - Redis-backed session state persistence
    - Bidirectional VAD with barge-in detection
    - English transcript normalization

Main Entry Point:
    leibniz_agent.services.stt_vad.app:app

Reference:
    leibniz_agent/docs/Cloud Transformation.md (lines 173-322)

Components:
    VADManager: Core VAD logic with speech capture and state management
    GeminiLiveSession: Gemini Live API connection manager
    VADConfig: Service configuration from environment variables
    normalize_transcript: Transcript normalization utility
"""

from vad_manager import VADManager
from config import VADConfig
from utils import normalize_english_transcript

__all__ = [
    "VADManager",
    "VADConfig",
    "normalize_english_transcript",
]
