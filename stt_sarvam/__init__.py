"""
STT/VAD Microservice for Leibniz Agent (Sarvam AI Saarika)

This package provides real-time speech-to-text transcription with voice activity
detection via Sarvam AI's Saarika v2.5 model. The service accepts PCM audio streams
over WebSocket and returns normalized transcripts in real-time.

Architecture:
    - FastAPI application with WebSocket streaming endpoint
    - Sarvam AI Saarika REST API integration
    - Redis-backed session state persistence
    - Energy-based VAD with barge-in detection
    - Multi-language support (11 Indian languages + automatic LID)

Main Entry Point:
    leibniz_agent.services.stt_sarvam.app:app

Reference:
    https://docs.sarvam.ai/api-reference-docs/getting-started/models/saarika

Components:
    VADManager: Core VAD logic with speech capture and state management
    SarvamSTTClient: Sarvam AI REST API client
    VADConfig: Service configuration from environment variables
    normalize_english_transcript: Transcript normalization utility
"""

from vad_manager import VADManager
from config import VADConfig
from utils import normalize_english_transcript
from sarvam_client import SarvamSTTClient, SarvamTranscriptionResult

__all__ = [
    "VADManager",
    "VADConfig",
    "normalize_english_transcript",
    "SarvamSTTClient",
    "SarvamTranscriptionResult",
]
