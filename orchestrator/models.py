"""
Pydantic models for StateManager Orchestrator

Defines data structures for WebSocket messages and state management.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class STTFragment(BaseModel):
    """Message from STT service"""
    type: str = Field(..., description="Message type: 'stt_fragment'")
    session_id: str = Field(..., description="Session identifier")
    text: str = Field(..., description="Transcribed text fragment")
    is_final: bool = Field(False, description="Whether this is the final fragment")
    timestamp: float = Field(..., description="Unix timestamp")


class VADEnd(BaseModel):
    """End-of-turn signal from VAD"""
    type: str = Field(..., description="Message type: 'vad_end'")
    session_id: str = Field(..., description="Session identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="VAD confidence")


class UserSpeaking(BaseModel):
    """Barge-in signal from VAD"""
    type: str = Field(..., description="Message type: 'user_speaking'")
    session_id: str = Field(..., description="Session identifier")
    timestamp: float = Field(..., description="Unix timestamp")


class OrchestrationResponse(BaseModel):
    """Response from orchestrator"""
    type: str = Field(..., description="Response type")
    session_id: str = Field(..., description="Session identifier")
    state: str = Field(..., description="Current FSM state")
    text: Optional[str] = Field(None, description="Response text")
    latency_breakdown: Optional[Dict[str, float]] = Field(None, description="Latency metrics")


class SessionMetrics(BaseModel):
    """Session metrics"""
    session_id: str
    state: str
    turn_number: int
    total_latency_ms: float
    intent_latency_ms: float
    rag_latency_ms: float
    llm_latency_ms: float
    tts_latency_ms: float




