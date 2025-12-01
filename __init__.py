"""
Leibniz Agent Microservices Package

This package contains all microservices for the Leibniz Agent architecture.

Subpackages:
    shared: Common utilities (Redis client, health checks)
    
Future microservices (to be added in subsequent phases):
    stt_vad: Speech-to-text and voice activity detection service
    intent: Intent classification service
    rag: RAG (Retrieval-Augmented Generation) service
    tts: Text-to-speech service
    appointment: Appointment scheduling FSM service
    orchestrator: Main orchestrator service
    tests: Integration tests
"""

__all__ = ["shared"]
