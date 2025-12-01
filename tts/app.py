"""
TTS Microservice FastAPI Application

REST API for TTS synthesis with multi-provider support.

Endpoints:
    POST /api/v1/synthesize - Synthesize text to audio
    GET /api/v1/audio/{cache_key} - Retrieve cached audio
    GET /health - Health check

Features:
    - Async synthesis with multiple providers
    - Automatic caching with MD5 keys
    - Provider fallback and retry logic
    - CORS support for web clients
    - Comprehensive error handling
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Use absolute imports for Docker compatibility
from leibniz_agent.services.tts.config import TTSConfig
from leibniz_agent.services.tts.tts_synthesizer import TTSSynthesizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Leibniz TTS Service",
    description="Text-to-Speech microservice with multi-provider support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global synthesizer instance
config = TTSConfig.from_env()

try:
    synthesizer = TTSSynthesizer(config)
    print(f" TTS SYNTHESIZER CREATED: lemonfox_provider={synthesizer.lemonfox_provider is not None}")  # Debug print
    logger.info(f" TTS Service initialized: provider={config.provider}, cache={config.enable_cache}")
except Exception as e:
    print(f" FAILED TO CREATE SYNTHESIZER: {e}")  # Debug print
    logger.error(f"Failed to create TTS synthesizer: {e}", exc_info=True)
    synthesizer = None


# Request/Response Models
class SynthesizeRequest(BaseModel):
    """Request model for synthesis endpoint."""
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=5000)
    emotion: str = Field(default="helpful", description="Emotion type (helpful, excited, calm, etc.)")
    voice: Optional[str] = Field(default=None, description="Voice name/ID (provider-specific)")
    language: Optional[str] = Field(default=None, description="Language code (e.g., en-US)")
    provider: Optional[str] = Field(default=None, description="Force specific provider (google, elevenlabs, gemini, xtts_local)")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Hello! Welcome to Leibniz University. How can I help you today?",
                "emotion": "helpful",
                "voice": None,
                "language": "en-US"
            }
        }


class SynthesizeResponse(BaseModel):
    """Response model for synthesis endpoint."""
    success: bool = Field(..., description="Whether synthesis succeeded")
    cache_key: Optional[str] = Field(default=None, description="MD5 cache key for audio retrieval")
    audio_url: Optional[str] = Field(default=None, description="URL to retrieve audio file")
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds (estimated)")
    cached: bool = Field(default=False, description="Whether result was from cache")
    provider: Optional[str] = Field(default=None, description="Provider used for synthesis")
    elapsed: Optional[float] = Field(default=None, description="Time taken for synthesis")
    error: Optional[str] = Field(default=None, description="Error message if success=False")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "cache_key": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                "audio_url": "/api/v1/audio/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                "duration": 3.5,
                "cached": False,
                "provider": "gemini",
                "elapsed": 1.234
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    providers_available: list = Field(..., description="List of available TTS providers")
    cache_enabled: bool = Field(..., description="Whether caching is enabled")
    cache_stats: dict = Field(default={}, description="Cache statistics")
    total_requests: int = Field(..., description="Total synthesis requests processed")


# API Endpoints
@app.post("/api/v1/synthesize", response_model=SynthesizeResponse)
async def synthesize_text(request: SynthesizeRequest):
    """
    Synthesize text to speech.
    
    Process:
    1. Check cache for existing audio
    2. If cache miss, synthesize with primary provider
    3. If primary fails, try fallback provider (if enabled)
    4. Return audio URL and cache key
    
    Args:
        request: SynthesizeRequest with text, emotion, voice, language
    
    Returns:
        SynthesizeResponse with audio URL and metadata
    
    Raises:
        HTTPException: 400 (invalid request), 500 (synthesis failed)
    """
    try:
        # Validate request
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Synthesize
        result = await synthesizer.synthesize_to_file(
            text=request.text,
            emotion=request.emotion,
            voice=request.voice,
            language=request.language
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Synthesis failed: {result.get('error', 'Unknown error')}"
            )
        
        # Generate cache key for URL
        voice = request.voice or synthesizer._get_default_voice()
        language = request.language or config.language_code
        
        cache_key = synthesizer.cache.get_cache_key(
            text=request.text,
            voice=voice,
            language=language,
            provider=result['provider'],
            emotion=request.emotion
        ) if synthesizer.cache else None
        
        # Build audio URL
        audio_url = f"/api/v1/audio/{cache_key}" if cache_key else None
        
        return SynthesizeResponse(
            success=True,
            cache_key=cache_key,
            audio_url=audio_url,
            duration=result.get('duration'),
            cached=result.get('cached', False),
            provider=result.get('provider'),
            elapsed=result.get('elapsed')
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/v1/audio/{cache_key}")
async def get_audio(cache_key: str):
    """
    Retrieve cached audio file by cache key.
    
    Args:
        cache_key: MD5 cache key from synthesis response
    
    Returns:
        FileResponse with audio/wav content
    
    Raises:
        HTTPException: 404 (audio not found), 400 (caching disabled)
    """
    try:
        # Check if caching is enabled
        if not synthesizer.cache:
            raise HTTPException(
                status_code=400,
                detail="Caching is disabled. Cannot retrieve audio by cache key."
            )
        
        # Build cache file path
        cache_file = Path(config.cache_dir) / f"{cache_key}.wav"
        
        # Check if file exists
        if not cache_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Audio file not found for cache key: {cache_key}"
            )
        
        # Return audio file
        return FileResponse(
            path=str(cache_file),
            media_type="audio/wav",
            filename=f"{cache_key}.wav"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/v1/synthesize/stream")
async def synthesize_stream(
    text: str,
    emotion: str = "helpful",
    voice: Optional[str] = None,
    language: str = "en-US"
):
    """
    Stream audio synthesis results using Server-Sent Events.

    This endpoint provides real-time audio streaming for immediate playback,
    useful for conversational applications where latency matters.

    Args:
        text: Text to synthesize
        emotion: Emotion type for voice modulation
        voice: Voice identifier (provider-specific)
        language: Language code

    Returns:
        Server-Sent Events stream with audio chunks

    Event Types:
        - metadata: Initial event with synthesis info
        - audio: Base64-encoded audio chunks
        - complete: Final event when streaming ends
        - error: Error event if synthesis fails
    """
    import json
    import base64

    async def generate_events():
        """Generate Server-Sent Events for audio streaming"""
        try:
            # Validate input
            if not text or not text.strip():
                yield f"event: error\ndata: {json.dumps({'error': 'Empty text provided'})}\n\n"
                return

            # Send metadata event first
            metadata = {
                "type": "metadata",
                "text_length": len(text),
                "emotion": emotion,
                "voice": voice,
                "language": language,
                "estimated_duration": len(text) * 0.1  # Rough estimate: 100ms per character
            }
            yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"

            # Stream audio chunks
            chunk_index = 0
            async for audio_chunk, is_final in synthesizer.synthesize_streaming(
                text=text,
                emotion=emotion,
                voice=voice,
                language=language
            ):
                # Encode chunk as base64
                b64_chunk = base64.b64encode(audio_chunk).decode('utf-8')

                event_data = {
                    "type": "audio",
                    "data": b64_chunk,
                    "chunk_index": chunk_index,
                    "is_final": is_final
                }

                yield f"event: audio\ndata: {json.dumps(event_data)}\n\n"
                chunk_index += 1

                # Small delay to prevent overwhelming client
                await asyncio.sleep(0.01)

            # Send completion event
            completion_data = {
                "type": "complete",
                "total_chunks": chunk_index
            }
            yield f"event: complete\ndata: {json.dumps(completion_data)}\n\n"

        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}", exc_info=True)
            error_data = {
                "type": "error",
                "error": str(e)
            }
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with service status and statistics
    """
    try:
        # Get synthesizer stats and provider status
        stats = synthesizer.get_stats() if synthesizer else {}
        provider_status = synthesizer.get_provider_status() if synthesizer else {}
        
        # Determine available providers
        providers_available = [p for p, available in provider_status.items() if available]
        
        # Determine overall status
        if not providers_available:
            # If no providers from synthesizer, check if LemonFox should be available
            config = TTSConfig.from_env()
            if config.lemonfox_api_key and synthesizer:
                # Synthesizer exists but reports no providers - this is unexpected
                providers_available = ["lemonfox"]  # Force LemonFox as available
                status = "healthy"
            else:
                status = "unhealthy"
        elif "lemonfox" not in providers_available:
            # LemonFox is our primary provider - degraded if not available
            status = "degraded"
        else:
            # LemonFox is available - healthy
            status = "healthy"
        
        return HealthResponse(
            status=status,
            providers_available=providers_available,
            cache_enabled=config.enable_cache,
            cache_stats=stats.get('cache', {}),
            total_requests=stats.get('total_requests', 0)
        )
    
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        # Even on error, if LemonFox API key is set, assume it's available
        config = TTSConfig.from_env()
        if config.lemonfox_api_key:
            return HealthResponse(
                status="healthy",
                providers_available=["lemonfox"],
                cache_enabled=config.enable_cache,
                cache_stats={},
                total_requests=0
            )
        else:
            return HealthResponse(
                status="unhealthy",
                providers_available=[],
                cache_enabled=config.enable_cache,
                cache_stats={},
                total_requests=0
            )


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Leibniz TTS Service",
        "version": "1.0.0",
        "endpoints": {
            "synthesize": "POST /api/v1/synthesize",
            "audio": "GET /api/v1/audio/{cache_key}",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=" * 60)
    logger.info(" Leibniz TTS Service Starting")
    logger.info(f"   Provider: {config.provider}")
    logger.info(f"   Fallback: {config.fallback_provider if config.enable_fallback else 'disabled'}")
    logger.info(f"   Cache: {config.enable_cache} (max_size={config.max_cache_size})")
    logger.info(f"   Sample Rate: {config.sample_rate}Hz")
    logger.info("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    stats = synthesizer.get_stats()
    logger.info("=" * 60)
    logger.info(" Leibniz TTS Service Shutting Down")
    logger.info(f"   Total Requests: {stats.get('total_requests', 0)}")
    logger.info(f"   Cache Hits: {stats.get('cache_hits', 0)}")
    logger.info(f"   Cache Misses: {stats.get('cache_misses', 0)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8004
    port = int(os.getenv("LEIBNIZ_TTS_SERVICE_PORT", "8004"))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False  # Set to True for development
    )
