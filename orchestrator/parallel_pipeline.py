"""
Parallel Intent+RAG+LLM processing

Executes Intent and RAG services concurrently to minimize latency.
Supports TARA mode for direct RAG queries (bypassing Intent service).

Pre-LLM Accumulation Optimization:
- During partial speech (is_final=false): Triggers RAG pre-processing
  (pattern detection, document retrieval, extraction, prompt building)
- During final speech (is_final=true): Uses pre-built prompt for fast generation
"""

import asyncio
import logging
import time
import json
import aiohttp
from typing import Dict, Any, Optional, AsyncGenerator

logger = logging.getLogger(__name__)

# Session chunk sequence tracking (in-memory fallback when state_manager not available)
_chunk_sequences: Dict[str, int] = {}


async def call_intent_service(text: str, session_id: str, intent_url: str) -> Dict[str, Any]:
    """Call Intent service (non-blocking)"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{intent_url}/api/v1/classify",
                json={"text": text},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"Intent service error: HTTP {resp.status}")
                    return {"intent": "unknown", "confidence": 0.0, "context": {}}
        except asyncio.TimeoutError:
            logger.error(f"Intent service timeout")
            return {"intent": "unknown", "confidence": 0.0, "context": {}}
        except Exception as e:
            logger.error(f"Intent service error: {e}")
            return {"intent": "unknown", "confidence": 0.0, "context": {}}


async def call_rag_service(text: str, session_id: str, rag_url: str, intent_context: Dict = None) -> Dict[str, Any]:
    """Call RAG service (non-blocking)"""
    async with aiohttp.ClientSession() as session:
        try:
            query_data = {"query": text}
            if intent_context:
                query_data["context"] = intent_context
            
            async with session.post(
                f"{rag_url}/api/v1/query",
                json=query_data,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"RAG service error: HTTP {resp.status}")
                    return {"answer": "", "sources": [], "confidence": 0.0}
        except asyncio.TimeoutError:
            logger.error(f"RAG service timeout")
            return {"answer": "", "sources": [], "confidence": 0.0}
        except Exception as e:
            logger.error(f"RAG service error: {e}")
            return {"answer": "", "sources": [], "confidence": 0.0}


async def call_llm(intent: Dict, rag_context: Dict, user_text: str) -> str:
    """Call LLM (Groq/Gemini) - Placeholder for future implementation"""
    # TODO: Implement LLM integration
    # For now, use RAG answer if available, otherwise use intent
    if rag_context.get("answer"):
        return rag_context["answer"]
    elif intent.get("context", {}).get("extracted_meaning"):
        return intent["context"]["extracted_meaning"]
    else:
        return f"Response to: {user_text}"


async def stream_rag_service(text: str, session_id: str, rag_url: str, intent_context: Dict = None) -> AsyncGenerator[str, None]:
    """Stream RAG service response"""
    async with aiohttp.ClientSession() as session:
        try:
            query_data = {"query": text}
            if intent_context:
                query_data["context"] = intent_context
            
            async with session.post(
                f"{rag_url}/api/v1/stream_query",
                json=query_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    async for line in resp.content:
                        if line:
                            try:
                                data = json.loads(line)
                                yield data.get("text", "")
                            except:
                                pass
                else:
                    logger.error(f"RAG stream error: HTTP {resp.status}")
                    yield "Sorry, I encountered an error."
        except Exception as e:
            logger.error(f"RAG stream error: {e}")
            yield "Sorry, I encountered an error."


async def process_intent_rag_llm(
    user_text: str, 
    session_id: str,
    intent_url: str,
    rag_url: str = None
) -> AsyncGenerator[str, None]:
    """
    Parallel execution of Intent + RAG with Streaming.
    Yields tokens as they are generated.
    """
    start = time.time()
    
    # Always call Intent service in background
    intent_task = asyncio.create_task(call_intent_service(user_text, session_id, intent_url))
    
    # Conditionally call RAG service if URL provided
    if rag_url:
        logger.info("⚡ Spawning parallel Intent + RAG Streaming...")
        
        # Stream from RAG
        async for token in stream_rag_service(user_text, session_id, rag_url):
            yield token
            
        # Wait for intent to finish (just for logging/consistency)
        try:
            intent_result = await intent_task
            logger.info(f"   Intent: {intent_result.get('intent', 'unknown')}")
        except Exception as e:
            logger.error(f"Intent task failed: {e}")
            
    else:
        logger.info("⚡ Processing Intent only (RAG not configured)...")
        # Only Intent service
        intent_result = await intent_task
        
        intent_time = (time.time() - start) * 1000
        logger.info(f"✅ Intent execution completed in {intent_time:.0f}ms")
        logger.info(f"   Intent: {intent_result.get('intent', 'unknown')}")
        
        # Generate response from intent only
        intent_context = intent_result.get('context', {})
        extracted_meaning = intent_context.get('extracted_meaning', '')
        user_goal = intent_context.get('user_goal', '')
        
        if extracted_meaning:
            llm_response = extracted_meaning
        elif user_goal:
            llm_response = f"I understand you want to {user_goal}."
        else:
            llm_response = f"Response to: {user_text}"
        
        # Yield the full response as a single chunk (or split it if we want to simulate streaming)
        yield llm_response


async def process_rag_direct(
    user_text: str,
    session_id: str,
    rag_url: str,
    language: str = "te-mixed",
    organization: str = "TASK"
) -> AsyncGenerator[str, None]:
    """
    Direct RAG call without Intent service for TARA mode (Streaming).
    
    Bypasses Intent classification and sends user query directly to RAG service
    with Telugu/TASK context for faster response times.
    """
    start = time.time()
    
    logger.info("=" * 70)
    logger.info("🇮🇳 TARA MODE: Direct RAG processing (skipping Intent)")
    logger.info(f"   User: {user_text[:100]}...")
    logger.info(f"   Language: {language} | Organization: {organization}")
    logger.info("=" * 70)
    
    # Build context for TARA Telugu mode
    tara_context = {
        "language": language,
        "organization": organization,
        "mode": "tara_telugu",
        "response_style": "mixed_telugu_english"
    }
    
    # Stream from RAG service directly
    full_response = ""
    async for token in stream_rag_service(
        text=user_text,
        session_id=session_id,
        rag_url=rag_url,
        intent_context=tara_context
    ):
        full_response += token
        yield token
    
    total_time = (time.time() - start) * 1000
    logger.info(f"✅ RAG Stream Complete ({total_time:.0f}ms)")
    logger.info(f"   Response: {full_response}")


# =============================================================================
# INCREMENTAL RAG (Pre-LLM Accumulation + Streaming Generation)
# =============================================================================

def get_next_chunk_sequence(session_id: str) -> int:
    """Get and increment chunk sequence for a session."""
    if session_id not in _chunk_sequences:
        _chunk_sequences[session_id] = 0
    _chunk_sequences[session_id] += 1
    return _chunk_sequences[session_id]


def reset_chunk_sequence(session_id: str) -> None:
    """Reset chunk sequence after final generation (new utterance)."""
    if session_id in _chunk_sequences:
        _chunk_sequences[session_id] = 0


async def buffer_rag_incremental(
    text: str,
    session_id: str,
    rag_url: str,
    language: str = "te-mixed",
    organization: str = "TASK",
    chunk_sequence: int = None
) -> Dict[str, Any]:
    """
    Fire-and-forget call to trigger RAG pre-LLM accumulation.
    Called on partial STT events (is_final=False).
    
    The RAG service now performs full pre-LLM processing:
    - Pattern detection
    - Document retrieval
    - Information extraction
    - Prompt construction
    
    Returns quickly with buffer status (processing happens async in RAG).
    
    Args:
        text: Partial transcription text
        session_id: Unique session identifier for buffer management
        rag_url: RAG service URL
        language: Response language (default: te-mixed for Telugu)
        organization: Organization context (default: TASK)
        chunk_sequence: Optional chunk sequence number for ordering
    """
    # Auto-increment sequence if not provided
    if chunk_sequence is None:
        chunk_sequence = get_next_chunk_sequence(session_id)
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        try:
            request_data = {
                "session_id": session_id,
                "text": text,
                "is_final": False,
                "context": {
                    "language": language,
                    "organization": organization,
                    "mode": "tara_telugu",
                    "chunk_sequence": chunk_sequence
                }
            }
            
            async with session.post(
                f"{rag_url}/api/v1/query/incremental",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=3)  # Short timeout for buffering
            ) as resp:
                duration_ms = (time.time() - start_time) * 1000
                
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(
                        f"[SESSION:{session_id}] 📦 Chunk {chunk_sequence}: RAG accumulation triggered "
                        f"({duration_ms:.0f}ms) - '{text[:30]}...'"
                    )
                    return result
                else:
                    logger.warning(
                        f"[SESSION:{session_id}] ⚠️ Chunk {chunk_sequence}: Buffer request failed HTTP {resp.status}"
                    )
                    return {"status": "error", "docs_retrieved": 0, "chunk_sequence": chunk_sequence}
                    
        except asyncio.TimeoutError:
            logger.warning(f"[SESSION:{session_id}] ⚠️ Chunk {chunk_sequence}: Buffer request timeout")
            return {"status": "timeout", "docs_retrieved": 0, "chunk_sequence": chunk_sequence}
        except Exception as e:
            logger.warning(f"[SESSION:{session_id}] ⚠️ Chunk {chunk_sequence}: Buffer request error: {e}")
            return {"status": "error", "docs_retrieved": 0, "chunk_sequence": chunk_sequence}


async def stream_rag_incremental(
    text: str,
    session_id: str,
    rag_url: str,
    language: str = "te-mixed",
    organization: str = "TASK"
) -> AsyncGenerator[str, None]:
    """
    Stream RAG response using pre-accumulated context.
    Called on final STT event (is_final=True).
    
    The RAG service uses the pre-built prompt from accumulation phase
    for fast LLM generation (~355ms vs ~400ms without optimization).
    
    Returns streaming response from pre-buffered context.
    """
    start = time.time()
    
    logger.info("=" * 70)
    logger.info(f"[SESSION:{session_id}] 🚀 OPTIMIZED: Requesting generation from pre-accumulated context")
    logger.info(f"   Final query: {text[:50]}...")
    logger.info("=" * 70)
    
    async with aiohttp.ClientSession() as session:
        try:
            request_data = {
                "session_id": session_id,
                "text": text,
                "is_final": True,
                "context": {
                    "language": language,
                    "organization": organization,
                    "mode": "tara_telugu",
                    "response_style": "mixed_telugu_english"
                }
            }
            
            async with session.post(
                f"{rag_url}/api/v1/query/incremental",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    full_response = ""
                    async for line in resp.content:
                        if line:
                            try:
                                data = json.loads(line)
                                token = data.get("text", "")
                                full_response += token
                                yield token
                            except json.JSONDecodeError:
                                pass
                    
                    total_time = (time.time() - start) * 1000
                    logger.info(f"[SESSION:{session_id}] ✅ Optimized generation complete ({total_time:.0f}ms)")
                    logger.info(f"   Response: {full_response[:100]}...")
                    
                    # Reset chunk sequence for next utterance
                    reset_chunk_sequence(session_id)
                else:
                    logger.error(f"[SESSION:{session_id}] ❌ Incremental RAG error: HTTP {resp.status}")
                    reset_chunk_sequence(session_id)
                    yield "Sorry, I encountered an error processing your request."
        except asyncio.TimeoutError:
            logger.error(f"[SESSION:{session_id}] ❌ Incremental RAG timeout")
            reset_chunk_sequence(session_id)
            yield "Sorry, the request timed out."
        except Exception as e:
            logger.error(f"[SESSION:{session_id}] ❌ Incremental RAG error: {e}")
            reset_chunk_sequence(session_id)
            yield "Sorry, I encountered an error."


async def process_rag_incremental(
    user_text: str,
    session_id: str,
    rag_url: str,
    is_final: bool,
    language: str = "te-mixed",
    organization: str = "TASK",
    chunk_sequence: int = None
) -> AsyncGenerator[str, None]:
    """
    Main entry point for incremental RAG processing with Pre-LLM Accumulation.
    
    Optimization Flow:
    - is_final=False: Trigger RAG pre-processing (pattern detection, retrieval,
      extraction, prompt building) - fire-and-forget, yields nothing
    - is_final=True: Request generation using pre-built prompt (fast ~355ms)
    
    Args:
        user_text: User's transcribed text
        session_id: Unique session identifier (must be consistent across chunks)
        rag_url: RAG service URL
        is_final: True for final text (trigger generation), False for partial
        language: Response language
        organization: Organization context
        chunk_sequence: Optional chunk sequence number
    """
    if not is_final:
        # Fire-and-forget pre-LLM accumulation (don't yield anything)
        asyncio.create_task(buffer_rag_incremental(
            text=user_text,
            session_id=session_id,
            rag_url=rag_url,
            language=language,
            organization=organization,
            chunk_sequence=chunk_sequence
        ))
        logger.debug(f"[SESSION:{session_id}] Pre-LLM accumulation triggered for partial chunk")
        # Yield nothing - this is a background operation
        return
    else:
        # Stream from pre-accumulated context (fast path)
        logger.info(f"[SESSION:{session_id}] Final chunk: requesting optimized generation")
        async for token in stream_rag_incremental(
            text=user_text,
            session_id=session_id,
            rag_url=rag_url,
            language=language,
            organization=organization
        ):
            yield token

