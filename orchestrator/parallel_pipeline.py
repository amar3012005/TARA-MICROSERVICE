"""
Parallel Intent+RAG+LLM processing

Executes Intent and RAG services concurrently to minimize latency.
"""

import asyncio
import logging
import time
import aiohttp
from typing import Dict, Any

logger = logging.getLogger(__name__)


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


async def process_intent_rag_llm(
    user_text: str, 
    session_id: str,
    intent_url: str,
    rag_url: str = None
) -> Dict[str, Any]:
    """
    CRITICAL: Parallel execution of Intent + RAG (if RAG URL provided)
    
    Without parallelization: Intent (50ms) + RAG (80ms) = 130ms sequential
    With parallelization: max(50ms, 80ms) = 80ms parallel
    
    Saves ~50ms per turn!
    
    If RAG URL is None, only processes Intent.
    """
    start = time.time()
    
    # Always call Intent service
    intent_task = asyncio.create_task(call_intent_service(user_text, session_id, intent_url))
    
    # Conditionally call RAG service if URL provided
    if rag_url:
        logger.info("⚡ Spawning parallel Intent + RAG tasks...")
        rag_task = asyncio.create_task(call_rag_service(user_text, session_id, rag_url))
        # Wait for both to complete
        intent_result, rag_result = await asyncio.gather(intent_task, rag_task)
        
        parallel_time = (time.time() - start) * 1000
        logger.info(f"✅ Parallel execution completed in {parallel_time:.0f}ms")
        logger.info(f"   Intent: {intent_result.get('intent', 'unknown')}")
        logger.info(f"   RAG: {len(rag_result.get('sources', []))} docs found")
        
        # Call LLM if needed (for now, use RAG answer if available)
        llm_response = await call_llm(intent_result, rag_result, user_text)
        
        return {
            "response": llm_response,
            "intent": intent_result,
            "rag": rag_result,
            "latency_ms": parallel_time
        }
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
        
        return {
            "response": llm_response,
            "intent": intent_result,
            "rag": None,
            "latency_ms": intent_time
        }


