"""
RAG Client for Gemini Live Experiment

Communicates with the RAG service to provide domain-specific knowledge
for Gemini Live bidirectional conversations.
"""

import asyncio
import logging
from typing import Optional, AsyncIterator
import httpx

logger = logging.getLogger(__name__)


class RAGClient:
    """
    Client for communicating with the RAG service.
    
    Handles incremental RAG queries for real-time conversation support.
    """
    
    def __init__(self, rag_service_url: str = "http://tara-task-rag:8003"):
        """
        Initialize RAG client.
        
        Args:
            rag_service_url: Base URL of the RAG service
        """
        self.rag_service_url = rag_service_url.rstrip('/')
        self.client = None
        logger.info(f"🔗 RAG Client initialized | Service: {self.rag_service_url}")
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if self.client is None:
            timeout = httpx.Timeout(30.0, connect=5.0)
            self.client = httpx.AsyncClient(timeout=timeout)
    
    async def send_incremental(
        self,
        session_id: str,
        text: str,
        is_final: bool = False
    ) -> AsyncIterator[str]:
        """
        Send incremental query to RAG service.
        
        Args:
            session_id: Session identifier for buffer management
            text: User text (partial or complete)
            is_final: True if this is the final text, triggers generation
            
        Yields:
            str: Text chunks from RAG response (streaming)
        """
        await self._ensure_client()
        
        url = f"{self.rag_service_url}/api/v1/query/incremental"
        
        payload = {
            "session_id": session_id,
            "text": text,
            "is_final": is_final
        }
        
        try:
            logger.debug(f"📤 Sending incremental RAG query | Session: {session_id} | Final: {is_final} | Text: '{text[:50]}...'")
            
            async with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        import json
                        data = json.loads(line)
                        text_chunk = data.get("text", "")
                        is_final_chunk = data.get("is_final", False)
                        
                        if text_chunk:
                            yield text_chunk
                        
                        if is_final_chunk:
                            logger.debug(f"✅ RAG response complete | Session: {session_id}")
                            break
                            
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Invalid JSON from RAG: {line}")
                        continue
                        
        except httpx.HTTPError as e:
            logger.error(f"❌ RAG HTTP error: {e}")
            yield f"Error: RAG service unavailable ({str(e)})"
        except Exception as e:
            logger.error(f"❌ RAG client error: {e}")
            yield f"Error: {str(e)}"
    
    async def get_full_response(
        self,
        session_id: str,
        text: str
    ) -> str:
        """
        Get full RAG response (aggregates streaming).
        
        Args:
            session_id: Session identifier
            text: Final query text
            
        Returns:
            str: Complete RAG response
        """
        full_text = ""
        async for chunk in self.send_incremental(session_id, text, is_final=True):
            full_text += chunk
        
        return full_text
    
    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.info("🔌 RAG Client closed")
