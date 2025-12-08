"""
RAG Service FastAPI Application

HTTP REST API for knowledge base queries with Redis caching.

Reference:
    - Cloud Transformation doc (lines 474-641) - RAG service specifications
    - services/intent/app.py - FastAPI pattern
"""

import os
import sys
import time
import logging
import json
import hashlib
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from leibniz_agent.services.shared.redis_client import get_redis_client, close_redis_client, ping_redis
from leibniz_agent.services.rag.config import RAGConfig
from leibniz_agent.services.rag.rag_engine import RAGEngine
from leibniz_agent.services.rag.index_builder import IndexBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    context: Optional[Dict[str, Any]] = Field(None, description="Context from intent service")
    enable_streaming: Optional[bool] = Field(None, description="Enable streaming response")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(..., description="Source document filenames")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Retrieval confidence")
    timing_breakdown: Dict[str, float] = Field(..., description="Timing metrics")
    cached: bool = Field(..., description="Whether result was served from cache")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    index_size: int
    cache_hit_rate: float
    redis_connected: bool
    gemini_available: bool
    uptime_seconds: float


class RebuildIndexRequest(BaseModel):
    knowledge_base_path: Optional[str] = Field(None, description="Override knowledge base path")


class RebuildIndexResponse(BaseModel):
    status: str
    documents_indexed: int
    categories: int
    build_time_seconds: float


# =============================================================================
# INCREMENTAL RAG MODELS (Buffered Retrieval)
# =============================================================================
class IncrementalQueryRequest(BaseModel):
    """Request model for incremental/buffered RAG queries"""
    session_id: str = Field(..., min_length=1, description="Session identifier for buffer management")
    text: str = Field(..., min_length=1, description="User text (partial or complete)")
    is_final: bool = Field(False, description="True if this is the final text, triggers generation")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context from intent service")


class IncrementalBufferResponse(BaseModel):
    """Response for non-final incremental queries (buffering phase)"""
    status: str = Field(..., description="'buffered', 'processing_async', or 'error'")
    session_id: str = Field(..., description="Session identifier")
    docs_retrieved: int = Field(0, description="Number of documents retrieved so far")
    buffer_size_chars: int = Field(0, description="Total characters in buffer")
    timing_ms: float = Field(0, description="Time taken for this operation")
    message: Optional[str] = Field(None, description="Status message")


# Global state (initialized in lifespan)
rag_engine: Optional[RAGEngine] = None
redis_client: Optional[redis.Redis] = None
cache_hits = 0
cache_misses = 0
app_start_time = 0.0

# Incremental buffer settings
INCREMENTAL_BUFFER_PREFIX = "rag:incremental:"
INCREMENTAL_BUFFER_TTL = 60  # 60 seconds TTL for incremental buffers


# =============================================================================
# INCREMENTAL BUFFER HELPERS (Redis-based)
# =============================================================================
async def get_incremental_buffer(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve buffered documents for a session from Redis.
    
    Returns:
        Dict with 'docs' (list of doc dicts), 'last_query' (str), 'created_at' (float)
        or None if no buffer exists
    """
    if not app.state.redis:
        return None
    
    try:
        buffer_key = f"{INCREMENTAL_BUFFER_PREFIX}{session_id}"
        data = await app.state.redis.get(buffer_key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.warning(f"⚠️ Failed to get incremental buffer for {session_id}: {e}")
        return None


async def set_incremental_buffer(session_id: str, docs: List[Dict], query: str) -> bool:
    """
    Store/update buffered documents for a session in Redis.
    Merges with existing docs (deduplicates by source).
    
    Args:
        session_id: Session identifier
        docs: List of document dicts with 'text', 'metadata', 'similarity'
        query: The query text that triggered this retrieval
        
    Returns:
        True if successful, False otherwise
    """
    if not app.state.redis:
        return False
    
    try:
        buffer_key = f"{INCREMENTAL_BUFFER_PREFIX}{session_id}"
        
        # Get existing buffer
        existing = await get_incremental_buffer(session_id)
        
        if existing:
            # Merge docs (deduplicate by source)
            existing_sources = set(d.get('metadata', {}).get('source', '') for d in existing.get('docs', []))
            merged_docs = existing.get('docs', [])
            
            for doc in docs:
                source = doc.get('metadata', {}).get('source', '')
                if source and source not in existing_sources:
                    merged_docs.append(doc)
                    existing_sources.add(source)
            
            buffer_data = {
                'docs': merged_docs,
                'last_query': query,
                'created_at': existing.get('created_at', time.time()),
                'updated_at': time.time()
            }
        else:
            # New buffer
            buffer_data = {
                'docs': docs,
                'last_query': query,
                'created_at': time.time(),
                'updated_at': time.time()
            }
        
        # Store with TTL
        await app.state.redis.setex(
            buffer_key,
            INCREMENTAL_BUFFER_TTL,
            json.dumps(buffer_data)
        )
        
        logger.debug(f"📦 Buffer updated for {session_id}: {len(buffer_data['docs'])} docs")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to set incremental buffer for {session_id}: {e}")
        return False


async def clear_incremental_buffer(session_id: str) -> bool:
    """Clear the incremental buffer for a session."""
    if not app.state.redis:
        return False
    
    try:
        buffer_key = f"{INCREMENTAL_BUFFER_PREFIX}{session_id}"
        await app.state.redis.delete(buffer_key)
        logger.debug(f"🗑️ Buffer cleared for {session_id}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Failed to clear incremental buffer for {session_id}: {e}")
        return False


# ============================================================================
# TRUE PARALLEL INCREMENTAL PROCESSING FUNCTIONS
# ============================================================================

INCREMENTAL_BUFFER_PREFIX = "incremental:buffer:"
INCREMENTAL_LOCK_PREFIX = "incremental:lock:"
INCREMENTAL_CHUNK_PREFIX = "incremental:chunk:"

# ============================================================================
# PRE-LLM ACCUMULATION HELPER FUNCTIONS
# ============================================================================

def detect_query_pattern(text: str) -> dict:
    """
    Fast pattern detection (1-5ms) using keyword matching.
    Aligned with T.A.S.K 12-category knowledge base structure.
    Returns pattern type and confidence score.
    """
    patterns = {
        # Category 01: Organization Overview
        'organization_info': ['what is task', 'about task', 'task organization', 'telangana academy', 
                              'mission', 'vision', 'history', 'established', 'టాస్క్ అంటే ఏమిటి', 
                              't.a.s.k', 'task అంటే', 'overview', 'introduction'],
        
        # Category 02: Student Categories / Eligibility
        'eligibility': ['eligibility', 'who can join', 'engineering students', 'polytechnic', 
                        'graduates', 'unemployed', 'ఎలిజిబిలిటీ', 'ఎవరు చేరవచ్చు', 'criteria',
                        'qualification', 'percentage', 'age limit', 'branches', 'year'],
        
        # Category 03: Programs & Courses
        'programs': ['programs', 'courses', 'training', 'AI', 'cloud', 'cybersecurity', 
                     'data science', 'ప్రోగ్రామ్స్', 'కోర్సులు', 'full stack', 'java',
                     'machine learning', 'aws', 'azure', 'blockchain', 'duration'],
        
        # Category 04: Industry Partners
        'partners': ['partners', 'companies', 'collaboration', 'industry', 'aws', 'microsoft',
                     'ibm', 'google', 'infosys', 'tcs', 'కంపెనీలు', 'partnership'],
        
        # Category 05: Registration
        'registration': ['register', 'registration', 'enrollment', 'how to apply', 'documents',
                         'నమోదు', 'ఎలా నమోదు', 'apply', 'sign up', 'join', 'portal',
                         'account', 'login', 'otp', 'verification'],
        
        # Category 06: Placement
        'placement': ['placement', 'job', 'salary', 'career', 'companies', 'hiring',
                      'ప్లేస్మెంట్', 'సెలరీ', 'lpa', 'package', 'internship', 'employment',
                      'placement rate', 'average salary', 'highest package'],
        
        # Category 07: Financial / Fees
        'fees': ['fee', 'cost', 'price', 'scholarship', 'payment', 'free', 'subsidy',
                 'ఫీజు', 'ఎంత', 'స్కాలర్షిప్', 'financial assistance', 'discount',
                 'upi', 'netbanking', 'paid', 'charges'],
        
        # Category 08: Contact
        'contact': ['contact', 'phone', 'email', 'address', 'location', 'office hours',
                    'సంప్రదించు', 'ఫోన్', 'చిరునామా', 'సమయం', 'hyderabad', 'number',
                    'enquiry', 'support', 'helpline', 'whatsapp'],
        
        # Category 09: FAQ / Help
        'faq_help': ['help', 'support', 'problem', 'issue', 'question', 'faq',
                     'సహాయం', 'మద్దతు', 'సమస్య', 'forgot password', 'technical',
                     'how to', 'what if'],
        
        # Category 10: Achievements
        'achievements': ['achievement', 'recognition', 'award', 'statistics', 'impact',
                         'students trained', 'success', 'milestone', 'reach'],
        
        # Category 11: Compliance
        'compliance': ['dpdp', 'privacy', 'data protection', 'security', 'compliance',
                       'certification', 'audit', 'consent'],
        
        # Category 12: News
        'news': ['news', 'announcement', 'update', 'latest', 'new program', 'event',
                 'వార్తలు', 'announcement'],
        
        # General greeting
        'general': ['hi', 'hello', 'namaste', 'నమస్కారం', 'thanks', 'goodbye', 'bye']
    }
    
    text_lower = text.lower()
    best_pattern = 'general'
    best_score = 0.0
    
    for pattern_name, keywords in patterns.items():
        if not keywords:
            continue
        matches = sum(1 for kw in keywords if kw in text_lower)
        score = matches / len(keywords) if keywords else 0
        if score > best_score:
            best_score = score
            best_pattern = pattern_name
    
    return {
        'type': best_pattern,
        'confidence': best_score,
        'detected_keywords': [kw for kw in patterns.get(best_pattern, []) if kw in text_lower]
    }


def extract_template_fields(text: str, docs: list, pattern: dict) -> dict:
    """
    Rule-based information extraction (10-50ms).
    Extracts dates, times, names, and other relevant fields.
    """
    import re as regex_module
    
    extracted = {
        'dates': [],
        'times': [],
        'names': [],
        'numbers': [],
        'entities': []
    }
    
    # Date patterns
    date_patterns = [
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        r'\b(today|tomorrow|yesterday)\b',
        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}\b'
    ]
    
    # Time patterns
    time_patterns = [
        r'\b(\d{1,2}:\d{2})\b',
        r'\b(\d{1,2}\s*(am|pm))\b',
        r'\b(morning|afternoon|evening)\b'
    ]
    
    text_lower = text.lower()
    
    for pattern_re in date_patterns:
        matches = regex_module.findall(pattern_re, text_lower, regex_module.IGNORECASE)
        if matches:
            for m in matches:
                if isinstance(m, tuple):
                    extracted['dates'].append(m[0])
                else:
                    extracted['dates'].append(m)
    
    for pattern_re in time_patterns:
        matches = regex_module.findall(pattern_re, text_lower, regex_module.IGNORECASE)
        if matches:
            for m in matches:
                if isinstance(m, tuple):
                    extracted['times'].append(m[0])
                else:
                    extracted['times'].append(m)
    
    # Extract entities from docs if available
    if docs:
        for doc in docs[:3]:  # Limit to top 3 docs
            content = doc.get('content', doc.get('chunk_text', doc.get('text', '')))
            # Extract any capitalized words that might be names/entities
            entities = regex_module.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content)
            extracted['entities'].extend(entities[:5])
    
    return extracted


def build_incremental_prompt(accumulated_text: str, extracted_info: dict, pattern: dict, context: str) -> str:
    """
    Build prompt incrementally based on pattern and extracted info.
    Aligned with T.A.S.K knowledge base categories for humanized Telugu-English responses.
    """
    pattern_type = pattern.get('type', 'general')
    
    # T.A.S.K-specific prompt templates with Telugu-English mixed style
    templates = {
        'organization_info': """మీరు TARA, T.A.S.K (Telangana Academy for Skill and Knowledge) customer service assistant.
User Query: {query}

About T.A.S.K:
{context}

Instructions:
- T.A.S.K అంటే Telangana Academy for Skill and Knowledge, Government of Telangana ITE&C Department establish చేసిన not-for-profit organization అని explain చేయండి.
- Mission: Youth skilling and employability improvement అని mention చేయండి.
- 85%+ placement rate, 50,000+ students trained, 80+ industry partners highlight చేయండి.
- Respond in friendly Telugu-English mixed style.""",

        'eligibility': """మీరు TARA, T.A.S.K customer service assistant.
User Query: {query}

Eligibility Information:
{context}

Instructions:
- Engineering students (any year, any branch, 50% aggregate) eligible అని చెప్పండి.
- Polytechnic students (50-60% aggregate) eligible అని mention చేయండి.
- Graduates (60% in 10th, Inter, Graduation) eligible అని explain చేయండి.
- Unemployed youth 18-35 years కూడా apply చేయవచ్చు అని add చేయండి.
- Friendly Telugu-English mixed style లో respond చేయండి.""",

        'programs': """మీరు TARA, T.A.S.K customer service assistant.
User Query: {query}

Program Information:
{context}

Instructions:
- T.A.S.K 50+ programs offer చేస్తుంది: AI & ML, Cloud Computing (AWS, Azure, GCP), Cybersecurity, Data Science, Full Stack Development, Java, etc.
- Program durations vary: Short (40-60 hrs), Medium (60-80 hrs), Long (3-4 months) అని explain చేయండి.
- Online, offline, and hybrid formats available అని mention చేయండి.
- Certifications from AWS, Microsoft, Google available అని highlight చేయండి.
- Telugu-English mixed style లో respond చేయండి.""",

        'partners': """మీరు TARA, T.A.S.K customer service assistant.
User Query: {query}

Industry Partners:
{context}

Instructions:
- T.A.S.K has 80+ industry partners అని mention చేయండి.
- Major partners: Infosys, TCS, IBM, Microsoft, Google, Amazon, AWS, Cognizant, Wipro, HCL అని list చేయండి.
- Tri-partite model: Government-Academia-Industry collaboration అని explain చేయండి.
- Telugu-English mixed style లో respond చేయండి.""",

        'registration': """మీరు TARA, T.A.S.K customer service assistant.
User Query: {query}

Registration Process:
{context}

Instructions:
- Step-by-step registration guide ఇవ్వండి:
  1. Visit https://www.task.telangana.gov.in
  2. Click "Register" or "Student Login"
  3. Create account with email and phone OTP
  4. Fill personal and academic details
  5. Select programs (primary + 3 secondary)
  6. Upload documents (College ID, Aadhar, Photo)
  7. Complete payment if applicable
- Registration 15-30 minutes తీసుకుంటుంది అని mention చేయండి.
- Support: registrations_task@telangana.gov.in, +91-40-35485290 అని provide చేయండి.""",

        'placement': """మీరు TARA, T.A.S.K customer service assistant.
User Query: {query}

Placement Information:
{context}

Instructions:
- Overall placement rate: 85%+ అని strongly highlight చేయండి.
- Average salaries: IT Rs.3.5-6.5 LPA, Cybersecurity Rs.4-5.5 LPA, Cloud Rs.5.5-6.5 LPA అని mention చేయండి.
- Highest package: Rs.11.5 LPA (Verisk Analytics) అని highlight చేయండి.
- Top recruiters: Infosys, TCS, IBM, Microsoft, Google, Amazon అని list చేయండి.
- Placement support: Resume building, interview prep, career counseling available అని add చేయండి.
- Telugu-English mixed style లో respond చేయండి.""",

        'fees': """మీరు TARA, T.A.S.K customer service assistant.
User Query: {query}

Cost Information:
{context}

Instructions:
- Most programs FREE or heavily subsidized (40-83% discounts) అని strongly emphasize చేయండి.
- AWS training is completely FREE అని mention చేయండి.
- Government funding available for many programs అని highlight చేయండి.
- Payment methods: UPI, NetBanking, Debit/Credit Card, NEFT అని list చేయండి.
- Scholarships and financial assistance for economically backward students available అని add చేయండి.
- Telugu-English mixed style లో respond చేయండి.""",

        'contact': """మీరు TARA, T.A.S.K customer service assistant.
User Query: {query}

Contact Information:
{context}

Instructions:
- Contact details direct గా provide చేయండి:
  Phone: +91-40-35485290, +91-40-48488275
  General: enquiry_task@telangana.gov.in
  Registration: registrations_task@telangana.gov.in
  Address: 1st Floor, Sanketika Vidya Bhavan, Masabtank, Hyderabad - 500028
  Hours: Monday-Friday, 9:30 AM - 5:00 PM IST
  Website: https://www.task.telangana.gov.in
- 24/7 support via email and WhatsApp available అని mention చేయండి.
- Telugu-English mixed style లో respond చేయండి.""",

        'faq_help': """మీరు TARA, T.A.S.K customer service assistant.
User Query: {query}

Support Information:
{context}

Instructions:
- Issue solve చేయడానికి helpful response ఇవ్వండి.
- Common issues కి solutions provide చేయండి.
- Contact support options mention చేయండి: enquiry_task@telangana.gov.in, +91-40-35485290
- Student portal: https://www.task.telangana.gov.in/login అని provide చేయండి.
- Password reset available on login page అని mention చేయండి.
- Telugu-English mixed style లో respond చేయండి.""",

        'achievements': """మీరు TARA, T.A.S.K customer service assistant.
User Query: {query}

Achievement Information:
{context}

Instructions:
- Key statistics highlight చేయండి:
  50,000+ students trained (5 years)
  10,000+ currently registered students
  85%+ placement rate
  80+ industry partners
  770+ academic institutions connected
  Rs.300+ crores aggregate earnings impact
  200+ startups founded by trainees
- Government recognition and DPDP 2025 compliance mention చేయండి.
- Telugu-English mixed style లో respond చేయండి.""",

        'compliance': """మీరు TARA, T.A.S.K customer service assistant.
User Query: {query}

Compliance Information:
{context}

Instructions:
- T.A.S.K is DPDP 2025 compliant అని confirm చేయండి.
- Data privacy and security measures explain చేయండి.
- User consent required for all data usage అని mention చేయండి.
- SSL/TLS encryption, regular security audits అని highlight చేయండి.
- Telugu-English mixed style లో respond చేయండి.""",

        'news': """మీరు TARA, T.A.S.K customer service assistant.
User Query: {query}

Latest Updates:
{context}

Instructions:
- Latest announcements and updates share చేయండి.
- Website announcements page: https://task.telangana.gov.in/announcements/ mention చేయండి.
- New programs and events highlight చేయండి.
- Telugu-English mixed style లో respond చేయండి.""",

        'general': """మీరు TARA, T.A.S.K (Telangana Academy for Skill and Knowledge) customer service assistant.
User Query: {query}

Context:
{context}

Instructions:
- Friendly, helpful response ఇవ్వండి.
- User ని T.A.S.K programs, registration, or support వైపు guide చేయండి if relevant.
- Telugu-English mixed style (Tenglish) లో respond చేయండి.
- Contact: +91-40-35485290, enquiry_task@telangana.gov.in always mention if they need more help."""
    }
    
    template = templates.get(pattern_type, templates['general'])
    
    return template.format(
        query=accumulated_text,
        dates=', '.join(extracted_info.get('dates', [])) or 'None detected',
        times=', '.join(extracted_info.get('times', [])) or 'None detected',
        context=context[:2000] if context else 'No additional context available'
    )


async def accumulate_pre_llm_context(session_id: str, chunk_text: str, is_final: bool = False) -> dict:
    """
    Main accumulation function called for each chunk.
    Performs all pre-LLM processing and stores in buffer.
    """
    import re as regex_module
    accumulation_start_time = time.time()
    
    # Get existing buffer
    buffer_key = f"{INCREMENTAL_BUFFER_PREFIX}{session_id}"
    logger.info(f"[PRE-LLM DEBUG] accumulate_pre_llm_context called for session: {session_id}")
    logger.info(f"[PRE-LLM DEBUG] Buffer key: {buffer_key}")
    logger.info(f"[PRE-LLM DEBUG] Chunk text: '{chunk_text[:50]}...' (is_final={is_final})")
    existing_buffer_raw = await app.state.redis.get(buffer_key)
    
    if existing_buffer_raw:
        buffer_data = json.loads(existing_buffer_raw)
    else:
        buffer_data = {
            'chunks': [],
            'accumulated_text': '',
            'relevance_scores': [],
            'context': '',
            'pattern': None,
            'extracted_info': {},
            'prompt_components': [],
            'pre_built_prompt': None,
            'processing_metadata': {
                'chunk_count': 0,
                'last_pattern_update': None,
                'confidence_score': 0.0
            },
            'docs': [],
            'session_id': session_id,
            'last_updated': time.time()
        }
    
    # Accumulate text
    if 'chunks' not in buffer_data:
        buffer_data['chunks'] = []
    buffer_data['chunks'].append(chunk_text)
    buffer_data['accumulated_text'] = ' '.join(buffer_data['chunks'])
    
    if 'processing_metadata' not in buffer_data:
        buffer_data['processing_metadata'] = {'chunk_count': 0, 'last_pattern_update': None, 'confidence_score': 0.0}
    buffer_data['processing_metadata']['chunk_count'] += 1
    
    # 1. PATTERN DETECTION (Fast: 1-5ms)
    pattern = detect_query_pattern(buffer_data['accumulated_text'])
    buffer_data['pattern'] = pattern
    buffer_data['processing_metadata']['confidence_score'] = pattern['confidence']
    
    # 2. DOCUMENT RETRIEVAL (50-200ms) - Only if text changed significantly
    accumulated_text = buffer_data['accumulated_text']
    if len(accumulated_text) > 10:  # Only retrieve if we have enough text
        try:
            # Use existing parallel retrieval
            docs = await retrieve_documents_parallel(accumulated_text, None)
            
            # Build context from docs
            context_parts = []
            for i, doc in enumerate(docs[:5]):
                content = doc.get('content', doc.get('chunk_text', doc.get('text', '')))
                score = doc.get('similarity', doc.get('score', 0))
                context_parts.append(f"[Doc {i+1}, Score: {score:.2f}]: {content[:500]}")
            
            buffer_data['context'] = '\n\n'.join(context_parts)
            buffer_data['relevance_scores'] = [doc.get('similarity', doc.get('score', 0)) for doc in docs[:5]]
            
            # Merge with existing docs using smart buffering
            if 'docs' not in buffer_data:
                buffer_data['docs'] = []
            buffer_data['docs'] = merge_and_deduplicate_docs(buffer_data.get('docs', []), docs)
            
        except Exception as e:
            logger.warning(f"Document retrieval failed during accumulation: {e}")
    
    # 3. INFORMATION EXTRACTION (10-50ms)
    docs_for_extraction = buffer_data.get('docs', [])
    buffer_data['extracted_info'] = extract_template_fields(
        accumulated_text,
        docs_for_extraction,
        pattern
    )
    
    # 4. PROMPT CONSTRUCTION (5-15ms)
    buffer_data['pre_built_prompt'] = build_incremental_prompt(
        accumulated_text,
        buffer_data['extracted_info'],
        pattern,
        buffer_data['context']
    )
    
    # Store updated buffer
    buffer_data['processing_metadata']['last_pattern_update'] = time.time()
    buffer_data['last_updated'] = time.time()
    redis_set_result = await app.state.redis.setex(buffer_key, 300, json.dumps(buffer_data, ensure_ascii=False))  # 5 min TTL
    
    processing_time = (time.time() - accumulation_start_time) * 1000
    logger.info(f"[PRE-LLM DEBUG] Redis SET result: {redis_set_result}")
    logger.info(f"[PRE-LLM DEBUG] Stored pre_built_prompt: {bool(buffer_data.get('pre_built_prompt'))}")
    logger.info(f"[PRE-LLM DEBUG] Prompt length: {len(buffer_data.get('pre_built_prompt', '')) if buffer_data.get('pre_built_prompt') else 0}")
    logger.info(f"🔄 Pre-LLM accumulation completed in {processing_time:.1f}ms for session {session_id}")
    
    return buffer_data


async def generate_with_prompt(prompt: str):
    """
    Direct LLM generation with pre-built prompt.
    Yields streaming text chunks.
    Includes TTFT (Time To First Token) timing for latency analysis.
    """
    try:
        import google.generativeai as genai
        
        # Detailed timing for Gemini API call
        api_start = time.perf_counter()
        first_token_time = None
        chunk_count = 0
        total_chars = 0
        
        # Use existing Gemini model from app state
        # Reduced max_output_tokens from 1024 to 512 for faster generation
        response = app.state.rag_engine.gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=150,  # ~200-300 chars for 2-3 sentences
            ),
            stream=True
        )
        
        for chunk in response:
            if hasattr(chunk, 'text') and chunk.text:
                chunk_count += 1
                total_chars += len(chunk.text)
                
                # Log time to first token (TTFT)
                if first_token_time is None:
                    first_token_time = (time.perf_counter() - api_start) * 1000
                    logger.info(f"⏱️ GEMINI TTFT (Time To First Token): {first_token_time:.1f}ms")
                
                yield chunk.text
        
        # Log total streaming time
        total_time = (time.perf_counter() - api_start) * 1000
        ttft = first_token_time or 0
        logger.info(f"⏱️ GEMINI STREAMING COMPLETE: {total_time:.1f}ms total, {chunk_count} chunks, {total_chars} chars")
        logger.info(f"⏱️ GEMINI BREAKDOWN: TTFT={ttft:.1f}ms, StreamingRest={total_time - ttft:.1f}ms")
                
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        yield f"Error generating response: {str(e)}"


async def process_chunk_parallel(session_id: str, chunk_text: str, context: Optional[Dict] = None, sequence_number: int = 0):
    """
    Process a single chunk in true parallel fashion.
    No blocking, no waiting - pure async processing.
    """
    try:
        chunk_id = f"{session_id}:{sequence_number}:{hash(chunk_text)}"
        lock_key = f"{INCREMENTAL_LOCK_PREFIX}{chunk_id}"

        # Check if this exact chunk is already being processed
        if await app.state.redis.get(lock_key):
            logger.debug(f"⚠️ Chunk {chunk_id} already processing - skipping")
            return

        # Set processing lock with short TTL
        await app.state.redis.setex(lock_key, 30, "processing")

        logger.info(f"🔄 Processing chunk {sequence_number} for session {session_id}")

        # Parallel document retrieval (async, non-blocking)
        docs = await retrieve_documents_parallel(chunk_text, context)

        # Smart buffering with deduplication
        await smart_buffer_documents(session_id, docs, chunk_text, sequence_number)

        # Clean up lock
        await app.state.redis.delete(lock_key)

        logger.info(f"✅ Chunk {sequence_number} processed: {len(docs)} docs buffered")

    except Exception as e:
        logger.error(f"❌ Parallel chunk processing error: {e}")
        # Clean up lock on error
        try:
            await app.state.redis.delete(lock_key)
        except:
            pass

async def retrieve_documents_parallel(query_text: str, context: Optional[Dict] = None) -> List[Dict]:
    """
    Parallel document retrieval optimized for Telugu text and incremental chunks.
    """
    try:
        # Telugu-aware query enrichment
        enriched_query = enrich_telugu_query(query_text, context)

        # Async embedding (runs in thread pool)
        loop = asyncio.get_running_loop()
        query_embedding = await loop.run_in_executor(
            None,
            lambda: app.state.rag_engine.embeddings.embed_query(enriched_query)
        )

        # Prepare for FAISS search
        import numpy as np
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Async FAISS search (runs in thread pool)
        distances, indices = await loop.run_in_executor(
            None,
            lambda: app.state.rag_engine.vector_store.search(query_embedding, k=app.state.rag_engine.config.top_k)
        )

        # Build document list with Telugu-aware processing
        docs = []
        for i, idx in enumerate(indices[0]):
            if idx < len(app.state.rag_engine.documents):
                distance = float(distances[0][i])
                similarity = 1.0 - (distance * distance / 2.0)

                if similarity < app.state.rag_engine.config.similarity_threshold:
                    continue

                doc_text = app.state.rag_engine.documents[idx]
                doc_meta = app.state.rag_engine.doc_metadata[idx] if idx < len(app.state.rag_engine.doc_metadata) else {}

                docs.append({
                    'text': doc_text,
                    'metadata': doc_meta,
                    'similarity': similarity,
                    'chunk_contribution': query_text,  # Track which chunk found this
                    'embedding_distance': distance
                })

        return docs

    except Exception as e:
        logger.error(f"Document retrieval error: {e}")
        return []

def enrich_telugu_query(query_text: str, context: Optional[Dict] = None) -> str:
    """Enrich query with Telugu-specific processing"""
    enriched = query_text

    # Add Telugu-specific terms if context provided
    if context:
        # Add key entities
        if 'key_entities' in context:
            entities = context['key_entities']
            entity_terms = ' '.join([f"{k} {v}" for k, v in entities.items()])
            enriched = f"{enriched} {entity_terms}"

        # Add user goal for better context
        if 'user_goal' in context and context['user_goal']:
            enriched = f"{enriched} {context['user_goal']}"

    return enriched.strip()

async def smart_buffer_documents(session_id: str, new_docs: List[Dict], chunk_text: str, sequence_number: int):
    """
    Smart buffering with deduplication, merging, and ranking.
    Optimized for Telugu text and incremental chunks.
    """
    try:
        buffer_key = f"{INCREMENTAL_BUFFER_PREFIX}{session_id}"

        # Get existing buffer
        existing_buffer = await get_incremental_buffer(session_id)
        existing_docs = existing_buffer.get('docs', []) if existing_buffer else []

        # Merge and deduplicate documents
        merged_docs = merge_and_deduplicate_docs(existing_docs, new_docs)

        # Sort by relevance (combine similarity scores)
        merged_docs.sort(key=lambda x: x.get('combined_similarity', x.get('similarity', 0)), reverse=True)

        # Limit buffer size to prevent memory bloat
        max_docs = min(len(merged_docs), 50)  # Configurable limit
        merged_docs = merged_docs[:max_docs]

        # Prepare buffer data
        buffer_data = {
            'session_id': session_id,
            'docs': merged_docs,
            'chunks_processed': existing_buffer.get('chunks_processed', 0) + 1 if existing_buffer else 1,
            'last_updated': time.time(),
            'total_chars': sum(len(d.get('text', '')) for d in merged_docs),
            'language': 'te-mixed',  # Telugu-aware
            'chunk_sequence': sequence_number
        }

        # Store in Redis with TTL (30 minutes for incremental sessions)
        await app.state.redis.setex(buffer_key, 1800, json.dumps(buffer_data, ensure_ascii=False))

        logger.debug(f"📦 Smart buffered: {len(merged_docs)} docs ({buffer_data['total_chars']} chars)")

    except Exception as e:
        logger.error(f"Smart buffering error: {e}")

def merge_and_deduplicate_docs(existing_docs: List[Dict], new_docs: List[Dict]) -> List[Dict]:
    """
    Merge document lists with intelligent deduplication.
    Combines similarity scores and preserves best matches.
    """
    # Create lookup by source document
    doc_lookup = {}

    # Process existing docs
    for doc in existing_docs:
        source = doc.get('metadata', {}).get('source', '')
        if source:
            doc_lookup[source] = doc

    # Process new docs and merge/deduplicate
    for new_doc in new_docs:
        source = new_doc.get('metadata', {}).get('source', '')

        if source in doc_lookup:
            # Merge: combine similarity scores
            existing = doc_lookup[source]
            existing_similarity = existing.get('similarity', 0)
            new_similarity = new_doc.get('similarity', 0)

            # Weighted average favoring higher similarity
            combined_similarity = max(existing_similarity, new_similarity)

            existing['combined_similarity'] = combined_similarity
            existing['chunks_contributed'] = existing.get('chunks_contributed', []) + [new_doc.get('chunk_contribution', '')]
        else:
            # New document
            new_doc['combined_similarity'] = new_doc.get('similarity', 0)
            new_doc['chunks_contributed'] = [new_doc.get('chunk_contribution', '')]
            doc_lookup[source] = new_doc

    return list(doc_lookup.values())

async def get_incremental_buffer(session_id: str) -> Optional[Dict]:
    """Get incremental buffer from Redis"""
    try:
        buffer_key = f"{INCREMENTAL_BUFFER_PREFIX}{session_id}"
        buffer_data = await app.state.redis.get(buffer_key)
        return json.loads(buffer_data) if buffer_data else None
    except Exception as e:
        logger.error(f"Buffer retrieval error: {e}")
        return None

async def clear_incremental_buffer(session_id: str) -> bool:
    """Clear the incremental buffer for a session."""
    try:
        buffer_key = f"{INCREMENTAL_BUFFER_PREFIX}{session_id}"
        await app.state.redis.delete(buffer_key)

        # Clear processing locks for this session (best effort)
        # Note: Redis SCAN is expensive, so we use a simple approach
        try:
            # Delete a few potential lock keys (not perfect but better than nothing)
            for i in range(10):  # Assume max 10 concurrent chunks
                lock_key = f"{INCREMENTAL_LOCK_PREFIX}{session_id}:{i}:*"
                # Redis doesn't support wildcards in DELETE, so we skip detailed cleanup
                pass
        except:
            pass

        logger.debug(f"🗑️ Smart buffer cleared for {session_id}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Failed to clear incremental buffer for {session_id}: {e}")
        return False

def build_telugu_context(buffered_docs: List[Dict]) -> str:
    """Build Telugu-aware context from smart buffered documents"""
    context_parts = []

    for doc in buffered_docs:
        text = doc.get('text', '')
        similarity = doc.get('combined_similarity', doc.get('similarity', 0))
        chunks = doc.get('chunks_contributed', [])

        # Telugu-aware context building
        if similarity > 0.3:  # Only include relevant docs
            # Add Telugu language hints for better processing
            context_parts.append(f"[Telugu Context - Similarity: {similarity:.2f}]\n{text}")

    return "\n\n".join(context_parts)

def extract_sources_from_buffer(buffered_docs: List[Dict]) -> List[str]:
    """Extract unique sources from smart buffer"""
    sources = set()
    for doc in buffered_docs:
        source = doc.get('metadata', {}).get('source', 'Unknown')
        if source and source != 'Unknown':
            sources.add(source)

    return list(sources)

async def generate_with_buffered_context(query: str, context_docs: List[Dict],
                                       context_text: str, intent_context: Optional[Dict],
                                       streaming_callback) -> Dict:
    """Generate response using smart buffered context with Telugu optimization"""

    # Prepare enriched context for Telugu-aware generation
    enriched_context = intent_context.copy() if intent_context else {}
    enriched_context.update({
        'language': 'te-mixed',
        'buffered_docs_count': len(context_docs),
        'context_quality': 'high',  # Smart buffer provides high quality context
        'telugu_optimized': True
    })

    # Try to use process_query_with_context if available
    try:
        return await app.state.rag_engine.process_query_with_context(
            query=query,
            context_docs=context_docs,
            intent_context=enriched_context,
            streaming_callback=streaming_callback
        )
    except AttributeError:
        # Fallback: Use standard process_query with enriched context
        logger.info("Using standard process_query with smart context")

        # Add buffered context to the query for better results
        enriched_query = f"{query}\n\nContext from previous chunks:\n{context_text[:2000]}"  # Limit context size

        return await app.state.rag_engine.process_query(
            enriched_query,
            enriched_context,
            streaming_callback=streaming_callback
        )

# Redis client utilities
# Removed custom implementation in favor of shared client
# async def get_redis_client() -> redis.Redis: ...
# async def close_redis_client(client: redis.Redis): ...
# async def ping_redis(client: redis.Redis) -> bool: ...


async def prepopulate_cache():
    """Pre-populate Redis cache with common T.A.S.K queries aligned to 12 knowledge base categories"""
    if not app.state.redis:
        logger.info("⚠️ Redis not available - skipping cache prepopulation")
        return
        
    logger.info("🔥 Pre-populating query cache with T.A.S.K knowledge base queries...")
    
    # T.A.S.K-specific common queries aligned to 12 knowledge base categories
    common_queries = [
        # Category 01: Organization Information
        {"query": "what is task", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "టాస్క్ అంటే ఏమిటి", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "task about", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "telangana academy for skill and knowledge", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        
        # Category 02: Eligibility
        {"query": "eligibility criteria ఎలిజిబిలిటీ", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "who can join ఎవరు చేరవచ్చు", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "engineering students eligibility", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "polytechnic students eligibility", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "percentage required", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        
        # Category 03: Programs & Courses
        {"query": "programs available ప్రోగ్రామ్స్", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "courses offered కోర్సులు", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "AI machine learning program", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "cloud computing program AWS Azure", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "cybersecurity program", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "data science course", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "full stack development", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        
        # Category 04: Industry Partners
        {"query": "industry partners కంపెనీలు", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "companies collaboration", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        
        # Category 05: Registration
        {"query": "how to register ఎలా నమోదు", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "registration process నమోదు ప్రక్రియ", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "documents required పత్రాలు", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "how to apply అప్లై చేయడం ఎలా", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "registration portal", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        
        # Category 06: Placement
        {"query": "placement statistics ప్లేస్మెంట్", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "placement rate ప్లేస్మెంట్ రేట్", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "salary package సెలరీ", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "average salary LPA", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "companies hiring కంపెనీలు", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "job opportunities", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        
        # Category 07: Fees & Financial
        {"query": "fee structure ఫీజు", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "program cost ఎంత", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "scholarship available స్కాలర్షిప్", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "free programs ఫ్రీ", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "payment methods", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        
        # Category 08: Contact Information
        {"query": "contact information కాంటాక్ట్ ఇన్ఫర్మేషన్", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "phone number ఫోన్ నంబర్", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "email address ఇమెయిల్", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "office location address చిరునామా", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "office hours timing సమయం", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "hyderabad office", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        
        # Category 09: FAQ & Help
        {"query": "help సహాయం", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "support మద్దతు", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "forgot password", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "technical support", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        
        # Category 10: Achievements
        {"query": "students trained", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "success statistics", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        
        # Greeting patterns
        {"query": "hi hello", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
        {"query": "నమస్కారం", "context": {"language": "te-mixed", "organization": "T.A.S.K"}},
    ]
    
    for i, query_data in enumerate(common_queries):
        try:
            # Generate cache key
            cache_key = f"rag:{hashlib.md5(query_data['query'].encode()).hexdigest()}"
            
            # Check if already cached
            existing = await app.state.redis.get(cache_key)
            if existing:
                continue
                
            # Process query to populate cache
            result = await app.state.rag_engine.process_query(
                query_data['query'],
                query_data['context']
            )
            
            # Cache the result
            cache_data = {
                'answer': result['answer'],
                'sources': result['sources'],
                'confidence': result['confidence'],
                'timing_breakdown': result['timing_breakdown'],
                'metadata': result['metadata']
            }
            
            await app.state.redis.setex(
                cache_key,
                app.state.rag_engine.config.cache_ttl,
                json.dumps(cache_data)
            )
            
            logger.info(f"✅ Cached query {i+1}/{len(common_queries)}: {query_data['query'][:30]}...")
            
            # Small delay to avoid overwhelming
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to cache query '{query_data['query']}': {e}")
    
    logger.info("✅ Cache prepopulation complete")

async def optimize_connections():
    """Optimize connection pools for better performance"""
    
    # Gemini connection pooling
    try:
        import google.auth.transport.requests
        # Pre-warm HTTP connection pool
        transport = google.auth.transport.requests.Request()
        logger.info("✅ Gemini connection pool optimized")
    except:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup/shutdown."""
    global rag_engine, redis_client, cache_hits, cache_misses, app_start_time
    
    # Startup
    logger.info("🚀 Starting RAG service with prewarming...")
    app_start_time = time.time()
    
    try:
        # Load config
        config = RAGConfig.from_env()
        logger.info("📋 Configuration loaded")
        
        # Create RAG engine
        rag_engine = RAGEngine(config)
        
        # If index not loaded, try to build it
        if not rag_engine.vector_store or not rag_engine.documents:
            logger.warning("⚠️ FAISS index not found, attempting to build...")
            from leibniz_agent.services.rag.index_builder import IndexBuilder
            builder = IndexBuilder(config)
            if builder.build_index():
                rag_engine.load_index()  # Reload after build
                logger.info(f"✅ Index built successfully: {len(rag_engine.documents)} documents")
            else:
                logger.error("❌ Index build failed")
        
        # If still no index, log warning but continue (degraded mode)
        if not rag_engine.vector_store or not rag_engine.documents:
            logger.error("❌ FAISS index not available - service in degraded mode")
        else:
            logger.info(f"📚 RAG engine initialized: {len(rag_engine.documents)} documents")
        
        # Connect to Redis (optional - service can run without it)
        try:
            # Use shared client which handles config from env vars
            redis_client = await asyncio.wait_for(get_redis_client(), timeout=15.0)
            await asyncio.wait_for(ping_redis(redis_client), timeout=5.0)
            logger.info("🔗 Redis connected successfully")
        except asyncio.TimeoutError:
            logger.warning("⏰ Redis connection timeout - service will run in degraded mode")
            redis_client = None
        except Exception as redis_error:
            logger.warning(f"❌ Redis connection failed: {redis_error} - caching disabled")
            redis_client = None
        
        # Initialize counters
        cache_hits = 0
        cache_misses = 0
        
        # ===========================================
        # PHASE 1: COMPONENT PREWARMING
        # ===========================================
        if config.enable_prewarming:
            logger.info("=" * 70)
            logger.info("🚀 STARTING RAG PREWARMING SEQUENCE")
            logger.info("=" * 70)
            
            # Pre-warm embeddings
            await rag_engine.warmup_embeddings()
            
            # Pre-warm Gemini
            await rag_engine.warmup_gemini()
            
            # Enable model persistence
            if config.enable_model_persistence:
                rag_engine.enable_model_persistence()
            
            # Pre-compute patterns
            rag_engine.precompute_patterns()
            
            # Optimize connections
            await optimize_connections()
        
        # ===========================================
        # PHASE 2: CACHE PREPOPULATION
        # ===========================================
        if config.prepopulate_cache and redis_client:
            # Run in background to not block startup completely if it takes long
            asyncio.create_task(prepopulate_cache())
        
        # Store in app state
        app.state.rag_engine = rag_engine
        app.state.redis = redis_client
        app.state.cache_hits = cache_hits
        app.state.cache_misses = cache_misses
        app.state.start_time = app_start_time
        
        logger.info("=" * 70)
        logger.info("✅ RAG SERVICE FULLY PREWARMED AND READY")
        logger.info("=" * 70)
        
        yield
        
        # Shutdown
        logger.info(" Shutting down RAG service...")
        
        # Log performance stats
        if rag_engine:
            stats = rag_engine.get_performance_stats()
            logger.info(f" Performance stats: {stats}")
        
        # Close Redis
        if redis_client:
            # Shared client manages its own lifecycle, but we can close our reference
            # Actually, shared client is singleton, so we shouldn't close it here if other services use it
            # But since this is microservice, we are the only user in this process.
            # However, close_redis_client in shared lib closes the global client.
            await close_redis_client() 
            logger.info(" Redis connection closed")
        
        logger.info(" RAG service shutdown complete")
    
    except Exception as e:
        logger.error(f" Startup error: {e}", exc_info=True)
        raise


# Create FastAPI app
app = FastAPI(
    title="Leibniz RAG Service",
    description="Context-aware knowledge base queries with FAISS retrieval and Gemini generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Process knowledge base query with context-aware retrieval.
    
    Uses Redis caching with 1-hour TTL. Returns answer, sources, confidence, and timing.
    """
    try:
        # Generate cache key
        cache_key = f"rag:{hashlib.md5(request.query.encode()).hexdigest()}"
        
        # Check Redis cache (only if connected)
        cached = None
        if app.state.redis:
            try:
                cached = await app.state.redis.get(cache_key)
            except Exception as cache_read_error:
                logger.warning(f" Cache read failed: {cache_read_error}")
                cached = None
        
        if cached:
            # Cache hit
            app.state.cache_hits += 1
            result = json.loads(cached)
            result['cached'] = True
            
            logger.info(f" CACHE HIT: {request.query[:50]}...")
            return QueryResponse(**result)
        
        # Cache miss
        app.state.cache_misses += 1
        
        # Process query
        result = await app.state.rag_engine.process_query(
            request.query,
            request.context,
            streaming_callback=None  # Streaming handled separately if needed
        )
        
        # Add cached flag
        result['cached'] = False
        
        # Cache result (only if Redis is available)
        if app.state.redis:
            try:
                await app.state.redis.setex(
                    cache_key,
                    app.state.rag_engine.config.cache_ttl,
                    json.dumps({
                        'answer': result['answer'],
                        'sources': result['sources'],
                        'confidence': result['confidence'],
                        'timing_breakdown': result['timing_breakdown'],
                        'metadata': result['metadata']
                    })
                )
            except Exception as cache_error:
                logger.warning(f"️ Cache write failed: {cache_error}")
        
        # Log query
        if app.state.rag_engine.config.log_queries:
            logger.info(
                f" QUERY: {request.query[:50]}... → "
                f"{result['confidence']:.2f} confidence, "
                f"{result['timing_breakdown']['total_ms']:.1f}ms"
            )
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f" Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/api/v1/stream_query")
async def stream_query_knowledge_base(request: QueryRequest):
    """
    Stream knowledge base query response.
    Returns a stream of JSON objects: {"text": "...", "is_final": bool}
    """
    async def event_generator():
        # Generate cache key
        cache_key = f"rag:{hashlib.md5(request.query.encode()).hexdigest()}"
        
        # Check Redis cache (only if connected)
        cached = None
        if app.state.redis:
            try:
                cached = await app.state.redis.get(cache_key)
            except Exception as cache_read_error:
                logger.warning(f" Cache read failed: {cache_read_error}")
                cached = None
        
        if cached:
            # Cache hit - simulate streaming
            app.state.cache_hits += 1
            result = json.loads(cached)
            answer = result.get('answer', '')
            
            logger.info(f" CACHE HIT (Streaming): {request.query[:50]}...")
            
            # Stream the cached answer in chunks to simulate natural typing/speech
            chunk_size = 20
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i+chunk_size]
                yield json.dumps({"text": chunk, "is_final": False}) + "\n"
                await asyncio.sleep(0.01)  # Small delay for realism
            
            # Final chunk
            yield json.dumps({"text": "", "is_final": True}) + "\n"
            return

        # Cache miss
        app.state.cache_misses += 1
        
        q = asyncio.Queue()
        loop = asyncio.get_running_loop()
        
        # Container for the full result to cache later
        full_result_container = {}
        
        def callback(text, is_final):
            loop.call_soon_threadsafe(q.put_nowait, (text, is_final))
            
        async def run_query():
            try:
                # Process query with streaming callback
                result = await app.state.rag_engine.process_query(
                    request.query,
                    request.context,
                    streaming_callback=callback
                )
                
                # Store result for caching
                full_result_container['data'] = result
                
            except Exception as e:
                logger.error(f"Streaming query error: {e}")
                loop.call_soon_threadsafe(q.put_nowait, (f"Error: {str(e)}", True))
            finally:
                await q.put(None) # Sentinel

        # Start query task
        asyncio.create_task(run_query())

        while True:
            item = await q.get()
            if item is None:
                break
            text, is_final = item
            yield json.dumps({"text": text, "is_final": is_final}) + "\n"
            
        # After streaming is done, cache the result if we have it
        if 'data' in full_result_container and app.state.redis:
            try:
                result = full_result_container['data']
                await app.state.redis.setex(
                    cache_key,
                    app.state.rag_engine.config.cache_ttl,
                    json.dumps({
                        'answer': result['answer'],
                        'sources': result['sources'],
                        'confidence': result['confidence'],
                        'timing_breakdown': result['timing_breakdown'],
                        'metadata': result['metadata']
                    })
                )
                logger.info(f" Cached streamed result for: {request.query[:30]}...")
            except Exception as cache_error:
                logger.warning(f"️ Cache write failed: {cache_error}")

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


# =============================================================================
# INCREMENTAL RAG ENDPOINT (Buffered Retrieval + Streaming Generation)
# =============================================================================
@app.post("/api/v1/query/incremental")
async def incremental_query(request: IncrementalQueryRequest):
    """
    TRUE PARALLEL Incremental RAG query with smart Redis buffering.

    Workflow:
    - is_final=False: Parallel document retrieval and smart buffering (no blocking)
    - is_final=True: Generate response using all buffered context

    Features:
    - True parallel processing (no sequential blocking)
    - Smart document deduplication and merging
    - Redis session buffering with TTL
    - Telugu text support
    - Async processing with background tasks
    """
    start_time = time.time()

    try:
        if not request.is_final:
            # =================================================================
            # PRE-LLM ACCUMULATION PHASE - Do all pre-LLM work during speech
            # =================================================================
            logger.info(f"⚡ PRE-LLM ACCUMULATION: Processing chunk for {request.session_id}: '{request.text[:30]}...'")

            # Fire-and-forget: Start background pre-LLM accumulation
            # This does pattern detection, retrieval, extraction, and prompt building
            asyncio.create_task(
                accumulate_pre_llm_context(
                    session_id=request.session_id,
                    chunk_text=request.text,
                    is_final=False
                )
            )

            # Return immediately - don't wait for processing
            return IncrementalBufferResponse(
                status="buffering",
                session_id=request.session_id,
                docs_retrieved=0,  # Will be updated asynchronously
                buffer_size_chars=0,  # Will be updated asynchronously
                timing_ms=(time.time() - start_time) * 1000,
                message="Chunk queued for pre-LLM accumulation"
            )
        
        else:
            # =================================================================
            # TRUE PARALLEL GENERATION PHASE: Use smart buffered context
            # =================================================================
            logger.info(f"🚀 TRUE PARALLEL: Final generation for {request.session_id}: '{request.text[:30]}...'")

            async def generate_from_smart_buffer():
                """
                OPTIMIZED: Generate response using pre-accumulated context.
                Fast path when pre-built prompt is available.
                """
                gen_start = time.time()

                # Get smart buffered documents with pre-computed data
                buffer_key_debug = f"{INCREMENTAL_BUFFER_PREFIX}{request.session_id}"
                logger.info(f"[PRE-LLM DEBUG] Final generation - Looking for cached prompt")
                logger.info(f"[PRE-LLM DEBUG] Session ID: {request.session_id}")
                logger.info(f"[PRE-LLM DEBUG] Buffer key being searched: {buffer_key_debug}")
                
                buffer = await get_incremental_buffer(request.session_id)
                logger.info(f"[PRE-LLM DEBUG] Buffer found: {buffer is not None}")
                
                if buffer:
                    buffered_docs = buffer.get('docs', [])
                    chunks_processed = buffer.get('chunks_processed', buffer.get('processing_metadata', {}).get('chunk_count', 0))
                    total_chars = buffer.get('total_chars', len(buffer.get('accumulated_text', '')))
                    pre_built_prompt = buffer.get('pre_built_prompt')
                    
                    logger.info(f"[PRE-LLM DEBUG] Pre-built prompt found: {bool(pre_built_prompt)}")
                    logger.info(f"[PRE-LLM DEBUG] Prompt length: {len(pre_built_prompt) if pre_built_prompt else 0}")
                    logger.info(f"📊 Using smart buffer: {len(buffered_docs)} docs, {chunks_processed} chunks, pre-built prompt: {bool(pre_built_prompt)}")
                else:
                    buffered_docs = []
                    pre_built_prompt = None
                    logger.warning(f"[PRE-LLM DEBUG] ⚠️ No smart buffer found for {request.session_id}")

                # FAST PATH: Use pre-built prompt if available
                if pre_built_prompt:
                    prep_time = (time.time() - gen_start) * 1000
                    logger.info(f"[PRE-LLM DEBUG] ✅ FAST PATH ACTIVATED - Using pre-built prompt!")
                    logger.info(f"🚀 FAST PATH: Using pre-built prompt, prep time: {prep_time:.1f}ms")
                    
                    # Check if final query differs from accumulated text
                    accumulated_text = buffer.get('accumulated_text', '')
                    if request.text and request.text.strip() != accumulated_text.strip():
                        # Re-build prompt with final query (quick operation)
                        logger.info("Re-building prompt with final query text")
                        pattern = buffer.get('pattern', {'type': 'general'})
                        extracted_info = buffer.get('extracted_info', {})
                        context = buffer.get('context', '')
                        pre_built_prompt = build_incremental_prompt(
                            request.text,
                            extracted_info,
                            pattern,
                            context
                        )
                    
                    # Direct LLM generation with pre-built prompt
                    # First, send a metadata chunk indicating fast path
                    yield json.dumps({"text": "", "is_final": False, "cached": True, "fast_path": True}) + "\n"
                    
                    async for chunk in generate_with_prompt(pre_built_prompt):
                        yield json.dumps({"text": chunk, "is_final": False}) + "\n"
                    
                    yield json.dumps({"text": "", "is_final": True, "cached": True}) + "\n"
                    
                    # Clean up smart buffer after generation
                    await clear_incremental_buffer(request.session_id)
                    
                    gen_ms = (time.time() - gen_start) * 1000
                    logger.info(f"✅ FAST PATH generation complete for {request.session_id} in {gen_ms:.0f}ms")
                    return

                # FALLBACK: Standard processing if no pre-built prompt
                logger.info("[PRE-LLM DEBUG] ❌ FALLBACK PATH - No pre-built prompt available")
                logger.info("⚠️ FALLBACK: No pre-built prompt, using standard processing")
                
                # Parallel generation with streaming
                q = asyncio.Queue()
                loop = asyncio.get_running_loop()
                full_result = {}

                def streaming_callback(text, is_final):
                    """Streaming callback for incremental text generation"""
                    loop.call_soon_threadsafe(q.put_nowait, (text, is_final))

                async def run_parallel_generation():
                    try:
                        # Use smart buffered context for Telugu-aware generation
                        if buffered_docs:
                            # Build enriched context from smart buffer
                            context_docs = buffered_docs[:app.state.rag_engine.config.top_n]
                            context_text = build_telugu_context(context_docs)
                            sources = extract_sources_from_buffer(buffered_docs)

                            logger.info(f"🌐 Generating with {len(context_docs)} smart buffered docs")

                            # Telugu-aware generation with pre-buffered context
                            result = await generate_with_buffered_context(
                                query=request.text,
                                context_docs=context_docs,
                                context_text=context_text,
                                intent_context=request.context,
                                streaming_callback=streaming_callback
                            )
                        else:
                            # Fallback to standard generation
                            logger.warning("Falling back to standard generation (no buffer)")
                            result = await app.state.rag_engine.process_query(
                                request.text,
                                request.context,
                                streaming_callback=streaming_callback
                            )

                        full_result['data'] = result

                    except Exception as e:
                        logger.error(f"❌ Parallel generation error: {e}")
                        loop.call_soon_threadsafe(q.put_nowait, (f"తప్పు: {str(e)}", True))
                    finally:
                        await q.put(None)  # Sentinel

                # Start parallel generation immediately
                asyncio.create_task(run_parallel_generation())

                # Stream results as they become available
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    text, is_final = item
                    yield json.dumps({"text": text, "is_final": is_final}) + "\n"

                # Clean up smart buffer after generation
                await clear_incremental_buffer(request.session_id)

                gen_ms = (time.time() - gen_start) * 1000
                logger.info(f"✅ TRUE PARALLEL generation complete for {request.session_id} in {gen_ms:.0f}ms")

            return StreamingResponse(generate_from_smart_buffer(), media_type="application/x-ndjson")
    
    except Exception as e:
        logger.error(f"❌ Incremental query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Incremental query failed: {str(e)}")



@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Service health check.
    
    Returns index status, cache hit rate, Redis/Gemini availability, and uptime.
    """
    try:
        # Calculate cache hit rate
        total_requests = app.state.cache_hits + app.state.cache_misses
        cache_hit_rate = app.state.cache_hits / total_requests if total_requests > 0 else 0.0
        
        # Check Redis health (if available)
        redis_connected = False
        if app.state.redis:
            redis_connected = await ping_redis(app.state.redis)
        
        # Get RAG engine stats
        index_loaded = app.state.rag_engine.vector_store is not None
        index_size = len(app.state.rag_engine.documents)
        gemini_available = app.state.rag_engine.gemini_model is not None
        
        # Calculate uptime
        uptime_seconds = time.time() - app.state.start_time
        
        # Determine status
        if not index_loaded:
            status = "unhealthy"
            status_code = 503
        elif not redis_connected:
            status = "degraded"
            status_code = 200
        else:
            status = "healthy"
            status_code = 200
        
        return HealthResponse(
            status=status,
            index_loaded=index_loaded,
            index_size=index_size,
            cache_hit_rate=cache_hit_rate,
            redis_connected=redis_connected,
            gemini_available=gemini_available,
            uptime_seconds=uptime_seconds
        )
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/performance")
async def performance_metrics():
    """Get detailed performance metrics"""
    
    warmup_status = {
        "embeddings_warmed": hasattr(app.state.rag_engine, 'warmup_embeddings_cache'),
        "patterns_cached": len(getattr(app.state.rag_engine, 'pattern_cache', {})),
        "model_persistence": app.state.rag_engine.config.enable_model_persistence
    }
    
    # Calculate cache hit rate
    total_requests = app.state.cache_hits + app.state.cache_misses
    cache_hit_rate = app.state.cache_hits / total_requests if total_requests > 0 else 0.0
    
    # Get pattern match rate (if tracked)
    pattern_match_rate = getattr(app.state.rag_engine, 'pattern_match_rate', 0.0)
    
    return {
        "warmup_status": warmup_status,
        "query_performance": {
            "total_queries": total_requests,
            "cache_hit_rate": round(cache_hit_rate, 3),
            "pattern_match_rate": round(pattern_match_rate, 3),
            "average_response_time_ms": getattr(app.state.rag_engine, 'avg_response_time', 0)
        },
        "cache_status": {
            "hits": app.state.cache_hits,
            "misses": app.state.cache_misses,
            "redis_connected": app.state.redis is not None
        },
        "uptime_seconds": time.time() - app.state.start_time
    }


@app.get("/metrics")
async def get_metrics():
    """
    Detailed performance metrics.
    
    Returns RAG engine stats, cache stats, and index stats.
    """
    try:
        # RAG engine stats
        rag_stats = app.state.rag_engine.get_performance_stats()
        
        # Cache stats
        total_requests = app.state.cache_hits + app.state.cache_misses
        cache_stats = {
            'cache_hits': app.state.cache_hits,
            'cache_misses': app.state.cache_misses,
            'cache_hit_rate': app.state.cache_hits / total_requests if total_requests > 0 else 0.0
        }
        
        # Index stats
        index_stats = {
            'total_documents': len(app.state.rag_engine.documents),
            'categories': len(set(m.get('category', '') for m in app.state.rag_engine.doc_metadata)),
            'embedding_dimension': app.state.rag_engine.vector_store.d if app.state.rag_engine.vector_store else 0
        }
        
        # Uptime
        uptime_seconds = time.time() - app.state.start_time
        
        return {
            'rag_engine': rag_stats,
            'cache': cache_stats,
            'index': index_stats,
            'uptime_seconds': uptime_seconds
        }
    
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@app.post("/api/v1/admin/rebuild_index", response_model=RebuildIndexResponse)
async def rebuild_index(request: RebuildIndexRequest):
    """
    Rebuild FAISS index from knowledge base (admin endpoint).
    
    Useful for knowledge base updates. Clears cache after rebuild.
    """
    try:
        build_start = time.time()
        
        # Create config (override path if provided)
        config = RAGConfig.from_env()
        if request.knowledge_base_path:
            config.knowledge_base_path = request.knowledge_base_path
        
        # Build index
        builder = IndexBuilder(config)
        success = builder.build_index()
        
        if not success:
            raise HTTPException(status_code=500, detail="Index build failed")
        
        # Reload index in RAG engine
        app.state.rag_engine.load_index()
        
        # Clear cache (only if Redis is available)
        if app.state.redis:
            try:
                # Delete all rag:* keys
                keys = await app.state.redis.keys("rag:*")
                if keys:
                    await app.state.redis.delete(*keys)
                    logger.info(f"️ Cleared {len(keys)} cached queries")
            except Exception as cache_error:
                logger.warning(f"️ Cache clear failed: {cache_error}")
        else:
            logger.info("️ Redis not available - skipping cache clear")
        
        # Get stats
        stats = builder.get_index_stats()
        build_time = time.time() - build_start
        
        logger.info(f" Index rebuilt: {stats['total_documents']} documents in {build_time:.2f}s")
        
        return RebuildIndexResponse(
            status="success",
            documents_indexed=stats['total_documents'],
            categories=stats['categories'],
            build_time_seconds=build_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rebuild error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Docker vs local: single worker to avoid FAISS index duplication
    # Rely on async concurrency for throughput
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8003")),
        workers=1,
        log_level="info"
    )
