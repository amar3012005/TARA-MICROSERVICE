"""
Intent Classification Core Logic - 3-Layer Architecture

Robust 3-layer intent classification system:
- Layer 1 (Regex): Fast pattern matching (<5ms) for high-confidence cases
- Layer 2 (Semantic): Enhanced keyword/semantic matching for medium complexity
- Layer 3 (LLM): Gemini fallback for complex/ambiguous queries

Ported from leibniz_intent_parser.py with enhancements for microservice deployment.

Reference:
    leibniz_agent/leibniz_intent_parser.py - Original 3-layer implementation
"""

import re
import json
import time
import asyncio
import logging
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
import google.generativeai as genai

from .config import IntentConfig
from .patterns import load_intent_patterns, compile_patterns, get_system_prompt

logger = logging.getLogger(__name__)

# Layer 2: DistilBERT imports
try:
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️ Transformers not available - Layer 2 (SLM) will be disabled")

# Spacy for semantic context extraction
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("⚠️ Spacy not available - will use regex fallback for context extraction")


class IntentClassifier:
    """
    3-Layer intent classifier for Leibniz University Agent.
    
    Classification Strategy:
        1. Layer 1 (Regex): Fast pattern matching (target >60% of requests, <5ms)
        2. Layer 2 (Semantic): Enhanced keyword/semantic matching (target >25% of requests, <50ms)
        3. Layer 3 (LLM): Gemini fallback for complex cases (target <15% of requests, <500ms)
    
    Intents:
        - APPOINTMENT_SCHEDULING: Book/schedule meetings
        - RAG_QUERY: Information requests (most common)
        - GREETING: Standalone greetings (strict - very rare)
        - EXIT: End conversation
        - UNCLEAR: Ambiguous/nonsensical input
    """
    
    def __init__(self, config: IntentConfig):
        """
        Initialize 3-layer intent classifier.
        
        Args:
            config: Intent configuration with thresholds, Gemini API key, etc.
        """
        self.config = config
        
        # Load and compile patterns
        self.patterns = load_intent_patterns()
        self.compiled_patterns = compile_patterns(self.patterns)
        
        # === Initialize Spacy for semantic context extraction ===
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                spacy_model_name = os.getenv("SPACY_MODEL_NAME", "en_core_web_sm")
                self.nlp = spacy.load(spacy_model_name)
                logger.info(f"✅ Spacy model loaded: {spacy_model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Spacy model: {e}. Will use regex fallback for context extraction.")
                self.nlp = None
        else:
            logger.info("ℹ️ Spacy not available - using regex fallback for context extraction")
        
        # === LAYER 2: Initialize DistilBERT Model ===
        self.slm_tokenizer = None
        self.slm_model = None
        self.slm_device = None
        self.slm_ready = False
        self.id2label = {
            0: "APPOINTMENT_SCHEDULING",
            1: "RAG_QUERY",
            2: "GREETING",
            3: "EXIT",
            4: "UNCLEAR"
        }
        
        if config.layer2_enabled and TRANSFORMERS_AVAILABLE:
            try:
                # Determine model path (relative to current file or absolute)
                model_path = os.getenv("LEIBNIZ_INTENT_SLM_MODEL_PATH", "./leibniz_distilbert_intent_v2")
                
                # Try multiple possible paths
                possible_paths = [
                    model_path,
                    os.path.join(os.path.dirname(__file__), "leibniz_distilbert_intent_v2"),
                    os.path.join(os.path.dirname(__file__), "..", "leibniz_distilbert_intent_v2"),
                    "/app/leibniz_agent/services/intent/leibniz_distilbert_intent_v2",  # Docker path
                ]
                
                model_path_found = None
                for path in possible_paths:
                    if os.path.exists(path) and os.path.isdir(path):
                        model_path_found = path
                        break
                
                if model_path_found:
                    logger.info(f"📦 Loading DistilBERT model from: {model_path_found}")
                    self.slm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.slm_tokenizer = DistilBertTokenizer.from_pretrained(model_path_found)
                    self.slm_model = DistilBertForSequenceClassification.from_pretrained(model_path_found)
                    self.slm_model.to(self.slm_device)
                    self.slm_model.eval()
                    self.slm_ready = True
                    logger.info(f"✅ Layer 2 (SLM/DistilBERT) initialized on {self.slm_device}")
                else:
                    logger.warning(f"⚠️ DistilBERT model not found at any of: {possible_paths}. Layer 2 disabled.")
            except Exception as e:
                logger.error(f"❌ Failed to load DistilBERT model: {e}", exc_info=True)
                self.slm_ready = False
        elif not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ Transformers library not available - Layer 2 (SLM) disabled")
        elif not config.layer2_enabled:
            logger.info("ℹ️ Layer 2 (SLM) disabled by configuration")
        
        # === LAYER 3: Initialize Gemini client ===
        self.model = None
        self.llm_ready = False
        if config.gemini_api_key:
            try:
                genai.configure(api_key=config.gemini_api_key)
                self.model = genai.GenerativeModel(config.gemini_model)
                self.llm_ready = True
                logger.info(f"✅ Layer 3 (LLM) initialized: {config.gemini_model}")
            except Exception as e:
                logger.warning(f"⚠️ Gemini initialization failed: {e}. Layer 3 disabled.")
        else:
            logger.warning("⚠️ No Gemini API key. Layer 3 (LLM) disabled.")
        
        # LRU Cache for repeated queries
        self.classification_cache = OrderedDict() if config.enable_cache else None
        self.cache_hits = 0
        
        # Performance tracking
        self.layer1_count = 0
        self.layer2_count = 0
        self.layer3_count = 0
        self.total_confidence = 0.0
        
        logger.info(
            f"✅ IntentClassifier initialized (3-Layer): "
            f"L1_threshold={config.layer1_regex_threshold}, "
            f"L2_threshold={config.layer2_slm_threshold}, "
            f"L2_ready={self.slm_ready}, "
            f"L3_ready={self.llm_ready}, "
            f"cache_enabled={config.enable_cache}"
        )
    
    async def classify_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify user intent with 3-layer approach.
        
        Args:
            text: User input text to classify
            context: Optional conversation context for better classification
            
        Returns:
            Dictionary with:
                - intent: str (APPOINTMENT_SCHEDULING, RAG_QUERY, GREETING, EXIT, UNCLEAR)
                - confidence: float (0.0-1.0)
                - context: dict with user_goal, key_entities, extracted_meaning
                - reasoning: str (classification explanation)
                - layer_type: str (L1, L2, L3, CACHE)
                - decision_path: str (description of classification path)
                - response_time: float (seconds)
        """
        start_time = time.time()
        
        # Validate input
        if not text or not text.strip():
            return {
                "intent": "UNCLEAR",
                "confidence": 0.0,
                "context": {
                    "user_goal": "empty input received",
                    "key_entities": {},
                    "extracted_meaning": ""
                },
                "reasoning": "Empty or whitespace-only input",
                "layer_type": "L1",
                "decision_path": "L1: Empty input",
                "response_time": time.time() - start_time
            }
        
        # Check cache first (if enabled)
        if self.config.enable_cache and self.classification_cache is not None:
            cache_key = hashlib.md5(text.lower().strip().encode()).hexdigest()
            
            if cache_key in self.classification_cache:
                cached_result, cache_time = self.classification_cache[cache_key]
                
                # Check TTL
                if time.time() - cache_time < self.config.cache_ttl_seconds:
                    # Cache hit - return immediately
                    self.cache_hits += 1
                    cached_result["response_time"] = time.time() - start_time
                    cached_result["layer_type"] = "CACHE"
                    cached_result["decision_path"] = f"CACHE: {cached_result.get('layer_type', 'L1')}"
                    
                    # Move to end (LRU)
                    self.classification_cache.move_to_end(cache_key)
                    
                    if self.config.log_classifications:
                        logger.debug(f" CACHE HIT: '{text[:50]}...' → {cached_result['intent']}")
                    
                    return cached_result
                else:
                    # Expired - remove from cache
                    del self.classification_cache[cache_key]
        
        try:
            # === LAYER 1: Fast Regex Classification ===
            layer1_result = self._layer_1_regex(text, context)
            
            if layer1_result and layer1_result["confidence"] > self.config.layer1_regex_threshold:
                # High confidence regex match - use it
                self.layer1_count += 1
                self.total_confidence += layer1_result["confidence"]
                layer1_result["layer_type"] = "L1"
                layer1_result["decision_path"] = f"L1: Regex match (conf={layer1_result['confidence']:.2f})"
                layer1_result["response_time"] = time.time() - start_time
                
                # Cache result
                self._cache_result(text, layer1_result)
                
                if self.config.log_classifications:
                    logger.info(
                        f" L1 (Regex): {layer1_result['intent']} "
                        f"(conf={layer1_result['confidence']:.2f}, "
                        f"time={layer1_result['response_time']*1000:.1f}ms)"
                    )
                
                return layer1_result
            
            # === LAYER 2: SLM (DistilBERT) Classification ===
            if self.config.layer2_enabled and self.slm_ready:
                layer2_result = self._layer_2_slm(text, context, layer1_result)
                
                if layer2_result and layer2_result["confidence"] > self.config.layer2_slm_threshold:
                    # Good semantic match - use it
                    self.layer2_count += 1
                    self.total_confidence += layer2_result["confidence"]
                    layer2_result["layer_type"] = "L2"
                    layer2_result["decision_path"] = f"L2: Semantic match (conf={layer2_result['confidence']:.2f})"
                    layer2_result["response_time"] = time.time() - start_time
                    
                    # Cache result
                    self._cache_result(text, layer2_result)
                    
                    if self.config.log_classifications:
                        logger.info(
                            f" L2 (SLM/DistilBERT): {layer2_result['intent']} "
                            f"(conf={layer2_result['confidence']:.2f}, "
                            f"time={layer2_result['response_time']*1000:.1f}ms)"
                        )
                    
                    return layer2_result
            
            # === LAYER 3: LLM Fallback (Gemini) ===
            if self.llm_ready:
                try:
                    layer3_result = await self._layer_3_llm(text, context)
                    self.layer3_count += 1
                    self.total_confidence += layer3_result["confidence"]
                    layer3_result["layer_type"] = "L3"
                    layer3_result["decision_path"] = f"L3: LLM fallback (conf={layer3_result['confidence']:.2f})"
                    layer3_result["response_time"] = time.time() - start_time
                    
                    # Cache result
                    self._cache_result(text, layer3_result)
                    
                    if self.config.log_classifications:
                        logger.info(
                            f" L3 (LLM): {layer3_result['intent']} "
                            f"(conf={layer3_result['confidence']:.2f}, "
                            f"time={layer3_result['response_time']*1000:.1f}ms)"
                        )
                    
                    return layer3_result
                
                except Exception as e:
                    logger.warning(f"⚠️ Layer 3 (LLM) failed: {e}")
                    # Fall through to default
            
            # === FALLBACK: Use Layer 1 result if available, else UNCLEAR ===
            if layer1_result:
                self.layer1_count += 1
                layer1_result["layer_type"] = "L1"
                layer1_result["decision_path"] = f"L1: Fallback (conf={layer1_result['confidence']:.2f})"
                layer1_result["response_time"] = time.time() - start_time
                self._cache_result(text, layer1_result)
                return layer1_result
            
            # Default: UNCLEAR
            unclear_result = {
                "intent": "UNCLEAR",
                "confidence": 0.3,
                "context": {
                    "user_goal": "unclear intent",
                    "key_entities": {},
                    "extracted_meaning": text.lower()
                },
                "reasoning": "No clear pattern matched across all layers",
                "layer_type": "L1",
                "decision_path": "L1: Default UNCLEAR",
                "response_time": time.time() - start_time
            }
            self._cache_result(text, unclear_result)
            return unclear_result
                
        except Exception as e:
            logger.error(f"❌ Classification error: {e}", exc_info=True)
            return {
                "intent": "UNCLEAR",
                "confidence": 0.2,
                "context": {
                    "user_goal": "classification error occurred",
                    "key_entities": {},
                    "extracted_meaning": text.lower()
                },
                "reasoning": f"Error during classification: {str(e)}",
                "layer_type": "ERROR",
                "decision_path": f"ERROR: {str(e)}",
                "response_time": time.time() - start_time
            }
    
    def _layer_1_regex(self, text: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Layer 1: Fast regex pattern matching (<5ms).
        
        Uses compiled regex patterns for instant classification of high-confidence cases.
        
        Args:
            text: User input text
            context: Optional conversation context
            
        Returns:
            Classification result dict or None if no match
        """
        # Light normalization
        text_normalized = text.lower().strip()
        text_normalized = re.sub(r'[?!.,;]', '', text_normalized)
        text_lower = text_normalized
        
        # Priority 1: Appointment Detection (highest priority)
        appointment_patterns = self.patterns["appointment"]
        
        # Negative keywords to exclude false positives
        appointment_negative_keywords = ["class schedule", "course schedule", "exam schedule", "schedule of classes"]
        has_negative_keyword = any(neg_kw in text_lower for neg_kw in appointment_negative_keywords)
        
        # Check keywords
        appointment_keywords_found = [kw for kw in appointment_patterns["keywords"] if kw in text_lower]
        
        # Check regex patterns
        appointment_regex_patterns = self.compiled_patterns["appointment"]["regex_patterns"]
        appointment_regex_match = any(pattern.search(text_lower) for pattern in appointment_regex_patterns)
        
        # Determine sentence length
        word_count = len(text_normalized.split())
        
        # Refined appointment detection
        is_appointment = False
        
        # Skip L1 appointment detection for long, complex sentences (>5 words) unless high-confidence keywords are present
        # This forces complex requests to Layer 2 (SLM) or Layer 3 (LLM)
        if word_count > 5:
             # Only allow if VERY specific keywords are present
             if any(kw in text_lower for kw in ["schedule", "book appointment", "booking"]):
                 # Still apply negative keyword check
                 if not has_negative_keyword and appointment_regex_match:
                     is_appointment = True
        elif not has_negative_keyword:
            if appointment_regex_match:
                is_appointment = True
            elif any(kw in ["appointment", "meeting", "meet"] for kw in appointment_keywords_found):
                is_appointment = True
            elif any(kw in ["schedule", "book", "booking"] for kw in appointment_keywords_found):
                has_person_office = re.search(r"\b(with|see|visit)\s+(an?|the)?\s*\w+\b", text_lower) or \
                                   re.search(r"\b(advisor|counselor|professor|dean|staff|office|department)\b", text_lower)
                if has_person_office:
                    is_appointment = True
        
        if is_appointment:
            # Extract entities
            key_entities = {}
            entity_patterns = self.compiled_patterns["appointment"]["entity_patterns"]
            
            dept_match = entity_patterns["department"].search(text_lower)
            if dept_match:
                key_entities["department"] = dept_match.group(0)
            
            datetime_match = entity_patterns["datetime"].search(text_lower)
            if datetime_match:
                key_entities["datetime"] = datetime_match.group(0)
            
            purpose_match = entity_patterns["purpose"].search(text_lower)
            if purpose_match:
                key_entities["purpose"] = purpose_match.group(0)
            
            # Build context
            user_goal = "wants to schedule appointment"
            if "department" in key_entities:
                user_goal += f" with {key_entities['department']}"
            if "purpose" in key_entities:
                user_goal += f" for {key_entities['purpose']}"
            
            # Build extracted meaning
            meaning_parts = ["schedule", "appointment"]
            if "department" in key_entities:
                meaning_parts.append(key_entities["department"])
            if "purpose" in key_entities:
                meaning_parts.append(key_entities["purpose"])
            extracted_meaning = " ".join(meaning_parts)
            
            return {
                "intent": "APPOINTMENT_SCHEDULING",
                "confidence": 0.95,
                "context": {
                    "user_goal": user_goal,
                    "key_entities": key_entities,
                    "extracted_meaning": extracted_meaning
                },
                "reasoning": f"Appointment keywords detected: {appointment_keywords_found}"
            }
        
        # Priority 2: Exit Detection
        # Skip L1 for exit on long sentences (avoid "exit" keyword false positives in complex queries)
        if len(text_normalized.split()) > 5:
             # Log skipping L1 for exit to avoid false positives on complex sentences
             logger.debug(f"Skipping L1 Exit detection for long sentence ({len(text_normalized.split())} words)")
        else:
            exit_patterns = self.patterns["exit"]
            exit_keywords_found = [kw for kw in exit_patterns["keywords"] if kw in text_lower]
            exit_regex_patterns = self.compiled_patterns["exit"]["regex_patterns"]
            exit_regex_match = any(pattern.search(text_lower) for pattern in exit_regex_patterns)
            
            if exit_keywords_found or exit_regex_match:
                return {
                    "intent": "EXIT",
                    "confidence": 0.90,  # Reduced from 0.95 to allow L2 override if needed
                    "context": {
                        "user_goal": "ending conversation",
                        "key_entities": {},
                        "extracted_meaning": ""
                    },
                    "reasoning": "Exit phrase detected"
                }
        
        # Priority 3: Greeting Detection (STRICT - must be standalone greeting)
        greeting_patterns = self.patterns["greeting"]
        greeting_keywords_found = [kw for kw in greeting_patterns["keywords"] if kw in text_lower]
        greeting_regex_patterns = self.compiled_patterns["greeting"]["regex_patterns"]
        greeting_regex_match = any(pattern.search(text_lower) for pattern in greeting_regex_patterns)
        
        # CRITICAL: Greetings must NOT contain question words or information request phrases
        has_question_words = re.search(r'\b(can you|could you|tell me|talk about|what|when|where|who|why|how|explain|describe)\b', text_lower)
        is_information_request = re.search(r'\b(about|information|help me with|assist|know more)\b', text_lower)
        
        if (greeting_keywords_found or greeting_regex_match) and not has_question_words and not is_information_request:
            word_count = len(text_normalized.split())
            if word_count <= 10:
                return {
                    "intent": "GREETING",
                    "confidence": 0.95,
                    "context": {
                        "user_goal": "greeting the assistant",
                        "key_entities": {},
                        "extracted_meaning": ""
                    },
                    "reasoning": "Greeting detected"
                }
        
        # Priority 4: General Query Detection (RAG routing)
        query_patterns = self.patterns["general_query"]
        
        has_question_word = any(text_lower.startswith(qw) or f" {qw} " in f" {text_lower} " 
                                for qw in query_patterns["question_words"])
        
        topics_found = [topic for topic in query_patterns["university_topics"] if topic in text_lower]
        
        query_regex_patterns = self.compiled_patterns["general_query"]["regex_patterns"]
        info_request_match = any(pattern.search(text_lower) for pattern in query_regex_patterns)
        
        # Skip L1 for complex queries (>5 words) unless it has very clear question markers
        # This forces long, narrative queries to L2 (SLM) or L3 (LLM) for better understanding
        if word_count > 5 and not has_question_word:
             logger.debug(f"Skipping L1 General Query detection for long sentence ({word_count} words)")
             return None
        
        if has_question_word or topics_found or info_request_match:
            # Extract entities
            key_entities = {}
            entity_patterns = self.compiled_patterns["entities"]
            
            program_match = entity_patterns["program"].search(text_lower)
            if program_match:
                key_entities["program"] = program_match.group(0)
            
            course_match = entity_patterns["course_code"].search(text)
            if course_match:
                key_entities["course"] = course_match.group(0)
            
            if topics_found:
                key_entities["topic"] = topics_found[0]
            
            # Build user_goal
            user_goal = "asking about"
            if "program" in key_entities:
                user_goal += f" {key_entities['program']} program"
            elif "topic" in key_entities:
                user_goal += f" {key_entities['topic']}"
            else:
                user_goal += " university information"
            
            # Build extracted_meaning (normalize query)
            filler_words = ["um", "uh", "like", "you know", "i mean", "well", "so"]
            meaning_text = text_lower
            for filler in filler_words:
                meaning_text = meaning_text.replace(filler, "")
            
            meaning_text = re.sub(r"^(what|when|where|who|why|how)\s+(is|are|do|does|can|could)\s+", "", meaning_text)
            meaning_text = re.sub(r"\s+", " ", meaning_text).strip()
            
            return {
                "intent": "RAG_QUERY",
                "confidence": 0.85,
                "context": {
                    "user_goal": user_goal,
                    "key_entities": key_entities,
                    "extracted_meaning": meaning_text
                },
                "reasoning": "General information query detected"
            }
        
        # No match - return None (will fall through to Layer 2)
        return None
    
    def _layer_2_slm(self, text: str, context: Optional[Dict[str, Any]] = None, layer1_result: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Layer 2: SLM-based classification using fine-tuned DistilBERT.
        
        Uses the fine-tuned DistilBERT model (leibniz_distilbert_intent_v2) for medium-complexity queries.
        This provides semantic understanding beyond regex patterns.
        
        Args:
            text: User input text
            context: Optional conversation context
            layer1_result: Optional Layer 1 result for smart gating
            
        Returns:
            Classification result dict or None if no match
        """
        if not self.slm_ready or not self.slm_model or not self.slm_tokenizer:
            return None
        
        try:
            # Tokenize input
            inputs = self.slm_tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.slm_device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.slm_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_id = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_id].item()
            
            # Map to intent label
            predicted_intent = self.id2label.get(predicted_id, "UNCLEAR")
            
            # Smart gating: If Layer 1 had a match, verify Layer 2 agrees
            if layer1_result and layer1_result.get("intent") != predicted_intent:
                # Conflict - prefer Layer 1 for explicit patterns, but boost Layer 2 if very confident
                if confidence > 0.85:
                    logger.debug(f"Smart Gating: L1={layer1_result['intent']} vs L2={predicted_intent}, using L2 (high confidence={confidence:.2f})")
                else:
                    # Layer 1 more reliable for explicit patterns
                    return None
            
            # Threshold check (match leibniz_intent_parser.py: > 0.7)
            if confidence <= self.config.layer2_slm_threshold:
                return None
            
            # Don't return UNCLEAR from Layer 2 (let Layer 3 handle it)
            if predicted_intent == "UNCLEAR":
                return None
            
            # Extract semantic context
            semantic_context = self._extract_semantic_context(text, predicted_intent)
            
            return {
                "intent": predicted_intent,
                "confidence": float(confidence),
                "context": semantic_context,
                "reasoning": f"DistilBERT SLM classification (conf={confidence:.3f}, predicted_id={predicted_id})"
            }
        
        except Exception as e:
            logger.warning(f"⚠️ Layer 2 (SLM) classification failed: {e}")
            return None
    
    async def _layer_3_llm(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Layer 3: LLM-based classification using Gemini with retry logic.
        
        Handles complex/ambiguous queries that Layers 1 and 2 couldn't classify confidently.
        Implements exponential backoff retry for 429 (rate limit) and 503 (service unavailable) errors.
        
        Args:
            text: User input text
            context: Optional conversation context
            
        Returns:
            Classification result dict
        """
        # Get system prompt
        system_prompt = get_system_prompt()
        
        # Build context information if provided
        context_info = ""
        if context:
            context_info = f"Context: {json.dumps(context)}. "
        
        # Build full prompt
        full_prompt = f"{system_prompt}\n\n{context_info}User input: '{text}'. Classify this input:"
        
        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                # Call Gemini API with timeout
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.model.generate_content,
                            full_prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.1,  # Low temperature for consistent classification
                                max_output_tokens=500
                            )
                        ),
                        timeout=self.config.gemini_timeout
                    )
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"⚠️ Gemini timeout (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        return {
                            "intent": "UNCLEAR",
                            "confidence": 0.3,
                            "context": {
                                "user_goal": "Gemini classification timeout",
                                "key_entities": {},
                                "extracted_meaning": text.lower()
                            },
                            "reasoning": f"Gemini classification timed out after {max_retries} attempts"
                        }
                
                # Success - break out of retry loop
                break
                
            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limit (429) or service unavailable (503) errors
                is_retryable = ("429" in error_str or "503" in error_str or 
                               "quota" in error_str or "rate limit" in error_str or
                               "service unavailable" in error_str)
                
                if is_retryable and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"⚠️ Gemini API error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}... Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Non-retryable error or max retries reached
                    logger.warning(f"⚠️ Gemini API error: {e}")
                    return {
                        "intent": "UNCLEAR",
                        "confidence": 0.2,
                        "context": {
                            "user_goal": "Gemini API error",
                            "key_entities": {},
                            "extracted_meaning": text.lower()
                        },
                        "reasoning": f"Gemini API error: {str(e)[:200]}"
                    }
        
        # Process response (only reached if successful)
        try:
            
            # Extract response text
            response_text = response.text.strip()
            
            # Robust JSON extraction: Remove code fences and extra wrappers
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            response_text = response_text.strip()
            
            # Try multiple extraction strategies
            json_text = None
            
            # Strategy 1: Try parsing the entire response as JSON
            try:
                result = json.loads(response_text)
                json_text = response_text
            except json.JSONDecodeError:
                # Strategy 2: Find first balanced JSON object
                json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    # Strategy 3: Find last JSON-looking block
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)
            
            if not json_text:
                raise ValueError("No JSON found in Gemini response")
            
            result = json.loads(json_text)
            
            # Validate result structure
            if "intent" not in result:
                result["intent"] = "UNCLEAR"
            if "confidence" not in result:
                result["confidence"] = 0.5
            if "context" not in result:
                result["context"] = {
                    "user_goal": "unclear intent",
                    "key_entities": {},
                    "extracted_meaning": text.lower()
                }
            else:
                # Ensure context has required fields
                if "user_goal" not in result["context"]:
                    result["context"]["user_goal"] = "unclear intent"
                if "key_entities" not in result["context"]:
                    result["context"]["key_entities"] = {}
                if "extracted_meaning" not in result["context"]:
                    result["context"]["extracted_meaning"] = text.lower()
            
            if "reasoning" not in result:
                result["reasoning"] = "Gemini classification"
            
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parse error in Gemini response: {e}")
            return {
                "intent": "UNCLEAR",
                "confidence": 0.4,
                "context": {
                    "user_goal": "JSON parse error",
                    "key_entities": {},
                    "extracted_meaning": text.lower()
                },
                "reasoning": "Failed to parse Gemini JSON response"
            }
        except Exception as e:
            logger.warning(f"⚠️ Unexpected error in Layer 3 (LLM): {e}")
            return {
                "intent": "UNCLEAR",
                "confidence": 0.2,
                "context": {
                    "user_goal": "classification error",
                    "key_entities": {},
                    "extracted_meaning": text.lower()
                },
                "reasoning": f"Unexpected error: {str(e)[:200]}"
            }
    
    def _extract_semantic_context(self, text: str, intent: str = "RAG_QUERY") -> Dict[str, Any]:
        """
        Extract rich semantic context for RAG system using Spacy NER.
        
        Uses Spacy for advanced NLP (NER, noun chunks, lemmatization) with regex fallback.
        Matches the implementation in leibniz_intent_parser.py for consistency.
        
        Args:
            text: User input text
            intent: Classified intent
            
        Returns:
            Context dict with user_goal, key_entities, extracted_meaning, and optional spacy_analysis
        """
        text_lower = text.lower()
        key_entities = {}
        
        # Use Spacy if available, otherwise fall back to regex
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                
                # Extract entities (NER)
                entities = {ent.label_: ent.text for ent in doc.ents}
                
                # Extract noun chunks (key phrases)
                noun_chunks = [chunk.text for chunk in doc.noun_chunks]
                
                # Extract lemmatized keywords (for search)
                keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
                
                # Extract core action (root verb)
                root_verb = [token.lemma_ for token in doc if token.dep_ == "ROOT" and token.pos_ == "VERB"]
                action = root_verb[0] if root_verb else "unknown"
                
                # Query type detection
                is_question = any(token.tag_ == "." and token.text == "?" for token in doc) or \
                            any(token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WRB" for token in doc)
                query_type = "question" if is_question else "statement"
                
                spacy_analysis = {
                    "original_text": text,
                    "entities": entities,
                    "noun_chunks": noun_chunks,
                    "keywords": keywords,
                    "action": action,
                    "query_type": query_type
                }
            except Exception as e:
                logger.warning(f"⚠️ Spacy processing failed: {e}. Using regex fallback.")
                spacy_analysis = None
        else:
            spacy_analysis = None
        
        # Build context based on intent (same logic for both Spacy and regex paths)
        if intent == "APPOINTMENT_SCHEDULING":
            # Extract appointment-related entities
            dept_match = re.search(r"\b(admissions?|registrar|financial aid|counseling|advising|career services?|academic|student services?|advisor|counselor|professor|dean)\b", text_lower)
            if dept_match:
                key_entities["department"] = dept_match.group(0)
            
            purpose_match = re.search(r"\b(admission|enrollment|transcript|financial|academic|career|degree|program|meeting|consultation)\b", text_lower)
            if purpose_match:
                key_entities["purpose"] = purpose_match.group(0)
            
            user_goal = "wants to schedule appointment"
            if "department" in key_entities:
                user_goal += f" with {key_entities['department']}"
            if "purpose" in key_entities:
                user_goal += f" for {key_entities['purpose']}"
            
            meaning_parts = ["schedule", "appointment"]
            if "department" in key_entities:
                meaning_parts.append(key_entities["department"])
            if "purpose" in key_entities:
                meaning_parts.append(key_entities["purpose"])
            extracted_meaning = " ".join(meaning_parts)
            
        elif intent == "RAG_QUERY":
            # Extract query-related entities
            program_match = re.search(r"\b(computer science|cs|engineering|business|biology|chemistry|physics|mathematics|english|history|economics|psychology)\b", text_lower)
            if program_match:
                key_entities["program"] = program_match.group(0)
            
            course_match = re.search(r"\b[A-Z]{2,4}\s*\d{3,4}[A-Z]?\b", text)
            if course_match:
                key_entities["course"] = course_match.group(0)
            
            topics = self.patterns["general_query"]["university_topics"]
            topic_found = None
            for topic in topics:
                if topic in text_lower:
                    topic_found = topic
                    break
            if topic_found:
                key_entities["topic"] = topic_found
            
            # Build user_goal
            if "program" in key_entities:
                user_goal = f"seeking information about {key_entities['program']} program"
            elif "topic" in key_entities:
                user_goal = f"seeking information about {key_entities['topic']}"
            else:
                user_goal = "seeking general university information"
            
            # Build extracted_meaning (normalize query)
            meaning_text = text_lower
            for filler in ["um ", "uh ", "like ", "you know ", "i mean ", "well ", "so "]:
                meaning_text = meaning_text.replace(filler, " ")
            
            meaning_text = re.sub(r"^(what|when|where|who|why|how)\s+(is|are|do|does|can|could)\s+", "", meaning_text)
            meaning_text = re.sub(r"\s+", " ", meaning_text).strip()
            
            if key_entities:
                entity_str = " ".join(f"{k} {v}" for k, v in key_entities.items())
                extracted_meaning = f"{user_goal}: {entity_str} {meaning_text}".strip()
            else:
                extracted_meaning = f"{user_goal}: {meaning_text}".strip()
            
        elif intent == "GREETING":
            user_goal = "greeting the assistant"
            extracted_meaning = ""
            
        elif intent == "EXIT":
            user_goal = "ending conversation"
            extracted_meaning = ""
            
        else:  # UNCLEAR
            user_goal = "unclear intent"
            extracted_meaning = text_lower
        
        result = {
            "user_goal": user_goal,
            "key_entities": key_entities,
            "extracted_meaning": extracted_meaning
        }
        
        # Add Spacy analysis if available (for debugging/advanced use)
        if spacy_analysis:
            result["spacy_analysis"] = spacy_analysis
        
        return result
    
    def _cache_result(self, text: str, result: Dict[str, Any]):
        """
        Cache classification result with LRU eviction.
        
        Args:
            text: Original input text (for cache key generation)
            result: Classification result to cache
        """
        if not self.config.enable_cache or self.classification_cache is None:
            return
        
        cache_key = hashlib.md5(text.lower().strip().encode()).hexdigest()
        
        # Remove response_time before caching (session-specific)
        result_copy = result.copy()
        result_copy.pop("response_time", None)
        
        # Store with timestamp
        self.classification_cache[cache_key] = (result_copy, time.time())
        
        # LRU eviction if over size
        if len(self.classification_cache) > self.config.cache_max_size:
            self.classification_cache.popitem(last=False)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get classification performance statistics.
        
        Returns:
            Dictionary with performance metrics:
                - total_requests: Total number of classifications
                - layer1_count: Number of Layer 1 (regex) hits
                - layer2_count: Number of Layer 2 (semantic) hits
                - layer3_count: Number of Layer 3 (LLM) hits
                - cache_hits: Number of cache hits
                - layer1_percentage: Percentage of Layer 1 hits
                - layer2_percentage: Percentage of Layer 2 hits
                - layer3_percentage: Percentage of Layer 3 hits
                - cache_hit_rate: Cache hit rate percentage
                - average_confidence: Average classification confidence
        """
        total_requests = self.layer1_count + self.layer2_count + self.layer3_count
        layer1_percentage = (self.layer1_count / total_requests * 100) if total_requests > 0 else 0
        layer2_percentage = (self.layer2_count / total_requests * 100) if total_requests > 0 else 0
        layer3_percentage = (self.layer3_count / total_requests * 100) if total_requests > 0 else 0
        cache_hit_rate = (self.cache_hits / (total_requests + self.cache_hits) * 100) if (total_requests + self.cache_hits) > 0 else 0
        average_confidence = (self.total_confidence / total_requests) if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "layer1_count": self.layer1_count,
            "layer2_count": self.layer2_count,
            "layer3_count": self.layer3_count,
            "cache_hits": self.cache_hits,
            "layer1_percentage": round(layer1_percentage, 2),
            "layer2_percentage": round(layer2_percentage, 2),
            "layer3_percentage": round(layer3_percentage, 2),
            "cache_hit_rate": round(cache_hit_rate, 2),
            "average_confidence": round(average_confidence, 3),
            "cache_enabled": self.config.enable_cache,
            "cache_size": len(self.classification_cache) if self.classification_cache else 0
        }
