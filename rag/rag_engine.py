"""
RAG Engine Core Logic

Ported from leibniz_rag.py for microservice deployment.

Reference:
    - leibniz_rag.py (lines 62-1262) - Original LeibnizRAG class
"""

import os
import json
import time
import logging
import hashlib
import re
import asyncio
from typing import Dict, Any, Optional, List, Callable
import numpy as np
import faiss
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings

from leibniz_agent.services.rag.config import RAGConfig

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Core RAG engine with FAISS retrieval and Gemini generation.
    
    Attributes:
        config: RAG configuration
        embeddings: HuggingFace embeddings model
        gemini_model: Gemini model instance
        vector_store: FAISS index
        documents: Document chunks
        doc_metadata: Chunk metadata
        query_count: Query counter
        total_query_time: Cumulative query time
    """
    
    def __init__(self, config: RAGConfig):
        """
        Initialize RAG engine with configuration.
        
        Args:
            config: RAG configuration instance
        """
        self.config = config
        
        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Gemini model
        self.gemini_model = None
        if config.gemini_api_key:
            try:
                genai.configure(api_key=config.gemini_api_key)
                self.gemini_model = genai.GenerativeModel(config.gemini_model)
                logger.info(f" Gemini model initialized: {config.gemini_model}")
            except Exception as e:
                logger.error(f" Gemini initialization failed: {e}")
        else:
            logger.warning("️ No Gemini API key - response generation unavailable")
        
        # Storage
        self.vector_store = None
        self.documents: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
        
        # HYBRID APPROACH: Rule-based patterns for instant context reduction
        # Reduces Gemini token count by 50-70% for common queries
        # CUSTOMIZED FOR TASK ORGANIZATION BASED ON KNOWLEDGE BASE
        self.quick_answer_patterns = {
            "organization_info": {
                "keywords": ["what is task", "about task", "task organization", "telangana academy", "task academy", "mission", "vision", "history", "established", "government organization"],
                "response_template": "T.A.S.K (Telangana Academy for Skill and Knowledge) is a {description}. {additional_info}",
                "faiss_boost": ["organization_overview", "general_information", "mission_vision"],
                "max_context_chars": 2500,
                "priority": 1
            },
            "contact_info": {
                "keywords": ["contact", "email", "phone", "call", "reach", "get in touch", "contact information", "office address", "location", "address", "head office"],
                "response_template": "You can contact T.A.S.K via {contact_methods}. {additional_info}",
                "faiss_boost": ["contact_access", "official_contact"],
                "max_context_chars": 1500,
                "priority": 1
            },
            "office_hours": {
                "keywords": ["office hours", "opening hours", "working hours", "open time", "office time", "hours of operation", "timing", "schedule", "when open", "business hours"],
                "response_template": "T.A.S.K office hours are {hours}. {additional_info}",
                "faiss_boost": ["contact_access", "official_contact"],
                "max_context_chars": 2000,
                "priority": 1
            },
            "registration_process": {
                "keywords": ["register", "registration", "enrollment", "enroll", "join", "apply", "how to register", "registration process", "sign up", "enrollment process"],
                "response_template": "To register for T.A.S.K programs: {steps}. {additional_info}",
                "faiss_boost": ["registration_enrollment", "registration_process"],
                "max_context_chars": 3000,
                "priority": 2
            },
            "program_eligibility": {
                "keywords": ["eligibility", "requirements", "who can join", "criteria", "qualifications", "minimum percentage", "age limit", "branches accepted"],
                "response_template": "T.A.S.K eligibility: {requirements}. {additional_info}",
                "faiss_boost": ["student_categories", "programs_courses"],
                "max_context_chars": 2500,
                "priority": 2
            },
            "program_costs": {
                "keywords": ["fees", "cost", "price", "payment", "tuition", "how much", "charges", "scholarship", "financial assistance", "free programs"],
                "response_template": "T.A.S.K program costs: {cost_info}. {additional_info}",
                "faiss_boost": ["financial_information"],
                "max_context_chars": 2000,
                "priority": 2
            },
            "placement_stats": {
                "keywords": ["placement", "placement rate", "job placement", "hiring", "companies", "salary", "average salary", "highest package", "placement statistics", "career"],
                "response_template": "T.A.S.K placement statistics: {stats}. {additional_info}",
                "faiss_boost": ["placement_career", "placement_statistics"],
                "max_context_chars": 2800,
                "priority": 2
            },
            "available_programs": {
                "keywords": ["programs", "courses", "what do you offer", "training programs", "skill development", "courses available", "program list", "ai programs", "cloud programs"],
                "response_template": "T.A.S.K offers {programs}. {additional_info}",
                "faiss_boost": ["programs_courses", "technology_programs"],
                "max_context_chars": 3000,
                "priority": 1
            }
        }
        
        # TASK Organization patterns (Telugu customer service - TARA mode)
        # These patterns include Telugu keywords for better matching
        # ENHANCED BASED ON TASK KNOWLEDGE BASE STRUCTURE
        self.task_patterns = {
            "task_organization": {
                "keywords": [
                    "what is task", "టాస్క్ అంటే ఏమిటి", "task ఏమిటి", "about task", "టాస్క్ గురించి",
                    "telangana academy", "టెలంగాణ అకాడమీ", "government organization", "సర్కార్ సంస్థ",
                    "mission", "మిషన్", "vision", "విజన్", "history", "చరిత్ర", "established", "స్థాపించబడింది"
                ],
                "response_template": "T.A.S.K (Telangana Academy for Skill and Knowledge) అనేది {description}. {additional_info}",
                "faiss_boost": ["organization_overview", "general_information", "mission_vision", "history_establishment"],
                "max_context_chars": 2500,
                "priority": 1
            },
            "task_contact": {
                "keywords": [
                    "contact", "సంప్రదించు", "phone", "ఫోన్", "email", "ఇమెయిల్",
                    "address", "చిరునామా", "location", "స్థానం", "reach", "చేరుకోవడం",
                    "head office", "హెడ్ ఆఫీస్", "hyderabad office", "హైదరాబాద్ ఆఫీస్"
                ],
                "response_template": "మీరు T.A.S.K ని {contact_methods} ద్వారా సంప్రదించవచ్చు. {additional_info}",
                "faiss_boost": ["contact_access", "official_contact"],
                "max_context_chars": 1500,
                "priority": 1
            },
            "task_timing": {
                "keywords": [
                    "timing", "సమయం", "hours", "గంటలు", "when open", "ఎప్పుడు తెరవబడుతుంది",
                    "working hours", "పని సమయాలు", "office hours", "ఆఫీస్ సమయాలు",
                    "business hours", "వ్యాపార సమయాలు", "operating hours", "ఆపరేటింగ్ గంటలు"
                ],
                "response_template": "T.A.S.K ఆఫీస్ సమయాలు {hours}. {additional_info}",
                "faiss_boost": ["contact_access", "official_contact"],
                "max_context_chars": 1800,
                "priority": 1
            },
            "task_registration": {
                "keywords": [
                    "register", "నమోదు", "enrollment", "నమోదు", "join", "చేరండి",
                    "apply", "దరఖాస్తు", "admission", "ప్రవేశం", "sign up", "సైన్ అప్",
                    "how to register", "ఎలా నమోదు చేయాలి", "registration process", "నమోదు ప్రక్రియ"
                ],
                "response_template": "T.A.S.K లో నమోదు చేయడానికి: {steps}. {additional_info}",
                "faiss_boost": ["registration_enrollment", "registration_process"],
                "max_context_chars": 3000,
                "priority": 2
            },
            "task_programs": {
                "keywords": [
                    "programs", "కార్యక్రమాలు", "courses", "కోర్సులు", "training", "ట్రైనింగ్",
                    "what do you offer", "మీరు ఏమి అందిస్తారు", "skill development", "స్కిల్ డెవలప్మెంట్",
                    "ai programs", "ఎఐ ప్రోగ్రామ్స్", "cloud computing", "క్లౌడ్ కంప్యూటింగ్",
                    "cybersecurity", "సైబర్ సెక్యూరిటీ", "data science", "డేటా సైన్స్"
                ],
                "response_template": "T.A.S.K {programs} అందిస్తుంది. {additional_info}",
                "faiss_boost": ["programs_courses", "technology_programs"],
                "max_context_chars": 3000,
                "priority": 1
            },
            "task_eligibility": {
                "keywords": [
                    "eligibility", "ఎలిజిబిలిటీ", "requirements", "రియుౖర్మెంట్స్", "who can join", "ఎవరు చేరవచ్చు",
                    "criteria", "క్రైటీరియా", "qualifications", "క్వాలిఫికేషన్స్", "minimum percentage", "మైనిమమ్ పర్సెంటేజ్",
                    "engineering students", "ఇంజినీరింగ్ స్టూడెంట్స్", "polytechnic", "పాలిటెక్నిక్",
                    "graduates", "గ్రాడ్యుయేట్స్", "unemployed", "అనెంప్లాయ్డ్"
                ],
                "response_template": "T.A.S.K ఎలిజిబిలిటీ: {requirements}. {additional_info}",
                "faiss_boost": ["student_categories", "programs_courses"],
                "max_context_chars": 2500,
                "priority": 2
            },
            "task_fees": {
                "keywords": [
                    "fee", "ఫీజు", "cost", "ధర", "price", "ధర", "payment", "చెల్లింపు",
                    "charges", "ఛార్జీలు", "how much", "ఎంత", "scholarship", "స్కాలర్షిప్",
                    "financial assistance", "ఫైనాన్షియల్ అసిస్టెన్స్", "free programs", "ఫ్రీ ప్రోగ్రామ్స్"
                ],
                "response_template": "T.A.S.K ప్రోగ్రామ్ ఫీజులు: {cost_info}. {additional_info}",
                "faiss_boost": ["financial_information"],
                "max_context_chars": 2000,
                "priority": 2
            },
            "task_placement": {
                "keywords": [
                    "placement", "ప్లేస్మెంట్", "placement rate", "ప్లేస్మెంట్ రేట్", "job placement", "జాబ్ ప్లేస్మెంట్",
                    "companies", "కంపెనీలు", "salary", "సెలరీ", "average salary", "అవరేజ్ సెలరీ",
                    "highest package", "హైయెస్ట్ ప్యాకేజ్", "hiring", "హైరింగ్", "career", "కెరీర్"
                ],
                "response_template": "T.A.S.K ప్లేస్మెంట్ స్టాటిస్టిక్స్: {stats}. {additional_info}",
                "faiss_boost": ["placement_career", "placement_statistics"],
                "max_context_chars": 2800,
                "priority": 2
            },
            "task_documents": {
                "keywords": [
                    "documents", "పత్రాలు", "papers", "కాగితాలు", "certificate", "సర్టిఫికేట్",
                    "required", "అవసరం", "bring", "తీసుకురావాలి", "submit", "సమర్పించండి",
                    "college id", "కాలేజ్ ఐడీ", "aadhar", "ఆధార్", "photo", "ఫోటో"
                ],
                "response_template": "నమోదు కోసం అవసరమైన పత్రాలు: {documents}. {additional_info}",
                "faiss_boost": ["registration_enrollment", "registration_process"],
                "max_context_chars": 2000,
                "priority": 2
            },
            "task_support": {
                "keywords": [
                    "help", "సహాయం", "support", "మద్దతు", "problem", "సమస్య",
                    "issue", "సమస్య", "complaint", "ఫిర్యాదు", "grievance", "ఫిర్యాదు",
                    "technical support", "టెక్నికల్ సపోర్ట్", "assistance", "అసిస్టెన్స్"
                ],
                "response_template": "మీ సమస్యకు సహాయం: {solution}. {additional_info}",
                "faiss_boost": ["faq", "contact_access"],
                "max_context_chars": 2000,
                "priority": 1
            },
            "task_status": {
                "keywords": [
                    "status", "స్థితి", "check", "తనిఖీ", "track", "ట్రాక్",
                    "application status", "దరఖాస్తు స్థితి", "progress", "పురోగతి",
                    "enrollment status", "నమోదు స్థితి", "portal", "పోర్టల్"
                ],
                "response_template": "మీ {item} స్థితిని తనిఖీ చేయడానికి: {steps}. {additional_info}",
                "faiss_boost": ["registration_enrollment", "faq"],
                "max_context_chars": 1800,
                "priority": 1
            }
        }
        
        # Load index
        self.load_index()

    async def warmup_embeddings(self):
        """Pre-warm sentence transformer to avoid first-query latency"""
        logger.info("🔥 Pre-warming sentence transformer...")
        start_time = time.time()
        
        # Pre-compute embeddings for common TASK queries - UPDATED FOR TASK ORGANIZATION
        warmup_queries = [
            "what is task", "contact information", "office hours",
            "how to register", "eligibility requirements", "program costs",
            "placement statistics", "available programs", "who can join task",
            "task contact", "registration process", "course fees", "placement records",
            "task programs", "task fees", "task placement", "task eligibility",
            # Telugu queries for TARA mode
            "టాస్క్ అంటే ఏమిటి", "సంప్రదించు", "ఎలా నమోదు", "ఫీజు ఎంత"
        ]
        
        try:
            # This will cache the model in memory
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.embeddings.embed_documents(warmup_queries[:self.config.warmup_queries_count])
            )
            
            warmup_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Sentence transformer warmed up in {warmup_time:.0f}ms")
            
            # Store warmup embeddings for potential reuse
            self.warmup_embeddings_cache = dict(zip(warmup_queries[:self.config.warmup_queries_count], embeddings))
            
        except Exception as e:
            logger.warning(f"⚠️ Embedding warmup failed: {e}")

    async def warmup_gemini(self):
        """Pre-warm Gemini model with a test generation"""
        if not self.gemini_model:
            return
            
        logger.info("🔥 Pre-warming Gemini model...")
        start_time = time.time()
        
        try:
            # Simple warmup query in Telugu-English mixed
            warmup_prompt = "నమస్కారం, TASK కస్టమర్ సర్వీస్ ఏజెంట్ TARA ఇక్కడ ఉన్నాను. మీకు ఎలా సహాయం చేయగలను?"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    warmup_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=10
                    )
                )
            )
            
            warmup_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Gemini model warmed up in {warmup_time:.0f}ms")
            
        except Exception as e:
            logger.warning(f"⚠️ Gemini warmup failed: {e}")

    def enable_model_persistence(self):
        """Enable model persistence to avoid reloads"""
        try:
            # Keep models in memory even during low usage
            import gc
            gc.disable()  # Prevent aggressive garbage collection
            
            logger.info("✅ Model persistence enabled")
            
        except Exception as e:
            logger.warning(f"⚠️ Model persistence setup failed: {e}")

    def precompute_patterns(self):
        """Pre-compute pattern detection results for common TASK queries"""
        logger.info("🔥 Pre-computing pattern detection...")

        # Cache common TASK query patterns - UPDATED FOR TASK ORGANIZATION
        self.pattern_cache = {}

        task_queries = [
            # English queries
            "what is task", "contact information", "office hours",
            "how to register", "eligibility requirements", "program costs",
            "placement statistics", "available programs", "who can join task",
            "task contact", "registration process", "course fees", "placement records",
            "task programs", "task fees", "task placement", "task eligibility",
            # Telugu queries
            "టాస్క్ అంటే ఏమిటి", "సంప్రదించు", "సమయం", "ఎలా నమోదు", "ఎలిజిబిలిటీ",
            "ఫీజు ఎంత", "ప్లేస్మెంట్", "ప్రోగ్రామ్స్ ఏమున్నాయి", "ఎవరు చేరవచ్చు",
            "నమస్కారం", "హలో", "help", "సహాయం", "హెల్ప్"  # Telugu/English greetings
        ]

        for query in task_queries:
            pattern = self._detect_query_pattern(query)
            if pattern:
                self.pattern_cache[query] = pattern

        logger.info(f"✅ Pre-computed {len(self.pattern_cache)} query patterns")
    
    def load_index(self) -> bool:
        """
        Load pre-built FAISS index from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            index_path = os.path.join(self.config.vector_store_path, "index.faiss")
            metadata_path = os.path.join(self.config.vector_store_path, "metadata.json")
            texts_path = os.path.join(self.config.vector_store_path, "texts.json")
            
            # Validate files exist
            if not all(os.path.exists(p) for p in [index_path, metadata_path, texts_path]):
                logger.error(f" Index files not found at {self.config.vector_store_path}")
                return False
            
            # Load FAISS index
            self.vector_store = faiss.read_index(index_path)
            
            # Load metadata and texts
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.doc_metadata = json.load(f)
            
            with open(texts_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            logger.info(f" Loaded FAISS index: {len(self.documents)} documents")
            return True
        
        except Exception as e:
            logger.error(f" Error loading index: {e}", exc_info=True)
            return False
    
    def _detect_query_pattern(self, query: str) -> Optional[Dict]:
        """
        Detect if query matches known patterns for hybrid retrieval (1-5ms)
        
        Args:
            query: User query text
            
        Returns:
            Pattern config dict with:
                - name: str
                - keywords: list
                - response_template: str
                - faiss_boost: list
                - max_context_chars: int
                - priority: int
            or None if no pattern match
        """
        query_lower = query.lower()
        
        # Add synonym mappings for fuzzy matching - CUSTOMIZED FOR TASK ORGANIZATION
        synonyms = {
            # Quick answer patterns (English-focused)
            "organization_info": [
                "what is task", "about task", "task organization", "telangana academy", "task academy",
                "mission", "vision", "history", "established", "government organization", "task info",
                "task details", "task overview", "what does task do", "task purpose"
            ],
            "contact_info": [
                "contact", "email", "phone", "call", "reach", "get in touch", "contact information",
                "contact details", "how to reach", "reach out", "speak to", "talk to", "contact task",
                "task contact", "reach task", "contact details"
            ],
            "office_hours": [
                "office hours", "opening hours", "working hours", "open time", "office time",
                "hours of operation", "when open", "timing", "slot availability", "available hours",
                "office timing", "work hours", "working time", "task hours", "task timing",
                "when is task open", "task office hours"
            ],
            "registration_process": [
                "register", "registration", "enrollment", "enroll", "join", "apply", "how to register",
                "registration process", "sign up", "enrollment process", "task registration",
                "join task", "apply for task", "register for task", "task enrollment"
            ],
            "program_eligibility": [
                "eligibility", "requirements", "who can join", "criteria", "qualifications",
                "minimum percentage", "age limit", "branches accepted", "task eligibility",
                "who can join task", "task requirements", "eligibility criteria"
            ],
            "program_costs": [
                "fees", "cost", "price", "payment", "tuition", "how much", "charges",
                "scholarship", "financial assistance", "free programs", "task fees",
                "task cost", "how much does task cost", "task payment", "task scholarships"
            ],
            "placement_stats": [
                "placement", "placement rate", "job placement", "hiring", "companies", "salary",
                "average salary", "highest package", "placement statistics", "career", "task placement",
                "task jobs", "task salary", "task companies", "placement at task"
            ],
            "available_programs": [
                "programs", "courses", "what do you offer", "training programs", "skill development",
                "courses available", "program list", "ai programs", "cloud programs", "task programs",
                "task courses", "what does task offer", "task training", "task skills"
            ],
            # TASK patterns (Telugu-English mixed)
            "task_organization": [
                "what is task", "టాస్క్ అంటే ఏమిటి", "task ఏమిటి", "about task", "టాస్క్ గురించి",
                "telangana academy", "టెలంగాణ అకాడమీ", "government organization", "సర్కార్ సంస్థ",
                "mission", "మిషన్", "vision", "విజన్", "history", "చరిత్ర", "established", "స్థాపించబడింది"
            ],
            "task_contact": [
                "contact", "సంప్రదించు", "phone", "ఫోన్", "email", "ఇమెయిల్",
                "address", "చిరునామా", "location", "స్థానం", "reach", "చేరుకోవడం",
                "head office", "హెడ్ ఆఫీస్", "hyderabad office", "హైదరాబాద్ ఆఫీస్"
            ],
            "task_timing": [
                "timing", "సమయం", "hours", "గంటలు", "when open", "ఎప్పుడు తెరవబడుతుంది",
                "working hours", "పని సమయాలు", "office hours", "ఆఫీస్ సమయాలు",
                "business hours", "వ్యాపార సమయాలు", "operating hours", "ఆపరేటింగ్ గంటలు"
            ],
            "task_registration": [
                "register", "నమోదు", "enrollment", "నమోదు", "join", "చేరండి",
                "apply", "దరఖాస్తు", "admission", "ప్రవేశం", "sign up", "సైన్ అప్",
                "how to register", "ఎలా నమోదు చేయాలి", "registration process", "నమోదు ప్రక్రియ"
            ],
            "task_programs": [
                "programs", "కార్యక్రమాలు", "courses", "కోర్సులు", "training", "ట్రైనింగ్",
                "what do you offer", "మీరు ఏమి అందిస్తారు", "skill development", "స్కిల్ డెవలప్మెంట్",
                "ai programs", "ఎఐ ప్రోగ్రామ్స్", "cloud computing", "క్లౌడ్ కంప్యూటింగ్",
                "cybersecurity", "సైబర్ సెక్యూరిటీ", "data science", "డేటా సైన్స్"
            ],
            "task_eligibility": [
                "eligibility", "ఎలిజిబిలిటీ", "requirements", "రియుౖర్మెంట్స్", "who can join", "ఎవరు చేరవచ్చు",
                "criteria", "క్రైటీరియా", "qualifications", "క్వాలిఫికేషన్స్", "minimum percentage", "మైనిమమ్ పర్సెంటేజ్",
                "engineering students", "ఇంజినీరింగ్ స్టూడెంట్స్", "polytechnic", "పాలిటెక్నిక్",
                "graduates", "గ్రాడ్యుయేట్స్", "unemployed", "అనెంప్లాయ్డ్"
            ],
            "task_fees": [
                "fee", "ఫీజు", "cost", "ధర", "price", "ధర", "payment", "చెల్లింపు",
                "charges", "ఛార్జీలు", "how much", "ఎంత", "scholarship", "స్కాలర్షిప్",
                "financial assistance", "ఫైనాన్షియల్ అసిస్టెన్స్", "free programs", "ఫ్రీ ప్రోగ్రామ్స్"
            ],
            "task_placement": [
                "placement", "ప్లేస్మెంట్", "placement rate", "ప్లేస్మెంట్ రేట్", "job placement", "జాబ్ ప్లేస్మెంట్",
                "companies", "కంపెనీలు", "salary", "సెలరీ", "average salary", "అవరేజ్ సెలరీ",
                "highest package", "హైయెస్ట్ ప్యాకేజ్", "hiring", "హైరింగ్", "career", "కెరీర్"
            ],
            "task_documents": [
                "documents", "పత్రాలు", "papers", "కాగితాలు", "certificate", "సర్టిఫికేట్",
                "required", "అవసరం", "bring", "తీసుకురావాలి", "submit", "సమర్పించండి",
                "college id", "కాలేజ్ ఐడీ", "aadhar", "ఆధార్", "photo", "ఫోటో"
            ],
            "task_support": [
                "help", "సహాయం", "support", "మద్దతు", "problem", "సమస్య",
                "issue", "సమస్య", "complaint", "ఫిర్యాదు", "grievance", "ఫిర్యాదు",
                "technical support", "టెక్నికల్ సపోర్ట్", "assistance", "అసిస్టెన్స్"
            ],
            "task_status": [
                "status", "స్థితి", "check", "తనిఖీ", "track", "ట్రాక్",
                "application status", "దరఖాస్తు స్థితి", "progress", "పురోగతి",
                "enrollment status", "నమోదు స్థితి", "portal", "పోర్టల్"
            ]
        }
        
        # Find best matching pattern (highest priority + fuzzy threshold)
        best_match = None
        best_priority = 0
        best_match_count = 0
        
        # Determine which patterns to check based on TARA mode
        patterns_to_check = self.quick_answer_patterns.copy()
        
        # Add TASK patterns if TARA mode is enabled
        if self.config.tara_mode:
            patterns_to_check.update(self.task_patterns)
            logger.debug("🇮🇳 TARA mode: Including TASK organization patterns")
        
        for pattern_name, pattern_config in patterns_to_check.items():
            # Use extended keywords from synonyms or pattern's own keywords
            extended_keywords = synonyms.get(pattern_name, pattern_config["keywords"])
            
            # Count keyword matches for fuzzy threshold
            match_count = sum(1 for keyword in extended_keywords if keyword in query_lower)
            
            # Fuzzy regex patterns for hours/timing
            if pattern_name in ["office_hours", "task_timing"]:
                # Check for time-related patterns
                time_pattern = r'\d{1,2}:\d{2}|\d{1,2}\s?(am|pm)|hours?|timing|schedule|open|available|సమయం|గంటలు'
                if re.search(time_pattern, query_lower):
                    match_count += 1
            
            # Lower threshold - require at least 1 match
            if match_count > 0:
                # Priority tie-breaker: more matches = better
                if (pattern_config.get("priority", 0) > best_priority or 
                    (pattern_config.get("priority", 0) == best_priority and match_count > best_match_count)):
                    best_match = {"name": pattern_name, **pattern_config}
                    best_priority = pattern_config.get("priority", 0)
                    best_match_count = match_count
        
        if best_match:
            logger.debug(f" Pattern detected: {best_match['name']} (match_count: {best_match_count})")
        
        return best_match
    
    def _retrieve_with_boosting(self, query: str, context: Dict, boost_categories: list, max_context_chars: int) -> tuple:
        """
        FAISS retrieval with category boosting and context truncation
        
        Args:
            query: User query text
            context: Structured context from intent parser
            boost_categories: List of categories/keywords to boost
            max_context_chars: Maximum total character count for context
            
        Returns:
            tuple: (relevant_docs: list, timing_dict: dict)
        """
        timing = {}
        
        # ENHANCED: Enrich query with entities AND specific keywords from query
        enriched_query = query
        
        # Extract specific department/office mentions from query
        dept_keywords = {
            "registrar": "registrar office registry student records enrollment verification transcript certificate exmatriculation",
            "admissions": "admissions office admission requirements application enroll apply eligibility",
            "academic": "academic affairs dean faculty program curriculum course",
            "student services": "student services counseling support advisory wellness",
            "international": "international office exchange study abroad visa foreign",
            "financial": "financial aid bafög scholarship tuition fees payment funding",
            "transcript": "transcript records registry registrar student services academic records",
            "certificate": "certificate certification registrar student services verification document",
        }
        
        # Add department-specific keywords if mentioned
        query_lower = query.lower()
        for dept, keywords in dept_keywords.items():
            if dept in query_lower:
                enriched_query = f"{query} {keywords}"
                logger.debug(f" Query enriched with department keywords: {dept}")
                break
        
        # Add entities from context
        if context and 'key_entities' in context:
            entities = context['key_entities']
            entity_terms = ' '.join([f"{k} {v}" for k, v in entities.items()])
            enriched_query = f"{enriched_query} {entity_terms}"
        
        # Embed query
        embed_start = time.time()
        query_embedding = self.embeddings.embed_query(enriched_query)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        timing['embedding_ms'] = (time.time() - embed_start) * 1000
        
        # FAISS search (retrieve more candidates for filtering)
        search_start = time.time()
        distances, indices = self.vector_store.search(query_embedding, k=self.config.top_k + 5)
        timing['search_ms'] = (time.time() - search_start) * 1000
        
        # Build candidates with boosting
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                distance = float(distances[0][i])
                similarity = 1.0 - (distance * distance / 2.0)
                
                # Skip low-similarity docs
                if similarity < self.config.similarity_threshold:
                    continue
                
                doc_text = self.documents[idx]
                doc_meta = self.doc_metadata[idx] if idx < len(self.doc_metadata) else {}
                
                # Apply category boosting
                boosted_similarity = similarity
                doc_category = doc_meta.get('category', '').lower()
                doc_source = doc_meta.get('source', '').lower()
                
                # Check if doc matches any boost category
                for boost_cat in boost_categories:
                    boost_cat_lower = boost_cat.lower()
                    if boost_cat_lower in doc_category or boost_cat_lower in doc_source:
                        boosted_similarity *= 1.5  # 50% boost!
                        logger.debug(f" Boosted {doc_source} (matched '{boost_cat}')")
                        break  # Only boost once
                
                candidates.append({
                    'text': doc_text,
                    'metadata': doc_meta,
                    'distance': distance,
                    'similarity': similarity,
                    'boosted_similarity': boosted_similarity
                })
        
        # Sort by boosted similarity
        candidates.sort(key=lambda x: x['boosted_similarity'], reverse=True)
        
        # Truncate to max_context_chars (keeps highest-scoring docs)
        total_chars = 0
        final_docs = []
        
        for doc in candidates:
            doc_length = len(doc['text'])
            if total_chars + doc_length > max_context_chars:
                # Try to include partial doc if it fits
                remaining = max_context_chars - total_chars
                if remaining > 100:  # Only if meaningful chunk remains
                    doc_partial = {**doc, 'text': doc['text'][:remaining] + '...'}
                    final_docs.append(doc_partial)
                break
            
            final_docs.append(doc)
            total_chars += doc_length
        
        logger.info(f" Hybrid retrieval: {len(final_docs)} docs ({total_chars} chars, limit: {max_context_chars})")
        
        return final_docs, timing
    
    def _extract_template_fields(self, docs: list, pattern: Dict) -> Dict:
        """
        Extract structured fields from retrieved docs for template filling
        
        Args:
            docs: Retrieved document dicts
            pattern: Pattern configuration dict
            
        Returns:
            dict: Extracted fields for template (e.g., {department, hours, ...})
        """
        extracted = {}
        pattern_name = pattern.get("name", "")
        
        # Combine all doc texts for extraction
        combined_text = "\n".join([doc.get('text', '') for doc in docs])
        
        # Pattern-specific extraction logic
        if pattern_name == "office_hours":
            # Extract hours pattern (e.g., "Monday-Friday 9:00-15:00")
            hours_match = re.search(r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*[-–]\s*(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?\s*\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2})', combined_text)
            if hours_match:
                extracted["hours"] = hours_match.group(1)
            
            # Find department mention
            dept_keywords = ["admissions", "academic", "student services", "registry", "enrollment"]
            for dept in dept_keywords:
                if dept in combined_text.lower():
                    extracted["department"] = dept.title()
                    break
            
            # Extract additional info (walk-in, appointment, etc.)
            if "walk-in" in combined_text.lower():
                walk_in_match = re.search(r'walk-in.*?(\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2})', combined_text, re.IGNORECASE)
                if walk_in_match:
                    extracted["additional_info"] = f"Walk-in hours: {walk_in_match.group(1)}"
        
        elif pattern_name == "admission_requirements":
            # ENHANCED: More specific extraction for admission requirements
            # Extract program mention (with context)
            programs = {
                "master": ["master", "master's", "msc", "m.sc"],
                "bachelor": ["bachelor", "bachelor's", "bsc", "b.sc"],
                "phd": ["phd", "doctoral", "doctorate"],
                "mba": ["mba", "business administration"]
            }
            for prog_key, prog_variants in programs.items():
                if any(variant in combined_text.lower() for variant in prog_variants):
                    extracted["program"] = prog_key.upper() if prog_key == "mba" else prog_key.title()
                    break
            
            # Extract specific requirements with categories
            requirements_found = []
            
            # Check for GPA/grades
            gpa_match = re.search(r'(GPA|grade|average).*?(\d+\.?\d*)', combined_text, re.IGNORECASE)
            if gpa_match:
                requirements_found.append(f"GPA requirement: {gpa_match.group(2)}")
            
            # Check for language requirements
            if re.search(r'(TOEFL|IELTS|language|English|German)', combined_text, re.IGNORECASE):
                lang_match = re.search(r'(TOEFL.*?(?:\d+)|IELTS.*?(?:\d+\.?\d*))', combined_text, re.IGNORECASE)
                if lang_match:
                    requirements_found.append(f"Language: {lang_match.group(1)}")
                else:
                    requirements_found.append("Language proficiency required")
            
            # Check for degree requirements
            degree_match = re.search(r'(bachelor[\'s]?|undergraduate)\s+degree', combined_text, re.IGNORECASE)
            if degree_match:
                requirements_found.append(f"Previous {degree_match.group(1)} degree required")
            
            # Extract bulleted/numbered requirements
            req_lines = [line.strip() for line in combined_text.split('\n') 
                        if line.strip() and (line.strip().startswith(('-', '•', '*')) or re.match(r'^\d+\.', line.strip()))]
            if req_lines:
                # Filter out too-long lines (likely not actual requirements)
                clean_reqs = [r for r in req_lines if len(r) < 150][:3]
                requirements_found.extend(clean_reqs)
            
            if requirements_found:
                extracted["requirements"] = "; ".join(requirements_found[:4])  # Top 4 requirements
            
            # Extract application link/portal
            link_match = re.search(r'(https?://[^\s]+(?:application|apply|admission)[^\s]*)', combined_text, re.IGNORECASE)
            if link_match:
                extracted["application_link"] = link_match.group(1)
        
        elif pattern_name == "contact_info":
            # ENHANCED: More aggressive rule-based extraction for contact info
            # Extract ALL emails (prioritize .uni-hannover.de)
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', combined_text)
            uni_emails = [e for e in emails if 'uni-hannover' in e.lower() or 'leibniz' in e.lower()]
            if uni_emails:
                extracted["email"] = uni_emails[0]  # Prioritize university email
            elif emails:
                extracted["email"] = emails[0]
            
            # Extract ALL phones (prioritize German format)
            phones = re.findall(r'(\+?\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4})', combined_text)
            if phones:
                extracted["phone"] = phones[0]
            
            # Extract office/department names
            dept_patterns = [
                r'(Registrar[\'s]?\s+Office)',
                r'(Student\s+Services)',
                r'(Academic\s+Affairs)',
                r'(Admissions\s+Office)',
                r'(International\s+Office)',
                r'(Central\s+Student\s+Advisory\s+Service)',
            ]
            for pattern_regex in dept_patterns:
                dept_match = re.search(pattern_regex, combined_text, re.IGNORECASE)
                if dept_match:
                    extracted["department"] = dept_match.group(1)
                    break
            
            # Extract office hours if mentioned
            hours_match = re.search(r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Mon|Tue|Wed|Thu|Fri)\s*[-–]\s*(?:Monday|Tuesday|Wednesday|Thursday|Friday|Mon|Tue|Wed|Thu|Fri)?\s*\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2})', combined_text, re.IGNORECASE)
            if hours_match:
                extracted["hours"] = hours_match.group(1)
            
            # Extract building/room numbers
            room_match = re.search(r'((?:Building|Room|Office)\s+[A-Z]?\d+[A-Za-z]?(?:\s*,\s*Room\s+\d+)?)', combined_text, re.IGNORECASE)
            if room_match:
                extracted["location"] = room_match.group(1)
            
            # Build contact summary
            contact_parts = []
            if "department" in extracted:
                contact_parts.append(f"the {extracted['department']}")
            if "email" in extracted:
                contact_parts.append(f"email: {extracted['email']}")
            if "phone" in extracted:
                contact_parts.append(f"phone: {extracted['phone']}")
            if "location" in extracted:
                contact_parts.append(f"location: {extracted['location']}")
            if "hours" in extracted:
                contact_parts.append(f"hours: {extracted['hours']}")
                
            if contact_parts:
                extracted["contact_summary"] = " | ".join(contact_parts)
        
        elif pattern_name == "appointment_scheduling":
            # Extract steps/instructions
            step_lines = [line.strip() for line in combined_text.split('\n')
                         if line.strip() and (re.match(r'^\d+\.', line.strip()) or 'step' in line.lower())]
            if step_lines:
                extracted["steps"] = " ".join(step_lines[:3])

        # TASK ORGANIZATION SPECIFIC EXTRACTION LOGIC
        elif pattern_name == "organization_info":
            # Extract organization description, mission, vision
            if "government of telangana" in combined_text.lower():
                extracted["description"] = "government-backed not-for-profit organization established by Telangana Government"
            elif "skill development" in combined_text.lower():
                extracted["description"] = "skill development organization focused on employability enhancement"
            elif "tri-partite partnership" in combined_text.lower():
                extracted["description"] = "organization following government-academia-industry collaboration model"

            # Extract additional info like statistics
            stats_match = re.search(r'(\d+(?:,\d+)*)\s+(?:students?|trainees?)', combined_text, re.IGNORECASE)
            if stats_match:
                extracted["additional_info"] = f"Served {stats_match.group(1)} students"

        elif pattern_name in ["contact_info", "task_contact"]:
            # Extract TASK-specific contact information
            # Extract emails (prioritize telangana.gov.in)
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', combined_text)
            telangana_emails = [e for e in emails if 'telangana.gov.in' in e.lower() or 'task.' in e.lower()]
            if telangana_emails:
                extracted["email"] = telangana_emails[0]
            elif emails:
                extracted["email"] = emails[0]

            # Extract phones (Indian format)
            phones = re.findall(r'(\+?91[\s-]?\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4})', combined_text)
            if phones:
                extracted["phone"] = phones[0]

            # Extract address/location
            address_patterns = [
                r'1st\s+Floor.*Masabtank.*Hyderabad',
                r'Hyderabad.*500028',
                r'Masabtank.*Hyderabad',
                r'Telangana.*India'
            ]
            for pattern in address_patterns:
                addr_match = re.search(pattern, combined_text, re.IGNORECASE)
                if addr_match:
                    extracted["location"] = addr_match.group(0)
                    break

            # Extract office hours
            hours_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\s*[-–]\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)', combined_text)
            if hours_match:
                extracted["hours"] = hours_match.group(1)

            # Build contact methods summary
            contact_parts = []
            if "phone" in extracted:
                contact_parts.append(f"phone: {extracted['phone']}")
            if "email" in extracted:
                contact_parts.append(f"email: {extracted['email']}")
            if "location" in extracted:
                contact_parts.append(f"address: {extracted['location']}")
            if "hours" in extracted:
                contact_parts.append(f"hours: {extracted['hours']}")

            if contact_parts:
                extracted["contact_methods"] = ", ".join(contact_parts)

        elif pattern_name in ["office_hours", "task_timing"]:
            # Extract TASK office hours
            hours_patterns = [
                r'Monday-Friday.*?(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\s*[-–]\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)',
                r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\s*[-–]\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?).*?IST',
                r'9:30\s*(?:AM|am)?\s*[-–]\s*5:00\s*(?:PM|pm)?'
            ]
            for pattern in hours_patterns:
                hours_match = re.search(pattern, combined_text, re.IGNORECASE)
                if hours_match:
                    extracted["hours"] = hours_match.group(1) if hours_match.groups() else hours_match.group(0)
                    break

            # Default to standard hours if not found
            if "hours" not in extracted:
                extracted["hours"] = "Monday-Friday, 9:30 AM - 5:00 PM IST"

            extracted["additional_info"] = "Weekend closed, government holidays observed"

        elif pattern_name in ["registration_process", "task_registration"]:
            # Extract registration steps from the process document
            steps = []
            step_patterns = [
                r'Step\s*\d+:\s*([^.\n]+)',
                r'###\s*([^.\n]+)',
                r'\d+\.\s*([^.\n]+)'
            ]
            for pattern in step_patterns:
                matches = re.findall(pattern, combined_text)
                if matches:
                    steps.extend(matches[:5])  # Limit to first 5 steps
                    break

            if steps:
                extracted["steps"] = "; ".join(steps)
            else:
                extracted["steps"] = "Visit task.telangana.gov.in, create account, fill personal/academic info, select programs, upload documents, complete payment if applicable"

        elif pattern_name in ["program_eligibility", "task_eligibility"]:
            # Extract eligibility criteria
            eligibility_parts = []

            # Academic requirements
            if "50%" in combined_text or "minimum percentage" in combined_text.lower():
                eligibility_parts.append("50% minimum aggregate for engineering/polytechnic students")
            if "60%" in combined_text:
                eligibility_parts.append("60% in 10th, Intermediate, and Graduation for graduates")

            # Student categories
            categories = []
            if "engineering students" in combined_text.lower():
                categories.append("engineering students")
            if "polytechnic" in combined_text.lower():
                categories.append("polytechnic students")
            if "graduates" in combined_text.lower():
                categories.append("graduates and unemployed youth")

            if categories:
                eligibility_parts.append(f"Eligible categories: {', '.join(categories)}")

            if eligibility_parts:
                extracted["requirements"] = "; ".join(eligibility_parts)

        elif pattern_name in ["program_costs", "task_fees"]:
            # Extract fee information
            cost_info = []

            if "free" in combined_text.lower() or "subsidized" in combined_text.lower():
                cost_info.append("Most programs are free or heavily subsidized (40-83% discounts)")

            if "financial assistance" in combined_text.lower():
                cost_info.append("Financial assistance available for economically backward students")

            if "government funded" in combined_text.lower():
                cost_info.append("Many programs fully funded by government or industry partners")

            if cost_info:
                extracted["cost_info"] = "; ".join(cost_info)
            else:
                extracted["cost_info"] = "Most programs are free or subsidized; financial assistance available"

        elif pattern_name in ["placement_stats", "task_placement"]:
            # Extract placement statistics
            stats_parts = []

            # Overall placement rate
            rate_match = re.search(r'(\d+(?:\.\d+)?)%\s*(?:placement|placement rate)', combined_text, re.IGNORECASE)
            if rate_match:
                stats_parts.append(f"Overall placement rate: {rate_match.group(1)}%")

            # Salary ranges
            salary_match = re.search(r'(₹[\d,]+(?:-[\d,]+)?\s*LPA)', combined_text)
            if salary_match:
                stats_parts.append(f"Average salary: {salary_match.group(1)}")

            # Number of placements
            placement_match = re.search(r'(\d+(?:,\d+)*)\s*(?:placements?|students?\s+placed)', combined_text, re.IGNORECASE)
            if placement_match:
                stats_parts.append(f"Students placed: {placement_match.group(1)}")

            # Companies
            if "infosys" in combined_text.lower() or "tcs" in combined_text.lower():
                stats_parts.append("Top companies: Infosys, TCS, IBM, Microsoft, and 80+ industry partners")

            if stats_parts:
                extracted["stats"] = "; ".join(stats_parts)
            else:
                extracted["stats"] = "85%+ overall placement rate; Average salary ₹3.5-6.5 LPA; 80+ industry partners"

        elif pattern_name in ["available_programs", "task_programs"]:
            # Extract available programs
            programs = []

            tech_programs = ["AI & Machine Learning", "Cloud Computing", "Cybersecurity", "Data Science",
                           "Full Stack Development", "Java Programming", "Blockchain", "IoT"]
            for program in tech_programs:
                if program.lower() in combined_text.lower():
                    programs.append(program)

            if not programs:
                programs = ["AI & Machine Learning", "Cloud Computing", "Cybersecurity", "Data Science",
                          "Full Stack Development", "Java Programming", "Manufacturing", "Healthcare"]

            extracted["programs"] = f"{len(programs)}+ programs including: {', '.join(programs[:6])}"
            if len(programs) > 6:
                extracted["programs"] += f" and {len(programs) - 6} more"

        elif pattern_name == "task_documents":
            # Extract required documents
            documents = ["College ID", "Aadhar Card", "Photo"]
            if "marksheets" in combined_text.lower():
                documents.append("Academic Marksheets")

            extracted["documents"] = ", ".join(documents)

        elif pattern_name == "task_support":
            # Extract support information
            support_methods = []
            if "email" in combined_text.lower():
                support_methods.append("email support")
            if "phone" in combined_text.lower():
                support_methods.append("phone support")
            if "portal" in combined_text.lower():
                support_methods.append("student portal")
            if "whatsapp" in combined_text.lower():
                support_methods.append("WhatsApp support")

            if support_methods:
                extracted["solution"] = f"Contact support via {', '.join(support_methods)}. 24/7 digital support available."
            else:
                extracted["solution"] = "Contact enquiry_task@telangana.gov.in or call +91-40-35485290. 24/7 digital support available."

        elif pattern_name == "task_status":
            # Extract status check information
            status_methods = []
            if "portal" in combined_text.lower():
                status_methods.append("student portal")
            if "email" in combined_text.lower():
                status_methods.append("email")
            if "phone" in combined_text.lower():
                status_methods.append("phone support")

            if status_methods:
                extracted["steps"] = f"Check your application status via {', '.join(status_methods)}"
            else:
                extracted["steps"] = "Check your application status via student portal or contact support"

        return extracted
    
    def _build_hybrid_prompt(self, query: str, extracted_info: Dict, pattern: Dict, context: Optional[Dict], retrieved_docs: list) -> str:
        """
        Build SHORT prompt with template-based structure (reduces tokens by 50-70%)
        Supports Telugu mixed responses for TARA mode.
        
        Args:
            query: User query
            extracted_info: Extracted template fields
            pattern: Pattern configuration
            context: Structured context from intent parser
            retrieved_docs: Retrieved documents
            
        Returns:
            str: Compact Gemini prompt
        """
        template = pattern["response_template"]
        pattern_name = pattern["name"]
        
        # Build compact context summary (only essential info)
        context_summary = ""
        if retrieved_docs:
            # Take first 300 chars of top doc (increased for better context)
            top_doc_text = retrieved_docs[0].get('text', '')[:300]
            context_summary = f"\nKey information: {top_doc_text}..."
        
        # Determine language instruction based on config and context
        language_instruction = ""
        if self.config.tara_mode or self.config.response_language.startswith("te"):
            language_instruction = """IMPORTANT: Respond in Telugu mixed with English (Tenglish style).
- DO NOT introduce yourself or use greetings (you're already in conversation)
- Use professional Telugu slang naturally: "అండి", "చూద్దాం", "అవును", "కదా", "గా", "లో", "తో"
- Use English for technical terms, numbers, and proper nouns
- Example: "Cybersecurity program placement rate 78% and average salary ₹4-5.5 LPA ఉంటుంది"
- Be warm, professional, and direct - like a helpful colleague, not a formal introduction
- Respond directly to the question without greetings"""
        elif context and context.get('language') == 'german':
            language_instruction = "Respond in German."
        elif self.config.response_language == "hi-mixed":
            language_instruction = "Respond in Hindi mixed with English (Hinglish style)."
        
        # TARA MODE: Telugu prompts for TASK patterns
        if self.config.tara_mode and pattern_name.startswith("task_"):
            return self._build_tara_prompt(query, extracted_info, pattern, context_summary, language_instruction)
        
        # ENHANCED: Pattern-specific prompt templates for better accuracy
        if pattern_name == "contact_info":
            # Force direct extraction-based response
            if extracted_info:
                prompt = f"""They asked: "{query}"

Here's the contact info you have:
{json.dumps(extracted_info, indent=2, ensure_ascii=False)}
{context_summary}

{language_instruction}

Give them the contact information directly - email, phone, location, hours, whatever's relevant. Keep it short and clear (2-3 sentences). Keep your response between 150-200 characters total. Just give them what they need.

Your response:"""
            else:
                # Fallback if extraction failed
                prompt = f"""They asked: "{query}"
{context_summary}

{language_instruction}

Give them the contact information from what you know. Be specific and direct. Keep your response between 150-200 characters total.

Your response:"""
        
        elif pattern_name == "admission_requirements":
            # Force specific, structured requirements response
            if extracted_info:
                prompt = f"""They asked: "{query}"

Requirements you found:
{json.dumps(extracted_info, indent=2, ensure_ascii=False)}
{context_summary}

{language_instruction}

List out the admission requirements clearly. Mention the program if you know it. Keep it structured (3-4 sentences). Keep your response between 150-200 characters total. Only suggest contacting admissions if you're missing key info.

Your response:"""
            else:
                prompt = f"""They asked: "{query}"
{context_summary}

{language_instruction}

Give them the admission requirements based on what you know. Be direct and clear. Keep your response between 150-200 characters total.

Your response:"""
        
        else:
            # Generic template for other patterns
            prompt = f"""They asked: "{query}"

Pattern type: {pattern_name.replace('_', ' ')}
Details: {json.dumps(extracted_info, ensure_ascii=False)}
{context_summary}

{language_instruction}

Answer directly using the details you have. Be friendly and conversational (2-3 sentences). Keep your response between 150-200 characters total. If you don't have complete info, suggest who they should contact. Don't mention any template structure.

Your response:"""
        
        return prompt
    
    def _build_tara_prompt(self, query: str, extracted_info: Dict, pattern: Dict, context_summary: str, language_instruction: str) -> str:
        """
        Build prompt specifically for TARA Telugu customer service agent.
        
        Args:
            query: User query (may be in Telugu)
            extracted_info: Extracted template fields
            pattern: Pattern configuration (TASK patterns)
            context_summary: Context from retrieved docs
            language_instruction: Language instructions
            
        Returns:
            str: Telugu-optimized prompt
        """
        pattern_name = pattern["name"]
        org_name = self.config.organization_name
        agent_name = self.config.agent_name
        
        # Base TARA prompt - conversational, no greetings (middle of conversation)
        tara_base = f"""మీరు {agent_name}, {org_name} customer service agent. మీరు already conversation middle lo unnaru, so greetings ivvakandi.
User question: "{query}"

{language_instruction}

CRITICAL INSTRUCTIONS:
- DO NOT introduce yourself or say greetings like "నమస్కారం", "హాయ్", "Hello"
- Respond directly to the question as if already in conversation
- Use professional Telugu slang naturally: "అండి", "చూద్దాం", "అవును", "కదా", "గా"
- Mix Telugu and English naturally (Tenglish style)
- Be warm but professional, like a helpful colleague
- Keep it concise and direct

Available information:
{json.dumps(extracted_info, indent=2, ensure_ascii=False)}
{context_summary}

"""
        
        # Pattern-specific Telugu prompts - direct, conversational, no greetings
        if pattern_name == "task_contact":
            prompt = tara_base + f"""
Contact details direct గా mention చేయండి - phone, email, address, timing.
Professional Telugu slang use చేయండి naturally. 2-3 sentences lo concise answer.

Response:"""
        
        elif pattern_name == "task_timing":
            prompt = tara_base + f"""
Office timings direct గా చెప్పండి - days and time clearly.
Professional slang use చేస్తూ helpful tone maintain చేయండి.

Response:"""
        
        elif pattern_name == "task_services":
            prompt = tara_base + f"""
{org_name} services గురించి explain చేయండి - clear and concise.
Professional Telugu slang తో natural response ఇవ్వండి.

Response:"""
        
        elif pattern_name == "task_registration":
            prompt = tara_base + f"""
Registration process step by step explain చేయండి.
Required documents mention చేయండి. Professional slang use చేస్తూ guide చేయండి.

Response:"""
        
        elif pattern_name == "task_fees":
            prompt = tara_base + f"""
Fees and charges clear గా mention చేయండి.
Payment methods available అయితే add చేయండి. Professional slang తో natural response.

Response:"""
        
        elif pattern_name == "task_support":
            prompt = tara_base + f"""
Customer issue కు solution direct గా ఇవ్వండి.
Empathetic tone maintain చేస్తూ clear steps mention చేయండి. Professional slang use చేయండి.

Response:"""
        
        elif pattern_name == "task_documents":
            prompt = tara_base + f"""
Required documents list చేయండి - each document purpose briefly mention చేయండి.
Professional slang తో natural response.

Response:"""
        
        elif pattern_name == "task_status":
            prompt = tara_base + f"""
Status check process explain చేయండి - online/phone/visit methods mention చేయండి.
Professional slang use చేస్తూ helpful tone.

Response:"""
        
        else:
            # Generic TARA prompt - direct and conversational
            prompt = tara_base + f"""
Query కు direct గా helpful answer ఇవ్వండి.
Professional Telugu slang use చేస్తూ natural, warm tone maintain చేయండి. 2-3 sentences sufficient.

Response:"""
        
        return prompt
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable[[str, bool], None]] = None
    ) -> Dict[str, Any]:
        """
        Process RAG query with context-aware retrieval.
        
        Args:
            query: User question
            context: Optional context from intent service (user_goal, key_entities, extracted_meaning)
            streaming_callback: Optional callback for streaming responses
            
        Returns:
            Dictionary with answer, sources, confidence, timing_breakdown, metadata
        """
        start_time = time.time()
        timing = {}
        
        try:
            # Step 1: Extract query from context
            query_text = query
            if context:
                # Priority: extracted_meaning > user_goal > raw query
                if 'extracted_meaning' in context and context['extracted_meaning']:
                    query_text = context['extracted_meaning']
                elif 'user_goal' in context and context['user_goal']:
                    query_text = context['user_goal']
            
            # Step 2: Validate components
            if not self.embeddings or not self.vector_store or not self.gemini_model:
                logger.warning("️ Components unavailable, falling back to Gemini-only")
                return await self.gemini_only_query(query_text, context, streaming_callback)
            
            # HYBRID APPROACH: Check for pattern-based optimization (NEW STEP 2.5)
            if self.config.enable_hybrid_search:
                pattern_start = time.time()
                detected_pattern = self._detect_query_pattern(query_text)
                timing['pattern_detection_ms'] = (time.time() - pattern_start) * 1000
                
                if detected_pattern:
                    # HYBRID PATH: Use rule-based boosting + reduced context
                    logger.info(f" Pattern detected: {detected_pattern['name']} (optimized hybrid path)")
                    
                    try:
                        # Retrieve with category boosting and context truncation
                        relevant_docs, retrieval_timing = self._retrieve_with_boosting(
                            query_text,
                            context or {},
                            boost_categories=detected_pattern.get("faiss_boost", []),
                            max_context_chars=detected_pattern.get("max_context_chars", 800)
                        )
                        timing.update(retrieval_timing)
                        
                        # Extract structured fields for template
                        extract_start = time.time()
                        extracted_info = self._extract_template_fields(relevant_docs, detected_pattern)
                        timing['extraction_ms'] = (time.time() - extract_start) * 1000
                        
                        # Build compact hybrid prompt (50-70% fewer tokens)
                        prompt = self._build_hybrid_prompt(
                            query_text,
                            extracted_info,
                            detected_pattern,
                            context,
                            relevant_docs
                        )
                        
                        # Generate response with compact prompt
                        gen_start = time.time()
                        
                        if streaming_callback:
                            # Streaming generation for Hybrid Path
                            accumulated_text = ""
                            try:
                                response_stream = self.gemini_model.generate_content(
                                    prompt,
                                    generation_config=genai.types.GenerationConfig(
                                        temperature=0.7,
                                        top_p=0.9,
                                        max_output_tokens=150,  # ~200-300 chars for 2-3 sentences
                                    ),
                                    stream=True
                                )
                                
                                sentence_buffer = ""
                                for chunk in response_stream:
                                    chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                                    accumulated_text += chunk_text
                                    sentence_buffer += chunk_text
                                    
                                    # Split by sentence boundaries
                                    sentences = re.split(r'[.!?]\s+', sentence_buffer)
                                    if len(sentences) > 1:
                                        for complete_sentence in sentences[:-1]:
                                            if complete_sentence.strip():
                                                streaming_callback(complete_sentence.strip() + '.', False)
                                        sentence_buffer = sentences[-1]
                                
                                if sentence_buffer.strip():
                                    streaming_callback(sentence_buffer.strip(), True)
                                    
                                answer = accumulated_text.strip()
                            except Exception as e:
                                logger.error(f"Hybrid streaming error: {e}")
                                # Fallback to non-streaming
                                response = self.gemini_model.generate_content(
                                    prompt,
                                    generation_config=genai.types.GenerationConfig(
                                        temperature=0.7,
                                        top_p=0.9,
                                        max_output_tokens=150,  # ~200-300 chars for 2-3 sentences
                                    )
                                )
                                answer = response.text.strip()
                        else:
                            response = self.gemini_model.generate_content(
                                prompt,
                                generation_config=genai.types.GenerationConfig(
                                    temperature=0.7,
                                    top_p=0.9,
                                    max_output_tokens=150,  # ~200-300 chars for 2-3 sentences  # 100-130 for 150-200 char responses
                                )
                            )
                            answer = response.text.strip()
                        timing['generation_ms'] = (time.time() - gen_start) * 1000
                        
                        # Get sources from retrieved docs
                        sources = list(set([doc['metadata'].get('source', 'Unknown') for doc in relevant_docs]))
                        
                        # Calculate confidence
                        avg_similarity = sum(d.get('similarity', 0) for d in relevant_docs) / len(relevant_docs) if relevant_docs else 0.0
                        quality = self.validate_response_quality(answer)
                        confidence = min(avg_similarity, quality.get('quality_score', 0.5))
                        
                        # Humanize response if enabled
                        is_first_turn = True
                        if context:
                            turn_number = context.get('turn_number', 1)
                            conversation_history = context.get('conversation_history', [])
                            is_first_turn = (turn_number <= 1 and len(conversation_history) == 0)
                        
                        final_response = self.humanize_response(answer, query_text, context, is_first_turn) if self.config.enable_humanization else answer
                        
                        timing['total_ms'] = (time.time() - start_time) * 1000
                        
                        # Update metrics
                        self.query_count += 1
                        self.total_query_time += timing['total_ms']
                        
                        return {
                            'answer': final_response,
                            'sources': sources,
                            'confidence': confidence,
                            'timing_breakdown': timing,
                            'metadata': {
                                'categories': list(set(d['metadata'].get('category', '') for d in relevant_docs)),
                                'quality_score': quality.get('quality_score', 0.0),
                                'num_docs_retrieved': len(relevant_docs),
                                'pattern': detected_pattern.get('name', 'unknown'),
                                'method': 'hybrid'
                            }
                        }
                        
                    except Exception as e:
                        logger.error(f" Hybrid generation failed: {e}")
                        # Fall through to standard RAG path
                        if detected_pattern:
                            logger.info(" Falling back to standard RAG (hybrid failed)")
            
            # STANDARD RAG PATH: No pattern match or hybrid failed
            # Step 3: Enrich query with entities
            enriched_query = query_text
            if context and 'key_entities' in context:
                entities = context['key_entities']
                entity_terms = ' '.join([f"{k} {v}" for k, v in entities.items()])
                enriched_query = f"{query_text} {entity_terms}"
            
            # Add user goal for context
            if context and 'user_goal' in context:
                user_goal = context['user_goal']
                enriched_query = f"{enriched_query} {user_goal}"
            
            # Step 4: Embed query
            embed_start = time.time()
            query_embedding = self.embeddings.embed_query(enriched_query)
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            timing['embedding_ms'] = (time.time() - embed_start) * 1000
            
            # Step 5: FAISS search
            search_start = time.time()
            distances, indices = self.vector_store.search(query_embedding, k=self.config.top_k)
            timing['search_ms'] = (time.time() - search_start) * 1000
            
            # Step 6: Filter by similarity and apply boosting
            relevant_docs = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    distance = float(distances[0][i])
                    similarity = 1.0 - (distance * distance / 2.0)
                    
                    if similarity < self.config.similarity_threshold:
                        continue
                    
                    doc_text = self.documents[idx]
                    doc_meta = self.doc_metadata[idx] if idx < len(self.doc_metadata) else {}
                    
                    relevant_docs.append({
                        'text': doc_text,
                        'metadata': doc_meta,
                        'distance': distance,
                        'similarity': similarity,
                        'priority_boost': 0
                    })
            
            # Entity-based boosting
            if context and 'key_entities' in context:
                entities = context['key_entities']
                for doc in relevant_docs:
                    category = doc['metadata'].get('category', '').lower()
                    
                    # Boost by category matching
                    if any(e in ['program', 'admission', 'enrollment'] for e in entities.keys()):
                        if 'admission' in category or 'enrollment' in category:
                            doc['priority_boost'] = 10
                        elif 'program' in category or 'academic' in category:
                            doc['priority_boost'] = 6
                    elif any(e in ['service', 'housing', 'financial'] for e in entities.keys()):
                        if 'student_services' in category:
                            doc['priority_boost'] = 10
                    elif any(e in ['course', 'class'] for e in entities.keys()):
                        if 'academic' in category or 'program' in category:
                            doc['priority_boost'] = 10
                    elif any(e in ['contact', 'office', 'emergency'] for e in entities.keys()):
                        if 'contact' in category:
                            doc['priority_boost'] = 10
                    else:
                        doc['priority_boost'] = doc['metadata'].get('priority', 0)
                
                # Re-rank by priority + similarity
                relevant_docs.sort(key=lambda x: (-x.get('priority_boost', 0), -x.get('similarity', 0)))
            
            # Select top N
            relevant_docs = relevant_docs[:self.config.top_n]
            
            # Step 7: Build prompt
            context_text = "\n\n".join([doc['text'] for doc in relevant_docs])
            sources = list(set([doc['metadata'].get('source', 'Unknown') for doc in relevant_docs]))
            
            user_goal_text = context.get('user_goal', 'general information') if context else 'general information'
            
            # Build prompt based on mode (TARA Telugu vs Standard)
            if self.config.tara_mode or self.config.response_language.startswith("te"):
                # TARA Telugu mode prompt
                prompt = f"""మీరు {self.config.agent_name}, {self.config.organization_name} యొక్క customer service agent.

User అడిగారు: "{query_text}"

Knowledge base information:
{context_text}

IMPORTANT: Respond in Telugu mixed with English (Tenglish style):
- Telugu for conversational parts, greetings, and common phrases
- English for technical terms, numbers, proper nouns, and specific details
- Example: "మీ application status check చేయడానికి, online portal లో login చేయండి"

Response guidelines:
- Be warm, helpful, and professional
- Give practical, accurate information
- If information is not available, politely say so in Telugu
- Keep response to 3-5 sentences
- Sound natural like a customer service representative

మీ response:"""
            else:
                # Standard English mode prompt
                prompt = f"""You're {self.config.agent_name}, the assistant at {self.config.organization_name}. A user just asked: "{query_text}"

Here's what you know from the knowledge base:
{context_text}

Respond naturally and directly. Keep it friendly but not overly enthusiastic. Focus on giving them practical, accurate information they can use. If the knowledge base doesn't cover their question, say so honestly and suggest where they might find the answer.

Key points:
- Be conversational but not chatty
- Get straight to the answer
- Use simple language
- Stay grounded in the knowledge base
- Aim for 3-5 sentences typically

Your response:"""
            
            # Step 8: Generate response
            gen_start = time.time()
            
            if streaming_callback:
                # Streaming generation
                accumulated_text = ""
                try:
                    response_stream = self.gemini_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            top_p=0.9,
                            top_k=40,
                            max_output_tokens=150,  # ~200-300 chars for 2-3 sentences
                        ),
                        stream=True
                    )
                    
                    sentence_buffer = ""
                    for chunk in response_stream:
                        chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                        accumulated_text += chunk_text
                        sentence_buffer += chunk_text
                        
                        # Split by sentence boundaries
                        sentences = re.split(r'[.!?]\s+', sentence_buffer)
                        if len(sentences) > 1:
                            for complete_sentence in sentences[:-1]:
                                if complete_sentence.strip():
                                    streaming_callback(complete_sentence.strip() + '.', False)
                            sentence_buffer = sentences[-1]
                    
                    if sentence_buffer.strip():
                        streaming_callback(sentence_buffer.strip(), True)
                    
                    raw_response = accumulated_text
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    response = self.gemini_model.generate_content(prompt)
                    raw_response = response.text if response else "Sorry, I couldn't generate a response."
            else:
                # Standard generation
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=150,  # ~200-300 chars for 2-3 sentences
                    )
                )
                raw_response = response.text if response else "Sorry, I couldn't generate a response."
            
            timing['generation_ms'] = (time.time() - gen_start) * 1000
            
            # Step 9: Validate quality
            quality = self.validate_response_quality(raw_response)
            if quality.get('retry', False) and quality.get('quality_score', 0) < self.config.min_quality_score:
                retry_prompt = prompt + "\n\nIMPORTANT: Please make the response more friendly and conversational, avoiding formal language."
                response = self.gemini_model.generate_content(retry_prompt)
                raw_response = response.text if response else raw_response
            
            # Step 10: Humanize response
            is_first_turn = True
            if context:
                turn_number = context.get('turn_number', 1)
                conversation_history = context.get('conversation_history', [])
                is_first_turn = (turn_number <= 1 and len(conversation_history) == 0)
            
            final_response = self.humanize_response(raw_response, query_text, context, is_first_turn) if self.config.enable_humanization else raw_response
            
            timing['total_ms'] = (time.time() - start_time) * 1000
            
            # Update metrics
            self.query_count += 1
            self.total_query_time += timing['total_ms']
            
            logger.info(f"✅ RAG Generation Complete ({timing['total_ms']:.0f}ms)")
            logger.info(f"   Response: {final_response}")
            
            # Calculate confidence
            avg_similarity = sum(d['similarity'] for d in relevant_docs) / len(relevant_docs) if relevant_docs else 0.0
            confidence = min(avg_similarity, quality.get('quality_score', 0.5))
            
            # Step 11: Return structured result
            return {
                'answer': final_response,
                'sources': sources,
                'confidence': confidence,
                'timing_breakdown': timing,
                'metadata': {
                    'categories': list(set(d['metadata'].get('category', '') for d in relevant_docs)),
                    'quality_score': quality.get('quality_score', 0.0),
                    'num_docs_retrieved': len(relevant_docs)
                }
            }
        
        except Exception as e:
            logger.error(f" RAG query error: {e}", exc_info=True)
            return {
                'answer': "I apologize, but I encountered an error while processing your question. Could you please try rephrasing it?",
                'sources': [],
                'confidence': 0.0,
                'timing_breakdown': {'total_ms': (time.time() - start_time) * 1000},
                'metadata': {'error': str(e)}
            }

    async def process_query_with_context(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        intent_context: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable[[str, bool], None]] = None
    ) -> Dict[str, Any]:
        """
        Process RAG query with PRE-RETRIEVED documents (for incremental buffering).
        
        This method skips the retrieval phase and uses provided documents directly.
        Used by the incremental endpoint to avoid redundant FAISS searches.
        
        Args:
            query: User question
            context_docs: Pre-retrieved document dicts with 'text', 'metadata', 'similarity'
            intent_context: Optional context from intent service
            streaming_callback: Optional callback for streaming responses
            
        Returns:
            Dictionary with answer, sources, confidence, timing_breakdown, metadata
        """
        start_time = time.time()
        timing = {}
        
        try:
            query_text = query
            if intent_context:
                if 'extracted_meaning' in intent_context and intent_context['extracted_meaning']:
                    query_text = intent_context['extracted_meaning']
                elif 'user_goal' in intent_context and intent_context['user_goal']:
                    query_text = intent_context['user_goal']
            
            # Use provided docs directly (skip retrieval)
            timing['retrieval_ms'] = 0  # Docs were pre-retrieved
            relevant_docs = context_docs[:self.config.top_n] if context_docs else []
            
            if not relevant_docs:
                logger.warning("⚠️ No context docs provided, falling back to Gemini-only")
                return await self.gemini_only_query(query_text, intent_context, streaming_callback)
            
            # Build context and sources
            context_text = "\n\n".join([doc.get('text', '') for doc in relevant_docs])
            sources = list(set([doc.get('metadata', {}).get('source', 'Unknown') for doc in relevant_docs]))
            
            # Build prompt (TARA mode vs Standard)
            if self.config.tara_mode or self.config.response_language.startswith("te"):
                prompt = f"""మీరు {self.config.agent_name}, {self.config.organization_name} యొక్క customer service agent.

User అడిగారు: "{query_text}"

Knowledge base information:
{context_text}

IMPORTANT: Respond in Telugu mixed with English (Tenglish style).
- Telugu for conversational parts
- English for technical terms, numbers, proper nouns
- Be warm, helpful, professional
- Keep response to 3-5 sentences

మీ response:"""
            else:
                prompt = f"""You're {self.config.agent_name}, the assistant at {self.config.organization_name}. A user asked: "{query_text}"

Here's what you know:
{context_text}

Respond naturally, directly, and helpfully. 3-5 sentences.

Your response:"""
            
            # Generate response
            gen_start = time.time()
            
            if streaming_callback:
                accumulated_text = ""
                try:
                    response_stream = self.gemini_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            top_p=0.9,
                            max_output_tokens=150,  # ~200-300 chars for 2-3 sentences
                        ),
                        stream=True
                    )
                    
                    sentence_buffer = ""
                    for chunk in response_stream:
                        chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                        accumulated_text += chunk_text
                        sentence_buffer += chunk_text
                        
                        sentences = re.split(r'[.!?।]\s+', sentence_buffer)
                        if len(sentences) > 1:
                            for complete_sentence in sentences[:-1]:
                                if complete_sentence.strip():
                                    streaming_callback(complete_sentence.strip() + '.', False)
                            sentence_buffer = sentences[-1]
                    
                    if sentence_buffer.strip():
                        streaming_callback(sentence_buffer.strip(), True)
                    
                    raw_response = accumulated_text
                except Exception as e:
                    logger.error(f"Streaming error in process_query_with_context: {e}")
                    response = self.gemini_model.generate_content(prompt)
                    raw_response = response.text if response else "Sorry, I couldn't generate a response."
            else:
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        max_output_tokens=150,  # ~200-300 chars for 2-3 sentences
                    )
                )
                raw_response = response.text if response else "Sorry, I couldn't generate a response."
            
            timing['generation_ms'] = (time.time() - gen_start) * 1000
            
            # Validate and humanize
            quality = self.validate_response_quality(raw_response)
            
            is_first_turn = True
            if intent_context:
                turn_number = intent_context.get('turn_number', 1)
                is_first_turn = turn_number <= 1
            
            final_response = self.humanize_response(raw_response, query_text, intent_context, is_first_turn) if self.config.enable_humanization else raw_response
            
            timing['total_ms'] = (time.time() - start_time) * 1000
            
            # Metrics
            self.query_count += 1
            self.total_query_time += timing['total_ms']
            
            avg_similarity = sum(d.get('similarity', 0.5) for d in relevant_docs) / len(relevant_docs) if relevant_docs else 0.5
            confidence = min(avg_similarity, quality.get('quality_score', 0.5))
            
            logger.info(f"✅ Incremental RAG Generation Complete ({timing['total_ms']:.0f}ms)")
            
            return {
                'answer': final_response,
                'sources': sources,
                'confidence': confidence,
                'timing_breakdown': timing,
                'metadata': {
                    'categories': list(set(d.get('metadata', {}).get('category', '') for d in relevant_docs)),
                    'quality_score': quality.get('quality_score', 0.0),
                    'num_docs_retrieved': len(relevant_docs),
                    'method': 'incremental_buffered'
                }
            }
        
        except Exception as e:
            logger.error(f"❌ process_query_with_context error: {e}", exc_info=True)
            return {
                'answer': "I apologize, but I encountered an error. Please try again.",
                'sources': [],
                'confidence': 0.0,
                'timing_breakdown': {'total_ms': (time.time() - start_time) * 1000},
                'metadata': {'error': str(e)}
            }
    
    async def gemini_only_query(
        self,
        query_text: str,
        context: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable[[str, bool], None]] = None
    ) -> Dict[str, Any]:
        """Fallback when embeddings/vector store unavailable."""
        start_time = time.time()
        
        try:
            if not self.documents:
                # Standalone Gemini mode
                prompt = f"""You are Lexi, a helpful university assistant for Leibniz University.
Answer this question to the best of your ability:

Question: {query_text}

Provide a helpful response even without specific knowledge base context."""
            else:
                # Keyword-based selection
                keywords = query_text.lower().split()
                if context and 'key_entities' in context:
                    for entity_value in context['key_entities'].values():
                        if isinstance(entity_value, str):
                            keywords.extend(entity_value.lower().split())
                
                doc_scores = []
                for i, doc_text in enumerate(self.documents):
                    score = sum(keyword in doc_text.lower() for keyword in keywords)
                    if score > 0:
                        doc_meta = self.doc_metadata[i] if i < len(self.doc_metadata) else {}
                        doc_scores.append({'text': doc_text, 'metadata': doc_meta, 'score': score})
                
                doc_scores.sort(key=lambda x: -x['score'])
                relevant_docs = doc_scores[:self.config.top_n]
                
                context_text = "\n\n".join([doc['text'] for doc in relevant_docs])
                sources = list(set([doc['metadata'].get('source', 'Unknown') for doc in relevant_docs]))
                
                prompt = f"""Answer this question using the provided context:

Context: {context_text}

Question: {query_text}

Response:"""
            
            gen_start = time.time()
            response = self.gemini_model.generate_content(prompt) if self.gemini_model else None
            raw_response = response.text.strip() if response else "Service unavailable."
            
            timing = {
                'generation_ms': (time.time() - gen_start) * 1000,
                'total_ms': (time.time() - start_time) * 1000
            }
            
            return {
                'answer': raw_response,
                'sources': sources if self.documents else [],
                'confidence': 0.5,
                'timing_breakdown': timing,
                'metadata': {'fallback_mode': 'gemini_only'}
            }
        
        except Exception as e:
            logger.error(f"Gemini-only query error: {e}")
            return {
                'answer': "I'm having trouble generating a response. Please try again.",
                'sources': [],
                'confidence': 0.0,
                'timing_breakdown': {'total_ms': (time.time() - start_time) * 1000},
                'metadata': {'error': str(e)}
            }
    
    def humanize_response(self, response: str, query: str, context: Optional[Dict[str, Any]] = None, is_first_turn: bool = True) -> str:
        """Make responses conversational. Supports Telugu for TARA mode."""
        if not response:
            return response
        
        # Remove formal prefixes (English)
        formal_prefixes = [
            "According to the context",
            "Based on the information provided",
            "The knowledge base states",
            "As per the document"
        ]
        for prefix in formal_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].lstrip(' ,:-')
        
        # TARA Telugu mode - remove greetings, add professional slang, make conversational
        if self.config.tara_mode or self.config.response_language.startswith("te"):
            # Remove common greetings and introductions (middle of conversation)
            greeting_patterns = [
                r'^నమస్కారం[!.]?\s*',
                r'^హాయ్[!.]?\s*',
                r'^Hello[!.]?\s*',
                r'^నమస్తే[!.]?\s*',
                r'^అలాగేనండి[!.]?\s*',
                r'నేను\s+TARA[!.]?\s*',
                r'నేను\s+T\.A\.S\.K[!.]?\s*',
                r'customer\s+service\s+agent[!.]?\s*',
                r'నేను\s+.*agent[!.]?\s*',
            ]
            
            for pattern in greeting_patterns:
                response = re.sub(pattern, '', response, flags=re.IGNORECASE)
            
            # Remove formal thank you phrases (middle of conversation)
            formal_thanks = [
                r'మీరు\s+అడిగినందుకు\s+ధన్యవాదాలు[!.]?\s*',
                r'గురించి\s+మీరు\s+అడిగినందుకు\s+ధన్యవాదాలు[!.]?\s*',
                r'ధన్యవాదాలు[!.]?\s*',
                r'Thank\s+you[!.]?\s*',
            ]
            for pattern in formal_thanks:
                response = re.sub(pattern, '', response, flags=re.IGNORECASE)
            
            # Remove "మీరు అడిగారు" / "మీరు అడుగుతున్నారా" patterns (redundant in conversation)
            response = re.sub(r'మీరు\s+అడిగారు[!.]?\s*', '', response, flags=re.IGNORECASE)
            response = re.sub(r'మీరు\s+అడుగుతున్నారా[!?]?\s*', '', response, flags=re.IGNORECASE)
            response = re.sub(r'మీరు\s+.*?గురించి\s+అడిగారు[!.]?\s*', '', response, flags=re.IGNORECASE)
            response = re.sub(r'.*?గురించి\s+మీరు\s+అడిగారు[!.]?\s*', '', response, flags=re.IGNORECASE)
            
            # Remove formal English starters
            formal_starters = [
                r'^Okay[!.,]?\s*',
                r'^Sure[!.,]?\s*',
                r'^Alright[!.,]?\s*',
            ]
            for pattern in formal_starters:
                response = re.sub(pattern, '', response, flags=re.IGNORECASE)
            
            # Remove "మీరు" at start if it's introducing/questioning
            if response.strip().startswith('మీరు'):
                # Remove if it's asking a question back
                response = re.sub(r'^మీరు\s+.*?[?]?\s*', '', response)
            
            # Ensure proper ending
            if response and response[-1] not in '.!?।':
                response += '.'
            
            # Clean up extra spaces
            response = re.sub(r'\s+', ' ', response).strip()
            
            return response
        
        # Add conversational starters for first turn (English mode only)
        if is_first_turn and self.config.response_style == "friendly_casual":
            query_lower = query.lower()
            
            # Use hash for deterministic selection
            hash_val = int(hashlib.md5(query.encode()).hexdigest(), 16)
            
            starters_how = ["Here's how you can", "Here's the process", "This is how it works"]
            starters_what = ["Here's what you need to know", "Basically", "Let me explain"]
            starters_where = ["You can find it at", "The location is", "It's located at"]
            starters_when = ["The schedule is", "It happens", "Timing-wise"]
            starters_why = ["The reason is", "This is because", "Here's why"]
            starters_default = ["Sure", "Absolutely", "Good question"]
            
            if query_lower.startswith('how'):
                starter = starters_how[hash_val % len(starters_how)]
            elif query_lower.startswith('what'):
                starter = starters_what[hash_val % len(starters_what)]
            elif query_lower.startswith('where'):
                starter = starters_where[hash_val % len(starters_where)]
            elif query_lower.startswith('when'):
                starter = starters_when[hash_val % len(starters_when)]
            elif query_lower.startswith('why'):
                starter = starters_why[hash_val % len(starters_why)]
            else:
                starter = starters_default[hash_val % len(starters_default)]
            
            if not response.startswith(starter):
                response = f"{starter}: {response[0].lower()}{response[1:]}"
        
        # Ensure proper capitalization
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        
        # Ensure proper ending
        if response and response[-1] not in '.!?':
            response += '.'
        
        # Add helpful ending for longer responses
        if len(response) > 150:
            helpful_endings = [
                " Let me know if you need more details!",
                " Feel free to ask if you have more questions.",
                " Hope that helps!"
            ]
            hash_val = int(hashlib.md5(response.encode()).hexdigest(), 16)
            ending = helpful_endings[hash_val % len(helpful_endings)]
            if not any(response.endswith(e.strip()) for e in helpful_endings):
                response += ending
        
        # Enforce max length
        if len(response) > self.config.max_response_length:
            sentences = re.split(r'[.!?]\s+', response)
            truncated = ""
            for sentence in sentences:
                if len(truncated) + len(sentence) < self.config.max_response_length:
                    truncated += sentence + '. '
                else:
                    break
            response = truncated.strip()
        
        return response
    
    def validate_response_quality(self, response: str) -> Dict[str, Any]:
        """Validate response quality."""
        issues = []
        quality_score = 1.0
        
        # Check for formal language
        formal_words = ['pursuant', 'aforementioned', 'hereby', 'henceforth', 'notwithstanding']
        if any(word in response.lower() for word in formal_words):
            issues.append("overly_formal")
            quality_score -= 0.3
        
        # Check for jargon
        jargon_words = ['matriculation', 'pedagogy', 'curriculum vitae']
        if any(word in response.lower() for word in jargon_words):
            issues.append("unexplained_jargon")
            quality_score -= 0.2
        
        # Check length
        if len(response) < 30:
            issues.append("too_short")
            quality_score -= 0.3
        elif len(response) > 500:
            issues.append("too_long")
            quality_score -= 0.1
        
        # Check for unhelpfulness
        unhelpful_phrases = ["i don't know", "not sure", "can't help"]
        if any(phrase in response.lower() for phrase in unhelpful_phrases):
            if "but" not in response.lower() and "however" not in response.lower():
                issues.append("unhelpful_without_alternative")
                quality_score -= 0.4
        
        retry = quality_score < 0.5
        
        return {
            'issues': issues,
            'quality_score': max(0.0, quality_score),
            'retry': retry
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Return performance metrics."""
        return {
            'total_queries': self.query_count,
            'average_query_time': self.total_query_time / self.query_count if self.query_count > 0 else 0.0,
            'vector_store_size': len(self.documents),
            'categories_loaded': len(set(m.get('category', '') for m in self.doc_metadata)),
            'embeddings_available': self.embeddings is not None,
            'gemini_available': self.gemini_model is not None
        }
