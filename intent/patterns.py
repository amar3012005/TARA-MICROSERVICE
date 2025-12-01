"""
Intent Classification Patterns for Leibniz University Agent

Extracted from leibniz_intent_parser.py for microservice deployment.

Reference:
    leibniz_agent/leibniz_intent_parser.py (lines 87-176) - Original pattern definitions
"""

from typing import Dict, List, Any
import re


def load_intent_patterns() -> Dict[str, Any]:
    """
    Load comprehensive English-only intent patterns for 3-layer classification.
    
    Ported from leibniz_intent_parser.py with enhanced coverage for robust classification.
    
    Pattern Categories:
        - appointment: Scheduling/booking meetings (highest priority)
        - greeting: Social pleasantries (strict - standalone only)
        - exit: Conversation termination
        - general_query: Information requests (RAG routing)
        - entities: Cross-intent entity extraction patterns
    
    Returns:
        Dictionary with pattern categories for fast classification
    """
    patterns = {
        # Appointment patterns - highest priority (expanded coverage)
        "appointment": {
            "keywords": [
                "appointment", "schedule", "book", "booking", "meeting", 
                "meet", "visit", "consultation", "advising", "counseling",
                "set up", "arrange", "make an appointment", "book a slot"
            ],
            "regex_patterns": [
                r"\b(schedule|book|make)\s+(an?\s+)?appointment\b",
                r"\bappointment\s+(with|for)\b",
                r"\b(want|need|like)\s+to\s+(schedule|book|meet)\b",
                r"\bwhen\s+can\s+i\s+(meet|see|visit)\b",
                r"\b(set up|arrange)\s+(a\s+)?(meeting|appointment)\b",
                r"\b(available\s+)?(time|slot|appointment)\s+(for|with)\b",
                r"\b(book|reserve)\s+(a\s+)?(meeting|consultation|session)\b"
            ],
            "entity_patterns": {
                "department": r"\b(admissions?|registrar|financial aid|counseling|advising|career services?|academic|student services?|advisor|counselor|professor|dean|staff|office)\b",
                "datetime": r"\b(today|tomorrow|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}\s*(?:am|pm)|\d{1,2}/\d{1,2}|\d{1,2}-\d{1,2})\b",
                "purpose": r"\b(admission|enrollment|transcript|financial|academic|career|degree|program|meeting|consultation|advising)\b"
            }
        },
        
        # Greeting patterns (STRICT - standalone only, expanded coverage)
        "greeting": {
            "keywords": [
                "hello", "hi", "hey", "greetings", "good morning", 
                "good afternoon", "good evening", "howdy", "hiya",
                "what's up", "yo", "sup", "greeting", "good day"
            ],
            "regex_patterns": [
                r"^(hello|hi|hey|hiya|greetings|howdy)\b",
                r"\bgood\s+(morning|afternoon|evening|day)\b",
                r"^(what'?s?\s+up|yo|sup)\b",
                r"^(greetings|salutations)\b"
            ]
        },
        
        # Exit patterns (expanded coverage for robust detection)
        "exit": {
            "keywords": [
                "bye", "goodbye", "exit", "quit", "stop", "end", 
                "thanks bye", "that's all", "no more questions",
                "see you", "i'm done", "that is all", "nothing else",
                "talk later", "gotta go", "have a good day",
                "thanks that's all", "thank you bye", "no thanks",
                "i'm finished", "all done", "done talking"
            ],
            "regex_patterns": [
                r"\b(bye|goodbye|see you|talk later|gotta go)\b",
                r"\b(that'?s?\s+all|no\s+more|i'?m\s+done|nothing\s+else)\b",
                r"\b(thanks?|thank you).*(bye|goodbye|all)\b",
                r"\bhave\s+a\s+good\s+(day|night|one)\b",
                r"\b(i'?m\s+)?(finished|done|complete)\b",
                r"\bno\s+thanks?\b"
            ]
        },
        
        # General query patterns (for RAG routing - expanded coverage)
        "general_query": {
            "question_words": [
                "what", "when", "where", "who", "why", "how", 
                "can", "could", "would", "is", "are", "do", "does",
                "should", "will", "tell", "explain", "describe"
            ],
            "university_topics": [
                "course", "class", "program", "degree", "admission", 
                "enrollment", "tuition", "fee", "scholarship", "campus", 
                "library", "dorm", "housing", "faculty", "professor", 
                "department", "major", "minor", "credit", "semester", 
                "exam", "grade", "transcript", "schedule", "registration",
                "building", "directions", "location", "map", "office",
                "requirements", "prerequisites", "deadline", "application",
                "financial aid", "grants", "loans", "work-study"
            ],
            "regex_patterns": [
                r"^(what|when|where|who|why|how)\b",
                r"\b(tell me|explain|describe|information about|talk about)\b",
                r"\b(how do i|how can i|where can i|what do i|when do i)\b",
                r"\b(can you|could you|would you)\s+(tell|explain|describe|help)\b",
                r"\b(i want|i need|i'm looking for)\s+(information|details|help)\b"
            ]
        },
        
        # Entity extraction patterns (used across all intents - expanded)
        "entities": {
            "course_code": r"\b[A-Z]{2,4}\s*\d{3,4}[A-Z]?\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            "program": r"\b(computer science|cs|engineering|business|biology|chemistry|physics|mathematics|english|history|economics|psychology|art|music|theater|drama|philosophy|political science|sociology|anthropology)\b",
            "department": r"\b(admissions?|registrar|financial aid|counseling|advising|career services?|academic|student services?|bursar|housing|residential life|dining|health services?|athletics?|alumni)\b"
        }
    }
    
    return patterns


def compile_patterns(patterns: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compile regex patterns for performance.
    
    Args:
        patterns: Raw patterns dictionary from load_intent_patterns()
    
    Returns:
        Patterns dictionary with compiled regex objects
    """
    compiled = {}
    
    for intent_name, intent_data in patterns.items():
        # Special handling for top-level "entities" dict (flat structure with regex strings)
        if intent_name == "entities":
            compiled[intent_name] = {
                entity_name: re.compile(entity_pattern, re.IGNORECASE)
                for entity_name, entity_pattern in intent_data.items()
            }
            continue
        
        compiled[intent_name] = {}
        
        # Copy non-regex data as-is
        for key, value in intent_data.items():
            if key == "regex_patterns":
                # Compile regex patterns
                compiled[intent_name][key] = [
                    re.compile(pattern, re.IGNORECASE) 
                    for pattern in value
                ]
            elif key == "entity_patterns":
                # Compile entity patterns
                compiled[intent_name][key] = {
                    entity_name: re.compile(entity_pattern, re.IGNORECASE)
                    for entity_name, entity_pattern in value.items()
                }
            else:
                # Copy keywords and other data
                compiled[intent_name][key] = value
    
    return compiled


def get_system_prompt() -> str:
    """
    Get Gemini system prompt for LLM fallback classification.
    
    Returns:
        System prompt string with classification rules
    """
    prompt = """You are an EXPERT intent classifier for Leibniz University Institute customer service with advanced natural language understanding.

 YOUR MISSION: Accurately classify user intent by deeply analyzing the SEMANTIC MEANING and CONTEXT, not just surface keywords.

CORE PRINCIPLE: Prioritize ACCURACY over speed. Think through the user's true intention before classifying.

═══════════════════════════════════════════════════════════════════════════════
CLASSIFICATION CATEGORIES (5 INTENTS)
═══════════════════════════════════════════════════════════════════════════════

1. ️ APPOINTMENT_SCHEDULING
   Definition: User explicitly wants to schedule, book, or arrange a meeting/appointment
   Key Indicators: "schedule", "book", "appointment", "meeting", "when can I meet"
   Confidence Threshold: Require explicit appointment-related action verbs
   
2.  RAG_QUERY (Most Common - Default for Information Requests)
   Definition: User seeking information, asking questions, or requesting explanations
   Key Indicators: Question words (what, how, why, where, when, who), "tell me", "explain", "describe"
   Confidence Threshold: ANY information-seeking behavior → RAG_QUERY
   
3.  GREETING (EXTREMELY STRICT - Rare)
   Definition: ONLY standalone social pleasantries with NO information request
   Key Indicators: "hi", "hello", "hey", "good morning" (ALONE, no follow-up)
   Confidence Threshold: Must be < 5 words AND contain zero question/request elements
   
4.  EXIT
   Definition: User wants to end the conversation
   Key Indicators: "bye", "goodbye", "thanks, that's all", "I'm done"
   Confidence Threshold: Clear termination intent
   
5.  UNCLEAR
   Definition: Genuinely ambiguous, incomplete, or nonsensical input
   Confidence Threshold: Use sparingly - most inputs have classifiable intent

═══════════════════════════════════════════════════════════════════════════════
CRITICAL CLASSIFICATION RULES (READ CAREFULLY)
═══════════════════════════════════════════════════════════════════════════════

 RULE 1: GREETING vs RAG_QUERY DISTINCTION (Most Common Error)

GREETING requires ALL of these conditions:
   Contains greeting word ("hi", "hello", "hey", "good morning")
   Standalone (no additional requests or questions)
   Word count ≤ 5 words
   NO question words (what, how, why, where, when, who, can, could, would)
   NO action requests ("tell me", "show me", "explain", "talk about")
   NO topic mentions (programs, courses, admission, etc.)

If ANY condition fails → Classify as RAG_QUERY, NOT GREETING

Examples of FALSE GREETINGS (actually RAG_QUERY):
   "can you specifically talk about talk about any" → RAG_QUERY (has "can you talk about")
   "can you tell me about programs" → RAG_QUERY (information request)
   "what can you help me with" → RAG_QUERY (question about services)
   "hello, how do I apply?" → RAG_QUERY (has follow-up question)
   "hey, what programs do you offer?" → RAG_QUERY (asking about programs)
   "hi there, I need information" → RAG_QUERY (information request)

Examples of TRUE GREETINGS:
   "hi" (standalone)
   "hello" (standalone)
   "good morning" (standalone)
   "hey there" (casual greeting only)
   "what's up" (colloquial greeting)

 RULE 2: RAG_QUERY is the DEFAULT for Information Requests

Classify as RAG_QUERY if user:
   Asks a question (contains what, how, why, where, when, who)
   Requests information ("tell me", "explain", "describe", "talk about")
   Seeks clarification ("can you...", "could you...", "would you...")
   Mentions university topics (programs, courses, admission, tuition, etc.)
   Uses imperative verbs ("show", "list", "give me", "provide")

Even if input is poorly formed or contains typos, extract the underlying information-seeking intent.

 RULE 3: APPOINTMENT_SCHEDULING Requires Explicit Scheduling Intent

Classify as APPOINTMENT_SCHEDULING ONLY if:
   Explicit scheduling verbs: "schedule", "book", "make", "arrange", "set up"
   Meeting/appointment nouns: "appointment", "meeting", "consultation", "visit"
   Time-related requests: "when can I meet", "available times", "book a slot"

DO NOT classify as appointment if:
   Asking ABOUT scheduling process (that's RAG_QUERY)
   General questions about appointments (that's RAG_QUERY)
   "How do I schedule..." without explicit action request (that's RAG_QUERY)

 RULE 4: Context Extraction is MANDATORY (Every Classification)

For EVERY intent, extract rich structured context:

user_goal: One clear sentence describing user's objective
  - Focus on the WHAT (what they want to achieve)
  - Use natural language, not keywords
  - Examples:
    * "wants to learn about computer science program requirements"
    * "seeking information on campus housing options and costs"
    * "trying to understand the admission application process"

key_entities: Dictionary of extracted entities
  - Program/major names: {"program": "computer science"}
  - Department names: {"department": "admissions"}
  - Time references: {"datetime": "next Monday"}
  - Contact info: {"email": "student@example.com"}
  - Course codes: {"course_code": "CS101"}
  - ANY relevant details from user's query

extracted_meaning: Paraphrased user query in clear, professional language
  - Remove filler words, typos, and grammatical errors
  - Preserve user's original intent
  - Examples:
    * Input: "hey uh can you like tell me about uh programs?"
    * Output: "Can you provide information about academic programs?"

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON)
═══════════════════════════════════════════════════════════════════════════════

Return ONLY valid JSON (no markdown, no code blocks):

{
  "intent": "APPOINTMENT_SCHEDULING" | "RAG_QUERY" | "GREETING" | "EXIT" | "UNCLEAR",
  "confidence": 0.0-1.0,
  "context": {
    "user_goal": "clear sentence describing user's objective",
    "key_entities": {
      "program": "extracted program name",
      "department": "extracted department",
      "datetime": "extracted time reference",
      ...
    },
    "extracted_meaning": "paraphrased clean version of user query"
  },
  "reasoning": "brief explanation of classification decision"
}

IMPORTANT: Return ONLY the JSON object. No explanations, no markdown formatting."""
    
    return prompt
