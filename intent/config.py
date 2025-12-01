"""
Configuration for Intent Classification Microservice

Extracted from leibniz_intent_parser.py for microservice deployment.

Reference:
    leibniz_agent/leibniz_intent_parser.py (lines 48-86) - Original configuration
    leibniz_agent/services/stt_vad/config.py - Configuration pattern
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# CRITICAL: Hardcoded Gemini API key for Docker deployment
# This ensures Layer 3 (LLM) is always available
DEFAULT_GEMINI_API_KEY = "AIzaSyC6cvyEl4FNjIQCV_p5_2wJkOa1cUObFHU"


@dataclass
class IntentConfig:
    """
    Configuration for Intent Classification service with 3-layer architecture.
    
    Attributes:
        gemini_api_key: Gemini API key for LLM fallback (Layer 3) (required)
        gemini_model: Gemini model name (default: gemini-2.0-flash-exp or gemini-2.5-flash-lite)
        layer1_regex_threshold: Layer 1 regex confidence threshold (default: 0.8)
        layer2_slm_threshold: Layer 2 SLM confidence threshold (default: 0.7)
        layer2_enabled: Enable Layer 2 semantic matching (default: True, uses lightweight keyword matching)
        gemini_timeout: Gemini API timeout in seconds (default: 5.0)
        enable_context_extraction: Enable context extraction (default: True)
        log_classifications: Log all classifications (default: True)
        minimal_output: Strip non-essential fields (default: False)
        fast_route_target: Target percentage for fast route (default: 0.8)
        enable_cache: Enable LRU cache for repeated queries (default: True)
        cache_max_size: Maximum cache size (default: 128)
        cache_ttl_seconds: Cache TTL in seconds (default: 300)
    """
    
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash-lite"
    layer1_regex_threshold: float = 0.8
    layer2_slm_threshold: float = 0.7
    layer2_enabled: bool = True
    gemini_timeout: float = 5.0
    enable_context_extraction: bool = True
    log_classifications: bool = True
    minimal_output: bool = False
    fast_route_target: float = 0.8
    enable_cache: bool = True
    cache_max_size: int = 128
    cache_ttl_seconds: float = 300.0
    
    # Backward compatibility alias
    @property
    def confidence_threshold(self) -> float:
        """Backward compatibility: maps to layer1_regex_threshold"""
        return self.layer1_regex_threshold
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Validate layer thresholds
        if not 0.0 <= self.layer1_regex_threshold <= 1.0:
            raise ValueError(
                f"layer1_regex_threshold must be between 0.0 and 1.0, got {self.layer1_regex_threshold}"
            )
        
        if not 0.0 <= self.layer2_slm_threshold <= 1.0:
            raise ValueError(
                f"layer2_slm_threshold must be between 0.0 and 1.0, got {self.layer2_slm_threshold}"
            )
        
        # Validate Gemini timeout
        if self.gemini_timeout <= 0:
            raise ValueError(
                f"gemini_timeout must be positive, got {self.gemini_timeout}"
            )
        
        # Validate fast route target
        if not 0.0 <= self.fast_route_target <= 1.0:
            raise ValueError(
                f"fast_route_target must be between 0.0 and 1.0, got {self.fast_route_target}"
            )
        
        # Validate cache settings
        if self.cache_max_size <= 0:
            raise ValueError(
                f"cache_max_size must be positive, got {self.cache_max_size}"
            )
        
        if self.cache_ttl_seconds <= 0:
            raise ValueError(
                f"cache_ttl_seconds must be positive, got {self.cache_ttl_seconds}"
            )
        
        # Warn if Gemini API key is missing
        if not self.gemini_api_key:
            logger.warning(
                "️ GEMINI_API_KEY not set. Layer 3 (LLM) fallback will be disabled. "
                "Only Layer 1 (regex) and Layer 2 (semantic) will be available."
            )
        
        # Log configuration if enabled
        if self.log_classifications:
            logger.info(
                f" IntentConfig loaded (3-Layer): model={self.gemini_model}, "
                f"L1_threshold={self.layer1_regex_threshold}, L2_threshold={self.layer2_slm_threshold}, "
                f"L2_enabled={self.layer2_enabled}, timeout={self.gemini_timeout}s, "
                f"cache_enabled={self.enable_cache}, cache_size={self.cache_max_size}"
            )
    
    @staticmethod
    def from_env() -> "IntentConfig":
        """
        Load configuration from environment variables.
        
        Environment Variables:
            GEMINI_API_KEY: Required Gemini API key for Layer 3 (LLM)
            GEMINI_MODEL: Gemini model name (default: gemini-2.0-flash-exp)
            LEIBNIZ_INTENT_LAYER1_THRESHOLD: Layer 1 regex threshold (default: 0.8)
            LEIBNIZ_INTENT_LAYER2_THRESHOLD: Layer 2 SLM threshold (default: 0.7)
            LEIBNIZ_INTENT_LAYER2_ENABLED: Enable Layer 2 semantic matching (default: true)
            LEIBNIZ_INTENT_PARSER_GEMINI_TIMEOUT: Gemini timeout in seconds (default: 5.0)
            LEIBNIZ_INTENT_PARSER_ENABLE_CONTEXT_EXTRACTION: Enable context extraction (default: true)
            LEIBNIZ_INTENT_PARSER_LOG_CLASSIFICATIONS: Log classifications (default: true)
            LEIBNIZ_INTENT_PARSER_MINIMAL_OUTPUT: Strip non-essential fields (default: false)
            LEIBNIZ_INTENT_PARSER_FAST_ROUTE_TARGET: Target fast route percentage (default: 0.8)
            LEIBNIZ_INTENT_CACHE_ENABLED: Enable LRU cache (default: true)
            LEIBNIZ_INTENT_CACHE_MAX_SIZE: Maximum cache size (default: 128)
            LEIBNIZ_INTENT_CACHE_TTL: Cache TTL in seconds (default: 300)
            
            # Backward compatibility (deprecated, use LAYER1_THRESHOLD)
            LEIBNIZ_INTENT_PARSER_CONFIDENCE_THRESHOLD: Maps to LAYER1_THRESHOLD
        
        Returns:
            IntentConfig instance loaded from environment
        """
        # Load Gemini API key (required for Layer 3 LLM fallback)
        # Use hardcoded default for Docker deployment
        gemini_api_key = os.getenv("GEMINI_API_KEY", DEFAULT_GEMINI_API_KEY)
        
        # Load Gemini model name (use gemini-2.5-flash-lite for optimal performance)
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        
        # Load Layer 1 threshold (backward compatibility with old env var)
        layer1_threshold_env = os.getenv("LEIBNIZ_INTENT_LAYER1_THRESHOLD") or os.getenv("LEIBNIZ_INTENT_PARSER_CONFIDENCE_THRESHOLD", "0.8")
        try:
            layer1_regex_threshold = float(layer1_threshold_env)
        except ValueError:
            logger.warning(
                "Invalid LEIBNIZ_INTENT_LAYER1_THRESHOLD, using default 0.8"
            )
            layer1_regex_threshold = 0.8
        
        # Load Layer 2 threshold
        try:
            layer2_slm_threshold = float(
                os.getenv("LEIBNIZ_INTENT_LAYER2_THRESHOLD", "0.7")
            )
        except ValueError:
            logger.warning(
                "Invalid LEIBNIZ_INTENT_LAYER2_THRESHOLD, using default 0.7"
            )
            layer2_slm_threshold = 0.7
        
        # Load Layer 2 enabled flag
        layer2_enabled = os.getenv(
            "LEIBNIZ_INTENT_LAYER2_ENABLED", "true"
        ).lower() in ("true", "1", "yes")
        
        # Load Gemini timeout with validation
        try:
            gemini_timeout = float(
                os.getenv("LEIBNIZ_INTENT_PARSER_GEMINI_TIMEOUT", "5.0")
            )
        except ValueError:
            logger.warning(
                "Invalid LEIBNIZ_INTENT_PARSER_GEMINI_TIMEOUT, using default 5.0"
            )
            gemini_timeout = 5.0
        
        # Load boolean flags
        enable_context_extraction = os.getenv(
            "LEIBNIZ_INTENT_PARSER_ENABLE_CONTEXT_EXTRACTION", "true"
        ).lower() in ("true", "1", "yes")
        
        log_classifications = os.getenv(
            "LEIBNIZ_INTENT_PARSER_LOG_CLASSIFICATIONS", "true"
        ).lower() in ("true", "1", "yes")
        
        minimal_output = os.getenv(
            "LEIBNIZ_INTENT_PARSER_MINIMAL_OUTPUT", "false"
        ).lower() in ("true", "1", "yes")
        
        # Load fast route target with validation
        try:
            fast_route_target = float(
                os.getenv("LEIBNIZ_INTENT_PARSER_FAST_ROUTE_TARGET", "0.8")
            )
        except ValueError:
            logger.warning(
                "Invalid LEIBNIZ_INTENT_PARSER_FAST_ROUTE_TARGET, using default 0.8"
            )
            fast_route_target = 0.8
        
        # Load cache settings
        enable_cache = os.getenv(
            "LEIBNIZ_INTENT_CACHE_ENABLED", "true"
        ).lower() in ("true", "1", "yes")
        
        try:
            cache_max_size = int(
                os.getenv("LEIBNIZ_INTENT_CACHE_MAX_SIZE", "128")
            )
        except ValueError:
            logger.warning(
                "Invalid LEIBNIZ_INTENT_CACHE_MAX_SIZE, using default 128"
            )
            cache_max_size = 128
        
        try:
            cache_ttl_seconds = float(
                os.getenv("LEIBNIZ_INTENT_CACHE_TTL", "300")
            )
        except ValueError:
            logger.warning(
                "Invalid LEIBNIZ_INTENT_CACHE_TTL, using default 300"
            )
            cache_ttl_seconds = 300.0
        
        return IntentConfig(
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            layer1_regex_threshold=layer1_regex_threshold,
            layer2_slm_threshold=layer2_slm_threshold,
            layer2_enabled=layer2_enabled,
            gemini_timeout=gemini_timeout,
            enable_context_extraction=enable_context_extraction,
            log_classifications=log_classifications,
            minimal_output=minimal_output,
            fast_route_target=fast_route_target,
            enable_cache=enable_cache,
            cache_max_size=cache_max_size,
            cache_ttl_seconds=cache_ttl_seconds,
        )
