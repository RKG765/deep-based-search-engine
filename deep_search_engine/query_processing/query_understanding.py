"""
query_understanding.py — Semantic Intent Understanding layer.

Before keyword extraction and BFS crawling, this module interprets
what the user actually *means*. It classifies the query type and
reformulates it into an optimal search-engine-friendly string.

This solves the core problem: "what is the capital of india" should
understand the user wants a factual answer about India's capital city,
NOT search for the isolated words "capital" and "india" separately.
"""

import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)


# ─── Query Intent Categories ────────────────────────────────────────
INTENT_FACTUAL = "factual"          # "what is X", "who invented Y"
INTENT_HOWTO = "howto"              # "how to deploy X", "steps to do Y"
INTENT_COMPARISON = "comparison"    # "X vs Y", "difference between X and Y"
INTENT_EXPLORATORY = "exploratory"  # "best practices for X", "explain X"
INTENT_INVESTIGATIVE = "investigative"  # complex multi-hop research queries

# ─── Patterns for intent detection ──────────────────────────────────
_FACTUAL_PATTERNS = [
    r"^(?:what|who|when|where)\s+(?:is|are|was|were)\s+",
    r"^(?:what|who|when|where)\s+(?:does|do|did)\s+",
    r"^(?:how)\s+(?:is|are|was|were)\s+",
    r"^(?:define|meaning of|definition of)\s+",
    r"^(?:capital|president|founder|ceo|inventor|creator)\s+of\s+",
    r"^(?:tell\s+me\s+about)\s+",
    r"^(?:explain)\s+",
]

_HOWTO_PATTERNS = [
    r"^how\s+(?:to|do|does|can|should)\s+",
    r"^(?:steps|guide|tutorial|way)\s+(?:to|for)\s+",
]

_COMPARISON_PATTERNS = [
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bdifference\s+between\b",
    r"\bcompare\b",
    r"\bcomparison\b",
]


def classify_intent(query: str) -> str:
    """
    Classify the user's query intent to guide search strategy.
    Returns one of the INTENT_* constants.
    """
    q = query.lower().strip()

    for pattern in _FACTUAL_PATTERNS:
        if re.search(pattern, q):
            return INTENT_FACTUAL

    for pattern in _HOWTO_PATTERNS:
        if re.search(pattern, q):
            return INTENT_HOWTO

    for pattern in _COMPARISON_PATTERNS:
        if re.search(pattern, q):
            return INTENT_COMPARISON

    # Short queries (< 6 words) are likely factual lookups
    if len(q.split()) <= 5:
        return INTENT_FACTUAL

    return INTENT_EXPLORATORY


def reformulate_query(query: str) -> Tuple[str, str]:
    """
    Reformulate the user query into a search-engine-optimized string.
    
    The key insight: search engines work best with noun-phrase-heavy queries,
    not natural language questions. But we must preserve the FULL semantic
    intent of the original query.
    
    Returns:
        (reformulated_query, intent)
    """
    intent = classify_intent(query)
    q = query.strip()

    if intent == INTENT_FACTUAL:
        # For factual queries, the FULL original query is the best search term.
        # "what is capital of india" -> search "capital of india" (keep prepositions!)
        reformulated = _strip_question_prefix(q)
        logger.info("Factual intent: '%s' → '%s'", q, reformulated)

    elif intent == INTENT_HOWTO:
        # Keep the full how-to context
        reformulated = _strip_question_prefix(q)
        logger.info("HowTo intent: '%s' → '%s'", q, reformulated)

    elif intent == INTENT_COMPARISON:
        # Keep full comparison context
        reformulated = q
        logger.info("Comparison intent: '%s'", q)

    else:
        # Exploratory / investigative — use as-is
        reformulated = q
        logger.info("Exploratory intent: '%s'", q)

    return reformulated, intent


def _strip_question_prefix(query: str) -> str:
    """
    Strip common question prefixes while keeping the meaningful part intact.
    
    "what is the capital of india" → "capital of india"
    "who invented the telephone" → "invented the telephone"
    "how does photosynthesis work" → "photosynthesis work"
    """
    q = query.lower().strip()

    # Remove leading question words + auxiliary verbs
    prefixes = [
        r"^(?:what|who|when|where|which)\s+(?:is|are|was|were|does|do|did)\s+(?:the\s+)?",
        r"^(?:tell\s+me\s+(?:about\s+)?(?:the\s+)?)",
        r"^(?:explain\s+(?:the\s+)?(?:concept\s+of\s+)?)",
        r"^(?:define\s+(?:the\s+)?)",
        r"^(?:describe\s+(?:the\s+)?)",
        r"^(?:how\s+(?:does|do|did|is|are|can|could|should)\s+)",
        r"^(?:can\s+you\s+(?:tell|explain|describe|show)\s+(?:me\s+)?(?:about\s+)?(?:the\s+)?)",
    ]

    for pattern in prefixes:
        result = re.sub(pattern, "", q).strip()
        if result and result != q:
            return result

    return q
