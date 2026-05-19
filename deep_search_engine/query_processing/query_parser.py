"""
query_parser.py — Tokenizes, normalizes, and removes stopwords from user queries.
Uses spaCy en_core_web_sm for linguistic processing.
"""

import logging
from typing import List

import spacy

from utils.text_utils import clean_text

logger = logging.getLogger(__name__)

# Interrogative / filler words that spaCy frequently misses as stopwords
_QUERY_STOPWORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "tell", "explain", "describe", "define", "list", "give", "show",
    "please", "also", "etc", "vs", "versus",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by",
}

# ─── Load spaCy model ────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )
    nlp = None


def normalize_query(query: str) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    return clean_text(query)


def tokenize(query: str) -> List[str]:
    """Tokenize query into individual words, removing stopwords and whitespace."""
    normalized = normalize_query(query)
    if nlp is None:
        # Fallback: simple whitespace split with custom stopword filtering
        return [w for w in normalized.split() if w.lower() not in _QUERY_STOPWORDS]
    doc = nlp(normalized)
    return [
        token.text for token in doc
        if not token.is_stop
        and not token.is_space
        and token.text.lower() not in _QUERY_STOPWORDS
    ]
