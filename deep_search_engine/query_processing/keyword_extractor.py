"""
keyword_extractor.py — Extracts key phrases using YAKE, spaCy noun chunks, and NER.
These feed into the Query Planner for BFS seed node generation.
"""

import logging
from typing import List

import yake

from query_processing.query_parser import nlp
from utils.text_utils import clean_text

logger = logging.getLogger(__name__)


def extract_noun_phrases(query: str) -> List[str]:
    """Extract noun chunks from the query using spaCy."""
    if nlp is None:
        return []
    doc = nlp(query)
    return list({chunk.text.lower().strip() for chunk in doc.noun_chunks})


def extract_named_entities(query: str) -> List[str]:
    """Extract named entities (ORG, TECH, GPE, PRODUCT, etc.) via spaCy NER."""
    if nlp is None:
        return []
    doc = nlp(query)
    return list({ent.text.lower().strip() for ent in doc.ents})


def extract_keywords_yake(query: str, top_n: int = 5) -> List[str]:
    """
    Extract keyphrases using YAKE (Yet Another Keyword Extractor).
    Lower YAKE score = more important keyword.
    """
    extractor = yake.KeywordExtractor(
        lan="en",
        n=2,              # up to bigrams
        dedupLim=0.9,     # deduplication threshold
        top=top_n,
        features=None,
    )
    keywords = extractor.extract_keywords(query)
    return [kw[0].lower() for kw in keywords]


def extract_all(query: str) -> List[str]:
    """
    Unified extraction: combine noun phrases + NER + YAKE keywords.
    Returns a deduplicated list sorted by phrase length (longer = more specific).
    """
    nouns = extract_noun_phrases(query)
    entities = extract_named_entities(query)
    yake_kws = extract_keywords_yake(query, top_n=5)

    combined = list(set(nouns + entities + yake_kws))
    # Filter out single-character noise
    combined = [kw for kw in combined if len(kw) > 1]
    # Sort by length descending — longer phrases are usually more descriptive
    combined.sort(key=len, reverse=True)

    logger.debug("Extracted keywords: %s", combined)
    return combined
