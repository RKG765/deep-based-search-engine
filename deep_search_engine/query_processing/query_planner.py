"""
query_planner.py — Query Expansion / Planner stage.
Takes extracted keywords and generates diverse BFS seed nodes
by recombining noun phrases, entities, and technical terms.

This is the NEW stage identified during architecture review:
Query → Decomposition → **Expansion** → BFS Search
"""

import logging
from typing import List

from app.config import settings
from query_processing.keyword_extractor import extract_all
from utils.text_utils import clean_text

logger = logging.getLogger(__name__)


def expand_query(original_query: str) -> List[str]:
    """
    Generate expanded search queries from the original user query.

    Strategy:
        1. Extract all keywords (noun phrases + NER + YAKE).
        2. Prefer multi-word phrases as expansions (they carry more context).
        3. Combine single keywords into meaningful pairs.
        4. Cap output at MAX_NEW_NODES (default: 3) to prevent tree explosion.

    Example:
        Input:  "what is capital of india"
        Output: ["capital india", "capital of india"]
    """
    keywords = extract_all(original_query)
    normalized_original = clean_text(original_query)

    # Separate multi-word phrases from single keywords
    multi_word = [kw for kw in keywords if len(kw.split()) >= 2]
    single_word = [kw for kw in keywords if len(kw.split()) == 1]

    expansions: List[str] = []

    # Priority 1: multi-word phrases (they carry the most context)
    for phrase in multi_word:
        if clean_text(phrase) != normalized_original and phrase not in expansions:
            expansions.append(phrase)
            if len(expansions) >= settings.MAX_NEW_NODES:
                break

    # Priority 2: combine single keywords into pairs for richer context
    if len(expansions) < settings.MAX_NEW_NODES and len(single_word) >= 2:
        for i in range(len(single_word)):
            for j in range(i + 1, len(single_word)):
                pair = f"{single_word[i]} {single_word[j]}"
                if clean_text(pair) != normalized_original and pair not in expansions:
                    expansions.append(pair)
                    if len(expansions) >= settings.MAX_NEW_NODES:
                        break
            if len(expansions) >= settings.MAX_NEW_NODES:
                break

    # Priority 3: only use single keywords as a last resort (avoid isolated words)
    if len(expansions) < settings.MAX_NEW_NODES:
        for kw in single_word:
            if kw not in expansions and clean_text(kw) != normalized_original:
                # Wrap single keyword with original query context
                contextual = f"{kw} {single_word[0]}" if kw != single_word[0] and len(single_word) > 1 else kw
                if contextual not in expansions:
                    expansions.append(contextual)
                    if len(expansions) >= settings.MAX_NEW_NODES:
                        break

    result = expansions[: settings.MAX_NEW_NODES]
    logger.info("Query expansion: '%s' → %s", original_query, result)
    return result


def generate_seed_nodes(query: str) -> List[str]:
    """
    Public API: produces the seed nodes that will initialize BFS at Depth 0/1.
    Always includes the original query as the root node.
    Returns: [original_query, expanded_node_1, expanded_node_2, ...]
    """
    expanded = expand_query(query)
    return [query] + expanded
