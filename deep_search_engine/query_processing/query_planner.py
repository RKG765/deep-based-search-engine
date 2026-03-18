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
        2. Recombine them into meaningful search phrases.
        3. Cap output at MAX_NEW_NODES (default: 3) to prevent tree explosion.

    Example:
        Input:  "deploy nodejs aws securely"
        Output: ["nodejs aws deployment",
                 "nodejs ec2 deploy",
                 "nodejs security configuration"]
    """
    keywords = extract_all(original_query)
    normalized_original = clean_text(original_query)

    # Filter out the exact original query
    expansions = [kw for kw in keywords if clean_text(kw) != normalized_original]

    # If we got fewer expansions than desired, supplement with pairs of keywords
    if len(expansions) < settings.MAX_NEW_NODES and len(keywords) >= 2:
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                pair = f"{keywords[i]} {keywords[j]}"
                if clean_text(pair) != normalized_original and pair not in expansions:
                    expansions.append(pair)
                    if len(expansions) >= settings.MAX_NEW_NODES:
                        break
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
