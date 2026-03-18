"""
node_pruning.py — Prevents Topic Drift by scoring candidate BFS nodes
and dropping those below the pruning threshold.

Formula:
    node_score = 0.4 * keyword_overlap + 0.3 * rank_score + 0.3 * similarity

Where:
    keyword_overlap  = |query_kws ∩ node_kws| / |query_kws|
    rank_score       = normalized inverse rank (1st → 1.0)
    similarity       = cosine_similarity(node_embedding, query_embedding)
                       using sentence-transformers/all-MiniLM-L6-v2
"""

import logging
from typing import List, Set

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


def _keyword_overlap(query_keywords: Set[str], node_keywords: Set[str]) -> float:
    """Jaccard-style overlap: |intersection| / |query_keywords|."""
    if not query_keywords:
        return 0.0
    return len(query_keywords & node_keywords) / len(query_keywords)


def _rank_score(rank: int, max_results: int) -> float:
    """Normalize rank into [0, 1]. Rank 1 → 1.0, last → close to 0."""
    if max_results <= 1:
        return 1.0
    return 1.0 - ((rank - 1) / max_results)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_node_score(
    query_keywords: Set[str],
    node_keywords: Set[str],
    rank: int,
    max_results: int,
    query_embedding: np.ndarray,
    node_embedding: np.ndarray,
) -> float:
    """
    Compute the composite pruning score for a single candidate node.
    Returns a float in [0, 1].
    """
    kw = _keyword_overlap(query_keywords, node_keywords)
    rs = _rank_score(rank, max_results)
    sim = _cosine_similarity(query_embedding, node_embedding)

    score = (
        settings.PRUNING_W_KEYWORD * kw
        + settings.PRUNING_W_RANK * rs
        + settings.PRUNING_W_SIMILARITY * sim
    )

    logger.debug(
        "Pruning score: kw=%.3f rank=%.3f sim=%.3f → total=%.3f",
        kw, rs, sim, score,
    )
    return score


def prune_nodes(
    candidates: List[dict],
    query_keywords: Set[str],
    query_embedding: np.ndarray,
    max_results: int,
) -> List[dict]:
    """
    Filter a list of candidate node dicts.
    Each dict must have: 'text', 'keywords' (set), 'rank' (int), 'embedding' (ndarray).
    Returns only nodes whose score ≥ PRUNING_THRESHOLD.
    """
    accepted: List[dict] = []
    for cand in candidates:
        score = compute_node_score(
            query_keywords=query_keywords,
            node_keywords=cand["keywords"],
            rank=cand["rank"],
            max_results=max_results,
            query_embedding=query_embedding,
            node_embedding=cand["embedding"],
        )
        cand["score"] = score
        if score >= settings.PRUNING_THRESHOLD:
            accepted.append(cand)
        else:
            logger.debug("Pruned node (score %.3f): %s", score, cand.get("text", ""))

    logger.info("Pruning: %d/%d nodes accepted", len(accepted), len(candidates))
    return accepted
