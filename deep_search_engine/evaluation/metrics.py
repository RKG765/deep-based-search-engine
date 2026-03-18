"""
metrics.py — Information Retrieval evaluation metrics.
Implements Precision@K, Recall@K, and NDCG for benchmarking
the recursive graph retrieval against a BM25 baseline.
"""

import math
import logging
from typing import List, Set

logger = logging.getLogger(__name__)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """
    Precision@K = |relevant ∩ retrieved[:K]| / K
    Measures how many of the top-K results are relevant.
    """
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """
    Recall@K = |relevant ∩ retrieved[:K]| / |relevant|
    Measures what fraction of all relevant docs were found in top-K.
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def dcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """
    Discounted Cumulative Gain at K.
    DCG@K = Σ (2^rel_i − 1) / log₂(i + 2)  for i in [0, K)
    """
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """
    Normalized DCG@K = DCG@K / IDCG@K.
    IDCG is DCG of the ideal ranking (sorted descending by relevance).
    """
    dcg = dcg_at_k(relevance_scores, k)
    ideal = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def topical_coverage(
    retrieved_keywords: List[Set[str]], query_keywords: Set[str]
) -> float:
    """
    Measures breadth: what fraction of query keywords appear
    across the union of all retrieved document keywords?
    """
    if not query_keywords:
        return 0.0
    covered = set()
    for kw_set in retrieved_keywords:
        covered |= (kw_set & query_keywords)
    return len(covered) / len(query_keywords)
