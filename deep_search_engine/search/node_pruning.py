"""
node_pruning.py — Prevents Topic Drift by scoring candidate BFS nodes
and dropping those below the pruning threshold.

Stage 1 Scoring Formula (6 components):
    node_score =
        0.25 * semantic_similarity   (embedding cosine)
        0.20 * keyword_overlap       (Jaccard vs query keywords)
        0.15 * serp_rank_score       (1/position, normalized)
        0.15 * domain_score          (authority heuristic)
        0.15 * content_quality       (composite quality score)
        0.10 * freshness_score       (recency proxy, uniform=0.5 default)
"""

import logging
import re
from typing import List, Set
from urllib.parse import urlparse

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ─── Explicit high-trust domain lookup ──────────────────────────────
# Exact netloc matches → guaranteed score. Catches domains that regex misses
# (e.g. cloudflare.com doesn't end in .com/special but IS authoritative).
_KNOWN_DOMAINS: dict[str, float] = {
    # Infrastructure / CDN / DevOps
    "cloudflare.com":               0.90,
    "kubernetes.io":                0.92,
    "golang.org":                   0.92,
    "go.dev":                       0.90,
    "rust-lang.org":                0.90,
    "python.org":                   0.95,
    "nodejs.org":                   0.90,
    "docker.com":                   0.88,
    "nginx.org":                    0.88,
    # Cloud docs
    "docs.aws.amazon.com":          0.92,
    "aws.amazon.com":               0.88,
    "cloud.google.com":             0.90,
    "learn.microsoft.com":          0.90,
    "azure.microsoft.com":          0.88,
    # Dev reference
    "developer.mozilla.org":        0.95,
    "developers.google.com":        0.90,
    "developer.apple.com":          0.90,
    # Academic
    "arxiv.org":                    0.97,
    "scholar.google.com":           0.95,
    "pubmed.ncbi.nlm.nih.gov":      0.97,
    "nature.com":                   0.97,
    "ieee.org":                     0.96,
    "acm.org":                      0.96,
    "springer.com":                 0.94,
    "sciencedirect.com":            0.94,
    # Community / reference
    "github.com":                   0.85,
    "stackoverflow.com":            0.85,
    "wikipedia.org":                0.88,
    "en.wikipedia.org":             0.88,
    # ML / AI
    "pytorch.org":                  0.90,
    "tensorflow.org":               0.90,
    "huggingface.co":               0.88,
    "openai.com":                   0.85,
    "anthropic.com":                0.85,
    "deepmind.com":                 0.87,
    # Blogs (medium quality)
    "medium.com":                   0.45,
    "dev.to":                       0.50,
    "hashnode.dev":                 0.48,
    "substack.com":                 0.45,
    "towardsdatascience.com":       0.55,
}

# ─── Fallback regex tiers ────────────────────────────────────────────
_TIER_HIGH = re.compile(
    r"(\.gov$|\.edu$|\.mil$)", re.IGNORECASE
)
_TIER_GOOD_DOCS = re.compile(
    r"(^docs\.|\.readthedocs\.io|developer\.|developers\.)", re.IGNORECASE
)
_TIER_ORG = re.compile(
    r"(\.org$|research\.)", re.IGNORECASE
)


def domain_authority(url: str) -> float:
    """
    Heuristic domain authority score [0, 1].
    Priority: exact known-domain lookup → regex tier fallback.
    """
    try:
        netloc = urlparse(url).netloc.lower()
        # Strip www. prefix for matching
        netloc_clean = netloc.lstrip("www.")
    except Exception:
        return 0.30

    # 1. Exact match in known-domain table
    if netloc_clean in _KNOWN_DOMAINS:
        return _KNOWN_DOMAINS[netloc_clean]
    if netloc in _KNOWN_DOMAINS:
        return _KNOWN_DOMAINS[netloc]

    # 2. Regex fallbacks
    if _TIER_HIGH.search(netloc):
        return 0.97
    if _TIER_GOOD_DOCS.search(netloc):
        return 0.88
    if _TIER_ORG.search(netloc):
        return 0.65

    return 0.30   # unknown / personal blog


def get_trust_label(
    domain_score: float,
    pagerank_score: float = 0.0,
    quality_score: float = 0.5,
    max_pagerank: float = 1.0,
) -> tuple[str, str]:
    """
    Composite trust classification.

    trust = 0.5×domain + 0.3×(pr/max_pr) + 0.2×quality

    Returns:
        (label, css_class)  e.g. ("Trusted", "sp-green")
    """
    pr_norm = (pagerank_score / max_pagerank) if max_pagerank > 0 else 0.0
    trust = 0.5 * domain_score + 0.3 * pr_norm + 0.2 * quality_score

    if trust >= 0.70:
        return "Trusted", "sp-green"
    elif trust >= 0.45:
        return "Moderate", "sp-amber"
    else:
        return "Unknown", "score-pill"


def _keyword_overlap(query_keywords: Set[str], node_keywords: Set[str]) -> float:
    """Jaccard-style overlap: |intersection| / |query_keywords|."""
    if not query_keywords:
        return 0.0
    return len(query_keywords & node_keywords) / len(query_keywords)


def _rank_score(rank: int, max_results: int) -> float:
    """Normalize SERP rank into [0, 1]. Position 1 → 1.0."""
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
    url: str = "",
    quality_score: float = 0.5,
    freshness_score: float = 0.5,
) -> float:
    """
    Compute the composite Stage 1 pruning score for a candidate node.
    Returns a float in [0, 1].
    """
    semantic  = _cosine_similarity(query_embedding, node_embedding)
    kw        = _keyword_overlap(query_keywords, node_keywords)
    serp      = _rank_score(rank, max_results)
    domain    = domain_authority(url) if url else 0.30
    quality   = quality_score
    freshness = freshness_score

    score = (
        settings.PRUNING_W_SEMANTIC  * semantic
        + settings.PRUNING_W_KEYWORD   * kw
        + settings.PRUNING_W_SERP      * serp
        + settings.PRUNING_W_DOMAIN    * domain
        + settings.PRUNING_W_QUALITY   * quality
        + settings.PRUNING_W_FRESHNESS * freshness
    )

    logger.debug(
        "Stage1 score: sem=%.3f kw=%.3f serp=%.3f dom=%.3f qual=%.3f fresh=%.3f → %.3f",
        semantic, kw, serp, domain, quality, freshness, score,
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
    Optional keys: 'url' (str), 'quality_score' (float).
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
            url=cand.get("url", ""),
            quality_score=cand.get("quality_score", 0.5),
            freshness_score=cand.get("freshness_score", 0.5),
        )
        cand["score"] = score
        if score >= settings.PRUNING_THRESHOLD:
            accepted.append(cand)
        else:
            logger.debug("Pruned node (score %.3f): %s", score, cand.get("text", ""))

    logger.info("Pruning: %d/%d nodes accepted", len(accepted), len(candidates))
    return accepted
