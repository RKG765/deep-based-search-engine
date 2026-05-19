"""
reranker.py — Stage 2 Re-ranking after Personalized PageRank.

Takes the top-K PageRank results and re-scores them with a
passage-level cosine similarity signal, producing a final ranking.

Final Score Formula:
    final_score =
        RERANK_W_COSINE    * passage_cosine_sim(query, doc_passage)
        RERANK_W_PAGERANK  * pagerank_score_normalized

Where:
    doc_passage = first 512 tokens of document content
    passage_cosine_sim = cosine(embed(query), embed(passage))
    pagerank_score_normalized = min(pr_score / max_pr_score, 1.0)

Default weights (from config):
    RERANK_W_COSINE   = 0.60
    RERANK_W_PAGERANK = 0.40
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from models.document import CleanDocument, RankedDocument, RerankResult

logger = logging.getLogger(__name__)

# Re-use the same model as recursive_search.py (cached globally)
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embed_model


def _truncate_to_passage(content: str, max_words: int = 512) -> str:
    """Take the first max_words words of content as the passage."""
    words = content.split()
    return " ".join(words[:max_words])


def rerank(
    ranked_docs: List[RankedDocument],
    documents: List[CleanDocument],
    query: str,
    top_k: int | None = None,
) -> List[RerankResult]:
    """
    Re-rank the top-K PageRank results using passage-level cosine similarity.

    Args:
        ranked_docs: sorted list from GraphRanker (best first)
        documents:   full list of CleanDocument (index-aligned with ranked_docs)
        query:       raw query string
        top_k:       re-rank window size (defaults to settings.RERANK_TOP_K)

    Returns:
        List of RerankResult sorted by final_score descending.
    """
    top_k = top_k or settings.RERANK_TOP_K
    candidates = ranked_docs[:top_k]

    if not candidates:
        return []

    model = _get_embed_model()

    # ── Embed query ──────────────────────────────────────────────────
    query_emb = model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]

    # ── Build passages from candidate documents ───────────────────────
    passages: List[str] = []
    valid_candidates: List[RankedDocument] = []

    for rd in candidates:
        if rd.index < len(documents):
            doc = documents[rd.index]
            passage = _truncate_to_passage(doc.content)
            passages.append(passage)
            valid_candidates.append(rd)
        else:
            logger.warning("RankedDocument index %d out of range (%d docs)", rd.index, len(documents))

    if not passages:
        return []

    # ── Embed all passages in one batch ──────────────────────────────
    passage_embs = model.encode(
        passages,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=32,
    )

    # ── Cosine similarity (normalized → dot product) ──────────────────
    cosine_scores = passage_embs @ query_emb  # shape (K,)

    # ── Normalize PageRank scores to [0, 1] ────────────────────────────
    pr_scores = np.array([rd.score for rd in valid_candidates])
    max_pr = pr_scores.max()
    pr_scores_norm = pr_scores / max_pr if max_pr > 0 else pr_scores

    # ── Blend scores (raw) ────────────────────────────────────────────
    # raw = 0.60 × cosine + 0.40 × pr_normalized
    # cosine ∈ [0,1], pr_norm ∈ [0,1] so raw ∈ [0,1]
    raw_scores = (
        settings.RERANK_W_COSINE    * cosine_scores
        + settings.RERANK_W_PAGERANK * pr_scores_norm
    )

    # ── Min-max normalize → confidence [0, 1] ─────────────────────────
    # Makes confidence RELATIVE to this result set (top doc = 100%)
    # Honest label: "Confidence relative to retrieved results"
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        confidence = (raw_scores - s_min) / (s_max - s_min)
    else:
        confidence = np.ones_like(raw_scores)   # all identical → all 100%

    # ── Build and sort results ────────────────────────────────────────
    results: List[RerankResult] = []
    for i, rd in enumerate(valid_candidates):
        doc = documents[rd.index] if rd.index < len(documents) else None
        results.append(RerankResult(
            index=rd.index,
            final_score=float(raw_scores[i]),
            confidence_pct=round(float(confidence[i]) * 100, 1),
            pagerank_score=float(pr_scores[i]),
            cosine_score=float(cosine_scores[i]),
            domain_score=doc.domain_score if doc else 0.0,
            quality_score=doc.quality_score if doc else 0.0,
            url=rd.url,
            title=rd.title,
        ))

    results.sort(key=lambda r: r.final_score, reverse=True)

    logger.info(
        "Re-ranking: %d candidates → top=%s (conf=%.1f%%)",
        len(results),
        results[0].url if results else "none",
        results[0].confidence_pct if results else 0,
    )
    return results
