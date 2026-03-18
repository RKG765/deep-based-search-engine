"""
graph_ranker.py — Topic-Sensitive (Personalized) PageRank.

Formula:
    P(t+1) = α · A · P(t) + (1 − α) · Q

Where:
    A = row-normalized adjacency matrix (document authority)
    Q = personalization vector (query relevance per document)
    α = 0.85 (damping factor)
    iterations = 20 (max)
    tolerance = 1e-6 (early stopping)
"""

import logging
from typing import List

import numpy as np

from app.config import settings
from models.document import RankedDocument

logger = logging.getLogger(__name__)


class GraphRanker:
    """Personalized PageRank over the document graph."""

    def rank_documents(
        self,
        adjacency_matrix: np.ndarray,
        doc_embeddings: np.ndarray,
        query_embedding: np.ndarray,
        urls: List[str],
        titles: List[str],
    ) -> List[RankedDocument]:
        """
        Run Topic-Sensitive PageRank and return documents sorted by score.

        Args:
            adjacency_matrix: (N, N) weighted adjacency matrix
            doc_embeddings:   (N, D) L2-normalized document embeddings
            query_embedding:  (D,)   L2-normalized query embedding
            urls:             list of document URLs
            titles:           list of document titles
        """
        n = adjacency_matrix.shape[0]
        if n == 0:
            return []

        # ── Build A: row-normalize the adjacency matrix ──────────────
        A = self._row_normalize(adjacency_matrix)

        # ── Build Q: personalization vector ──────────────────────────
        Q = self._build_personalization_vector(doc_embeddings, query_embedding)

        # ── Power iteration ──────────────────────────────────────────
        P = np.ones(n) / n  # uniform initialization

        for iteration in range(1, settings.PR_ITERATIONS + 1):
            P_new = settings.PR_ALPHA * A.T @ P + (1 - settings.PR_ALPHA) * Q

            # Check convergence
            delta = np.linalg.norm(P_new - P, ord=1)
            P = P_new

            if delta < settings.PR_TOLERANCE:
                logger.info("PageRank converged at iteration %d (δ=%.2e)", iteration, delta)
                break
        else:
            logger.info("PageRank finished %d iterations (δ=%.2e)", settings.PR_ITERATIONS, delta)

        # ── Sort by score descending ─────────────────────────────────
        ranked = [
            RankedDocument(index=i, score=float(P[i]), url=urls[i], title=titles[i])
            for i in range(n)
        ]
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked

    @staticmethod
    def _row_normalize(matrix: np.ndarray) -> np.ndarray:
        """Row-normalize a matrix so each row sums to 1 (stochastic matrix)."""
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero for dangling nodes
        return matrix / row_sums

    @staticmethod
    def _build_personalization_vector(
        doc_embeddings: np.ndarray, query_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Build Q: the personalization/teleportation vector.
        Q_i = cosine_sim(doc_i, query) normalized so ΣQ = 1.
        """
        # Inner product (cosine sim for L2-normalized vectors)
        similarities = doc_embeddings @ query_embedding
        # Clamp negatives to zero (irrelevant docs should not pull rank)
        similarities = np.maximum(similarities, 0)
        total = similarities.sum()
        if total > 0:
            return similarities / total
        # Fallback: uniform
        return np.ones(len(similarities)) / len(similarities)
