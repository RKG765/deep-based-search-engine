"""
graph_ranker.py — Topic-Sensitive (Personalized) PageRank.

Formula:
    P(t+1) = α · A · P(t) + (1 − α) · Q

Where:
    A = row-normalized adjacency matrix (document authority)
    Q = personalization vector — blends three signals:
          Q_i = 0.60 * cosine_sim(doc_i, query)
              + 0.25 * domain_score_i
              + 0.15 * (1 / serp_rank_i, normalized)
    α = 0.85 (damping factor)
    iterations = 30 (max)
    tolerance  = 1e-6 (early stopping)
"""

import logging
from typing import List

import numpy as np

from app.config import settings
from models.document import CleanDocument, RankedDocument

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
        documents: List[CleanDocument] | None = None,
    ) -> List[RankedDocument]:
        """
        Run Topic-Sensitive PageRank and return documents sorted by score.

        Args:
            adjacency_matrix: (N, N) weighted adjacency matrix
            doc_embeddings:   (N, D) L2-normalized document embeddings
            query_embedding:  (D,)   L2-normalized query embedding
            urls:             list of document URLs
            titles:           list of document titles
            documents:        optional list of CleanDocument (for domain/serp signals)
        """
        n = adjacency_matrix.shape[0]
        if n == 0:
            return []

        # ── Build A: row-normalize the adjacency matrix ──────────────
        A = self._row_normalize(adjacency_matrix)

        # ── Build Q: personalization vector ──────────────────────────
        Q = self._build_personalization_vector(
            doc_embeddings=doc_embeddings,
            query_embedding=query_embedding,
            documents=documents,
            n=n,
        )

        # ── Power iteration ──────────────────────────────────────────
        P = np.ones(n) / n  # uniform initialization
        delta = float("inf")

        for iteration in range(1, settings.PR_ITERATIONS + 1):
            P_new = settings.PR_ALPHA * A.T @ P + (1 - settings.PR_ALPHA) * Q

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
        doc_embeddings: np.ndarray,
        query_embedding: np.ndarray,
        documents: List[CleanDocument] | None,
        n: int,
    ) -> np.ndarray:
        """
        Build Q: the personalization/teleportation vector.

        Q_i = 0.60 * cosine_sim(doc_i, query)
            + 0.25 * domain_score_i
            + 0.15 * serp_rank_signal_i   (1/rank, normalized)

        All components are individually normalized then blended.
        """
        # ── Component 1: semantic similarity ─────────────────────────
        similarities = doc_embeddings @ query_embedding
        similarities = np.maximum(similarities, 0)

        # ── Component 2: domain authority ────────────────────────────
        if documents is not None:
            domain_scores = np.array([doc.domain_score for doc in documents], dtype=float)
        else:
            domain_scores = np.full(n, 0.30)  # default: unknown domain

        # ── Component 3: SERP rank signal ────────────────────────────
        if documents is not None:
            ranks = np.array([
                doc.serp_rank if doc.serp_rank > 0 else n
                for doc in documents
            ], dtype=float)
            serp_signals = 1.0 / ranks   # higher rank → lower position → smaller value
        else:
            serp_signals = np.ones(n)

        def _normalize(arr: np.ndarray) -> np.ndarray:
            total = arr.sum()
            return arr / total if total > 0 else np.ones(len(arr)) / len(arr)

        Q = (
            settings.PR_Q_W_SEMANTIC * _normalize(similarities)
            + settings.PR_Q_W_DOMAIN   * _normalize(domain_scores)
            + settings.PR_Q_W_SERP     * _normalize(serp_signals)
        )

        # Final normalization so Q sums to 1
        return _normalize(Q)
