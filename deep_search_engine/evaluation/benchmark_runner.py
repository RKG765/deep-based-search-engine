"""
benchmark_runner.py — Evaluates the recursive graph retrieval system
against a BM25 baseline using standard IR metrics.

Usage:
    python -m evaluation.benchmark_runner

Compares:
    1. BM25 (lexical baseline)
    2. Recursive Graph Retrieval (our system)

on Precision@10, Recall@10, NDCG@10, and Topical Coverage.
"""

import logging
import math
from typing import Dict, List, Set, Tuple

from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, topical_coverage

logger = logging.getLogger(__name__)

# ─── Simple BM25 baseline ──────────────────────────────────────────

class BM25:
    """
    Minimal BM25 implementation for baseline comparison.
    Operates over pre-tokenized documents.
    """

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / self.doc_count if self.doc_count else 0
        self.doc_freqs: Dict[str, int] = {}
        self._compute_doc_freqs()

    def _compute_doc_freqs(self):
        for doc in self.corpus:
            seen = set()
            for token in doc:
                if token not in seen:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen.add(token)

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc = self.corpus[doc_idx]
        dl = len(doc)
        score = 0.0
        term_freqs: Dict[str, int] = {}
        for t in doc:
            term_freqs[t] = term_freqs.get(t, 0) + 1

        for qt in query_tokens:
            tf = term_freqs.get(qt, 0)
            idf = self._idf(qt)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (numerator / denominator)
        return score

    def rank(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """Return top-k (doc_index, score) tuples sorted by BM25 score."""
        scores = [(i, self.score(query_tokens, i)) for i in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ─── Benchmark runner ───────────────────────────────────────────────

def run_benchmark(
    queries: List[str],
    query_tokens: List[List[str]],
    corpus_tokens: List[List[str]],
    corpus_ids: List[str],
    relevance_map: Dict[str, Set[str]],
    graph_results: Dict[str, List[str]],
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Compare BM25 baseline vs Recursive Graph Retrieval.

    Args:
        queries:         list of query strings
        query_tokens:    tokenized versions of queries
        corpus_tokens:   tokenized corpus (for BM25)
        corpus_ids:      document identifiers matching corpus_tokens
        relevance_map:   {query → set of relevant doc_ids}
        graph_results:   {query → ranked list of doc_ids from our system}
        k:               evaluation cutoff

    Returns:
        {
            "bm25":  {"precision@k": ..., "recall@k": ..., "ndcg@k": ...},
            "graph": {"precision@k": ..., "recall@k": ..., "ndcg@k": ...},
        }
    """
    bm25 = BM25(corpus_tokens)
    bm25_metrics = {"precision@k": 0.0, "recall@k": 0.0, "ndcg@k": 0.0}
    graph_metrics = {"precision@k": 0.0, "recall@k": 0.0, "ndcg@k": 0.0}

    for qi, query in enumerate(queries):
        relevant = relevance_map.get(query, set())

        # BM25 ranking
        bm25_ranked = bm25.rank(query_tokens[qi], top_k=k)
        bm25_ids = [corpus_ids[idx] for idx, _ in bm25_ranked]
        bm25_rels = [1.0 if did in relevant else 0.0 for did in bm25_ids]

        bm25_metrics["precision@k"] += precision_at_k(bm25_ids, relevant, k)
        bm25_metrics["recall@k"] += recall_at_k(bm25_ids, relevant, k)
        bm25_metrics["ndcg@k"] += ndcg_at_k(bm25_rels, k)

        # Graph retrieval ranking
        g_ids = graph_results.get(query, [])[:k]
        g_rels = [1.0 if did in relevant else 0.0 for did in g_ids]

        graph_metrics["precision@k"] += precision_at_k(g_ids, relevant, k)
        graph_metrics["recall@k"] += recall_at_k(g_ids, relevant, k)
        graph_metrics["ndcg@k"] += ndcg_at_k(g_rels, k)

    # Average across queries
    n = max(len(queries), 1)
    for key in bm25_metrics:
        bm25_metrics[key] /= n
        graph_metrics[key] /= n

    logger.info("BM25  metrics: %s", bm25_metrics)
    logger.info("Graph metrics: %s", graph_metrics)

    return {"bm25": bm25_metrics, "graph": graph_metrics}
