"""
document_graph.py — Constructs a weighted graph over documents.

Nodes = documents (identified by index)
Edges = relationships between document pairs

Edge weight formula (simplified, embedding-dominant):
    W_ij = 0.70 * embedding_similarity(i,j)
          + 0.20 * keyword_overlap(i,j)
          + 0.10 * link_presence(i,j)

Two-phase edge creation strategy:
    Phase 1 — Threshold edges: connect any pair where W_ij > EDGE_THRESHOLD (0.25)
    Phase 2 — Top-K guarantee: for each node with < TOP_K_NEIGHBORS edges,
               force-connect to its K most similar neighbors (by embedding).
               This guarantees the graph is NEVER disconnected.

Rules:
    - Cap edges per node at MAX_EDGES_PER_NODE (10)
    - Embeddings must be L2-normalized before calling build_graph()
"""

import logging
from typing import List, Set
from urllib.parse import urlparse

import networkx as nx
import numpy as np

from app.config import settings
from models.document import CleanDocument
from query_processing.query_parser import tokenize

logger = logging.getLogger(__name__)


def _extract_domain(url: str) -> str:
    """Extract the netloc from a URL."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


class DocumentGraph:
    """Weighted undirected graph over a document corpus."""

    def __init__(self):
        self.graph = nx.Graph()

    def build_graph(
        self,
        documents: List[CleanDocument],
        embeddings: np.ndarray,
    ) -> nx.Graph:
        """
        Build the document graph from clean documents and their L2-normalized embeddings.

        Args:
            documents: list of CleanDocument objects
            embeddings: (N, D) array — MUST be L2-normalized
        """
        n = len(documents)
        logger.info("Building document graph for %d documents", n)

        if n == 0:
            return self.graph

        # ── Verify normalization, fix if needed ──────────────────────
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        # ── Pre-compute pairwise similarity matrix (vectorized) ───────
        # Shape: (N, N) — cosine similarity for all pairs at once
        sim_matrix = embeddings @ embeddings.T

        # ── Pre-compute keyword sets ─────────────────────────────────
        keyword_sets: List[Set[str]] = []
        for doc in documents:
            snippet = doc.title + " " + " ".join(doc.headings) + " " + doc.content[:200]
            keyword_sets.append(set(tokenize(snippet)))

        # ── Pre-compute domain sets ───────────────────────────────────
        domain_sets: List[Set[str]] = []
        for doc in documents:
            domains = {_extract_domain(lnk) for lnk in doc.outbound_links if lnk}
            domain_sets.append(domains)

        # ── Add nodes ────────────────────────────────────────────────
        for idx, doc in enumerate(documents):
            self.graph.add_node(idx, url=doc.url, title=doc.title)

        # ── Phase 1: Threshold-based edges ───────────────────────────
        edge_counts = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if self.graph.degree(i) >= settings.MAX_EDGES_PER_NODE:
                    break
                if self.graph.degree(j) >= settings.MAX_EDGES_PER_NODE:
                    continue

                w = self._compute_weight(
                    sim=sim_matrix[i, j],
                    kw_i=keyword_sets[i], kw_j=keyword_sets[j],
                    dom_i=domain_sets[i], dom_j=domain_sets[j],
                    url_i=documents[i].url, url_j=documents[j].url,
                )
                if w > settings.EDGE_THRESHOLD:
                    self.graph.add_edge(i, j, weight=float(w))
                    edge_counts[i] += 1
                    edge_counts[j] += 1

        edges_after_phase1 = self.graph.number_of_edges()
        logger.info("Phase 1 (threshold %.2f): %d edges", settings.EDGE_THRESHOLD, edges_after_phase1)

        # ── Phase 2: Top-K guarantee ─────────────────────────────────
        # For every node with fewer than TOP_K_NEIGHBORS edges, force-connect
        # it to its most similar (by embedding) unconnected neighbors.
        # This GUARANTEES the graph is never fully disconnected.
        k = settings.TOP_K_NEIGHBORS
        for i in range(n):
            if self.graph.degree(i) >= k:
                continue

            # Rank all other nodes by embedding similarity
            sims = [(j, sim_matrix[i, j]) for j in range(n) if j != i]
            sims.sort(key=lambda x: x[1], reverse=True)

            for j, sim_val in sims:
                if self.graph.degree(i) >= k:
                    break
                if self.graph.has_edge(i, j):
                    continue
                if self.graph.degree(j) >= settings.MAX_EDGES_PER_NODE:
                    continue

                # Use raw embedding similarity as weight for forced edges
                forced_w = float(max(sim_val, 0.01))
                self.graph.add_edge(i, j, weight=forced_w)

        edges_after_phase2 = self.graph.number_of_edges()
        logger.info(
            "Phase 2 (top-K=%d guarantee): added %d forced edges → total %d edges",
            k, edges_after_phase2 - edges_after_phase1, edges_after_phase2,
        )

        # ── Log similarity stats for debugging ───────────────────────
        if n > 1:
            upper = sim_matrix[np.triu_indices(n, k=1)]
            logger.info(
                "Embedding similarity — min: %.3f, avg: %.3f, max: %.3f",
                float(upper.min()), float(upper.mean()), float(upper.max()),
            )

        logger.info(
            "Document graph final: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    def _compute_weight(
        self,
        sim: float,
        kw_i: Set[str], kw_j: Set[str],
        dom_i: Set[str], dom_j: Set[str],
        url_i: str, url_j: str,
    ) -> float:
        """
        Compute edge weight: 0.70 * emb + 0.20 * kw + 0.10 * link
        Embedding-dominant because other signals are too sparse at this scale.
        """
        # 1. Embedding cosine (pre-computed, passed in)
        emb_sim = float(sim)

        # 2. Keyword Jaccard over title + heading + content snippet
        union = kw_i | kw_j
        kw_overlap = len(kw_i & kw_j) / len(union) if union else 0.0

        # 3. Domain-level link presence (0.6 partial credit, not binary)
        dom_a = _extract_domain(url_i)
        dom_b = _extract_domain(url_j)
        link_signal = 0.6 if (dom_b in dom_i or dom_a in dom_j) else 0.0

        return (
            settings.EDGE_W_EMBEDDING * emb_sim
            + settings.EDGE_W_KEYWORD   * kw_overlap
            + settings.EDGE_W_HYPERLINK * link_signal
        )

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return the weighted adjacency matrix as a dense numpy array."""
        return nx.to_numpy_array(self.graph, weight="weight")
