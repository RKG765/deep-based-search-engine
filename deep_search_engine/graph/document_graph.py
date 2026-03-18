"""
document_graph.py — Constructs a weighted graph over documents.

Nodes = documents (identified by index)
Edges = relationships between document pairs

Edge weight formula:
    W_ij = 0.4 * keyword_overlap(i,j)
         + 0.3 * hyperlink_exists(i,j)
         + 0.3 * embedding_similarity(i,j)

Rules:
    - Only connect if W_ij > EDGE_THRESHOLD (0.5)
    - Cap edges per node at MAX_EDGES_PER_NODE (10)
"""

import logging
from typing import List, Set

import networkx as nx
import numpy as np

from app.config import settings
from models.document import CleanDocument
from query_processing.query_parser import tokenize

logger = logging.getLogger(__name__)


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
        Build the document graph from a list of clean documents and their embeddings.

        Args:
            documents: list of CleanDocument objects
            embeddings: (N, D) array of L2-normalized document vectors
        """
        n = len(documents)
        logger.info("Building document graph for %d documents", n)

        # Pre-compute keyword sets for each document
        keyword_sets: List[Set[str]] = []
        for doc in documents:
            kws = set(tokenize(doc.title + " " + " ".join(doc.headings)))
            keyword_sets.append(kws)

        # Pre-compute outbound link sets for hyperlink detection
        link_sets: List[Set[str]] = []
        for doc in documents:
            link_sets.append(set(doc.outbound_links))

        # Add nodes
        for idx, doc in enumerate(documents):
            self.graph.add_node(idx, url=doc.url, title=doc.title)

        # Add edges (O(N²) but N ≤ 80 so manageable)
        for i in range(n):
            edge_count_i = 0
            # Collect candidates with scores
            candidates = []
            for j in range(i + 1, n):
                weight = self._compute_edge_weight(
                    kw_i=keyword_sets[i],
                    kw_j=keyword_sets[j],
                    links_i=link_sets[i],
                    links_j=link_sets[j],
                    url_i=documents[i].url,
                    url_j=documents[j].url,
                    emb_i=embeddings[i],
                    emb_j=embeddings[j],
                )
                if weight > settings.EDGE_THRESHOLD:
                    candidates.append((j, weight))

            # Sort by weight descending to keep only strongest edges
            candidates.sort(key=lambda x: x[1], reverse=True)
            for j, w in candidates:
                if edge_count_i >= settings.MAX_EDGES_PER_NODE:
                    break
                # Also check j's edge count
                if self.graph.degree(j) >= settings.MAX_EDGES_PER_NODE:
                    continue
                self.graph.add_edge(i, j, weight=w)
                edge_count_i += 1

        logger.info(
            "Document graph: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    def _compute_edge_weight(
        self,
        kw_i: Set[str],
        kw_j: Set[str],
        links_i: Set[str],
        links_j: Set[str],
        url_i: str,
        url_j: str,
        emb_i: np.ndarray,
        emb_j: np.ndarray,
    ) -> float:
        """
        Compute the composite edge weight between two documents.
        """
        # 1. Keyword overlap (Jaccard)
        union = kw_i | kw_j
        kw_overlap = len(kw_i & kw_j) / len(union) if union else 0.0

        # 2. Hyperlink reference (binary: does one link to the other?)
        hyperlink = 1.0 if (url_j in links_i or url_i in links_j) else 0.0

        # 3. Embedding cosine similarity (vectors are already L2-normalized)
        emb_sim = float(np.dot(emb_i, emb_j))

        return (
            settings.EDGE_W_KEYWORD * kw_overlap
            + settings.EDGE_W_HYPERLINK * hyperlink
            + settings.EDGE_W_EMBEDDING * emb_sim
        )

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return the weighted adjacency matrix as a dense numpy array."""
        return nx.to_numpy_array(self.graph, weight="weight")
