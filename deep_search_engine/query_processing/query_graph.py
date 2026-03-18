"""
query_graph.py — Builds the BFS query exploration graph.
Nodes represent search queries. Edges represent parent → child expansion.
Uses NetworkX DiGraph with depth tracking on each node.
"""

import logging
from typing import List

import networkx as nx

from query_processing.query_planner import generate_seed_nodes

logger = logging.getLogger(__name__)


class QueryGraph:
    """Directed graph modelling the recursive query expansion tree."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.root: str | None = None

    def build_from_query(self, query: str) -> nx.DiGraph:
        """
        Build the initial query graph.
        Depth 0 → original query
        Depth 1 → expanded seed nodes (max MAX_NEW_NODES)
        """
        seeds = generate_seed_nodes(query)
        self.root = seeds[0]

        # Root node at depth 0
        self.graph.add_node(self.root, depth=0, node_type="original")
        logger.info("Root query node: '%s'", self.root)

        # Expanded nodes at depth 1
        for seed in seeds[1:]:
            self.graph.add_node(seed, depth=1, node_type="expanded")
            self.graph.add_edge(self.root, seed, relation="expansion")
            logger.debug("  Depth-1 node: '%s'", seed)

        return self.graph

    def add_expanded_nodes(
        self, parent: str, children: List[str], depth: int
    ) -> None:
        """
        Attach new child nodes discovered during recursive scraping.
        Called at Depth 2 when we extract new terms from scraped content.
        """
        for child in children:
            if child not in self.graph:
                self.graph.add_node(child, depth=depth, node_type="recursive")
                self.graph.add_edge(parent, child, relation="recursive_expansion")

    def get_nodes_at_depth(self, depth: int) -> List[str]:
        """Return all query node labels at a given BFS depth."""
        return [
            n for n, d in self.graph.nodes(data=True) if d.get("depth") == depth
        ]

    def all_nodes(self) -> List[str]:
        """Return every query node in the graph."""
        return list(self.graph.nodes)
