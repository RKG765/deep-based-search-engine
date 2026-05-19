"""
recursive_search.py — The BFS recursive search engine.
Core loop that explores query nodes breadth-first, scrapes results,
extracts new terms, prunes candidates, and collects unique documents.

Blueprint constraints:
    max_depth            = 2
    max_nodes_per_level  = 5
    max_results_per_search = 5
    max_total_docs       = 200

Deduplication is performed via scraping.deduplicator (MinHash).
"""

import asyncio
import logging
from typing import List, Set, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from models.document import CleanDocument, SearchResult
from models.query import Query
from query_processing.query_graph import QueryGraph
from query_processing.query_parser import tokenize
from query_processing.keyword_extractor import extract_all
from query_processing.query_understanding import reformulate_query, INTENT_FACTUAL
from search.search_client import search
from search.node_pruning import prune_nodes, domain_authority
from scraping.scraper import fetch_and_extract
from scraping.content_cleaner import clean_scraped_page
from scraping.deduplicator import Deduplicator

logger = logging.getLogger(__name__)

# ─── Lazy-loaded embedding model ────────────────────────────────────
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embed_model


def _embed(texts: List[str]) -> np.ndarray:
    """Batch-encode texts into dense vectors."""
    model = _get_embed_model()
    return model.encode(texts, show_progress_bar=False)


class RecursiveSearchEngine:
    """
    Breadth-First Search engine that recursively explores query nodes.

    Depth 0: original query          → search + scrape
    Depth 1: expanded seed nodes     → search + scrape
    Depth 2: nodes extracted from D1 → search + scrape (terminal)
    """

    def __init__(self):
        self.query_graph = QueryGraph()
        self.collected_docs: List[CleanDocument] = []
        self.seen_urls: Set[str] = set()
        self.deduplicator = Deduplicator()
        self.query_intent: str = ""

    async def run(self, query: Query) -> List[CleanDocument]:
        """
        Main entry point. Accepts a structured Query object.
        Runs full BFS pipeline and returns deduplicated documents.
        """
        max_depth = min(query.depth, settings.MAX_DEPTH)

        # ── Step 0: Understand intent & reformulate ──────────────────
        reformulated, intent = reformulate_query(query.raw)
        self.query_intent = intent
        logger.info(
            "Query understanding: intent='%s', reformulated='%s'",
            intent, reformulated,
        )

        # For factual queries, use the reformulated version as the primary search query
        # This preserves "capital of india" instead of splitting into "capital" + "india"
        search_query = reformulated if intent == INTENT_FACTUAL else query.raw

        logger.info(
            "Starting recursive search for '%s' (search_query='%s', max_depth=%d, pruning=%s)",
            query.raw, search_query, max_depth, query.pruning,
        )

        # Build query graph (Depth 0 + Depth 1 seeds) using the search query
        self.query_graph.build_from_query(search_query)

        # Compute query embedding for pruning
        query_embedding = _embed([query.raw])[0]
        query.embedding = query_embedding
        query_keywords = set(tokenize(query.raw))
        query.keywords = query_keywords

        # BFS level-by-level
        for current_depth in range(max_depth + 1):
            nodes = self.query_graph.get_nodes_at_depth(current_depth)
            nodes = nodes[: settings.MAX_NODES_PER_LEVEL]

            if not nodes:
                logger.info("No nodes at depth %d — stopping BFS", current_depth)
                break

            logger.info("Depth %d: processing %d nodes: %s", current_depth, len(nodes), nodes)

            # ── Search all nodes in parallel ──────────────────────────
            search_tasks = [search(node) for node in nodes]
            all_results: List[List[SearchResult]] = await asyncio.gather(
                *search_tasks, return_exceptions=True
            )

            # ── Scrape + clean + dedup results ────────────────────────
            for node_query, results in zip(nodes, all_results):
                if isinstance(results, Exception):
                    logger.warning("Search failed for node '%s': %s", node_query, results)
                    continue

                for sr in results:
                    if len(self.collected_docs) >= settings.MAX_TOTAL_DOCS:
                        logger.info("Hit MAX_TOTAL_DOCS (%d) — stopping", settings.MAX_TOTAL_DOCS)
                        return self.collected_docs

                    if sr.url in self.seen_urls:
                        continue
                    self.seen_urls.add(sr.url)

                    # Scrape (sync call — scrapling is synchronous)
                    page = await asyncio.get_event_loop().run_in_executor(
                        None, fetch_and_extract, sr.url
                    )
                    doc = clean_scraped_page(page)
                    if doc is None:
                        continue

                    # ── Plumb ranking signals ──────────────────────
                    doc.serp_rank   = sr.rank                  # SERP position
                    doc.domain_score = domain_authority(sr.url) # authority heuristic

                    # MinHash deduplication (via dedicated Deduplicator)
                    if self.deduplicator.is_duplicate(doc):
                        continue

                    self.collected_docs.append(doc)

                # ── Generate Depth+1 expansion nodes (if not terminal) ──
                if current_depth < max_depth and len(self.collected_docs) > 0:
                    last_docs = self.collected_docs[-len(results):]
                    new_terms = self._extract_expansion_terms(last_docs, query_keywords)

                    if query.pruning and new_terms:
                        # Prune: embed new terms and score them
                        new_embeddings = _embed([t["text"] for t in new_terms])
                        for t, emb in zip(new_terms, new_embeddings):
                            t["embedding"] = emb

                        new_terms = prune_nodes(
                            candidates=new_terms,
                            query_keywords=query_keywords,
                            query_embedding=query_embedding,
                            max_results=settings.MAX_RESULTS_PER_SEARCH,
                        )

                    for t in new_terms[: settings.MAX_NODES_PER_LEVEL]:
                        self.query_graph.add_expanded_nodes(
                            parent=node_query,
                            children=[t["text"]],
                            depth=current_depth + 1,
                        )

        logger.info(
            "BFS complete: %d documents collected from %d URLs explored",
            len(self.collected_docs),
            len(self.seen_urls),
        )
        return self.collected_docs

    def _extract_expansion_terms(
        self, docs: List[CleanDocument], existing_keywords: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract new search terms from recently scraped documents.
        These become candidate BFS nodes for the next depth level.
        """
        candidates: List[Dict[str, Any]] = []
        seen_terms: Set[str] = set()

        for doc in docs:
            # Use first 500 chars for keyword extraction to stay fast
            snippet = doc.content[:500]
            terms = extract_all(snippet)

            for term in terms:
                if term in seen_terms or term in existing_keywords:
                    continue
                seen_terms.add(term)
                term_keywords = set(tokenize(term))
                candidates.append({
                    "text": term,
                    "keywords": term_keywords,
                    "rank": 1,  # default rank since these come from content
                })

                if len(candidates) >= settings.MAX_NEW_NODES:
                    return candidates

        return candidates
