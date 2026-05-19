"""
models/document.py — Shared document data structures used across modules.
Centralised here to prevent circular imports between scraping, storage, and graph.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ScrapedPage:
    """Raw extraction result from a single web page (scraper.py output)."""
    url: str
    title: str = ""
    headings: List[str] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class CleanDocument:
    """A cleaned, graph-ready document (content_cleaner.py output)."""
    url: str
    title: str
    content: str
    headings: List[str] = field(default_factory=list)
    outbound_links: List[str] = field(default_factory=list)
    word_count: int = 0
    # ── Ranking signals (plumbed from pipeline) ──────────────────────
    serp_rank: int = 0          # 1-indexed SERP position (0 = unknown)
    quality_score: float = 0.0  # composite content quality [0, 1]
    domain_score: float = 0.0   # domain authority heuristic [0, 1]


@dataclass
class SearchResult:
    """A single SERP result returned by the search client."""
    title: str
    url: str
    snippet: str
    rank: int           # 1-indexed position in SERP


@dataclass
class RankedDocument:
    """A document with its PageRank score (graph_ranker.py output)."""
    index: int
    score: float
    url: str
    title: str


@dataclass
class RerankResult:
    """A document after the final re-ranking stage."""
    index: int
    final_score: float          # raw blended score: 0.60×cosine + 0.40×pr_norm
    confidence_pct: float       # min-max normalized relative confidence [0–100]
    pagerank_score: float       # raw PageRank score (sums to 1 across corpus)
    cosine_score: float         # passage-level cosine similarity [0–1]
    domain_score: float         # domain authority heuristic [0–1]
    quality_score: float        # composite content quality [0–1]
    url: str
    title: str
