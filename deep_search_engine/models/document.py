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
