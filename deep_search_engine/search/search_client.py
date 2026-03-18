"""
search_client.py — Unified search interface.
Primary:  Brave Search API
Fallback: DuckDuckGo (via duckduckgo-search library)

Returns a flat list of SearchResult dataclasses for each query.
"""

import logging
from typing import List

import httpx
from duckduckgo_search import DDGS

from app.config import settings
from models.document import SearchResult

logger = logging.getLogger(__name__)


async def _search_brave(query: str, max_results: int) -> List[SearchResult]:
    """
    Hit the Brave Search API (web search endpoint).
    Requires BRAVE_API_KEY to be set.
    """
    if not settings.BRAVE_API_KEY:
        raise ValueError("BRAVE_API_KEY not configured")

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": settings.BRAVE_API_KEY,
    }
    params = {"q": query, "count": max_results}

    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

    results: List[SearchResult] = []
    for idx, item in enumerate(data.get("web", {}).get("results", []), start=1):
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                rank=idx,
            )
        )
    return results[:max_results]


def _search_ddg(query: str, max_results: int) -> List[SearchResult]:
    """
    Fallback: use DuckDuckGo text search.
    This is synchronous but wrapped in the caller.
    """
    results: List[SearchResult] = []
    with DDGS() as ddgs:
        for idx, r in enumerate(ddgs.text(query, max_results=max_results), start=1):
            results.append(
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    rank=idx,
                )
            )
    return results[:max_results]


async def search(query: str, max_results: int | None = None) -> List[SearchResult]:
    """
    Public search API.
    Tries Brave first; falls back to DuckDuckGo on failure.
    """
    max_results = max_results or settings.MAX_RESULTS_PER_SEARCH

    # Try Brave
    if settings.BRAVE_API_KEY:
        try:
            results = await _search_brave(query, max_results)
            logger.info("Brave returned %d results for '%s'", len(results), query)
            return results
        except Exception as exc:
            logger.warning("Brave search failed for '%s': %s — falling back to DDG", query, exc)

    # Fallback: DuckDuckGo (sync, so we run in executor to avoid blocking)
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, _search_ddg, query, max_results)
        logger.info("DDG returned %d results for '%s'", len(results), query)
        return results
    except Exception as exc:
        logger.error("DuckDuckGo search also failed for '%s': %s", query, exc)
        return []
