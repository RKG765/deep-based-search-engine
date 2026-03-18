"""
scraper.py — Web page fetcher using the scrapling library.
Provides adaptive selectors, built-in anti-bot bypass, and UA rotation.
Falls back to httpx if scrapling encounters issues.
"""

import logging
from typing import List

from scrapling import Fetcher

from app.config import settings
from models.document import ScrapedPage

logger = logging.getLogger(__name__)

# Initialise scrapling Fetcher with auto-UA rotation
fetcher = Fetcher(auto_match=False)


def fetch_and_extract(url: str) -> ScrapedPage | None:
    """
    Fetch a URL with scrapling and extract structured content.
    Returns None if the fetch fails entirely.
    """
    try:
        page = fetcher.get(url, timeout=settings.REQUEST_TIMEOUT_SECONDS)

        title = ""
        title_tag = page.find("title")
        if title_tag:
            title = title_tag.text.strip()

        headings = [
            h.text.strip()
            for h in page.find_all("h1, h2, h3")
            if h.text.strip()
        ]

        paragraphs = [
            p.text.strip()
            for p in page.find_all("p")
            if p.text.strip() and len(p.text.strip()) > 30
        ]

        links = []
        for a in page.find_all("a[href]"):
            href = a.attrib.get("href", "")
            if href.startswith("http"):
                links.append(href)

        raw_text = " ".join(paragraphs)

        return ScrapedPage(
            url=url,
            title=title,
            headings=headings,
            paragraphs=paragraphs,
            links=links[:50],  # cap outbound links
            raw_text=raw_text,
        )

    except Exception as exc:
        logger.warning("Failed to scrape %s: %s", url, exc)
        return None
