"""
scraper.py — Web page fetcher using scrapling + httpx fallback.
Provides adaptive selectors, built-in anti-bot bypass, and UA rotation.
Falls back to httpx if scrapling encounters SSL or other issues.
"""

import logging
import ssl
from typing import List

import httpx
from scrapling import Fetcher

from app.config import settings
from models.document import ScrapedPage

logger = logging.getLogger(__name__)

# Initialise scrapling Fetcher
fetcher = Fetcher()


def _parse_html_with_scrapling(html: str, url: str) -> ScrapedPage | None:
    """Parse raw HTML string using scrapling's parser."""
    from scrapling import Adaptor
    page = Adaptor(html, url=url)

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
        links=links[:50],
        raw_text=raw_text,
    )


def _fetch_with_httpx(url: str) -> str | None:
    """Fallback: fetch raw HTML via httpx with SSL verification disabled."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        with httpx.Client(verify=False, timeout=settings.REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.text
    except Exception as exc:
        logger.warning("httpx fallback failed for %s: %s", url, exc)
        return None


def fetch_and_extract(url: str) -> ScrapedPage | None:
    """
    Fetch a URL and extract structured content.
    Tries scrapling first, falls back to httpx on SSL/network errors.
    Returns None if both fail.
    """
    # Attempt 1: scrapling (has anti-bot features)
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
            links=links[:50],
            raw_text=raw_text,
        )

    except Exception as exc:
        logger.warning("Scrapling failed for %s: %s — trying httpx fallback", url, exc)

    # Attempt 2: httpx with SSL disabled
    html = _fetch_with_httpx(url)
    if html:
        try:
            return _parse_html_with_scrapling(html, url)
        except Exception as exc:
            logger.warning("HTML parsing failed for %s: %s", url, exc)

    return None
