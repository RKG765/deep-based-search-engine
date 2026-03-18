"""
content_cleaner.py — Cleans scraped HTML into readable, graph-ready text.
Strips boilerplate, navigation, and ads to isolate article content.
"""

import re
import logging

from models.document import ScrapedPage, CleanDocument

logger = logging.getLogger(__name__)


def clean_scraped_page(page: ScrapedPage | None) -> CleanDocument | None:
    """
    Convert a ScrapedPage into a CleanDocument.
    Filters out extremely short pages that are unlikely to be useful articles.
    """
    if page is None:
        return None

    # Merge paragraphs into a single body
    body = " ".join(page.paragraphs)

    # Strip excessive whitespace and control characters
    body = re.sub(r"\s+", " ", body).strip()

    # Minimum quality gate: skip pages with fewer than 50 words
    word_count = len(body.split())
    if word_count < 50:
        logger.debug("Skipping thin page (%d words): %s", word_count, page.url)
        return None

    return CleanDocument(
        url=page.url,
        title=page.title,
        content=body,
        headings=page.headings,
        outbound_links=page.links,
        word_count=word_count,
    )
