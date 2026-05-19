"""
content_cleaner.py — Cleans scraped HTML into readable, graph-ready text.
Strips boilerplate, navigation, and ads to isolate article content.

Quality gate formula:
    quality_score =
        0.30 * text_length_score     (words/2000, capped at 1.0)
        0.20 * unique_token_ratio    (unique / total tokens)
        0.20 * low_link_density      (1 - link_density, floored at 0)
        0.20 * readability_proxy     (avg sentence length in [10,25] window)
        0.10 * title_relevance       (overlap between title words & headings)

Hard filters (drops page entirely):
    - word_count < MIN_WORD_COUNT (300)
    - unique_token_ratio < MIN_UNIQUE_TOKEN_RATIO (0.30)
    - link_density > MAX_LINK_DENSITY (0.05)
"""

import re
import logging
from typing import List

from models.document import ScrapedPage, CleanDocument
from app.config import settings

logger = logging.getLogger(__name__)

# Boilerplate patterns to strip before analysis
_BOILERPLATE_RE = re.compile(
    r"(cookie policy|accept cookies|privacy policy|terms of service|"
    r"all rights reserved|subscribe to|sign up for our newsletter|"
    r"gdpr|we use cookies|this site uses cookies)",
    re.IGNORECASE,
)


def _text_length_score(word_count: int) -> float:
    """Longer articles score higher, capped at 2000 words → 1.0."""
    return min(word_count / 2000.0, 1.0)


def _unique_token_ratio(text: str) -> float:
    """Ratio of unique tokens to total tokens. Low → SEO spam."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _link_density(link_count: int, word_count: int) -> float:
    """Links per word. High → navigation/directory page."""
    if word_count == 0:
        return 1.0
    return link_count / word_count


def _readability_proxy(text: str) -> float:
    """
    Proxy readability score based on average sentence length.
    Ideal range: 10–25 words/sentence. Outside → score decreases.
    """
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.5
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    # Triangle peak at 17 words
    if avg_len < 10:
        return avg_len / 10.0
    elif avg_len <= 25:
        return 1.0
    else:
        return max(0.0, 1.0 - (avg_len - 25) / 50.0)


def _title_relevance(title: str, headings: List[str]) -> float:
    """Word overlap between title and headings — measures page coherence."""
    title_words = set(title.lower().split())
    if not title_words or not headings:
        return 0.5  # neutral when missing
    heading_words = set(" ".join(headings).lower().split())
    common = title_words & heading_words
    return len(common) / len(title_words)


def compute_quality_score(
    body: str,
    title: str,
    headings: List[str],
    link_count: int,
    word_count: int,
) -> float:
    """
    Compute a composite [0, 1] quality score for a document.
    Higher = more likely to be a real, useful article.
    """
    tl = _text_length_score(word_count)
    utr = _unique_token_ratio(body)
    ld = _link_density(link_count, word_count)
    low_link = max(0.0, 1.0 - ld / settings.MAX_LINK_DENSITY)  # inverted + scaled
    rd = _readability_proxy(body)
    tr = _title_relevance(title, headings)

    score = (
        0.30 * tl
        + 0.20 * utr
        + 0.20 * low_link
        + 0.20 * rd
        + 0.10 * tr
    )
    return round(min(score, 1.0), 4)


def clean_scraped_page(page: ScrapedPage | None) -> CleanDocument | None:
    """
    Convert a ScrapedPage into a CleanDocument.
    Applies hard filters and computes a composite quality_score.
    Returns None if the page fails any hard filter.
    """
    if page is None:
        return None

    # Merge paragraphs, strip boilerplate patterns
    raw_body = " ".join(page.paragraphs)
    body = _BOILERPLATE_RE.sub(" ", raw_body)
    body = re.sub(r"\s+", " ", body).strip()

    word_count = len(body.split())
    link_count = len(page.links)

    # ── Hard filter 1: minimum article length ────────────────────────
    if word_count < settings.MIN_WORD_COUNT:
        logger.debug("Thin page (%d words), dropping: %s", word_count, page.url)
        return None

    # ── Hard filter 2: unique token ratio (SEO spam detector) ────────
    utr = _unique_token_ratio(body)
    if utr < settings.MIN_UNIQUE_TOKEN_RATIO:
        logger.debug("Low token diversity (%.2f), dropping: %s", utr, page.url)
        return None

    # ── Hard filter 3: link density (nav/directory page detector) ────
    ld = _link_density(link_count, word_count)
    if ld > settings.MAX_LINK_DENSITY:
        logger.debug("High link density (%.3f), dropping: %s", ld, page.url)
        return None

    quality = compute_quality_score(body, page.title, page.headings, link_count, word_count)

    return CleanDocument(
        url=page.url,
        title=page.title,
        content=body,
        headings=page.headings,
        outbound_links=page.links,
        word_count=word_count,
        quality_score=quality,
    )
