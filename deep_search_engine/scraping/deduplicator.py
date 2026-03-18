"""
deduplicator.py — Near-duplicate detection using MinHash (datasketch).
Runs AFTER scraping/cleaning and BEFORE graph construction.

Threshold: similarity > 0.9 → duplicate (dropped).
"""

import logging
from typing import List

from datasketch import MinHash, MinHashLSH

from app.config import settings
from models.document import CleanDocument
from utils.text_utils import shingle

logger = logging.getLogger(__name__)


class Deduplicator:
    """MinHash-based near-duplicate detector for scraped documents."""

    def __init__(self):
        self.lsh = MinHashLSH(
            threshold=settings.DEDUP_THRESHOLD,
            num_perm=settings.MINHASH_NUM_PERM,
        )
        self._counter = 0

    def _build_minhash(self, text: str) -> MinHash:
        """Create a MinHash signature from text k-shingles."""
        m = MinHash(num_perm=settings.MINHASH_NUM_PERM)
        for s in shingle(text):
            m.update(s.encode("utf-8"))
        return m

    def is_duplicate(self, doc: CleanDocument) -> bool:
        """Check if a document is a near-duplicate of anything already seen."""
        mh = self._build_minhash(doc.content)
        try:
            hits = self.lsh.query(mh)
            if hits:
                logger.debug("Duplicate detected: %s", doc.url)
                return True
        except Exception:
            pass

        # Register as new
        self._counter += 1
        self.lsh.insert(f"doc_{self._counter}", mh)
        return False

    def deduplicate(self, documents: List[CleanDocument]) -> List[CleanDocument]:
        """
        Filter a batch of documents, returning only unique ones.
        Order is preserved; first occurrence wins.
        """
        unique: List[CleanDocument] = []
        for doc in documents:
            if not self.is_duplicate(doc):
                unique.append(doc)

        dropped = len(documents) - len(unique)
        if dropped > 0:
            logger.info(
                "Deduplication: kept %d / %d documents (dropped %d)",
                len(unique), len(documents), dropped,
            )
        return unique
