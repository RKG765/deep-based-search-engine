"""
document_store.py — In-memory document persistence layer.
Stores CleanDocument objects for lookup by URL or ID during graph construction.
"""

import logging
from typing import Dict, List, Optional

from models.document import CleanDocument

logger = logging.getLogger(__name__)


class DocumentStore:
    """Simple in-memory registry of clean documents."""

    def __init__(self):
        self._by_url: Dict[str, CleanDocument] = {}
        self._by_index: List[CleanDocument] = []

    def add(self, doc: CleanDocument) -> int:
        """Store a document and return its integer index."""
        if doc.url in self._by_url:
            return self._by_index.index(self._by_url[doc.url])
        idx = len(self._by_index)
        self._by_index.append(doc)
        self._by_url[doc.url] = doc
        return idx

    def add_batch(self, docs: List[CleanDocument]) -> List[int]:
        """Store multiple documents, returning their indices."""
        return [self.add(d) for d in docs]

    def get_by_url(self, url: str) -> Optional[CleanDocument]:
        return self._by_url.get(url)

    def get_by_index(self, idx: int) -> Optional[CleanDocument]:
        if 0 <= idx < len(self._by_index):
            return self._by_index[idx]
        return None

    def all_docs(self) -> List[CleanDocument]:
        return list(self._by_index)

    def count(self) -> int:
        return len(self._by_index)
