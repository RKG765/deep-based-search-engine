"""
cache_manager.py — Orchestrates vector cache TTL sweeps and content_hash lookups.
Wraps VectorStore eviction for use by the API layer.
"""

import logging

from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class CacheManager:
    """Thin wrapper around VectorStore for cache lifecycle management."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def run_eviction(self) -> int:
        """Sweep stale entries beyond TTL. Returns count of removed docs."""
        removed = self.vector_store.evict_stale()
        if removed > 0:
            logger.info("Cache eviction removed %d stale entries", removed)
        return removed

    def is_duplicate(self, content_hash: str) -> bool:
        """Check if a content_hash already exists in the cache."""
        return content_hash in self.vector_store.content_hashes

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "total_documents": self.vector_store.index.ntotal,
            "unique_hashes": len(self.vector_store.content_hashes),
        }
