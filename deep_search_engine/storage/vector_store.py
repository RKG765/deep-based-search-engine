"""
vector_store.py — FAISS-backed dense vector store with metadata index.
Used for:
    - building the Q (query relevance) vector for PageRank
    - semantic cache: detect similar past queries
    - cross-query deduplication via content_hash

Blueprint constraints:
    Embedding model : all-MiniLM-L6-v2 (384-dim)
    max_documents   : 100,000
    cache_ttl       : 7 days
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from utils.text_utils import compute_content_hash

logger = logging.getLogger(__name__)

# ─── Lazy-loaded embedding model ────────────────────────────────────
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        import os
        # Increase timeout drastically for slow or restricted networks
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
        # Optional: Disable proxy SSL verification if still failing
        os.environ["CURL_CA_BUNDLE"] = ""
        _embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embed_model


@dataclass
class DocumentMeta:
    """Metadata payload stored alongside each FAISS vector."""
    document_id: str
    url: str
    title: str
    content: str
    content_hash: str
    timestamp: float            # Unix epoch
    source_query: str


class VectorStore:
    """FAISS flat index + in-memory metadata map."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)   # inner-product (cosine on normalized vecs)
        self.metadata: Dict[int, DocumentMeta] = {} # faiss row → meta
        self.content_hashes: set = set()            # fast hash-based dedup
        self._next_id = 0

    # ── Embeddings ────────────────────────────────────────────────────

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts and L2-normalize for cosine similarity via inner product."""
        model = _get_embed_model()
        vectors = model.encode(texts, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(vectors)
        return vectors

    # ── Add ───────────────────────────────────────────────────────────

    def add_documents(
        self,
        contents: List[str],
        urls: List[str],
        titles: List[str],
        source_query: str,
    ) -> List[str]:
        """
        Embed + index a batch of documents.
        Skips duplicates by content_hash.
        Returns list of new document_ids added.
        """
        new_ids: List[str] = []
        texts_to_embed: List[str] = []
        metas_to_store: List[DocumentMeta] = []

        for content, url, title in zip(contents, urls, titles):
            chash = compute_content_hash(content)
            if chash in self.content_hashes:
                logger.debug("Skipping duplicate (hash) %s", url)
                continue
            if self.index.ntotal >= settings.FAISS_MAX_DOCUMENTS:
                logger.warning("FAISS store full (%d docs) — skipping", settings.FAISS_MAX_DOCUMENTS)
                break

            doc_id = str(uuid.uuid4())
            meta = DocumentMeta(
                document_id=doc_id,
                url=url,
                title=title,
                content=content[:5000],     # cap stored content
                content_hash=chash,
                timestamp=time.time(),
                source_query=source_query,
            )
            self.content_hashes.add(chash)
            texts_to_embed.append(content)
            metas_to_store.append(meta)
            new_ids.append(doc_id)

        if texts_to_embed:
            vectors = self.embed_texts(texts_to_embed)
            start_id = self._next_id
            self.index.add(vectors)
            for i, meta in enumerate(metas_to_store):
                self.metadata[start_id + i] = meta
            self._next_id += len(texts_to_embed)
            logger.info("Added %d documents to FAISS (total: %d)", len(texts_to_embed), self.index.ntotal)

        return new_ids

    # ── Search ────────────────────────────────────────────────────────

    def search_similar(self, query: str, top_k: int = 10) -> List[DocumentMeta]:
        """Return top_k most similar documents to the query string."""
        if self.index.ntotal == 0:
            return []
        vec = self.embed_texts([query])
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vec, k)
        results = []
        for idx in indices[0]:
            if idx >= 0 and idx in self.metadata:
                results.append(self.metadata[idx])
        return results

    def get_all_embeddings(self) -> np.ndarray:
        """Reconstruct all stored vectors."""
        if self.index.ntotal == 0:
            return np.empty((0, self.dimension), dtype="float32")
        return self.index.reconstruct_n(0, self.index.ntotal)

    def get_all_metadata(self) -> List[DocumentMeta]:
        """Return metadata in insertion order."""
        return [self.metadata[i] for i in sorted(self.metadata.keys())]

    # ── Eviction ──────────────────────────────────────────────────────

    def evict_stale(self) -> int:
        """
        Remove documents older than CACHE_TTL_DAYS.
        NOTE: FAISS does not support item removal, so we rebuild the index.
        """
        cutoff = time.time() - (settings.CACHE_TTL_DAYS * 86400)
        keep_ids = [i for i, m in self.metadata.items() if m.timestamp >= cutoff]
        removed = self.index.ntotal - len(keep_ids)

        if removed <= 0:
            return 0

        # Rebuild
        vectors = np.vstack([self.index.reconstruct(i) for i in keep_ids]).astype("float32")
        new_meta = {new_i: self.metadata[old_i] for new_i, old_i in enumerate(keep_ids)}
        self.content_hashes = {m.content_hash for m in new_meta.values()}

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self.metadata = new_meta
        self._next_id = len(keep_ids)

        logger.info("Evicted %d stale documents (TTL=%dd)", removed, settings.CACHE_TTL_DAYS)
        return removed
