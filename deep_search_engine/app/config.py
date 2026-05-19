"""
config.py — Central configuration for the Deep Recursive Research Search Engine.
All tunable parameters from the architectural blueprint are defined here.
Uses pydantic-settings for environment variable overrides.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings. Override any value via environment variables."""

    # ─── App ───────────────────────────────────────────────────────────
    APP_NAME: str = "Deep Recursive Research Search Engine"
    DEBUG: bool = True

    # ─── Search API Keys ──────────────────────────────────────────────
    BRAVE_API_KEY: str = os.getenv("BRAVE_API_KEY", "")

    # ─── Concurrency & Resilience ─────────────────────────────────────
    SEARCH_CONCURRENCY_LIMIT: int = 10
    SEARCH_RETRY_ATTEMPTS: int = 3
    REQUEST_TIMEOUT_SECONDS: int = 10

    # ─── BFS Recursive Search Limits ──────────────────────────────────
    MAX_NEW_NODES: int = 6           # query expansion cap (was 3)
    MAX_DEPTH: int = 3               # BFS depth limit (was 2)
    MAX_NODES_PER_LEVEL: int = 12    # max nodes explored per BFS level (was 5)
    MAX_RESULTS_PER_SEARCH: int = 10 # SERP results per node query (was 5)
    MAX_TOTAL_DOCS: int = 200        # absolute document cap

    # ─── Content Quality ─────────────────────────────────────────────
    MIN_WORD_COUNT: int = 300        # hard filter — drop thin pages (was 50)
    MIN_UNIQUE_TOKEN_RATIO: float = 0.30   # drop SEO-spam pages
    MAX_LINK_DENSITY: float = 0.05   # drop nav/directory pages

    # ─── Node Pruning ────────────────────────────────────────────────
    PRUNING_THRESHOLD: float = 0.55  # kill noise early (was 0.3)
    # 6-component scoring weights (must sum to 1.0)
    PRUNING_W_SEMANTIC: float = 0.25
    PRUNING_W_KEYWORD: float = 0.20
    PRUNING_W_SERP: float = 0.15
    PRUNING_W_DOMAIN: float = 0.15
    PRUNING_W_QUALITY: float = 0.15
    PRUNING_W_FRESHNESS: float = 0.10

    # ─── Document Graph ──────────────────────────────────────────────
    EDGE_THRESHOLD: float = 0.25     # lowered to guarantee edges form (was 0.40)
    MAX_EDGES_PER_NODE: int = 10
    TOP_K_NEIGHBORS: int = 5         # force at least K edges per node regardless of threshold
    EDGE_W_EMBEDDING: float = 0.70   # dominant signal — sparse docs need this
    EDGE_W_KEYWORD: float = 0.20
    EDGE_W_HYPERLINK: float = 0.10   # reduced: domain-level links still too sparse
    EDGE_W_ANCHOR: float = 0.00      # disabled: slug-words too noisy at small N

    # ─── PageRank ────────────────────────────────────────────────────
    PR_ALPHA: float = 0.85
    PR_ITERATIONS: int = 30          # more iterations (was 20)
    PR_TOLERANCE: float = 1e-6
    # Q-vector blending weights
    PR_Q_W_SEMANTIC: float = 0.60
    PR_Q_W_DOMAIN: float = 0.25
    PR_Q_W_SERP: float = 0.15

    # ─── Re-ranking ──────────────────────────────────────────────────
    RERANK_TOP_K: int = 20           # re-rank window post-PageRank
    RERANK_W_COSINE: float = 0.60    # passage-level cosine weight
    RERANK_W_PAGERANK: float = 0.40  # pagerank score weight

    # ─── Deduplication ───────────────────────────────────────────────
    DEDUP_THRESHOLD: float = 0.9
    MINHASH_NUM_PERM: int = 128

    # ─── Vector Cache / FAISS ────────────────────────────────────────
    CACHE_TTL_DAYS: int = 7
    FAISS_MAX_DOCUMENTS: int = 100_000
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ─── Groq LLM (Primary) ─────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # ─── Ollama / LLM (Fallback) ─────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"


settings = Settings()
