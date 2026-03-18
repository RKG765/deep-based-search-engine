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
    MAX_NEW_NODES: int = 3          # query expansion cap
    MAX_DEPTH: int = 2              # BFS depth limit
    MAX_NODES_PER_LEVEL: int = 5    # max nodes explored per BFS level
    MAX_RESULTS_PER_SEARCH: int = 5 # SERP results per node query
    MAX_TOTAL_DOCS: int = 200       # absolute document cap

    # ─── Node Pruning ────────────────────────────────────────────────
    PRUNING_THRESHOLD: float = 0.3
    PRUNING_W_KEYWORD: float = 0.4
    PRUNING_W_RANK: float = 0.3
    PRUNING_W_SIMILARITY: float = 0.3

    # ─── Document Graph ──────────────────────────────────────────────
    EDGE_THRESHOLD: float = 0.5
    MAX_EDGES_PER_NODE: int = 10
    EDGE_W_KEYWORD: float = 0.4
    EDGE_W_HYPERLINK: float = 0.3
    EDGE_W_EMBEDDING: float = 0.3

    # ─── PageRank ────────────────────────────────────────────────────
    PR_ALPHA: float = 0.85
    PR_ITERATIONS: int = 20
    PR_TOLERANCE: float = 1e-6

    # ─── Deduplication ───────────────────────────────────────────────
    DEDUP_THRESHOLD: float = 0.9
    MINHASH_NUM_PERM: int = 128

    # ─── Vector Cache / FAISS ────────────────────────────────────────
    CACHE_TTL_DAYS: int = 7
    FAISS_MAX_DOCUMENTS: int = 100_000
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ─── Ollama / LLM ───────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"


settings = Settings()
