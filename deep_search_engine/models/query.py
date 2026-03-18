"""
models/query.py — Shared Query data structure.
Carries structured data through the pipeline instead of raw strings.
"""

from dataclasses import dataclass, field
from typing import List, Set

import numpy as np


@dataclass
class Query:
    """Structured query object passed through the entire pipeline."""

    raw: str                                    # original user input
    normalized: str = ""                        # cleaned / lowercased text
    tokens: List[str] = field(default_factory=list)        # stopword-free tokens
    keywords: Set[str] = field(default_factory=set)        # extracted key phrases
    expanded_queries: List[str] = field(default_factory=list)  # BFS seed sub-queries
    embedding: np.ndarray | None = None         # dense vector (384-dim)

    # Pipeline metadata
    depth: int = 2
    pruning: bool = True
