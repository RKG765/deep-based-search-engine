"""
text_utils.py — Shared text processing helpers used across multiple modules.
"""

import re
import hashlib
from typing import List, Set


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation (preserving tech terms like c++, c#, .net), and collapse whitespace."""
    text = text.lower()
    # Preserve common programming symbols: +, #, . (in context like c++, c#, .net)
    # First protect known tech terms
    _TECH_TERMS = {
        "c++": "CPLUS_CPLUS",
        "c#": "CSHARP_LANG",
        ".net": "DOTNET_LANG",
        "node.js": "NODEJS_LANG",
        "react.js": "REACTJS_LANG",
        "vue.js": "VUEJS_LANG",
        "next.js": "NEXTJS_LANG",
    }
    for term, placeholder in _TECH_TERMS.items():
        text = text.replace(term, placeholder)
    # Strip remaining punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Restore tech terms
    for term, placeholder in _TECH_TERMS.items():
        text = text.replace(placeholder, term)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_content_hash(content: str) -> str:
    """SHA-256 hash of content for fast cross-query dedup."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def shingle(text: str, k: int = 3) -> Set[str]:
    """Generate character-level k-shingles for MinHash input."""
    tokens = text.split()
    if len(tokens) < k:
        return set(tokens)
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def extract_domain(url: str) -> str:
    """Pull base domain from a URL string."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.netloc
