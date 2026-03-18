"""
summarizer.py — Answer generation from top-ranked documents.

Primary:  LLaMA via Ollama (abstractive)
Fallback: sumy TextRank (extractive)

Output includes structured answer text plus citations.
"""

import logging
from dataclasses import dataclass
from typing import List

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A single source citation."""
    url: str
    title: str


@dataclass
class AnswerResponse:
    """Final structured answer returned to the API."""
    answer: str
    citations: List[Citation]


# ─── Ollama / LLaMA ─────────────────────────────────────────────────

async def _check_ollama() -> bool:
    """Check if the Ollama server is reachable."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


async def _summarize_ollama(context: str, query: str) -> str:
    """Call Ollama's generate endpoint with the ranked document context."""
    prompt = (
        f"You are a research assistant. Based on the following documents, "
        f"provide a comprehensive answer to the query.\n\n"
        f"Query: {query}\n\n"
        f"Documents:\n{context}\n\n"
        f"Provide a detailed, well-structured answer with key findings."
    )
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


# ─── sumy TextRank Fallback ─────────────────────────────────────────

def _summarize_textrank(text: str, sentence_count: int = 5) -> str:
    """Extractive summarization using TextRank via sumy."""
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in sentences)


# ─── Public API ──────────────────────────────────────────────────────

async def generate_answer(
    ranked_docs: List[dict],
    query: str,
) -> AnswerResponse:
    """
    Generate a final answer from the top-ranked documents.
    Each entry in ranked_docs should have: 'title', 'url', 'content'.

    Tries Ollama first; falls back to TextRank.
    """
    # Build context block from top documents
    context_parts = []
    citations = []
    for i, doc in enumerate(ranked_docs[:10], start=1):
        title = doc.get("title", "Untitled")
        url = doc.get("url", "")
        content = doc.get("content", "")[:1500]  # cap per-doc context
        context_parts.append(f"[{i}] {title}\nURL: {url}\n{content}\n")
        citations.append(Citation(url=url, title=title))

    context = "\n---\n".join(context_parts)

    # Try Ollama
    ollama_online = await _check_ollama()
    if ollama_online:
        try:
            answer = await _summarize_ollama(context, query)
            logger.info("Generated answer via Ollama (%d chars)", len(answer))
            return AnswerResponse(answer=answer, citations=citations)
        except Exception as exc:
            logger.warning("Ollama generation failed: %s — falling back to TextRank", exc)

    # Fallback: TextRank extractive summary
    combined_text = " ".join(part for part in context_parts)
    answer = _summarize_textrank(combined_text, sentence_count=5)
    logger.info("Generated answer via TextRank fallback (%d chars)", len(answer))
    return AnswerResponse(answer=answer, citations=citations)
