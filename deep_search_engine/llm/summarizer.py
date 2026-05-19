"""
summarizer.py — Answer generation from top-ranked documents.

Priority chain:
    1. Groq API  (LLaMA 3.3 70B — fast cloud inference)
    2. Ollama    (local LLaMA 3 — offline fallback)
    3. TextRank  (extractive — zero-dependency last resort)

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
    provider: str = "unknown"  # "groq", "ollama", or "textrank"


# ─── Shared prompt builder ──────────────────────────────────────────

def _build_prompt(context: str, query: str) -> str:
    return (
        "You are a research assistant. Based on the following documents, "
        "provide a comprehensive, well-structured answer to the query.\n\n"
        f"Query: {query}\n\n"
        f"Documents:\n{context}\n\n"
        "Instructions:\n"
        "- Synthesize information across multiple sources\n"
        "- Cite sources using [1], [2], etc. matching the document numbers\n"
        "- Be detailed but concise\n"
        "- Use bullet points or numbered lists for clarity when appropriate\n"
        "- If sources disagree, note the different perspectives"
    )


# ─── Groq API (Primary) ────────────────────────────────────────────

async def _check_groq() -> bool:
    """Verify that the Groq API key is set and reachable."""
    if not settings.GROQ_API_KEY:
        return False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"{settings.GROQ_BASE_URL}/models",
                headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
            )
            return resp.status_code == 200
    except Exception:
        return False


async def _summarize_groq(context: str, query: str) -> str:
    """Call Groq's OpenAI-compatible chat completions endpoint."""
    prompt = _build_prompt(context, query)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{settings.GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a research assistant that provides comprehensive, cited answers based on provided documents."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 2048,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ─── Ollama / LLaMA (Fallback 1) ───────────────────────────────────

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
    prompt = _build_prompt(context, query)
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


# ─── sumy TextRank Fallback (Last Resort) ──────────────────────────

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

    Tries Groq first → Ollama second → TextRank last resort.
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

    # ── Try 1: Groq API ──────────────────────────────────────────────
    groq_online = await _check_groq()
    if groq_online:
        try:
            answer = await _summarize_groq(context, query)
            logger.info("Generated answer via Groq/%s (%d chars)", settings.GROQ_MODEL, len(answer))
            return AnswerResponse(answer=answer, citations=citations, provider="groq")
        except Exception as exc:
            logger.warning("Groq generation failed: %s — trying Ollama", exc)

    # ── Try 2: Ollama ────────────────────────────────────────────────
    ollama_online = await _check_ollama()
    if ollama_online:
        try:
            answer = await _summarize_ollama(context, query)
            logger.info("Generated answer via Ollama (%d chars)", len(answer))
            return AnswerResponse(answer=answer, citations=citations, provider="ollama")
        except Exception as exc:
            logger.warning("Ollama generation failed: %s — falling back to TextRank", exc)

    # ── Try 3: TextRank extractive summary ───────────────────────────
    combined_text = " ".join(part for part in context_parts)
    answer = _summarize_textrank(combined_text, sentence_count=5)
    logger.info("Generated answer via TextRank fallback (%d chars)", len(answer))
    return AnswerResponse(answer=answer, citations=citations, provider="textrank")
