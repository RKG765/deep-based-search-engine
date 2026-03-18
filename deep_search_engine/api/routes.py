"""
routes.py — FastAPI route definitions.
Single endpoint: POST /deep-search
Orchestrates the full pipeline from query to answer.
"""

import logging
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from models.query import Query
from search.recursive_search import RecursiveSearchEngine
from storage.vector_store import VectorStore
from storage.document_store import DocumentStore
from graph.document_graph import DocumentGraph
from graph.graph_ranker import GraphRanker
from llm.summarizer import generate_answer

logger = logging.getLogger(__name__)
router = APIRouter()

# ─── Request / Response schemas ──────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    depth: Optional[int] = 2
    pruning: Optional[bool] = True


class SourceDoc(BaseModel):
    url: str
    title: str


class SearchResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    documents_explored: int
    latency: float


# ─── Pipeline endpoint ──────────────────────────────────────────────

@router.post("/deep-search", response_model=SearchResponse)
async def deep_search(request: SearchRequest):
    """
    Full deep recursive search pipeline:
    1. Query Processing & Expansion
    2. BFS Recursive Search (parallel scraping + pruning + dedup)
    3. Document Graph Construction
    4. Personalized PageRank Ranking
    5. Answer Generation (LLM or TextRank)
    """
    start = time.time()
    raw_query = request.query.strip()
    if not raw_query:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    logger.info("═" * 60)
    logger.info("Deep search request: '%s' (depth=%d, pruning=%s)", raw_query, request.depth, request.pruning)

    # ── Build structured Query object ────────────────────────────────
    query = Query(
        raw=raw_query,
        depth=request.depth,
        pruning=request.pruning,
    )

    # ── Step 1+2: Recursive BFS search ───────────────────────────────
    engine = RecursiveSearchEngine()
    documents = await engine.run(query)

    if not documents:
        return SearchResponse(
            answer="No relevant documents found for this query.",
            sources=[],
            documents_explored=0,
            latency=time.time() - start,
        )

    # ── Step 3: Index into FAISS + build doc graph ───────────────────
    vector_store = VectorStore()
    doc_store = DocumentStore()
    doc_store.add_batch(documents)

    contents = [d.content for d in documents]
    urls = [d.url for d in documents]
    titles = [d.title for d in documents]

    vector_store.add_documents(contents, urls, titles, source_query=raw_query)
    doc_embeddings = vector_store.get_all_embeddings()
    query_embedding = vector_store.embed_texts([raw_query])[0]

    doc_graph = DocumentGraph()
    doc_graph.build_graph(documents, doc_embeddings)
    adjacency_matrix = doc_graph.get_adjacency_matrix()

    # ── Step 4: Personalized PageRank ────────────────────────────────
    ranker = GraphRanker()
    ranked = ranker.rank_documents(
        adjacency_matrix=adjacency_matrix,
        doc_embeddings=doc_embeddings,
        query_embedding=query_embedding,
        urls=urls,
        titles=titles,
    )

    # ── Step 5: Generate answer from top-ranked docs ─────────────────
    top_docs = []
    for r in ranked[:10]:
        doc = doc_store.get_by_index(r.index)
        if doc:
            top_docs.append({
                "title": doc.title,
                "url": doc.url,
                "content": doc.content,
            })

    answer_resp = await generate_answer(top_docs, raw_query)
    latency = time.time() - start

    logger.info(
        "Deep search complete: %d docs explored, %.2fs latency",
        len(documents), latency,
    )

    return SearchResponse(
        answer=answer_resp.answer,
        sources=[SourceDoc(url=c.url, title=c.title) for c in answer_resp.citations],
        documents_explored=len(documents),
        latency=round(latency, 2),
    )
