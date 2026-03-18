# Deep Recursive Research Search Engine

A production-quality **Graph-based Retrieval Engine** built in Python. Instead of naive single-pass retrieval (`Query → SERP → LLM`), this system uses recursive BFS exploration, web scraping, document graph construction, and **Topic-Sensitive PageRank** to answer complex, long-form queries comprehensively.

---

## System Architecture

```
User Query
  │
  ▼
┌─────────────────────────┐
│  Query Processing       │  spaCy + YAKE + NER
│  query_parser.py        │  Tokenize, normalize, stopwords
│  keyword_extractor.py   │  Extract noun phrases, entities
│  query_planner.py       │  Expand into BFS seed nodes (max 3)
│  query_graph.py         │  NetworkX adjacency list
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  BFS Recursive Search   │  max_depth=2, max_nodes_per_level=5
│  recursive_search.py    │  Breadth-first exploration
│  search_client.py       │  Brave API (primary) / DDG (fallback)
│  node_pruning.py        │  Score threshold ≥ 0.3
│  async_executor.py      │  Concurrency=10, backoff, 3 retries
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  Scraping & Cleaning    │  scrapling (anti-bot, UA rotation)
│  scraper.py             │  Fetch DOM, extract title/headings/links
│  content_cleaner.py     │  Quality filter (min 50 words)
│  MinHash Dedup          │  datasketch, threshold > 0.9
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  Graph + Ranking        │  NetworkX + FAISS + numpy
│  document_graph.py      │  Edges: W_ij > 0.5, max 10/node
│  graph_ranker.py        │  P(t+1) = αAP(t) + (1-α)Q
│  vector_store.py        │  all-MiniLM-L6-v2, TTL 7d
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  Answer Generation      │  Ollama LLaMA / sumy TextRank
│  summarizer.py          │  Structured answer + citations
└─────────────────────────┘
```

---

## Key Algorithms

### Personalized PageRank
```
P(t+1) = α · A · P(t) + (1 − α) · Q
α = 0.85, iterations = 20, tolerance = 1e-6
```

### Node Pruning
```
score = 0.4 * keyword_overlap + 0.3 * rank_score + 0.3 * cosine_similarity
```

### Document Graph Edge Weight
```
W_ij = 0.4 * keyword_overlap + 0.3 * hyperlink + 0.3 * embedding_similarity
Connect only if W_ij > 0.5
```

---

## Setup

```bash
# Clone and enter directory
cd deep_search_engine

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Environment Variables
Create a `.env` file or set:
```
BRAVE_API_KEY=your_brave_search_api_key
```

---

## Running the Server

```bash
cd deep_search_engine
uvicorn app.main:app --reload --port 8000
```

---

## API Usage

### POST `/api/v1/deep-search`

**Request:**
```json
{
    "query": "deploy nodejs aws securely best practices",
    "depth": 2,
    "pruning": true
}
```

**Response:**
```json
{
    "answer": "Based on analysis of 45 documents...",
    "sources": [
        {"url": "https://...", "title": "AWS Node.js Deployment Guide"},
        {"url": "https://...", "title": "EC2 Security Best Practices"}
    ],
    "documents_explored": 45,
    "latency": 8.34
}
```

### GET `/health`
Returns `{"status": "ok"}`.

---

## Evaluation

The system is benchmarked against **BM25** using:
- **Precision@10** — Relevance of top results
- **Recall@10** — Coverage of relevant documents
- **NDCG@10** — Rank quality
- **Topical Coverage** — Breadth of retrieved knowledge

---

## Technology Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Async I/O | asyncio + httpx |
| Scraping | scrapling |
| Graph | NetworkX |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS (CPU) |
| NLP | spaCy + YAKE |
| Dedup | datasketch (MinHash) |
| LLM | Ollama (LLaMA) / sumy (fallback) |

---

## Project Structure

```
deep_search_engine/
├── app/            # FastAPI entry, config
├── api/            # Route definitions
├── query_processing/  # Parser, extractor, planner, graph
├── search/         # BFS engine, client, pruning
├── scraping/       # scrapling fetcher, cleaner
├── graph/          # Document graph, PageRank
├── storage/        # FAISS store, doc store, cache
├── llm/            # Summarizer (Ollama/TextRank)
├── evaluation/     # Metrics, BM25 benchmark
├── utils/          # Async executor, text helpers
├── requirements.txt
└── README.md
```
