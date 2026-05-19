<h1 align="center">🕸️ DeepSearch — Graph-Based Research Engine</h1>

<p align="center">
<strong>A production-grade information retrieval system that combines recursive BFS web exploration, semantic document graphs, Personalized PageRank, and LLM-powered answer generation.</strong>
</p>

<p align="center">
<a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
<a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-0.103+-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://networkx.org"><img src="https://img.shields.io/badge/NetworkX-Graph%20Engine-orange" alt="NetworkX"></a>
<a href="https://github.com/facebookresearch/faiss"><img src="https://img.shields.io/badge/FAISS-Vector%20Search-blue" alt="FAISS"></a>
</p>

<p align="center">
<em>Unlike traditional search engines that perform a single SERP query, DeepSearch recursively explores the web using BFS, builds a weighted document relationship graph, ranks results with Topic-Sensitive PageRank, and generates grounded answers with citations.</em>
</p>


---

## 🎯 What Problem Does This Solve?

Traditional search workflows follow a linear path: `Query → SERP → Top 10 Links`. This approach:

- ❌ Misses relevant documents beyond the first page of results
- ❌ Cannot discover relationships between documents
- ❌ Treats every result equally regardless of domain authority
- ❌ Provides no synthesized answer — just a list of links

**DeepSearch** solves this with a **7-stage retrieval pipeline** that explores, filters, connects, ranks, and synthesizes:

```
Query → Understanding → BFS Expansion → Scrape + Filter → Graph Build → PageRank → Re-rank → Answer
```

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          USER QUERY                                      │
│                     "deploy nodejs aws securely"                         │
└─────────────────────────────┬────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ① QUERY PROCESSING                   spaCy · YAKE · NER               │
│  ┌──────────────┐ ┌────────────────┐ ┌────────────────┐                 │
│  │ query_parser │ │ keyword_extract│ │ query_planner  │                 │
│  │ Tokenize     │ │ Noun phrases   │ │ BFS seed nodes │                 │
│  │ Normalize    │ │ Named entities │ │ max 6 seeds    │                 │
│  └──────────────┘ └────────────────┘ └────────────────┘                 │
└─────────────────────────────┬───────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ② BFS RECURSIVE SEARCH              Concurrency=10 · 3 retries        │
│  ┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐            │
│  │ recursive_search │ │ search_client│ │ node_pruning     │            │
│  │ depth=3, nodes=12│ │ DuckDuckGo   │ │ 6-component score│            │
│  │ per level        │ │ + Brave API  │ │ threshold ≥ 0.55 │            │
│  └──────────────────┘ └──────────────┘ └──────────────────┘            │
└─────────────────────────────┬───────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ③ SCRAPING & QUALITY FILTERING       scrapling · datasketch            │
│  ┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐            │
│  │ scraper          │ │ content_clean│ │ MinHash Dedup    │            │
│  │ Anti-bot bypass  │ │ Min 300 words│ │ threshold > 0.9  │            │
│  │ UA rotation      │ │ Quality gate │ │ 128 permutations │            │
│  └──────────────────┘ └──────────────┘ └──────────────────┘            │
└─────────────────────────────┬───────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ④ GRAPH CONSTRUCTION & RANKING       NetworkX · FAISS · numpy          │
│  ┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐            │
│  │ document_graph   │ │ graph_ranker │ │ vector_store     │            │
│  │ Edge: W > 0.25   │ │ PageRank α=  │ │ all-MiniLM-L6-v2│            │
│  │ Top-K guarantee  │ │ 0.85, 30 iter│ │ FAISS index, 7d  │            │
│  └──────────────────┘ └──────────────┘ └──────────────────┘            │
└─────────────────────────────┬───────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ⑤ RE-RANKING & ANSWER GENERATION     Groq · Ollama · sumy              │
│  ┌──────────────────┐ ┌──────────────────────────────────┐             │
│  │ reranker         │ │ summarizer                       │             │
│  │ 0.60 cosine +    │ │ Groq LLaMA 3.3 70B (primary)    │             │
│  │ 0.40 PageRank    │ │ Ollama LLaMA 3 (local fallback) │             │
│  │ Top-20 window    │ │ TextRank extractive (last resort)│             │
│  └──────────────────┘ └──────────────────────────────────┘             │
└─────────────────────────────┬───────────────────────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  STRUCTURED      │
                    │  ANSWER +        │
                    │  CITATIONS       │
                    └──────────────────┘
```

---

## 🔬 Key Algorithms & Formulas

### Personalized PageRank (Topic-Sensitive)

```
P(t+1) = α · A · P(t) + (1 − α) · Q

Where:
  α = 0.85 (damping factor)
  A = row-normalized adjacency matrix
  Q = personalization vector (biased toward query-relevant documents)
  Iterations = 30, Tolerance = 1e-6

Q-vector blending:
  Q_i = 0.60 × semantic_sim(doc_i, query)
      + 0.25 × domain_authority(doc_i)
      + 0.15 × serp_position(doc_i)
```

### Document Graph Edge Weight

```
W_ij = 0.70 × cosine(embedding_i, embedding_j)
     + 0.20 × jaccard(keywords_i, keywords_j)
     + 0.10 × hyperlink_signal(i, j)

Phase 1: Connect if W_ij > 0.25
Phase 2: Top-K guarantee (min 5 edges per node regardless of threshold)
```

### Node Pruning (6-Component Scoring)

```
score = 0.25 × semantic_similarity
      + 0.20 × keyword_overlap
      + 0.15 × serp_rank_score
      + 0.15 × domain_authority
      + 0.15 × content_quality
      + 0.10 × freshness

Threshold: score ≥ 0.55 → keep, else prune
```

### Content Quality Gate

```
Q_quality = 0.30 × S_length + 0.20 × S_unique + 0.20 × S_link
          + 0.20 × S_readability + 0.10 × S_title

Hard filters (instant drop):
  • word_count < 300
  • unique_token_ratio < 0.30
  • link_density > 0.05
```

### 2-Stage Re-Ranking

```
final_score = 0.60 × passage_cosine_sim(query, doc_passage)
            + 0.40 × pagerank_score_normalized

  doc_passage = first 512 tokens of content
  Re-rank window: top 20 PageRank results
```

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI + Uvicorn | Async REST API with OpenAPI docs |
| **Async I/O** | asyncio + httpx | Non-blocking HTTP with retry + backoff |
| **Web Scraping** | scrapling | Anti-bot bypass, UA rotation, DOM extraction |
| **Graph Engine** | NetworkX | Document relationship graph + PageRank |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | 384-dim semantic embeddings |
| **Vector Search** | FAISS (CPU) | Approximate nearest neighbor indexing |
| **NLP Pipeline** | spaCy + YAKE | Tokenization, NER, keyword extraction |
| **Deduplication** | datasketch (MinHash LSH) | Near-duplicate detection (128 perms) |
| **LLM (Primary)** | Groq API — LLaMA 3.3 70B | Fast cloud inference for answer generation |
| **LLM (Fallback)** | Ollama — LLaMA 3 | Local offline inference |
| **LLM (Last Resort)** | sumy TextRank | Extractive summarization, zero-dependency |
| **Frontend** | Streamlit | Interactive live demo with real-time pipeline visualization |

---

## 📊 Evaluation & Benchmarking

The engine is benchmarked against a **BM25 baseline** using standard IR metrics:

| Metric | Description |
|---|---|
| **Precision@10** | Fraction of top-10 results that are relevant |
| **Recall@10** | Coverage of all relevant documents in top-10 |
| **NDCG@10** | Normalized Discounted Cumulative Gain (rank quality) |
| **Topical Coverage** | Breadth of query keywords covered across retrieved documents |

---

## 📁 Project Structure

```
├── app/                    # FastAPI entry point + global config
│   ├── main.py             # App factory, CORS, health check
│   └── config.py           # All tunable parameters (pydantic-settings)
├── api/
│   └── routes.py           # POST /deep-search endpoint
├── query_processing/       # Stage 1: Query understanding
│   ├── query_parser.py     # Tokenizer, normalizer, stopwords
│   ├── keyword_extractor.py# spaCy NER + YAKE keyword extraction
│   ├── query_planner.py    # BFS seed node generation
│   └── query_understanding.py # Intent detection + query reformulation
├── search/                 # Stage 2: BFS recursive retrieval
│   ├── recursive_search.py # BFS engine (depth, branching factor)
│   ├── search_client.py    # DuckDuckGo + Brave API abstraction
│   ├── node_pruning.py     # 6-component relevance scoring
│   └── reranker.py         # 2-stage passage-level re-ranking
├── scraping/               # Stage 3: Content extraction
│   ├── scraper.py          # scrapling-based web scraper
│   └── content_cleaner.py  # Quality filtering + text cleanup
├── graph/                  # Stage 4: Document graph
│   ├── document_graph.py   # Weighted graph construction
│   └── graph_ranker.py     # Personalized PageRank implementation
├── storage/                # Persistence layer
│   ├── vector_store.py     # FAISS index + embedding cache
│   └── document_store.py   # In-memory document store
├── llm/                    # Stage 5: Answer generation
│   └── summarizer.py       # Groq → Ollama → TextRank fallback chain
├── evaluation/             # IR metrics + benchmarking
│   ├── metrics.py          # P@K, R@K, NDCG@K, topical coverage
│   └── benchmark_runner.py # BM25 baseline comparison
├── utils/
│   ├── async_executor.py   # Async concurrency pool (limit=10)
│   └── text_utils.py       # Text processing helpers
├── models/
│   ├── query.py            # Query data model
│   └── document.py         # Document, RankedDocument, RerankResult
├── live_demo.py            # Streamlit interactive demo
├── requirements.txt        # Python dependencies
├── HOWTODEPLOY.md          # Full deployment guide
└── README.md               # This file
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/RKG765/deep-based-search-engine.git
cd deep-based-search-engine/deep_search_engine

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy language model
python -m spacy download en_core_web_sm

# 5. Set environment variables
#    Create a .env file or export directly:
#    GROQ_API_KEY=your_groq_api_key
#    BRAVE_API_KEY=your_brave_search_key  (optional)
```

### Run the API Server

```bash
uvicorn app.main:app --reload --port 8000
```

### Run the Streamlit Demo

```bash
streamlit run live_demo.py
```

---

## 📡 API Reference

### `POST /api/v1/deep-search`

Execute a full deep search pipeline.

**Request:**
```json
{
    "query": "deploy nodejs aws securely best practices",
    "depth": 2,
    "pruning": true
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | string | *required* | Natural language search query |
| `depth` | int | `2` | BFS exploration depth (0–3). Higher = more thorough, slower |
| `pruning` | bool | `true` | Enable 6-component node pruning to filter noise |

**Response:**
```json
{
    "answer": "Based on analysis of 45 documents across 12 sources...",
    "sources": [
        {"url": "https://docs.aws.amazon.com/...", "title": "AWS Node.js Deployment Guide"},
        {"url": "https://owasp.org/...", "title": "OWASP Security Best Practices"}
    ],
    "documents_explored": 45,
    "latency": 8.34
}
```

### `GET /health`

Liveness probe.

```json
{"status": "ok", "app_name": "Deep Recursive Research Search Engine"}
```

---

## ⚙️ Configuration

All parameters are configurable via environment variables or `app/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `MAX_DEPTH` | 3 | BFS recursion depth |
| `MAX_NODES_PER_LEVEL` | 12 | Nodes explored per BFS level |
| `MAX_TOTAL_DOCS` | 200 | Absolute document cap |
| `PRUNING_THRESHOLD` | 0.55 | Minimum score to survive pruning |
| `EDGE_THRESHOLD` | 0.25 | Minimum edge weight for graph connection |
| `TOP_K_NEIGHBORS` | 5 | Guaranteed minimum edges per node |
| `PR_ALPHA` | 0.85 | PageRank damping factor |
| `PR_ITERATIONS` | 30 | PageRank convergence iterations |
| `RERANK_TOP_K` | 20 | Re-ranking window size |
| `DEDUP_THRESHOLD` | 0.9 | MinHash near-duplicate threshold |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence embedding model |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Recommended | Groq API key for LLaMA 3.3 70B inference |
| `BRAVE_API_KEY` | Optional | Brave Search API key (DuckDuckGo used as fallback) |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
<strong>Built with ❤️ as an Information Retrieval End-Semester Project</strong><br>
<em>BML Munjal University · Semester VI · 2026</em>
</p>
