"""
live_demo.py — Premium Streamlit UI for Graph-Based Research Assistant.
Run: streamlit run live_demo.py
"""
import sys, os, asyncio, time
import networkx as nx
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd


def render_mermaid(chart: str, height: int = 400):
    """Render a Mermaid diagram inside Streamlit via an inline iframe."""
    escaped = chart.replace('"', '&quot;').replace("'", "&#39;")
    iframe_html = f"""<iframe srcdoc='
        <!DOCTYPE html><html><head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({{startOnLoad:true,theme:"dark",themeVariables:{{
            primaryColor:"#7c3aed",primaryTextColor:"#e2e8f0",primaryBorderColor:"#a78bfa",
            lineColor:"#4b5563",secondaryColor:"#1e1b4b",tertiaryColor:"#0f0f23",
            fontSize:"13px",fontFamily:"Inter,sans-serif",
            nodeBorder:"#a78bfa",mainBkg:"#1a1a2e",clusterBkg:"#0f0f23",clusterBorder:"#374151"
        }}}});</script>
        </head><body style="background:transparent;display:flex;justify-content:center;margin:0;padding:16px 0;">
        <div class="mermaid">{escaped}</div>
        </body></html>'
        style="width:100%;height:{height}px;border:none;background:transparent;"
        scrolling="no"></iframe>"""
    st.markdown(iframe_html, unsafe_allow_html=True)

st.set_page_config(
    page_title="DeepSearch — Graph IR Engine",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;0,14..32,800;1,14..32,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ─────────────────────────────────────────────── */
*, *::before, *::after { font-family: 'Inter', sans-serif; box-sizing: border-box; }
.stApp {
    background: #080812;
    min-height: 100vh;
}

/* Overlay gradient orbs for depth */
.stApp::before {
    content: '';
    position: fixed;
    top: -200px; left: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(124,58,237,0.12) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
    animation: orb1 15s ease-in-out infinite;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -150px; right: -150px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
    animation: orb2 20s ease-in-out infinite;
}
@keyframes orb1 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(60px,40px); } }
@keyframes orb2 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(-40px,-60px); } }

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(10, 10, 20, 0.98) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 10px; }

/* ── Hero ────────────────────────────────────────────── */
.hero-wrap { padding: 2.5rem 0 0; position: relative; }
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.72rem; font-weight: 600;
    color: #a78bfa; letter-spacing: 1.2px;
    text-transform: uppercase; margin-bottom: 1.1rem;
}
.hero-title {
    font-size: 3.2rem; font-weight: 800; line-height: 1.05;
    letter-spacing: -1.5px;
    background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 40%, #00d4ff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: titleShimmer 6s linear infinite;
    margin-bottom: 0.8rem;
}
@keyframes titleShimmer { to { background-position: 200% center; } }
.hero-sub {
    color: #4b5563; font-size: 1rem; font-weight: 400;
    line-height: 1.6; max-width: 560px;
}

/* ── Pipeline flow bar ───────────────────────────────── */
.pipeline-bar {
    display: flex; align-items: center; flex-wrap: wrap;
    gap: 0; margin: 2rem 0 2.5rem; padding: 0;
}
.pipe-step {
    display: flex; align-items: center; gap: 0;
}
.pipe-node {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px; padding: 6px 14px;
    font-size: 0.74rem; font-weight: 500; color: #6b7280;
    white-space: nowrap; transition: all 0.3s ease;
    cursor: default;
}
.pipe-node:hover {
    background: rgba(124,58,237,0.1);
    border-color: rgba(124,58,237,0.35);
    color: #c4b5fd;
}
.pipe-arrow {
    color: rgba(255,255,255,0.15);
    font-size: 0.8rem; padding: 0 4px;
}

/* ── Search input — use data-baseweb selector (Streamlit 1.30+) ──── */
[data-baseweb="input"] > div {
    background: rgba(10, 10, 25, 0.95) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    transition: all 0.3s ease !important;
}
[data-baseweb="input"]:focus-within > div {
    border-color: rgba(124, 58, 237, 0.55) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.1) !important;
}
[data-baseweb="input"] input {
    background: transparent !important;
    color: #f9fafb !important;
    font-size: 1rem !important;
    caret-color: #7c3aed !important;
}
[data-baseweb="input"] input::placeholder {
    color: #4b5563 !important;
}
/* Legacy fallback */
.stTextInput > div > div > input {
    background: rgba(10, 10, 25, 0.95) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    color: #f9fafb !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.35) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 25px rgba(124,58,237,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── How-it-works cards (landing) ────────────────────── */
.how-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin-top: 1rem;
}
.how-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 1.3rem 1.1rem;
    position: relative; overflow: hidden;
    transition: all 0.35s ease;
}
.how-card:hover {
    background: rgba(255,255,255,0.045);
    border-color: rgba(124,58,237,0.25);
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}
.how-num {
    font-size: 0.65rem; font-weight: 700;
    color: rgba(124,58,237,0.6);
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.how-icon { font-size: 1.6rem; margin-bottom: 0.6rem; }
.how-title { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.3rem; }
.how-desc  { font-size: 0.73rem; color: #6b7280; line-height: 1.5; }
.how-accent {
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(124,58,237,0.5), transparent);
    opacity: 0; transition: opacity 0.3s;
}
.how-card:hover .how-accent { opacity: 1; }

/* ── Result card ─────────────────────────────────────── */
.result-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 1.1rem 1.3rem;
    margin: 8px 0; position: relative; overflow: hidden;
    transition: all 0.3s ease;
}
.result-card:hover {
    border-color: rgba(124,58,237,0.3);
    background: rgba(124,58,237,0.04);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.result-rank {
    font-size: 0.65rem; font-weight: 700;
    color: #7c3aed; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 4px;
}
.result-title { font-size: 0.95rem; font-weight: 600; color: #f1f5f9; }
.result-url   { font-size: 0.73rem; color: #4b5563; font-family: 'JetBrains Mono', monospace; margin-top: 3px; }
.result-scores { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.score-pill {
    font-size: 0.68rem; font-weight: 600;
    padding: 2px 8px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
}
.sp-blue   { background: rgba(0,212,255,0.1); color: #00d4ff; border: 1px solid rgba(0,212,255,0.2); }
.sp-purple { background: rgba(124,58,237,0.1); color: #a78bfa; border: 1px solid rgba(124,58,237,0.2); }
.sp-green  { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.2); }
.sp-amber  { background: rgba(245,158,11,0.1); color: #f59e0b; border: 1px solid rgba(245,158,11,0.2); }
.sp-moved  { background: rgba(239,68,68,0.1); color: #f87171; border: 1px solid rgba(239,68,68,0.2); }

/* Domain tier badge */
.dom-tier-1 { color: #10b981; } /* .gov/.edu */
.dom-tier-2 { color: #3b82f6; } /* github/docs */
.dom-tier-3 { color: #a78bfa; } /* .org */
.dom-tier-4 { color: #f59e0b; } /* medium */
.dom-tier-5 { color: #6b7280; } /* unknown */

/* Score bar */
.score-bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 4px; height: 4px; margin: 6px 0;
}
.score-bar-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #7c3aed, #00d4ff);
    transition: width 0.6s ease;
}

/* ── Graph stat box ──────────────────────────────────── */
.graph-stat {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 1rem;
    text-align: center; transition: all 0.3s ease;
}
.graph-stat:hover { border-color: rgba(0,212,255,0.2); }
.gs-val { font-size: 2rem; font-weight: 800; color: #00d4ff; font-family: 'JetBrains Mono', monospace; }
.gs-label { font-size: 0.72rem; color: #4b5563; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* ── Sidebar elements ────────────────────────────────── */
.sb-section-title {
    font-size: 0.65rem; font-weight: 700; color: #374151;
    letter-spacing: 2px; text-transform: uppercase;
    padding: 0 0 8px; margin-bottom: 4px;
}
.sb-metric {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 10px; border-radius: 8px;
    margin-bottom: 4px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.04);
    transition: border-color 0.2s;
}
.sb-metric:hover { border-color: rgba(124,58,237,0.2); }
.sb-key { font-size: 0.73rem; color: #6b7280; }
.sb-val { font-size: 0.73rem; font-weight: 600; color: #a78bfa; font-family: 'JetBrains Mono', monospace; }
.sb-sample-btn button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #9ca3af !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    padding: 0.4rem 0.8rem !important;
    font-weight: 400 !important;
    box-shadow: none !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
}
.sb-sample-btn button:hover {
    border-color: rgba(124,58,237,0.35) !important;
    color: #c4b5fd !important;
    background: rgba(124,58,237,0.06) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Divider ─────────────────────────────────────────── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
    margin: 1.5rem 0;
}

/* ── Stage header ────────────────────────────────────── */
.stage-hdr {
    display: flex; align-items: center; gap: 10px;
    margin: 1.5rem 0 0.8rem;
}
.stage-num {
    width: 26px; height: 26px; border-radius: 50%;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.68rem; font-weight: 700; color: white;
    flex-shrink: 0;
}
.stage-title { font-size: 0.9rem; font-weight: 600; color: #d1d5db; }
.stage-desc  { font-size: 0.75rem; color: #4b5563; margin-top: 1px; }

/* ── Token / keyword pills ───────────────────────────── */
.token-pill {
    display: inline-block;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px; padding: 3px 9px;
    font-size: 0.75rem; color: #9ca3af;
    margin: 2px; font-family: 'JetBrains Mono', monospace;
}
.kw-pill {
    display: inline-block;
    background: rgba(124,58,237,0.08);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 6px; padding: 3px 9px;
    font-size: 0.75rem; color: #a78bfa;
    margin: 2px;
}
.seed-pill {
    display: inline-block;
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 6px; padding: 3px 9px;
    font-size: 0.75rem; color: #67e8f9;
    margin: 2px;
}

/* ── Metric delta override ───────────────────────────── */
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_modules():
    from app.config import settings
    from models.query import Query
    from query_processing.query_parser import tokenize, normalize_query
    from query_processing.keyword_extractor import extract_all
    from query_processing.query_planner import generate_seed_nodes
    from search.recursive_search import RecursiveSearchEngine
    from search.reranker import rerank
    from storage.vector_store import VectorStore
    from storage.document_store import DocumentStore
    from graph.document_graph import DocumentGraph
    from graph.graph_ranker import GraphRanker
    from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, topical_coverage
    from evaluation.benchmark_runner import BM25
    from llm.summarizer import generate_answer
    return {
        "settings": settings, "Query": Query,
        "tokenize": tokenize, "normalize_query": normalize_query,
        "extract_all": extract_all, "generate_seed_nodes": generate_seed_nodes,
        "RecursiveSearchEngine": RecursiveSearchEngine, "rerank": rerank,
        "VectorStore": VectorStore, "DocumentStore": DocumentStore,
        "DocumentGraph": DocumentGraph, "GraphRanker": GraphRanker,
        "precision_at_k": precision_at_k, "recall_at_k": recall_at_k,
        "ndcg_at_k": ndcg_at_k, "topical_coverage": topical_coverage,
        "BM25": BM25, "generate_answer": generate_answer,
    }

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def domain_badge(url: str) -> str:
    """Return a colored badge span based on domain authority tier."""
    from urllib.parse import urlparse
    import re
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        netloc = ""
    if re.search(r"(\.gov|\.edu|arxiv|ieee|acm|springer)", netloc):
        return '<span class="score-pill sp-green">★ High Auth</span>'
    if re.search(r"(github|stackoverflow|wikipedia|readthedocs|docs\.|developer\.)", netloc):
        return '<span class="score-pill sp-blue">⬡ Trusted</span>'
    if re.search(r"(\.org|openai|anthropic|research\.)", netloc):
        return '<span class="score-pill sp-purple">◈ .org</span>'
    if re.search(r"(medium|dev\.to|hashnode|substack)", netloc):
        return '<span class="score-pill sp-amber">◷ Blog</span>'
    return '<span class="score-pill" style="background:rgba(255,255,255,0.05);color:#4b5563;border:1px solid rgba(255,255,255,0.08);">◌ Unknown</span>'

m = load_modules()

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR — all native widgets (no custom HTML — prevents bleed on collapse)
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🕸️ DeepSearch")
    st.caption("Graph IR Engine · IR End-Sem Project")
    st.divider()

    st.markdown("**⚙️ Pipeline Controls**")
    depth   = st.slider("BFS Depth", 0, 3, 1,
        help="Depth 1 = fast (~30s) · Depth 3 = thorough (~3min)")
    pruning = st.toggle("Node Pruning", value=True,
        help="Removes off-topic BFS nodes using 6-component scoring")

    st.divider()
    st.markdown("**📊 Live Config**")

    cfg_rows = [
        ("Max Nodes/Level",   m['settings'].MAX_NODES_PER_LEVEL),
        ("Results/Search",    m['settings'].MAX_RESULTS_PER_SEARCH),
        ("Pruning Threshold", m['settings'].PRUNING_THRESHOLD),
        ("Edge Threshold",    m['settings'].EDGE_THRESHOLD),
        ("Top-K Min Edges",   m['settings'].TOP_K_NEIGHBORS),
        ("PageRank α",        m['settings'].PR_ALPHA),
        ("PR Iterations",     m['settings'].PR_ITERATIONS),
        ("Re-rank Top-K",     m['settings'].RERANK_TOP_K),
    ]
    # Use st.code for the config block — renders cleanly, hides on collapse
    config_text = "\n".join(f"{k:<22} = {v}" for k, v in cfg_rows)
    st.code(config_text, language="yaml")

    st.divider()
    st.markdown("**🎯 Try a Query**")
    samples = [
        "how does pagerank algorithm work",
        "information retrieval BM25 vs TF-IDF",
        "deploy nodejs on aws securely",
        "transformer self attention mechanism",
        "python ssl certificate verification",
    ]
    for sq in samples:
        if st.button(sq, key=f"sample_{sq}", use_container_width=True):
            st.session_state["query_input"] = sq
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">🎓 &nbsp;IR End-Semester Project · BML Sem VI</div>
    <div class="hero-title">Graph-Based<br>Research Assistant</div>
    <p class="hero-sub">
        Combines BFS document discovery, semantic embeddings, domain authority scoring,
        Personalized PageRank, and passage-level re-ranking to surface the best answers.
    </p>
    <div class="pipeline-bar">
        <div class="pipe-step"><div class="pipe-node">① Query Understanding</div><span class="pipe-arrow">→</span></div>
        <div class="pipe-step"><div class="pipe-node">② BFS Expansion</div><span class="pipe-arrow">→</span></div>
        <div class="pipe-step"><div class="pipe-node">③ Scrape + Filter</div><span class="pipe-arrow">→</span></div>
        <div class="pipe-step"><div class="pipe-node">④ Graph Construction</div><span class="pipe-arrow">→</span></div>
        <div class="pipe-step"><div class="pipe-node">⑤ Personalized PageRank</div><span class="pipe-arrow">→</span></div>
        <div class="pipe-step"><div class="pipe-node">⑥ Re-ranking</div><span class="pipe-arrow">→</span></div>
        <div class="pipe-node">⑦ Answer Generation</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# SEARCH INPUT
# ═══════════════════════════════════════════════════════════════════════
col_in, col_btn = st.columns([5, 1])
with col_in:
    query_text = st.text_input(
        "query",
        value=st.session_state.get("query_input", ""),
        placeholder="  🔍  Ask anything — e.g. 'how does PageRank work in search engines'",
        label_visibility="collapsed",
        key="main_query_input",
    )
with col_btn:
    search_clicked = st.button("Search →", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# LANDING STATE (no search yet)
# ═══════════════════════════════════════════════════════════════════════
if not search_clicked and not query_text.strip():
    st.markdown("""
    <div style="margin-top: 0.5rem; margin-bottom: 1rem;">
        <div style="font-size:0.75rem;font-weight:600;color:#374151;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">
            How it works
        </div>
        <div class="how-grid">
            <div class="how-card">
                <div class="how-num">Step 01</div>
                <div class="how-icon">🧠</div>
                <div class="how-title">Query Understanding</div>
                <div class="how-desc">Detects intent (factual/exploratory), extracts entities, and reformulates the query for better SERP results.</div>
                <div class="how-accent" style="--accent:#7c3aed;"></div>
            </div>
            <div class="how-card">
                <div class="how-num">Step 02</div>
                <div class="how-icon">🌐</div>
                <div class="how-title">BFS Document Discovery</div>
                <div class="how-desc">Breadth-first search across expanded query nodes. Fetches up to 200 pages using Brave API + DuckDuckGo fallback.</div>
                <div class="how-accent" style="--accent:#2563eb;"></div>
            </div>
            <div class="how-card">
                <div class="how-num">Step 03</div>
                <div class="how-icon">⚗️</div>
                <div class="how-title">Quality Filtering</div>
                <div class="how-desc">Q = 0.30·Length + 0.20·Uniqueness + 0.20·LinkDensity + 0.20·Readability + 0.10·TitleRelevance. Hard-drops thin pages, SEO spam, and nav pages.</div>
                <div class="how-accent" style="--accent:#059669;"></div>
            </div>
            <div class="how-card">
                <div class="how-num">Step 04</div>
                <div class="how-icon">🕸️</div>
                <div class="how-title">Graph Construction</div>
                <div class="how-desc">Builds a weighted document graph using embeddings (0.70), keywords (0.20), and link signals (0.10). Top-K guarantee ensures no isolated nodes.</div>
                <div class="how-accent" style="--accent:#7c3aed;"></div>
            </div>
            <div class="how-card">
                <div class="how-num">Step 05</div>
                <div class="how-icon">📊</div>
                <div class="how-title">Personalized PageRank</div>
                <div class="how-desc">Topic-sensitive PageRank with a Q vector blending semantic similarity (0.60), domain authority (0.25), and SERP position (0.15).</div>
                <div class="how-accent" style="--accent:#2563eb;"></div>
            </div>
            <div class="how-card">
                <div class="how-num">Step 06</div>
                <div class="how-icon">🎯</div>
                <div class="how-title">2-Stage Re-ranking</div>
                <div class="how-desc">Re-scores top-20 PageRank results using passage-level cosine similarity (0.60) blended with PageRank score (0.40).</div>
                <div class="how-accent" style="--accent:#059669;"></div>
            </div>
            <div class="how-card">
                <div class="how-num">Step 07</div>
                <div class="how-icon">✍️</div>
                <div class="how-title">Grounded Answer</div>
                <div class="how-desc">Top 5–8 documents are passed to a local LLM (Ollama / llama3) to generate a cited, grounded answer.</div>
                <div class="how-accent" style="--accent:#d97706;"></div>
            </div>
            <div class="how-card">
                <div class="how-num">Why</div>
                <div class="how-icon">🏛️</div>
                <div class="how-title">vs. BM25 / TF-IDF</div>
                <div class="how-desc">Graph ranking propagates authority through links. Semantic embeddings handle synonymy. Domain scoring boosts trusted sources.</div>
                <div class="how-accent" style="--accent:#7c3aed;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════════
if search_clicked and query_text.strip():
    pipeline_start = time.time()

    # ── Stage 1: Query Processing ─────────────────────────────────────
    st.markdown("""
    <div class="stage-hdr">
        <div class="stage-num">1</div>
        <div><div class="stage-title">Query Processing</div>
        <div class="stage-desc">Intent detection · Keyword extraction · Seed node generation</div></div>
    </div>""", unsafe_allow_html=True)

    with st.status("Processing query...", expanded=True) as s1:
        t0 = time.time()
        from query_processing.query_understanding import reformulate_query
        reformulated, intent = reformulate_query(query_text)
        normalized  = m["normalize_query"](query_text)
        tokens      = m["tokenize"](query_text)
        keywords    = m["extract_all"](query_text)
        seeds       = m["generate_seed_nodes"](reformulated if intent == "factual" else query_text)

        intent_map = {
            "factual": ("🎯", "#a78bfa"),
            "howto":   ("🔧", "#60a5fa"),
            "comparison": ("⚖️", "#34d399"),
            "exploratory": ("🔍", "#f59e0b"),
            "investigative": ("🕵️", "#f87171"),
        }
        intent_icon, intent_color = intent_map.get(intent, ("❓", "#9ca3af"))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div style="font-size:0.75rem;color:#4b5563;margin-bottom:6px;">DETECTED INTENT</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:1.4rem;font-weight:700;color:{intent_color};">{intent_icon} {intent.upper()}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.78rem;color:#6b7280;margin-top:4px;">Search: <em>{reformulated}</em></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin-bottom:6px;">TOKENS</div>', unsafe_allow_html=True)
            st.markdown(" ".join(f'<span class="token-pill">{t}</span>' for t in tokens[:12]), unsafe_allow_html=True)
        with c3:
            st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin-bottom:6px;">BFS SEED NODES</div>', unsafe_allow_html=True)
            st.markdown(" ".join(f'<span class="seed-pill">{s}</span>' for s in seeds[:6]), unsafe_allow_html=True)

        if keywords:
            st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin:8px 0 4px;">KEYWORDS (spaCy + YAKE)</div>', unsafe_allow_html=True)
            st.markdown(" ".join(f'<span class="kw-pill">{k}</span>' for k in keywords[:15]), unsafe_allow_html=True)
        s1.update(label=f"✅ Query processed in {time.time()-t0:.2f}s", state="complete", expanded=False)

    # ── Stage 2: BFS Search ──────────────────────────────────────────
    st.markdown("""
    <div class="stage-hdr">
        <div class="stage-num">2</div>
        <div><div class="stage-title">BFS Document Discovery</div>
        <div class="stage-desc">Recursive web search · Anti-bot scraping · Quality filtering · Deduplication</div></div>
    </div>""", unsafe_allow_html=True)

    with st.status("Running BFS search pipeline...", expanded=True) as s2:
        t0 = time.time()
        query_obj = m["Query"](raw=query_text, depth=depth, pruning=pruning)
        engine = m["RecursiveSearchEngine"]()
        documents = run_async(engine.run(query_obj))

        doc_count = len(documents)
        url_count = len(engine.seen_urls)
        avg_words = int(np.mean([d.word_count for d in documents])) if documents else 0
        avg_qual  = round(float(np.mean([d.quality_score for d in documents])), 3) if documents else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📄 Documents", doc_count)
        c2.metric("🌐 URLs Explored", url_count)
        c3.metric("📝 Avg Words", avg_words)
        c4.metric("⭐ Avg Quality", avg_qual)

        if documents:
            rows = [{"#": i+1, "Title": d.title[:55] + ("…" if len(d.title) > 55 else ""),
                     "Words": d.word_count, "Quality": round(d.quality_score, 3),
                     "Domain Auth": round(d.domain_score, 2), "URL": d.url[:60]}
                    for i, d in enumerate(documents[:12])]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("📐 Quality Gate Formula", expanded=False):
            render_mermaid("""
            graph LR
                subgraph Signals
                    A["S_length<br/>0.30 × min(words/2000, 1)"]
                    B["S_unique<br/>0.20 × unique/total tokens"]
                    C["S_link<br/>0.20 × 1 − link_density/τ"]
                    D["S_read<br/>0.20 × readability proxy"]
                    E["S_title<br/>0.10 × title∩headings overlap"]
                end
                A --> F["Q_quality ∈ 0,1"]
                B --> F
                C --> F
                D --> F
                E --> F
                F --> G{"Hard Filters"}
                G -->|words ≥ 300| H["✅ Accept"]
                G -->|unique ≥ 0.30| H
                G -->|link_den ≤ 0.05| H
                G -->|FAIL| I["❌ Drop"]
            """, height=350)
            st.markdown(r"""
$$
Q_{\text{quality}} = 0.30 \cdot S_{\text{length}} + 0.20 \cdot S_{\text{unique}} + 0.20 \cdot S_{\text{link}} + 0.20 \cdot S_{\text{read}} + 0.10 \cdot S_{\text{title}}
$$
""")
            st.markdown("""
<div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
     border-radius:12px;padding:1.2rem;margin-top:0.5rem;">
  <div style="font-size:0.7rem;color:#7c3aed;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;font-weight:700;">Signal Components</div>
  <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:8px 6px;color:#a78bfa;font-weight:600;width:80px;">0.30</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">S<sub>length</sub></td>
      <td style="padding:8px 6px;color:#6b7280;">min(word_count / 2000, 1.0) — longer articles score higher, capped at 2000 words</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:8px 6px;color:#a78bfa;font-weight:600;">0.20</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">S<sub>unique</sub></td>
      <td style="padding:8px 6px;color:#6b7280;">|unique tokens| / |total tokens| — low ratio → SEO spam / keyword stuffing</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:8px 6px;color:#a78bfa;font-weight:600;">0.20</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">S<sub>link</sub></td>
      <td style="padding:8px 6px;color:#6b7280;">max(0, 1 − link_density / τ) — high link-to-word ratio → navigation / directory page</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:8px 6px;color:#a78bfa;font-weight:600;">0.20</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">S<sub>read</sub></td>
      <td style="padding:8px 6px;color:#6b7280;">Readability proxy — peak at 17 words/sentence, triangle window [10, 25]</td>
    </tr>
    <tr>
      <td style="padding:8px 6px;color:#a78bfa;font-weight:600;">0.10</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">S<sub>title</sub></td>
      <td style="padding:8px 6px;color:#6b7280;">|title ∩ headings| / |title| — word overlap measures page coherence</td>
    </tr>
  </table>
</div>

<div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.15);
     border-radius:12px;padding:1.2rem;margin-top:12px;">
  <div style="font-size:0.7rem;color:#f87171;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;font-weight:700;">Hard Filters (page dropped entirely)</div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;">
    <span style="font-size:0.76rem;padding:4px 12px;border-radius:8px;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);color:#fca5a5;font-family:'JetBrains Mono',monospace;">word_count &lt; 300</span>
    <span style="font-size:0.76rem;padding:4px 12px;border-radius:8px;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);color:#fca5a5;font-family:'JetBrains Mono',monospace;">unique_ratio &lt; 0.30</span>
    <span style="font-size:0.76rem;padding:4px 12px;border-radius:8px;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);color:#fca5a5;font-family:'JetBrains Mono',monospace;">link_density &gt; 0.05</span>
  </div>
</div>
""", unsafe_allow_html=True)

        s2.update(label=f"✅ {doc_count} documents collected in {time.time()-t0:.1f}s", state="complete", expanded=False)

    if not documents:
        st.error("❌ No documents collected. Check your internet connection or try a different query.")
        st.stop()

    # ── Stage 3: Graph Construction ──────────────────────────────────
    st.markdown("""
    <div class="stage-hdr">
        <div class="stage-num">3</div>
        <div><div class="stage-title">Document Graph Construction</div>
        <div class="stage-desc">Embedding similarity · Keyword overlap · Top-K neighbor guarantee</div></div>
    </div>""", unsafe_allow_html=True)

    with st.status("Building document graph...", expanded=True) as s3:
        t0 = time.time()
        vector_store = m["VectorStore"]()
        doc_store    = m["DocumentStore"]()
        doc_store.add_batch(documents)

        contents  = [d.content for d in documents]
        urls      = [d.url     for d in documents]
        titles    = [d.title   for d in documents]

        vector_store.add_documents(contents, urls, titles, source_query=query_text)
        doc_embeddings  = vector_store.get_all_embeddings()
        query_embedding = vector_store.embed_texts([query_text])[0]

        doc_graph  = m["DocumentGraph"]()
        graph      = doc_graph.build_graph(documents, doc_embeddings)
        adj_matrix = doc_graph.get_adjacency_matrix()

        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        density = round(nx.density(graph) * 100, 1) if n_nodes > 1 else 0
        avg_deg = round(sum(dict(graph.degree()).values()) / max(n_nodes, 1), 1)

        weights  = [d["weight"] for _, _, d in graph.edges(data=True)]
        avg_w    = round(float(np.mean(weights)), 3) if weights else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🔵 Nodes", n_nodes)
        c2.metric("🔗 Edges", n_edges, delta="Top-K guaranteed" if n_edges > 0 else "⚠️ sparse")
        c3.metric("📊 Density", f"{density}%")
        c4.metric("〰️ Avg Degree", avg_deg)
        c5.metric("⚖️ Avg Weight", avg_w)

        if n_edges == 0:
            st.warning("⚠️ No edges formed. Top-K fallback should prevent this — check logs.")
        else:
            st.success(f"✅ Graph connected: {n_nodes} nodes × {n_edges} edges (density {density}%)")
        s3.update(label=f"✅ Graph — {n_nodes} nodes, {n_edges} edges in {time.time()-t0:.1f}s", state="complete", expanded=False)

    with st.expander("📐 Edge Weight Formula", expanded=False):
        render_mermaid("""
        graph LR
            subgraph Edge Weight W_ij
                A["Embedding Cosine<br/>0.70 × cos(e_i, e_j)"]
                B["Keyword Jaccard<br/>0.20 × Ki∩Kj / Ki∪Kj"]
                C["Link Presence<br/>0.10 × domain cross-link"]
            end
            A --> D["W_ij ∈ 0,1"]
            B --> D
            C --> D
            D --> E{"Phase 1<br/>W_ij > 0.25?"}
            E -->|Yes| F["Add Edge"]
            E -->|No| G{"Phase 2<br/>degree < K?"}
            G -->|Yes| H["Force-connect<br/>Top-K neighbors"]
            G -->|No| I["Skip"]
        """, height=320)
        st.markdown(r"""
$$
W_{ij} = 0.70 \cdot \text{cos}(\mathbf{e}_i, \mathbf{e}_j) + 0.20 \cdot \text{Jaccard}(K_i, K_j) + 0.10 \cdot \text{Link}(i,j)
$$
""")
        st.markdown("""
<div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
     border-radius:12px;padding:1.2rem;margin-top:0.5rem;">
  <div style="font-size:0.7rem;color:#00d4ff;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;font-weight:700;">Signal Components</div>
  <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:8px 6px;color:#67e8f9;font-weight:600;width:80px;">0.70</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">Embedding Cosine</td>
      <td style="padding:8px 6px;color:#6b7280;">cos(e<sub>i</sub>, e<sub>j</sub>) — L2-normalized document embeddings, computed via dot product</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:8px 6px;color:#67e8f9;font-weight:600;">0.20</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">Keyword Jaccard</td>
      <td style="padding:8px 6px;color:#6b7280;">|K<sub>i</sub> ∩ K<sub>j</sub>| / |K<sub>i</sub> ∪ K<sub>j</sub>| — title + headings + first 200 chars tokenized</td>
    </tr>
    <tr>
      <td style="padding:8px 6px;color:#67e8f9;font-weight:600;">0.10</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">Link Presence</td>
      <td style="padding:8px 6px;color:#6b7280;">0.6 if domain-level cross-link exists, else 0.0 — partial credit, not binary</td>
    </tr>
  </table>
</div>

<div style="background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.12);
     border-radius:12px;padding:1.2rem;margin-top:12px;">
  <div style="font-size:0.7rem;color:#00d4ff;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;font-weight:700;">Two-Phase Edge Strategy</div>
  <div style="font-size:0.82rem;color:#9ca3af;line-height:1.7;">
    <strong style="color:#e2e8f0;">Phase 1 — Threshold:</strong> Connect pairs where W<sub>ij</sub> &gt; 0.25 (EDGE_THRESHOLD). Cap at 10 edges/node.<br>
    <strong style="color:#e2e8f0;">Phase 2 — Top-K Guarantee:</strong> Nodes with &lt; K edges get force-connected to their K most similar neighbors by embedding. Guarantees the graph is <strong>never disconnected</strong>.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Stage 4: Personalized PageRank ───────────────────────────────
    st.markdown("""
    <div class="stage-hdr">
        <div class="stage-num">4</div>
        <div><div class="stage-title">Personalized PageRank</div>
        <div class="stage-desc">Q = 0.60×semantic + 0.25×domain + 0.15×SERP · α=0.85 · 30 iterations</div></div>
    </div>""", unsafe_allow_html=True)

    with st.status("Running PageRank...", expanded=True) as s4:
        t0 = time.time()
        ranker = m["GraphRanker"]()
        ranked = ranker.rank_documents(
            adjacency_matrix=adj_matrix,
            doc_embeddings=doc_embeddings,
            query_embedding=query_embedding,
            urls=urls, titles=titles,
            documents=documents,
        )
        st.markdown(f'<div style="font-size:0.78rem;color:#4b5563;margin-bottom:8px;">Top 5 before re-ranking</div>', unsafe_allow_html=True)
        for i, r in enumerate(ranked[:5], 1):
            pct = min(r.score * 1500, 100)
            st.markdown(f"""
            <div class="result-card" style="padding:0.8rem 1rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:0.7rem;color:#7c3aed;font-weight:700;">PR#{i}</span>
                    <span class="score-pill sp-purple">{r.score:.5f}</span>
                </div>
                <div style="font-size:0.85rem;font-weight:500;color:#d1d5db;margin-top:4px;">{r.title[:70]}</div>
                <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct:.0f}%;"></div></div>
            </div>""", unsafe_allow_html=True)
        s4.update(label=f"✅ PageRank done in {time.time()-t0:.2f}s", state="complete", expanded=False)

    with st.expander("📐 PageRank & Pruning Formulas", expanded=False):
        render_mermaid("""
        graph TD
            subgraph Personalization Q
                S1["Semantic Sim<br/>0.60 × cos(doc, query)"]
                S2["Domain Auth<br/>0.25 × heuristic tier"]
                S3["SERP Rank<br/>0.15 × 1/position"]
            end
            S1 --> Q["Q vector<br/>L1-normalized"]
            S2 --> Q
            S3 --> Q
            A["A: Row-normalized<br/>adjacency matrix"] --> PR
            Q --> PR["Power Iteration<br/>P = αAP + (1−α)Q"]
            PR --> C{"Converged?<br/>δ < 1e-6"}
            C -->|No, iter < 30| PR
            C -->|Yes| R["Ranked Documents"]
        """, height=420)
        st.markdown(r"""
$$
\mathbf{P}^{(t+1)} = \alpha \cdot \mathbf{A} \cdot \mathbf{P}^{(t)} + (1 - \alpha) \cdot \mathbf{Q}
$$
""")
        st.markdown("""
<div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
     border-radius:12px;padding:1.2rem;margin-top:0.5rem;">
  <div style="font-size:0.7rem;color:#a78bfa;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;font-weight:700;">PageRank Components</div>
  <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:8px 6px;color:#a78bfa;font-weight:600;width:60px;">A</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">Transition Matrix</td>
      <td style="padding:8px 6px;color:#6b7280;">Row-normalized adjacency matrix — encodes document authority structure</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:8px 6px;color:#a78bfa;font-weight:600;">α</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">Damping Factor</td>
      <td style="padding:8px 6px;color:#6b7280;">0.85 — probability of following a link vs. teleporting to Q</td>
    </tr>
    <tr>
      <td style="padding:8px 6px;color:#a78bfa;font-weight:600;">P</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">Score Vector</td>
      <td style="padding:8px 6px;color:#6b7280;">Initialized uniform (1/N), converges via power iteration (max 30 iters, tol 1e-6)</td>
    </tr>
  </table>
</div>
""", unsafe_allow_html=True)

        st.markdown(r"""
$$
Q_i = 0.60 \cdot \text{cos}(\mathbf{d}_i, \mathbf{q}) + 0.25 \cdot \text{DomainAuth}_i + 0.15 \cdot \frac{1}{\text{rank}_i}
$$
""")
        st.markdown("""
<div style="background:rgba(124,58,237,0.04);border:1px solid rgba(124,58,237,0.12);
     border-radius:12px;padding:1.2rem;margin-top:0.5rem;">
  <div style="font-size:0.7rem;color:#a78bfa;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;font-weight:700;">Personalization Vector Q</div>
  <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:8px 6px;color:#c4b5fd;font-weight:600;width:80px;">0.60</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">Semantic Similarity</td>
      <td style="padding:8px 6px;color:#6b7280;">cos(doc embedding, query embedding) — topic relevance</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:8px 6px;color:#c4b5fd;font-weight:600;">0.25</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">Domain Authority</td>
      <td style="padding:8px 6px;color:#6b7280;">Heuristic [0,1] — .gov/.edu=0.97, arxiv=0.97, github=0.85, unknown=0.30</td>
    </tr>
    <tr>
      <td style="padding:8px 6px;color:#c4b5fd;font-weight:600;">0.15</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">SERP Rank Signal</td>
      <td style="padding:8px 6px;color:#6b7280;">1/rank (normalized) — earlier SERP results get higher teleportation probability</td>
    </tr>
  </table>
  <div style="font-size:0.75rem;color:#4b5563;margin-top:10px;border-top:1px solid rgba(255,255,255,0.04);padding-top:8px;">
    Each component is L1-normalized independently, then blended. Final Q is re-normalized to sum to 1.
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**🔀 Node Pruning (Anti-Drift) — 6 Components**")
        st.markdown(r"""
$$
S_{\text{node}} = 0.25 \cdot \text{sem} + 0.20 \cdot \text{kw} + 0.15 \cdot \text{serp} + 0.15 \cdot \text{dom} + 0.15 \cdot \text{qual} + 0.10 \cdot \text{fresh}
$$
""")
        st.markdown("""
<div style="background:rgba(245,158,11,0.04);border:1px solid rgba(245,158,11,0.12);
     border-radius:12px;padding:1.2rem;margin-top:0.5rem;">
  <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:6px;color:#fbbf24;font-weight:600;width:60px;">0.25</td>
      <td style="padding:6px;color:#e2e8f0;">Semantic similarity (embedding cosine vs query)</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:6px;color:#fbbf24;font-weight:600;">0.20</td>
      <td style="padding:6px;color:#e2e8f0;">Keyword overlap (Jaccard vs query keywords)</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:6px;color:#fbbf24;font-weight:600;">0.15</td>
      <td style="padding:6px;color:#e2e8f0;">SERP rank score (1/position, normalized)</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:6px;color:#fbbf24;font-weight:600;">0.15</td>
      <td style="padding:6px;color:#e2e8f0;">Domain authority (heuristic tier lookup)</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:6px;color:#fbbf24;font-weight:600;">0.15</td>
      <td style="padding:6px;color:#e2e8f0;">Content quality (the 5-signal Q<sub>quality</sub> score)</td>
    </tr>
    <tr>
      <td style="padding:6px;color:#fbbf24;font-weight:600;">0.10</td>
      <td style="padding:6px;color:#e2e8f0;">Freshness (recency proxy, default 0.5)</td>
    </tr>
  </table>
  <div style="font-size:0.75rem;color:#4b5563;margin-top:10px;border-top:1px solid rgba(255,255,255,0.04);padding-top:8px;">
    Nodes scoring below <span style="color:#fbbf24;font-family:'JetBrains Mono',monospace;">PRUNING_THRESHOLD</span> are dropped to prevent BFS topic drift.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Stage 5: Re-ranking ──────────────────────────────────────────
    st.markdown("""
    <div class="stage-hdr">
        <div class="stage-num">5</div>
        <div><div class="stage-title">Passage-Level Re-ranking</div>
        <div class="stage-desc">Final score = 0.60×cosine + 0.40×PageRank · Top-20 window</div></div>
    </div>""", unsafe_allow_html=True)

    with st.status("Re-ranking results...", expanded=True) as s5:
        t0 = time.time()
        from search.node_pruning import get_trust_label
        import html as _html

        reranked = m["rerank"](ranked_docs=ranked, documents=documents, query=query_text)

        # Max PR for normalization → graph score shown as % of top doc's score
        max_pr = max((r.pagerank_score for r in reranked), default=1e-9)
        rerank_time = time.time() - t0
        s5.update(label=f"✅ Re-ranking complete in {rerank_time:.2f}s", state="complete", expanded=False)

    with st.expander("📐 Re-ranking & Trust Formulas", expanded=False):
        render_mermaid("""
        graph TD
            subgraph Re-rank Blend
                P1["Passage Cosine<br/>0.60 × cos(query, passage_512)"]
                P2["PageRank Normalized<br/>0.40 × PR_i / max PR"]
            end
            P1 --> F["F_i: Final Score"]
            P2 --> F
            F --> N["Min-Max Normalize<br/>conf = F−Fmin / Fmax−Fmin"]
            N --> R["Confidence %"]
            subgraph Trust Score
                T1["Domain Auth<br/>0.50 × tier score"]
                T2["Graph Auth<br/>0.30 × PR normalized"]
                T3["Content Quality<br/>0.20 × Q_quality"]
            end
            T1 --> T["T_i: Trust"]
            T2 --> T
            T3 --> T
            T --> D1["T ≥ 0.70 → Trusted"]
            T --> D2["T ≥ 0.45 → Moderate"]
            T --> D3["T < 0.45 → Unknown"]
        """, height=500)
        st.markdown(r"""
$$
F_i = 0.60 \cdot \text{cos}(\text{embed}(q),\, \text{embed}(p_i)) + 0.40 \cdot \frac{\text{PR}_i}{\max(\text{PR})}
$$
""")
        st.markdown("""
<div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
     border-radius:12px;padding:1.2rem;margin-top:0.5rem;">
  <div style="font-size:0.7rem;color:#10b981;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;font-weight:700;">Re-rank Blend</div>
  <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:8px 6px;color:#34d399;font-weight:600;width:80px;">0.60</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">Passage Cosine</td>
      <td style="padding:8px 6px;color:#6b7280;">cos(query, first-512-words passage) — passage-level relevance re-check</td>
    </tr>
    <tr>
      <td style="padding:8px 6px;color:#34d399;font-weight:600;">0.40</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">PageRank (norm)</td>
      <td style="padding:8px 6px;color:#6b7280;">PR<sub>i</sub> / max(PR) — preserves graph authority signal, normalized to [0,1]</td>
    </tr>
  </table>
  <div style="font-size:0.75rem;color:#4b5563;margin-top:10px;border-top:1px solid rgba(255,255,255,0.04);padding-top:8px;">
    Confidence % is min-max normalized across results: <span style="color:#34d399;font-family:'JetBrains Mono',monospace;">conf = (F - F<sub>min</sub>) / (F<sub>max</sub> - F<sub>min</sub>)</span>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown(r"""
$$
T_i = 0.50 \cdot \text{DomainAuth} + 0.30 \cdot \frac{\text{PR}_i}{\max(\text{PR})} + 0.20 \cdot Q_{\text{quality}}
$$
""")
        st.markdown("""
<div style="background:rgba(16,185,129,0.04);border:1px solid rgba(16,185,129,0.12);
     border-radius:12px;padding:1.2rem;margin-top:0.5rem;">
  <div style="font-size:0.7rem;color:#10b981;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;font-weight:700;">Trust Classification</div>
  <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:8px 6px;color:#34d399;font-weight:600;width:80px;">0.50</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">Domain Authority</td>
      <td style="padding:8px 6px;color:#6b7280;">Heuristic domain tier score [0,1]</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
      <td style="padding:8px 6px;color:#34d399;font-weight:600;">0.30</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">Graph Authority</td>
      <td style="padding:8px 6px;color:#6b7280;">Normalized PageRank score</td>
    </tr>
    <tr>
      <td style="padding:8px 6px;color:#34d399;font-weight:600;">0.20</td>
      <td style="padding:8px 6px;color:#e2e8f0;font-weight:600;">Content Quality</td>
      <td style="padding:8px 6px;color:#6b7280;">The 5-signal quality gate score</td>
    </tr>
  </table>
  <div style="display:flex;gap:10px;margin-top:12px;flex-wrap:wrap;">
    <span style="font-size:0.76rem;padding:4px 12px;border-radius:8px;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);color:#6ee7b7;font-family:'JetBrains Mono',monospace;">T ≥ 0.70 → Trusted</span>
    <span style="font-size:0.76rem;padding:4px 12px;border-radius:8px;background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);color:#fcd34d;font-family:'JetBrains Mono',monospace;">T ≥ 0.45 → Moderate</span>
    <span style="font-size:0.76rem;padding:4px 12px;border-radius:8px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);color:#6b7280;font-family:'JetBrains Mono',monospace;">T &lt; 0.45 → Unknown</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Render re-ranked cards OUTSIDE st.status (fixes HTML rendering) ──
    st.markdown(
        '<div style="font-size:0.78rem;color:#4b5563;margin-bottom:12px;">'
        'Confidence % = min-max normalized across retrieved results (top result = 100%)'
        '</div>',
        unsafe_allow_html=True,
    )

    for i, r in enumerate(reranked[:10], 1):
        # ── Safe-escape all user text ────────────────────────────────
        safe_title  = _html.escape(r.title[:90])
        safe_netloc = _html.escape(r.url.split("/")[2] if "//" in r.url else r.url[:40])

        # ── Position change vs raw PageRank ──────────────────────────
        pr_pos    = next((j+1 for j, rd in enumerate(ranked) if rd.url == r.url), "?")
        moved_tag = ""
        if isinstance(pr_pos, int) and pr_pos != i:
            arrow     = "↑" if pr_pos > i else "↓"
            delta     = abs(pr_pos - i)
            moved_tag = (
                f'<span class="score-pill sp-moved">{arrow}{delta} vs PR</span>'
            )

        # ── Domain authority tier label (not a % of 0-1) ─────────────
        da = max(r.domain_score, 0.30)      # floor at 0.30 (never "0%")
        if da >= 0.92:   auth_label, auth_cls = "Tier 1 · Academic/Gov", "sp-green"
        elif da >= 0.85: auth_label, auth_cls = "Tier 2 · Official Docs", "sp-blue"
        elif da >= 0.65: auth_label, auth_cls = "Tier 3 · Org/Research",  "sp-purple"
        elif da >= 0.45: auth_label, auth_cls = "Tier 4 · Community",     "sp-amber"
        else:            auth_label, auth_cls = "Tier 5 · Unknown",        "score-pill"

        # ── Scores for display ────────────────────────────────────────
        conf_pct   = r.confidence_pct                         # [0–100] normalized
        rel_pct    = round(r.cosine_score * 100)              # passage relevance [0–100]
        graph_pct  = round((r.pagerank_score / max_pr) * 100) # relative graph strength [0–100]
        qual_score = r.quality_score if r.quality_score > 0 else 0.5  # fallback

        # ── Composite trust label ─────────────────────────────────────
        trust_label, trust_cls = get_trust_label(
            domain_score=da,
            pagerank_score=r.pagerank_score,
            quality_score=qual_score,
            max_pagerank=max_pr,
        )

        # ── "Why this result?" explanation ────────────────────────────
        reasons = []
        if r.cosine_score > 0.70:  reasons.append("Highly relevant to query")
        elif r.cosine_score > 0.45: reasons.append("Moderately relevant")
        else:                       reasons.append("Low query relevance")
        if da >= 0.92:   reasons.append(f"Academic/gov source ({safe_netloc})")
        elif da >= 0.85: reasons.append(f"Official documentation ({safe_netloc})")
        elif da >= 0.65: reasons.append("Reputable domain")
        if graph_pct >= 60: reasons.append("Strong graph connections")
        elif graph_pct >= 30: reasons.append("Moderate graph connections")
        if isinstance(pr_pos, int) and pr_pos > i:
            reasons.append(f"Promoted {pr_pos - i} spot(s) by passage re-ranking")
        why_text = _html.escape(" · ".join(reasons))

        # ── Render card (NO HTML comments — they show as text) ─────────
        st.markdown(f"""
<div class="result-card">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px;margin-bottom:6px;">
    <div style="display:flex;align-items:center;gap:8px;">
      <div class="result-rank">#{i}</div>
      <span class="score-pill {trust_cls}">{trust_label}</span>
      {moved_tag}
    </div>
    <div style="text-align:right;">
      <div style="font-size:1.5rem;font-weight:800;color:#f9fafb;line-height:1;">{conf_pct:.0f}%</div>
      <div style="font-size:0.6rem;color:#4b5563;letter-spacing:0.5px;text-transform:uppercase;">Confidence</div>
    </div>
  </div>
  <div class="result-title">{safe_title}</div>
  <div class="result-url">{safe_netloc}</div>
  <div class="result-scores" style="margin-top:8px;">
    <span class="score-pill sp-green">Relevance {rel_pct}%</span>
    <span class="score-pill {auth_cls}">{auth_label}</span>
    <span class="score-pill sp-purple">Graph {graph_pct}%</span>
    <span class="score-pill sp-amber">Score {r.final_score:.3f}</span>
  </div>
  <div class="score-bar-bg" style="margin-top:8px;">
    <div class="score-bar-fill" style="width:{conf_pct:.0f}%;"></div>
  </div>
  <div style="margin-top:8px;font-size:0.72rem;color:#4b5563;border-top:1px solid rgba(255,255,255,0.04);padding-top:6px;">
    <span style="color:#6b7280;font-weight:600;">Why: </span>{why_text}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Stage 6: Evaluation ──────────────────────────────────────────
    st.markdown("""
    <div class="stage-hdr">
        <div class="stage-num">6</div>
        <div><div class="stage-title">Evaluation — Graph+PR vs BM25</div>
        <div class="stage-desc">Precision@10 · Recall@10 · NDCG@10 · Topical Coverage</div></div>
    </div>""", unsafe_allow_html=True)

    with st.status("Computing evaluation metrics...", expanded=True) as s6:
        t0 = time.time()
        corpus_tokens = [m["tokenize"](c[:500]) for c in contents]
        q_tokens      = m["tokenize"](query_text)
        bm25          = m["BM25"](corpus_tokens)
        bm25_ranked   = bm25.rank(q_tokens, top_k=10)
        bm25_ids      = [urls[idx] for idx, _ in bm25_ranked]
        graph_ids     = [r.url for r in reranked[:10]]

        relevant_set  = set(graph_ids[:5]) | set(bm25_ids[:3])
        bm25_rels     = [1.0 if u in relevant_set else 0.0 for u in bm25_ids]
        graph_rels    = [1.0 if u in relevant_set else 0.0 for u in graph_ids]

        metrics = {}
        for name, fn in [("Precision@10", m["precision_at_k"]), ("Recall@10", m["recall_at_k"])]:
            metrics[name] = {"bm25": fn(bm25_ids, relevant_set, 10), "graph": fn(graph_ids, relevant_set, 10)}
        metrics["NDCG@10"] = {"bm25": m["ndcg_at_k"](bm25_rels, 10), "graph": m["ndcg_at_k"](graph_rels, 10)}
        q_kws       = set(m["tokenize"](query_text))
        ret_kw_sets = [set(m["tokenize"](d.content[:300])) for d in documents[:10]]
        metrics["Topical Cov."] = {"bm25": 0, "graph": m["topical_coverage"](ret_kw_sets, q_kws)}

        cols = st.columns(4)
        for ci, (metric, vals) in enumerate(metrics.items()):
            with cols[ci]:
                delta = vals["graph"] - vals["bm25"] if vals["bm25"] > 0 else None
                d_str = f"+{delta:.3f}" if delta and delta > 0 else (f"{delta:.3f}" if delta else None)
                st.metric(metric, f"{vals['graph']:.4f}", delta=d_str)
                if metric != "Topical Cov.":
                    st.caption(f"BM25: {vals['bm25']:.4f}")

        cdf = pd.DataFrame({
            "Metric": [k for k in metrics if k != "Topical Cov."],
            "BM25":      [metrics[k]["bm25"]  for k in metrics if k != "Topical Cov."],
            "Graph+PR":  [metrics[k]["graph"] for k in metrics if k != "Topical Cov."],
        }).set_index("Metric")
        st.bar_chart(cdf, color=["#7c3aed", "#00d4ff"])
        s6.update(label=f"✅ Evaluation complete in {time.time()-t0:.2f}s", state="complete", expanded=False)

    # ── Stage 7: Answer Generation ───────────────────────────────────
    st.markdown("""
    <div class="stage-hdr">
        <div class="stage-num">7</div>
        <div><div class="stage-title">Answer Generation</div>
        <div class="stage-desc">Grounded synthesis from top-8 re-ranked documents · Groq LLaMA 3.3 70B / Ollama / TextRank fallback</div></div>
    </div>""", unsafe_allow_html=True)

    with st.status("Generating answer...", expanded=True) as s7:
        t0 = time.time()
        top_docs = []
        for r in reranked[:8]:
            doc = doc_store.get_by_index(r.index)
            if doc:
                top_docs.append({"title": doc.title, "url": doc.url, "content": doc.content})

        answer_resp = run_async(m["generate_answer"](top_docs, query_text))
        provider_label = getattr(answer_resp, 'provider', 'unknown').upper()

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
             border-radius:14px;padding:1.4rem;margin-top:0.5rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
                <div style="font-size:0.7rem;color:#4b5563;letter-spacing:1.5px;text-transform:uppercase;">
                    ✍️ Generated Answer
                </div>
                <span class="score-pill sp-purple">⚡ {provider_label}</span>
            </div>
            <div style="color:#e2e8f0;font-size:0.93rem;line-height:1.75;">
                {answer_resp.answer[:3000]}
            </div>
        </div>""", unsafe_allow_html=True)

        if answer_resp.citations:
            st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin:1rem 0 6px;letter-spacing:1px;text-transform:uppercase;">Sources</div>', unsafe_allow_html=True)
            for cite in answer_resp.citations[:6]:
                db = domain_badge(cite.url)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                    {db}
                    <a href="{cite.url}" target="_blank"
                       style="font-size:0.78rem;color:#60a5fa;text-decoration:none;">
                        {cite.title[:70]}
                    </a>
                </div>""", unsafe_allow_html=True)
        s7.update(label=f"✅ Answer generated in {time.time()-t0:.1f}s", state="complete", expanded=False)

    # ── Pipeline Summary ─────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    st.markdown(f"""
    <div style="margin-top:2.5rem;padding:1.5rem;
         background:rgba(124,58,237,0.06);
         border:1px solid rgba(124,58,237,0.15);
         border-radius:16px;text-align:center;">
        <div style="font-size:0.65rem;color:#7c3aed;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">
            Pipeline Complete
        </div>
        <div style="font-size:2rem;font-weight:800;color:#f9fafb;">{total_time:.1f}s</div>
        <div style="font-size:0.82rem;color:#4b5563;margin-top:6px;">
            {doc_count} docs · {n_nodes} graph nodes · {n_edges} edges · {len(reranked)} re-ranked · LLM answer generated
        </div>
    </div>
    """, unsafe_allow_html=True)

elif search_clicked and not query_text.strip():
    st.warning("Please enter a query to search.")
