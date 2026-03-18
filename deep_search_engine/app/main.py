"""
main.py — FastAPI application entry point.
Serves the Deep Recursive Research Search Engine API.
"""

import logging
import sys

from fastapi import FastAPI

from app.config import settings
from api.routes import router as search_router

# ─── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    stream=sys.stdout,
)

# ─── FastAPI app ─────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "A deep recursive document retrieval system that explores the web "
        "using BFS, builds a document relationship graph, and ranks results "
        "with Topic-Sensitive PageRank."
    ),
    version="1.0.0",
)

app.include_router(search_router, prefix="/api/v1", tags=["search"])


@app.get("/health")
async def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "app_name": settings.APP_NAME}
