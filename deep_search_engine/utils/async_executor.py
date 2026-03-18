"""
async_executor.py — Parallel task executor using asyncio with semaphore-based
concurrency limiting and exponential backoff retry logic.

Blueprint constraints:
    - max_concurrent_requests = 10
    - retry_attempts = 3
    - request_timeout = 10s
"""

import asyncio
import logging
import random
from typing import Any, Coroutine, List

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class AsyncExecutor:
    """Runs a pool of async coroutines with controlled concurrency."""

    def __init__(self):
        self.semaphore = asyncio.Semaphore(settings.SEARCH_CONCURRENCY_LIMIT)
        self.client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialise a shared httpx client."""
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.REQUEST_TIMEOUT_SECONDS),
                follow_redirects=True,
            )
        return self.client

    async def fetch_with_backoff(self, url: str) -> httpx.Response | None:
        """
        GET request with exponential backoff.
        Returns None if all retries exhausted.
        """
        client = await self._get_client()
        for attempt in range(1, settings.SEARCH_RETRY_ATTEMPTS + 1):
            try:
                async with self.semaphore:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    return resp
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                    attempt,
                    settings.SEARCH_RETRY_ATTEMPTS,
                    url,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

        logger.error("All retries exhausted for %s", url)
        return None

    async def run_parallel(self, tasks: List[Coroutine]) -> List[Any]:
        """
        Execute a list of coroutines concurrently, respecting the semaphore.
        Returns results in the same order as the input tasks.
        """
        async def _wrap(coro: Coroutine) -> Any:
            async with self.semaphore:
                return await coro

        return await asyncio.gather(*[_wrap(t) for t in tasks], return_exceptions=True)

    async def close(self):
        """Shutdown the shared httpx client."""
        if self.client and not self.client.is_closed:
            await self.client.aclose()
