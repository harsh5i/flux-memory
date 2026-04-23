"""Booth architecture and service orchestration (§1A.7, §1A.7a).

Three booth types:
  Read booth  — ThreadPoolExecutor(max_workers=READ_WORKERS) handles flux_retrieve
                calls concurrently. Reads are safe to parallelize because lazy
                decay is a pure computation at read time, not a mutation.

  Write booth — serial Queue; a single writer thread processes flux_store calls.
                SQLite WAL already serialises writes; the queue makes backpressure
                explicit and bounded.

  Feedback queue — async serial Queue; flux_feedback is applied after the caller
                   receives its retrieval response, decoupling feedback latency
                   from retrieval latency.

Ingestion limits (§1A.7a):
  MAX_GRAINS_PER_CALL  — per-call batch cap (default 100)
  MAX_WRITE_QUEUE_DEPTH — write queue depth cap (default 1000)
  MAX_GRAINS_PER_MINUTE — per-caller sliding-window rate limit (default 500)

Usage:
    svc = FluxService(store, llm, emb, cfg)
    svc.start()          # launch background workers
    svc.stop()           # graceful shutdown
    result = svc.retrieve("my query", caller_id="claude")
    grain_id = svc.store("some fact", caller_id="claude")
    svc.feedback(trace_id, grain_id, True, caller_id="claude")
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .config import Config, DEFAULT_CONFIG
from .embedding import EmbeddingBackend, SentenceTransformerBackend
from .graph import utcnow
from .health import log_event
from .llm import LLMBackend, OllamaBackend
from .retrieval import FeedbackResult, RetrievalResult, flux_feedback, flux_retrieve, flux_store
from .storage import FluxStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- rate limiter

class _SlidingWindowRateLimiter:
    """Per-caller sliding-window rate limiter (counts items, not requests)."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._windows: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check_and_record(self, caller_id: str, count: int = 1) -> bool:
        """Return True if allowed, False if rate limit exceeded."""
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            dq = self._windows[caller_id]
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            current = sum(c for _, c in dq)
            if current + count > self._max:
                return False
            dq.append((now, count))
            return True


# ---------------------------------------------------------------- write item

@dataclass
class _WriteItem:
    content: str
    provenance: str
    caller_id: str
    result_future: "queue.Queue[str | Exception]"


@dataclass
class _FeedbackItem:
    trace_id: str
    grain_id: str
    useful: bool
    caller_id: str


# ---------------------------------------------------------------- service

class FluxService:
    """Orchestrates read workers, write queue, and feedback queue."""

    def __init__(
        self,
        store: FluxStore,
        llm: LLMBackend | None = None,
        emb: EmbeddingBackend | None = None,
        cfg: Config = DEFAULT_CONFIG,
    ) -> None:
        self._store = store
        self._llm = llm or OllamaBackend(cfg)
        self._emb = emb or SentenceTransformerBackend(cfg.EMBEDDING_MODEL_NAME)
        self._cfg = cfg

        self._rate_limiter = _SlidingWindowRateLimiter(cfg.MAX_GRAINS_PER_MINUTE)
        self._write_queue: queue.Queue[_WriteItem | None] = queue.Queue()
        self._feedback_queue: queue.Queue[_FeedbackItem | None] = queue.Queue()
        self._executor: ThreadPoolExecutor | None = None
        self._write_thread: threading.Thread | None = None
        self._feedback_thread: threading.Thread | None = None
        self._running = False

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._executor = ThreadPoolExecutor(
            max_workers=self._cfg.READ_WORKERS,
            thread_name_prefix="flux-read",
        )
        self._write_thread = threading.Thread(
            target=self._write_worker, name="flux-write", daemon=True
        )
        self._feedback_thread = threading.Thread(
            target=self._feedback_worker, name="flux-feedback", daemon=True
        )
        self._write_thread.start()
        self._feedback_thread.start()
        log_event(self._store, "system", "startup", {"service": "FluxService"})
        logger.info("FluxService started (read_workers=%d)", self._cfg.READ_WORKERS)

    def stop(self, timeout: float = 10.0) -> None:
        if not self._running:
            return
        self._running = False
        # Poison pills.
        self._write_queue.put(None)
        self._feedback_queue.put(None)
        if self._write_thread:
            self._write_thread.join(timeout=timeout)
        if self._feedback_thread:
            self._feedback_thread.join(timeout=timeout)
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
        log_event(self._store, "system", "shutdown", {"service": "FluxService"})
        logger.info("FluxService stopped")

    # ---------------------------------------------------------------- public API

    def retrieve(self, query: str, caller_id: str = "default") -> RetrievalResult:
        """Submit a retrieve to the read booth (blocking, returns result)."""
        future = self._executor.submit(self._retrieve_worker, query)
        return future.result()

    def _retrieve_worker(self, query: str) -> RetrievalResult:
        """Run one retrieve using a worker-local SQLite connection."""
        if self._store.db_path == ":memory:":
            return flux_retrieve(
                query,
                store=self._store,
                llm=self._llm,
                emb=self._emb,
                cfg=self._cfg,
            )

        with FluxStore(self._store.db_path) as read_store:
            return flux_retrieve(
                query,
                store=read_store,
                llm=self._llm,
                emb=self._emb,
                cfg=self._cfg,
            )

    def store(self, content: str, provenance: str = "ai_stated",
              caller_id: str = "default") -> str:
        """Queue a store request. Blocks until processed or raises on limit."""
        if not content or not content.strip():
            raise ValueError("content must not be empty")
        # Batch cap: single item, but enforces same rule.
        if self._write_queue.qsize() >= self._cfg.MAX_WRITE_QUEUE_DEPTH:
            raise RuntimeError(
                "Flux write queue is full (backpressure). Retry later."
            )
        if not self._rate_limiter.check_and_record(caller_id, count=1):
            raise RuntimeError(
                f"Rate limit exceeded for caller '{caller_id}'. "
                f"Max {self._cfg.MAX_GRAINS_PER_MINUTE} grains/min."
            )
        result_q: queue.Queue[str | Exception] = queue.Queue()
        self._write_queue.put(_WriteItem(content, provenance, caller_id, result_q))
        outcome = result_q.get()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def store_batch(self, items: list[dict], caller_id: str = "default") -> list[str]:
        """Store multiple grains. Raises if batch exceeds MAX_GRAINS_PER_CALL."""
        if len(items) > self._cfg.MAX_GRAINS_PER_CALL:
            raise ValueError(
                f"Batch of {len(items)} exceeds MAX_GRAINS_PER_CALL "
                f"({self._cfg.MAX_GRAINS_PER_CALL}). Split into smaller calls."
            )
        return [
            self.store(
                item.get("content", ""),
                item.get("provenance", "ai_stated"),
                caller_id=caller_id,
            )
            for item in items
        ]

    def feedback(self, trace_id: str, grain_id: str, useful: bool,
                 caller_id: str = "default") -> None:
        """Queue feedback for async processing (non-blocking)."""
        self._feedback_queue.put(
            _FeedbackItem(trace_id, grain_id, useful, caller_id)
        )

    def feedback_sync(self, trace_id: str, grain_id: str, useful: bool,
                      caller_id: str = "default") -> FeedbackResult:
        """Apply feedback synchronously (for SDK callers that want the result)."""
        return flux_feedback(trace_id, grain_id, useful,
                             store=self._store, cfg=self._cfg)

    def health(self) -> dict:
        from .health import flux_health
        return flux_health(self._store, self._cfg)

    def list_grains(self, status: str | None = None, limit: int = 50) -> list[dict]:
        """Return grains filtered by status (read-only)."""
        valid = {"active", "dormant", "quarantined", "archived"}
        if status and status not in valid:
            raise ValueError(f"Invalid status '{status}'. Valid: {sorted(valid)}")
        if status:
            rows = self._store.conn.execute(
                "SELECT id, content, status, provenance, created_at "
                "FROM grains WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._store.conn.execute(
                "SELECT id, content, status, provenance, created_at "
                "FROM grains ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "content_snippet": r["content"][:120],
                "status": r["status"],
                "provenance": r["provenance"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ---------------------------------------------------------------- workers

    def _write_worker(self) -> None:
        while True:
            item = self._write_queue.get()
            if item is None:
                break
            try:
                gid = flux_store(
                    item.content,
                    provenance=item.provenance,
                    store=self._store,
                    llm=self._llm,
                    emb=self._emb,
                    cfg=self._cfg,
                )
                item.result_future.put(gid)
            except Exception as exc:
                item.result_future.put(exc)

    def _feedback_worker(self) -> None:
        while True:
            item = self._feedback_queue.get()
            if item is None:
                break
            try:
                flux_feedback(
                    item.trace_id, item.grain_id, item.useful,
                    store=self._store, cfg=self._cfg,
                )
            except Exception as exc:
                logger.error("feedback worker error: %s", exc)
