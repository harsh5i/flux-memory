"""Embedding backend and vector fallback (Sections 4.7–4.8, Track 2).

Embeddings are used in two places only:
  1. Bootstrap: one embedding per grain at insertion time, used to find the k
     nearest existing grains and create initial conduits (§4.7).
  2. Vector fallback: when propagation confidence < FALLBACK_CONFIDENCE_THRESHOLD,
     query the stored embeddings for nearest neighbours and return them as
     supplementary results (§4.8).

The embedding model is loaded once and kept in memory. Production default:
sentence-transformers (all-MiniLM-L6-v2, 384-dim). Test backends live in tests/mocks.py.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from typing import Protocol, runtime_checkable

import numpy as np

from .config import Config, DEFAULT_CONFIG
from .graph import new_id, utcnow
from .storage import FluxStore

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- protocol

@runtime_checkable
class EmbeddingBackend(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# ------------------------------------------------------ sentence-transformers

class SentenceTransformerBackend:
    """Wraps sentence-transformers for local embedding (§11.2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _model_instance(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        return self._model_instance().encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model_instance().encode(texts, convert_to_numpy=True).tolist()

    @property
    def model_name(self) -> str:
        return self._model_name


# ---------------------------------------------------------- storage helpers

def store_embedding(
    store: FluxStore,
    grain_id: str,
    embedding: list[float],
    model_name: str,
    now=None,
) -> None:
    """Persist a grain's embedding to grain_embeddings table.

    Stored as a float32 blob (~4 bytes/dim). Legacy rows hold JSON text;
    decode_embedding handles both.
    """
    now = now or utcnow()
    from .graph import iso
    blob = np.asarray(embedding, dtype=np.float32).tobytes()
    store.conn.execute(
        """
        INSERT INTO grain_embeddings (grain_id, embedding, model_name, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(grain_id) DO UPDATE SET embedding=excluded.embedding,
            model_name=excluded.model_name, created_at=excluded.created_at
        """,
        (grain_id, blob, model_name, iso(now)),
    )


def decode_embedding(value) -> np.ndarray:
    """Decode a grain_embeddings.embedding value: float32 blob or legacy JSON."""
    if isinstance(value, (bytes, memoryview)):
        return np.frombuffer(bytes(value), dtype=np.float32)
    return np.asarray(json.loads(value), dtype=np.float32)


def load_all_embeddings(store: FluxStore) -> tuple[list[str], np.ndarray]:
    """Load all stored grain embeddings as (grain_ids, matrix) where matrix[i]
    is the embedding for grain_ids[i]. Only returns grains with status='active'."""
    rows = store.conn.execute(
        """
        SELECT ge.grain_id, ge.embedding
        FROM grain_embeddings ge
        JOIN grains g ON g.id = ge.grain_id
        WHERE g.status = 'active'
        """
    ).fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    grain_ids = [r["grain_id"] for r in rows]
    matrix = np.stack([decode_embedding(r["embedding"]) for r in rows])
    return grain_ids, matrix


# ----------------------------------------------------------- cosine similarity

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def top_k_nearest(
    query_embedding: list[float],
    grain_ids: list[str],
    matrix: np.ndarray,
    k: int,
) -> list[tuple[str, float]]:
    """Return the k nearest grain IDs by cosine similarity.

    Returns list of (grain_id, similarity) sorted descending.
    """
    if len(grain_ids) == 0 or matrix.size == 0:
        return []
    q = np.array(query_embedding, dtype=np.float64)
    nq = np.linalg.norm(q)
    if nq == 0:
        return []
    q = q / nq
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = matrix / norms.astype(np.float64)
    sims = normed @ q
    top_indices = np.argsort(sims)[::-1][:k]
    return [(grain_ids[i], float(sims[i])) for i in top_indices]


# ----------------------------------------------------------- in-memory index

class EmbeddingIndex:
    """Thread-safe in-memory mirror of grain_embeddings for active grains.

    Avoids re-loading and JSON-parsing the full embedding matrix from SQLite
    on every store (dedup + bootstrap wiring) and every vector fallback.
    Appended on store; refreshed periodically (e.g. on the service health
    tick) to drop grains that went dormant/archived since the last refresh.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._grain_ids: list[str] = []
        self._matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def refresh(self, store: FluxStore) -> int:
        """Reload from the database. Returns the number of grains indexed."""
        grain_ids, matrix = load_all_embeddings(store)
        with self._lock:
            self._grain_ids = grain_ids
            self._matrix = matrix
            return len(grain_ids)

    def append(self, grain_id: str, embedding: list[float]) -> None:
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        with self._lock:
            if grain_id in self._grain_ids:
                return
            if self._matrix.size == 0:
                self._grain_ids = [grain_id]
                self._matrix = vec
            else:
                self._grain_ids.append(grain_id)
                self._matrix = np.vstack([self._matrix, vec])

    def snapshot(self) -> tuple[list[str], np.ndarray]:
        """Return (grain_ids, matrix) consistent with each other. The caller
        must treat both as read-only — append/refresh replace, never mutate."""
        with self._lock:
            return self._grain_ids[:], self._matrix

    def top_k(self, embedding: list[float], k: int) -> list[tuple[str, float]]:
        grain_ids, matrix = self.snapshot()
        return top_k_nearest(embedding, grain_ids, matrix, k)

    def __len__(self) -> int:
        with self._lock:
            return len(self._grain_ids)


# ------------------------------------------------------------------ vector fallback

def vector_fallback(
    store: FluxStore,
    query_text: str,
    backend: EmbeddingBackend,
    existing_results: list[tuple[str, float]],
    cfg: Config = DEFAULT_CONFIG,
    index: EmbeddingIndex | None = None,
) -> list[tuple[str, float]]:
    """Vector fallback retrieval (§4.8).

    Embeds the query, finds nearest grains by cosine similarity, merges with
    existing graph results (highest score wins duplicates), returns top-K.
    Fires when propagation confidence < FALLBACK_CONFIDENCE_THRESHOLD.

    When ``index`` is provided the in-memory matrix is used instead of
    re-loading all embeddings from SQLite.

    Returns merged list of (grain_id, score) sorted descending.
    """
    if index is not None:
        grain_ids, matrix = index.snapshot()
    else:
        grain_ids, matrix = load_all_embeddings(store)
    if not grain_ids:
        return existing_results

    try:
        query_embedding = backend.embed(query_text)
    except Exception as exc:
        logger.error("vector_fallback: embedding failed: %s", exc)
        return existing_results

    candidates = top_k_nearest(query_embedding, grain_ids, matrix, k=cfg.VECTOR_FALLBACK_K)
    scaled = [(gid, sim * cfg.VECTOR_FALLBACK_SCALE) for gid, sim in candidates]

    # Merge: union with dedup, highest score wins.
    merged: dict[str, float] = {gid: score for gid, score in existing_results}
    for gid, score in scaled:
        if gid not in merged or score > merged[gid]:
            merged[gid] = score

    return sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[: cfg.TOP_K]
