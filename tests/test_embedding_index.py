"""Tests for the in-memory EmbeddingIndex, its use in the store path, the
fast (LLM-free) graph rebuild, and the service health tick."""
from __future__ import annotations

import time

from mocks import MockEmbeddingBackend

from flux.config import Config
from flux.embedding import EmbeddingIndex
from flux.extraction import rebuild_missing_graph
from flux.graph import Grain
from flux.retrieval import flux_store_ex


# ----------------------------------------------------------- index behaviour

def test_index_refresh_and_top_k(store):
    emb = MockEmbeddingBackend()
    gid, _ = flux_store_ex("Paris is the capital of France.",
                           store=store, llm=None, emb=emb)
    index = EmbeddingIndex()
    assert index.refresh(store) == 1
    assert len(index) == 1
    hits = index.top_k(emb.embed("Paris is the capital of France."), k=1)
    assert hits[0][0] == gid
    assert hits[0][1] > 0.99


def test_index_append_and_idempotency(store):
    emb = MockEmbeddingBackend()
    index = EmbeddingIndex()
    index.append("g1", emb.embed("alpha"))
    index.append("g1", emb.embed("alpha"))
    index.append("g2", emb.embed("beta"))
    assert len(index) == 2


def test_index_refresh_drops_inactive_grains(store):
    emb = MockEmbeddingBackend()
    gid, _ = flux_store_ex("Paris is the capital of France.",
                           store=store, llm=None, emb=emb)
    index = EmbeddingIndex()
    index.refresh(store)
    store.conn.execute("UPDATE grains SET status='dormant' WHERE id=?", (gid,))
    assert index.refresh(store) == 0


# ------------------------------------------------------- store via the index

def test_store_uses_index_for_dedup(store):
    emb = MockEmbeddingBackend()
    index = EmbeddingIndex()
    a, status_a = flux_store_ex("Paris is the capital of France.",
                                store=store, llm=None, emb=emb, index=index)
    assert status_a == "stored_wired"
    assert len(index) == 1  # appended on store
    b, status_b = flux_store_ex("Paris is the capital of France.",
                                store=store, llm=None, emb=emb, index=index)
    assert status_b == "duplicate"
    assert b == a
    assert len(index) == 1


def test_store_with_index_wires_conduits(store):
    emb = MockEmbeddingBackend()
    index = EmbeddingIndex()
    flux_store_ex("Paris is the capital of France.",
                  store=store, llm=None, emb=emb, index=index)
    gid, _ = flux_store_ex("Olive oil smoke point is around 190C.",
                           store=store, llm=None, emb=emb, index=index)
    conduits = store.conn.execute(
        "SELECT COUNT(*) FROM conduits WHERE from_id=? OR to_id=?", (gid, gid),
    ).fetchone()[0]
    assert conduits > 0


# ----------------------------------------------------------- fast rebuild

def test_rebuild_without_llm_backfills_bare_grains(store):
    emb = MockEmbeddingBackend()
    # Store bare (no emb backend) -> orphan
    gid, status = flux_store_ex("A bare grain with no wiring.", store=store)
    assert status == "stored_bare"

    stats = rebuild_missing_graph(store, None, emb)
    assert stats["grains_rebuilt"] >= 1
    assert stats["embeddings_created"] >= 1
    row = store.conn.execute(
        "SELECT 1 FROM grain_embeddings WHERE grain_id=?", (gid,),
    ).fetchone()
    assert row is not None


# ----------------------------------------------------------- health tick

def test_health_tick_runs_periodically(tmp_path):
    from flux.service import FluxService
    from flux.storage import FluxStore
    from mocks import MockLLMBackend

    cfg = Config(HEALTH_TICK_MINUTES=0.002)  # ~0.12s
    svc = FluxService(
        FluxStore(tmp_path / "tick.db"),
        llm=MockLLMBackend(),
        emb=MockEmbeddingBackend(),
        cfg=cfg,
    )
    svc.start()
    try:
        time.sleep(0.5)
        count = svc._store.conn.execute(
            "SELECT COUNT(*) FROM events WHERE event='health_computed'",
        ).fetchone()[0]
        assert count >= 1
    finally:
        svc.stop()


def test_health_tick_disabled_with_zero(tmp_path):
    from flux.service import FluxService
    from flux.storage import FluxStore
    from mocks import MockLLMBackend

    cfg = Config(HEALTH_TICK_MINUTES=0)
    svc = FluxService(
        FluxStore(tmp_path / "notick.db"),
        llm=MockLLMBackend(),
        emb=MockEmbeddingBackend(),
        cfg=cfg,
    )
    svc.start()
    try:
        assert svc._health_thread is None
    finally:
        svc.stop()
