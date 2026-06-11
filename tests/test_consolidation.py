"""Tests for the dream cycle (memory consolidation)."""
from __future__ import annotations

import json
from datetime import timedelta

from mocks import MockEmbeddingBackend

from flux.config import Config
from flux.consolidation import consolidate_clump, dream_cycle, find_candidates
from flux.graph import iso, utcnow
from flux.retrieval import flux_store_ex


class EchoLLM:
    """Synthesizes by echoing the shared phrase — embeds near the centroid."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self.reply


CFG = Config(
    DREAM_MIN_CLUSTER=4,
    DREAM_SIMILARITY=0.95,
    DREAM_CENTROID_MIN=0.90,
    DREAM_MIN_GRAIN_AGE_HOURS=0.0,
    DEDUP_SIMILARITY_THRESHOLD=1.01,  # let near-identical test grains store
)


def _seed_clump(store, emb, text, n):
    ids = []
    for i in range(n):
        gid, _ = flux_store_ex(text, store=store, llm=None, emb=emb, cfg=CFG)
        ids.append(gid)
    return ids


def test_find_candidates_groups_similar_grains(store):
    emb = MockEmbeddingBackend()
    clump_ids = _seed_clump(store, emb, "The gateway restarts every morning.", 5)
    flux_store_ex("Completely unrelated olive oil fact.", store=store, llm=None,
                  emb=emb, cfg=CFG)
    clumps = find_candidates(store, CFG)
    assert len(clumps) == 1
    assert set(clumps[0]) == set(clump_ids)


def test_candidates_respect_min_cluster(store):
    emb = MockEmbeddingBackend()
    _seed_clump(store, emb, "Only three of these exist in memory.", 3)
    assert find_candidates(store, CFG) == []


def test_consolidate_creates_synthesis_with_evidence(store):
    emb = MockEmbeddingBackend()
    clump_ids = _seed_clump(store, emb, "The gateway restarts every morning.", 5)
    llm = EchoLLM("The gateway restarts every morning.")
    gid = consolidate_clump(store, clump_ids, llm, emb, cfg=CFG)
    assert gid is not None and llm.calls == 1

    new = store.conn.execute("SELECT provenance FROM grains WHERE id=?", (gid,)).fetchone()
    assert new["provenance"] == "ai_inferred"
    # evidence conduits to every source
    n = store.conn.execute(
        "SELECT COUNT(*) FROM conduits WHERE from_id=?", (gid,)).fetchone()[0]
    assert n >= len(clump_ids)
    # sources tagged, never consolidated twice
    tags = store.conn.execute(
        "SELECT source_tags FROM grains WHERE id=?", (clump_ids[0],)).fetchone()[0]
    assert "consolidated" in json.loads(tags)
    assert find_candidates(store, CFG) == []


def test_centroid_gate_rejects_hallucination(store):
    emb = MockEmbeddingBackend()
    clump_ids = _seed_clump(store, emb, "The gateway restarts every morning.", 5)
    llm = EchoLLM("Bananas are an excellent source of potassium nonsense.")
    gid = consolidate_clump(store, clump_ids, llm, emb, cfg=CFG)
    assert gid is None
    row = store.conn.execute(
        "SELECT COUNT(*) FROM events WHERE event='consolidation_rejected'").fetchone()
    assert row[0] == 1


def test_dream_cycle_end_to_end(store):
    emb = MockEmbeddingBackend()
    _seed_clump(store, emb, "Olive keeps timing out on compaction.", 5)
    llm = EchoLLM("Olive keeps timing out on compaction.")
    stats = dream_cycle(store, llm, emb, cfg=CFG)
    assert stats["clumps_found"] == 1
    assert len(stats["consolidated"]) == 1
    row = store.conn.execute(
        "SELECT COUNT(*) FROM events WHERE event='dream_cycle_completed'").fetchone()
    assert row[0] == 1


def test_dream_disabled_via_interval_age_gate(store):
    emb = MockEmbeddingBackend()
    cfg = Config(DREAM_MIN_CLUSTER=4, DREAM_SIMILARITY=0.95,
                 DREAM_MIN_GRAIN_AGE_HOURS=48.0,
                 DEDUP_SIMILARITY_THRESHOLD=1.01)
    _seed_clump(store, emb, "Fresh grains must not be consolidated yet.", 5)
    assert find_candidates(store, cfg) == []  # all too young
