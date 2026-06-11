"""Tests for the epistemic layer: contradiction, confidence, typed conduits,
tombstones."""
from __future__ import annotations

from datetime import timedelta

from mocks import MockEmbeddingBackend

from flux.config import Config
from flux.epistemics import (
    check_on_store, confirm, effective_confidence, leave_tombstone,
)
from flux.graph import Conduit, Grain, utcnow
from flux.retrieval import flux_store_ex


CFG = Config(EPISTEMIC_CHECK_SIMILARITY=0.0,  # force the judge to run in tests
             DEDUP_SIMILARITY_THRESHOLD=1.01)


class VerdictLLM:
    def __init__(self, verdict): self.verdict = verdict; self.calls = 0
    def complete(self, prompt): self.calls += 1; return self.verdict


def _store(store, text, prov="user_stated"):
    gid, _ = flux_store_ex(text, store=store, llm=None, emb=MockEmbeddingBackend(),
                           provenance=prov, cfg=CFG)
    return gid


def _conf(store, gid):
    return store.conn.execute("SELECT confidence FROM grains WHERE id=?", (gid,)).fetchone()[0]


# ------------------------------------------------------------ contradiction

def test_contradiction_creates_typed_conduit_and_lowers_confidence(store):
    emb = MockEmbeddingBackend()
    old = _store(store, "The gateway listens on port 7462.")
    new = Grain(content="The gateway listens on port 8080.", provenance="user_stated",
                created_at=utcnow())
    store.insert_grain(new)
    res = check_on_store(store, new, VerdictLLM("CONTRADICT"), emb, cfg=CFG)
    assert res and res["relation"] == "contradicts"
    rel = store.conn.execute(
        "SELECT relation FROM conduits WHERE relation='contradicts'").fetchone()
    assert rel is not None
    # older grain lost confidence
    assert _conf(store, old) < 1.0


def test_supersede_marks_old_stale(store):
    emb = MockEmbeddingBackend()
    old = _store(store, "Olive runs model gemma.")
    new = Grain(content="Olive now runs model kimi.", provenance="user_stated",
                created_at=utcnow())
    store.insert_grain(new)
    res = check_on_store(store, new, VerdictLLM("SUPERSEDE"), emb, cfg=CFG)
    assert res["relation"] == "supersedes"
    assert _conf(store, old) < 1.0
    assert store.conn.execute(
        "SELECT 1 FROM conduits WHERE relation='supersedes'").fetchone()


def test_independent_does_nothing(store):
    emb = MockEmbeddingBackend()
    _store(store, "Trading happens on Nifty.")
    new = Grain(content="The sky is blue today.", provenance="user_stated",
                created_at=utcnow())
    store.insert_grain(new)
    assert check_on_store(store, new, VerdictLLM("INDEPENDENT"), emb, cfg=CFG) is None


def test_disabled_or_no_llm_skips(store):
    emb = MockEmbeddingBackend()
    new = Grain(content="x", provenance="user_stated", created_at=utcnow())
    store.insert_grain(new)
    assert check_on_store(store, new, None, emb, cfg=CFG) is None
    cfg2 = Config(EPISTEMIC_CHECK_ENABLED=False)
    assert check_on_store(store, new, VerdictLLM("CONTRADICT"), emb, cfg=cfg2) is None


# --------------------------------------------------------------- confidence

def test_ai_inferred_confidence_decays_with_age(store):
    cfg = Config()
    now = utcnow()
    fresh = {"confidence": 1.0, "provenance": "ai_inferred",
             "created_at": now.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"}
    old = {"confidence": 1.0, "provenance": "ai_inferred",
           "created_at": (now - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"}
    c_fresh = effective_confidence(fresh, cfg, now)
    c_old = effective_confidence(old, cfg, now)
    assert c_fresh > c_old
    # user_stated does not age-decay
    us = {"confidence": 1.0, "provenance": "user_stated",
          "created_at": old["created_at"]}
    assert effective_confidence(us, cfg, now) == 1.0


def test_confirm_raises_confidence(store):
    gid = _store(store, "A fact to confirm.")
    store.conn.execute("UPDATE grains SET confidence=0.5 WHERE id=?", (gid,))
    confirm(store, gid, Config())
    assert _conf(store, gid) > 0.5


def test_confidence_scales_retrieval_score(store):
    # low confidence should reduce a grain's adjusted score vs full confidence
    cfg = Config()
    high = effective_confidence(
        {"confidence": 1.0, "provenance": "user_stated", "created_at": "x"}, cfg, utcnow())
    low = effective_confidence(
        {"confidence": 0.3, "provenance": "user_stated", "created_at": "x"}, cfg, utcnow())
    assert high > low


# --------------------------------------------------------------- tombstones

def test_tombstone_left_for_connected_grain(store):
    now = utcnow()
    g = Grain(content="Important gateway knowledge that mattered.",
              provenance="user_stated", created_at=now)
    store.insert_grain(g)
    store.insert_conduit(Conduit(from_id=g.id, to_id="other1", weight=0.5,
                                 created_at=now, last_used=now))
    store.insert_conduit(Conduit(from_id=g.id, to_id="other2", weight=0.5,
                                 created_at=now, last_used=now))
    tid = leave_tombstone(store, g, Config(), now)
    assert tid is not None
    tomb = store.get_grain(tid)
    assert "once knew" in tomb.content
    tags = store.conn.execute("SELECT source_tags FROM grains WHERE id=?", (tid,)).fetchone()[0]
    assert "tombstone" in tags


def test_no_tombstone_for_isolated_grain(store):
    now = utcnow()
    g = Grain(content="Trivial isolated grain.", provenance="ai_stated", created_at=now)
    store.insert_grain(g)
    assert leave_tombstone(store, g, Config(), now) is None
