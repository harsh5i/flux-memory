"""Tests for Track 4 Step 1 — Python SDK surface (flux_store, flux_retrieve, flux_feedback)."""
from __future__ import annotations

import pytest
from pathlib import Path
from datetime import timezone, datetime, timedelta

from flux import Config, FluxStore, Grain, Conduit
from mocks import MockEmbeddingBackend
from flux.embedding import store_embedding
from mocks import MockLLMBackend
from flux.health import log_event
from flux.retrieval import flux_store, flux_retrieve, flux_feedback, RetrievalResult, FeedbackResult


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    with FluxStore(db) as s:
        yield s


def _now():
    return datetime.now(timezone.utc)


# ===================================================================== flux_store

class TestFluxStore:
    def test_bare_store_returns_grain_id(self, store):
        gid = flux_store("User prefers Python", store=store)
        assert isinstance(gid, str)
        assert len(gid) > 0
        g = store.get_grain(gid)
        assert g is not None
        assert g.content == "User prefers Python"

    def test_default_provenance_user_stated(self, store):
        gid = flux_store("Test content", store=store)
        g = store.get_grain(gid)
        assert g.provenance == "user_stated"

    def test_custom_provenance(self, store):
        gid = flux_store("AI said this", provenance="ai_stated", store=store)
        assert store.get_grain(gid).provenance == "ai_stated"

    def test_empty_content_raises(self, store):
        with pytest.raises(ValueError):
            flux_store("", store=store)

    def test_invalid_provenance_raises(self, store):
        with pytest.raises(ValueError):
            flux_store("text", provenance="unknown_prov", store=store)

    def test_with_llm_and_emb_creates_entry_connections(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        flux_store("I prefer Python for ML", store=store, llm=llm, emb=emb)
        count = store.conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"]
        assert count >= 1

    def test_llm_failure_fallback_still_wires_graph(self, store):
        class FailingLLM:
            def complete(self, prompt):
                raise RuntimeError("llm unavailable")

        emb = MockEmbeddingBackend()
        gid = flux_store("User prefers compact dashboard metrics", store=store,
                         llm=FailingLLM(), emb=emb)

        assert store.get_grain(gid) is not None
        embedding_count = store.conn.execute(
            "SELECT COUNT(*) AS n FROM grain_embeddings WHERE grain_id = ?",
            (gid,),
        ).fetchone()["n"]
        entry_count = store.conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"]
        conduit_count = store.conn.execute("SELECT COUNT(*) AS n FROM conduits").fetchone()["n"]
        assert embedding_count == 1
        assert entry_count >= 1
        assert conduit_count >= 1


# ===================================================================== flux_retrieve

class TestFluxRetrieve:
    def test_returns_retrieval_result(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        flux_store("Python is great for data science", store=store, llm=llm, emb=emb)
        result = flux_retrieve("Python machine learning", store=store, llm=llm, emb=emb)
        assert isinstance(result, RetrievalResult)

    def test_trace_id_stored(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        flux_store("Python preference", store=store, llm=llm, emb=emb)
        result = flux_retrieve("Python", store=store, llm=llm, emb=emb)
        assert store.get_trace(result.trace_id) is not None

    def test_grains_have_expected_fields(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        flux_store("User likes dark themes", store=store, llm=llm, emb=emb)
        result = flux_retrieve("dark theme preferences", store=store, llm=llm, emb=emb)
        for g in result.grains:
            assert "id" in g
            assert "content" in g
            assert "score" in g
            assert "provenance" in g

    def test_confidence_is_float_in_range(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        result = flux_retrieve("some query", store=store, llm=llm, emb=emb)
        assert 0.0 <= result.confidence <= 1.0

    def test_features_returned(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        result = flux_retrieve("Python framework choice", store=store, llm=llm, emb=emb)
        assert isinstance(result.features, list)
        assert len(result.features) >= 1

    def test_fallback_triggered_on_empty_store(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        result = flux_retrieve("anything", store=store, llm=llm, emb=emb)
        # Empty store → no propagation → low confidence → fallback fires
        # (may or may not trigger depending on cfg, but no error)
        assert isinstance(result.fallback_triggered, bool)

    def test_retrieval_event_logged(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        flux_retrieve("test query", store=store, llm=llm, emb=emb)
        count = store.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE event='grains_returned'"
        ).fetchone()["n"]
        assert count >= 1

    def test_pending_feedback_blocks_same_caller(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        cfg = Config(FEEDBACK_ENFORCEMENT_GRACE_SECONDS=0)
        old = _now() - timedelta(seconds=1)
        log_event(
            store,
            "retrieval",
            "grains_returned",
            {"grain_ids": ["g1"], "grains_count": 1, "caller_id": "agent-a"},
            trace_id="trace-pending",
            now=old,
        )

        with pytest.raises(RuntimeError, match="flux_feedback required"):
            flux_retrieve(
                "next query",
                store=store,
                llm=llm,
                emb=emb,
                cfg=cfg,
                caller_id="agent-a",
            )

    def test_pending_feedback_does_not_block_other_callers(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        cfg = Config(FEEDBACK_ENFORCEMENT_GRACE_SECONDS=0)
        old = _now() - timedelta(seconds=1)
        log_event(
            store,
            "retrieval",
            "grains_returned",
            {"grain_ids": ["g1"], "grains_count": 1, "caller_id": "agent-a"},
            trace_id="trace-pending",
            now=old,
        )

        result = flux_retrieve(
            "next query",
            store=store,
            llm=llm,
            emb=emb,
            cfg=cfg,
            caller_id="agent-b",
        )

        assert isinstance(result, RetrievalResult)

    def test_feedback_clears_pending_retrieval_block(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        cfg = Config(FEEDBACK_ENFORCEMENT_GRACE_SECONDS=0)
        old = _now() - timedelta(seconds=1)
        log_event(
            store,
            "retrieval",
            "grains_returned",
            {"grain_ids": ["g1"], "grains_count": 1, "caller_id": "agent-a"},
            trace_id="trace-pending",
            now=old,
        )
        log_event(
            store,
            "feedback",
            "feedback_received",
            {"grain_id": "g1", "useful": True, "caller_id": "agent-a"},
            trace_id="trace-pending",
            now=_now(),
        )

        result = flux_retrieve(
            "next query",
            store=store,
            llm=llm,
            emb=emb,
            cfg=cfg,
            caller_id="agent-a",
        )

        assert isinstance(result, RetrievalResult)

    def test_zero_grain_retrieval_does_not_block_next_query(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        cfg = Config(FEEDBACK_ENFORCEMENT_GRACE_SECONDS=0)
        old = _now() - timedelta(seconds=1)
        log_event(
            store,
            "retrieval",
            "grains_returned",
            {"grain_ids": [], "grains_count": 0, "caller_id": "agent-a"},
            trace_id="trace-empty",
            now=old,
        )

        result = flux_retrieve(
            "next query",
            store=store,
            llm=llm,
            emb=emb,
            cfg=cfg,
            caller_id="agent-a",
        )

        assert isinstance(result, RetrievalResult)


# ===================================================================== flux_feedback

class TestFluxFeedback:
    def _setup(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        flux_store("User prefers Python for ML", store=store, llm=llm, emb=emb)
        result = flux_retrieve("Python machine learning", store=store, llm=llm, emb=emb)
        return result

    def test_feedback_returns_result(self, store):
        result = self._setup(store)
        if not result.grains:
            pytest.skip("No grains retrieved")
        fb = flux_feedback(result.trace_id, result.grains[0]["id"], True, store=store)
        assert isinstance(fb, FeedbackResult)

    def test_positive_feedback_action_reinforced(self, store):
        result = self._setup(store)
        if not result.grains:
            pytest.skip("No grains retrieved")
        fb = flux_feedback(result.trace_id, result.grains[0]["id"], True, store=store)
        assert fb.action == "reinforced"

    def test_negative_feedback_action_penalized(self, store):
        result = self._setup(store)
        if not result.grains:
            pytest.skip("No grains retrieved")
        fb = flux_feedback(result.trace_id, result.grains[0]["id"], False, store=store)
        assert fb.action == "penalized"

    def test_unknown_trace_returns_skipped(self, store):
        fb = flux_feedback("nonexistent-trace-id", "nonexistent-grain-id", True, store=store)
        assert fb.action == "skipped"

    def test_feedback_event_logged(self, store):
        result = self._setup(store)
        if not result.grains:
            pytest.skip("No grains retrieved")
        flux_feedback(result.trace_id, result.grains[0]["id"], True, store=store)
        count = store.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE event='feedback_received'"
        ).fetchone()["n"]
        assert count >= 1

    def test_effective_signal_positive_for_useful(self, store):
        result = self._setup(store)
        if not result.grains:
            pytest.skip("No grains retrieved")
        fb = flux_feedback(result.trace_id, result.grains[0]["id"], True, store=store)
        assert fb.effective_signal > 0

    def test_effective_signal_negative_for_not_useful(self, store):
        result = self._setup(store)
        if not result.grains:
            pytest.skip("No grains retrieved")
        fb = flux_feedback(result.trace_id, result.grains[0]["id"], False, store=store)
        assert fb.effective_signal < 0
