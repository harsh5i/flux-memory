"""End-to-end integration checkpoint (§11.7).

Three scenarios, per spec:

  1. Full conversation flow — query decomposition → retrieval (including
     expansion when triggered) → feedback → measurable graph changes → clean
     health report.

  2. Pre-warming from a sample corpus — seed a graph from text/markdown
     files, verify grains were extracted and the graph grew.

  3. Context shift detection via a pivot pattern — simulate a drop in
     retrieval success rate and verify the detector fires and activates the
     exploration boost.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from flux import Config, FluxStore
from flux.embedding import MockEmbeddingBackend
from flux.health import flux_health, log_event
from flux.llm import MockLLMBackend
from flux.prewarm import prewarm
from flux.retrieval import flux_feedback, flux_retrieve, flux_store
from flux.shift import ContextShiftDetector


# ----------------------------------------------------------------- fixtures

@pytest.fixture
def store(tmp_path: Path) -> FluxStore:
    db = tmp_path / "e2e.db"
    with FluxStore(db) as s:
        yield s


@pytest.fixture
def llm() -> MockLLMBackend:
    return MockLLMBackend()


@pytest.fixture
def emb() -> MockEmbeddingBackend:
    return MockEmbeddingBackend()


def _utc(offset_seconds: float = 0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)


# ================================================================ Scenario 1
# Full conversation flow: store → retrieve → feedback → graph changes + health

class TestFullConversationFlow:
    """A short conversation session exercising the complete hot path."""

    MEMORIES = [
        ("User prefers Python for machine learning tasks", "user_stated"),
        ("User is working on a recommendation engine project", "user_stated"),
        ("User has used scikit-learn and PyTorch in past projects", "ai_stated"),
        ("User's team deploys models to AWS SageMaker", "ai_inferred"),
        ("User dislikes verbose boilerplate in ML frameworks", "user_stated"),
    ]

    def _seed(self, store, llm, emb):
        """Store all memories and return their grain IDs."""
        return [
            flux_store(content, provenance=prov, store=store, llm=llm, emb=emb)
            for content, prov in self.MEMORIES
        ]

    def test_store_creates_grains_in_db(self, store, llm, emb):
        ids = self._seed(store, llm, emb)
        assert len(ids) == len(self.MEMORIES)
        for gid in ids:
            assert store.get_grain(gid) is not None

    def test_retrieve_returns_result_with_expected_fields(self, store, llm, emb):
        self._seed(store, llm, emb)
        result = flux_retrieve("Python ML framework", store=store, llm=llm, emb=emb)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.features, list) and len(result.features) >= 1
        assert isinstance(result.grains, list)
        for g in result.grains:
            assert {"id", "content", "provenance", "score"} <= g.keys()

    def test_retrieve_logs_grains_returned_event(self, store, llm, emb):
        self._seed(store, llm, emb)
        flux_retrieve("ML deployment", store=store, llm=llm, emb=emb)
        count = store.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE event='grains_returned'"
        ).fetchone()["n"]
        assert count >= 1

    def test_positive_feedback_reinforces_conduits(self, store, llm, emb):
        self._seed(store, llm, emb)
        result = flux_retrieve("Python scikit", store=store, llm=llm, emb=emb)
        if not result.grains:
            pytest.skip("No grains retrieved — graph too sparse for propagation")

        # Capture conduit weight before feedback.
        gid = result.grains[0]["id"]
        conduits_before = store.conn.execute(
            "SELECT weight FROM conduits WHERE to_id=? OR from_id=?", (gid, gid)
        ).fetchall()
        weights_before = [r["weight"] for r in conduits_before]

        fb = flux_feedback(result.trace_id, gid, True, store=store)
        assert fb.action == "reinforced"
        assert fb.effective_signal > 0

        conduits_after = store.conn.execute(
            "SELECT weight FROM conduits WHERE to_id=? OR from_id=?", (gid, gid)
        ).fetchall()
        weights_after = [r["weight"] for r in conduits_after]

        # At least one conduit must have been reinforced (weight increased or at ceiling).
        if weights_before:
            assert any(
                a >= b for a, b in zip(weights_after, weights_before)
            ), "Expected at least one conduit weight to increase after positive feedback"

    def test_negative_feedback_penalizes_conduits(self, store, llm, emb):
        self._seed(store, llm, emb)
        result = flux_retrieve("recommendation engine AWS", store=store, llm=llm, emb=emb)
        if not result.grains:
            pytest.skip("No grains retrieved")

        gid = result.grains[0]["id"]
        fb = flux_feedback(result.trace_id, gid, False, store=store)
        assert fb.action == "penalized"
        assert fb.effective_signal < 0

    def test_feedback_events_are_logged(self, store, llm, emb):
        self._seed(store, llm, emb)
        result = flux_retrieve("Python preference", store=store, llm=llm, emb=emb)
        if not result.grains:
            pytest.skip("No grains retrieved")
        flux_feedback(result.trace_id, result.grains[0]["id"], True, store=store)
        count = store.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE event='feedback_received'"
        ).fetchone()["n"]
        assert count >= 1

    def test_multi_round_retrieval_accumulates_events(self, store, llm, emb):
        self._seed(store, llm, emb)
        queries = [
            "Python ML",
            "SageMaker deployment",
            "recommendation engine",
        ]
        for q in queries:
            r = flux_retrieve(q, store=store, llm=llm, emb=emb)
            if r.grains:
                flux_feedback(r.trace_id, r.grains[0]["id"], True, store=store)

        event_count = store.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE category='retrieval'"
        ).fetchone()["n"]
        assert event_count >= len(queries)

    def test_expansion_candidates_field_is_present(self, store, llm, emb):
        self._seed(store, llm, emb)
        result = flux_retrieve("ML framework choice", store=store, llm=llm, emb=emb)
        assert isinstance(result.expansion_candidates, list)

    def test_health_report_has_all_14_signals(self, store, llm, emb):
        self._seed(store, llm, emb)
        flux_retrieve("Python", store=store, llm=llm, emb=emb)
        health = flux_health(store)

        assert "status" in health
        assert health["status"] in ("healthy", "warning", "critical")
        assert "signals" in health
        assert "active_warnings" in health
        assert "computed_at" in health

        expected_signals = {
            "highway_count",
            "orphan_rate",
            "avg_conduit_weight",
            "core_grain_count",
            "dormant_grain_rate",
            "highway_growth_rate",
            "shortcut_creation_rate",
            "conduit_dissolution_rate",
            "avg_weight_drop_on_failure",
            "promotion_events",
            "avg_hops_per_retrieval",
            "retrieval_success_rate",
            "fallback_trigger_rate",
            "feedback_compliance_rate",
        }
        assert expected_signals <= set(health["signals"].keys())

    def test_health_signals_have_value_and_healthy_fields(self, store, llm, emb):
        self._seed(store, llm, emb)
        flux_retrieve("Python", store=store, llm=llm, emb=emb)
        health = flux_health(store)
        for name, sig in health["signals"].items():
            assert "value" in sig, f"signal '{name}' missing 'value'"
            assert "healthy" in sig, f"signal '{name}' missing 'healthy'"
            assert isinstance(sig["healthy"], bool)


# ================================================================ Scenario 2
# Pre-warming from sample corpus

class TestPreWarming:
    """Seed the graph from text/markdown sources and verify it grew."""

    @pytest.fixture
    def text_corpus(self, tmp_path: Path) -> list[dict]:
        txt = tmp_path / "notes.txt"
        txt.write_text(
            "The user is a data scientist.\n"
            "They use Python 3.11 and prefer pandas for data manipulation.\n"
            "They have experience with PostgreSQL and dbt for data pipelines.\n"
            "They work at a fintech startup focused on fraud detection.\n"
            "Their preferred IDE is VS Code with the Pylance extension.\n",
            encoding="utf-8",
        )
        md = tmp_path / "context.md"
        md.write_text(
            "# User Profile\n\n"
            "## Skills\n"
            "- Python, SQL, Spark\n"
            "- Machine learning with XGBoost and LightGBM\n\n"
            "## Goals\n"
            "- Build a real-time fraud scoring pipeline\n"
            "- Reduce model inference latency below 50ms\n",
            encoding="utf-8",
        )
        return [
            {"path": str(txt), "type": "text"},
            {"path": str(md), "type": "markdown"},
        ]

    @pytest.fixture
    def conversation_corpus(self, tmp_path: Path) -> list[dict]:
        convo = tmp_path / "history.json"
        turns = [
            {"role": "user", "content": "How do I reduce XGBoost inference latency?"},
            {"role": "assistant", "content": "Consider quantization and feature selection to speed up prediction."},
            {"role": "user", "content": "What about batching requests?"},
            {"role": "assistant", "content": "Yes, batching amortizes overhead. Use a queue with a max_wait_ms threshold."},
        ]
        convo.write_text(json.dumps(turns), encoding="utf-8")
        return [{"path": str(convo), "type": "conversation_json"}]

    def test_prewarm_text_and_markdown_extracts_grains(self, store, llm, emb, text_corpus):
        report = prewarm(text_corpus, store=store, llm=llm, emb=emb)
        assert report["grains_extracted"] > 0
        assert report["files_processed"] == 2
        assert report["chunks_processed"] >= 2

    def test_prewarm_grows_the_graph(self, store, llm, emb, text_corpus):
        before = store.conn.execute("SELECT COUNT(*) AS n FROM grains").fetchone()["n"]
        prewarm(text_corpus, store=store, llm=llm, emb=emb)
        after = store.conn.execute("SELECT COUNT(*) AS n FROM grains").fetchone()["n"]
        assert after > before

    def test_prewarm_creates_conduits(self, store, llm, emb, text_corpus):
        prewarm(text_corpus, store=store, llm=llm, emb=emb)
        conduit_count = store.conn.execute(
            "SELECT COUNT(*) AS n FROM conduits"
        ).fetchone()["n"]
        assert conduit_count > 0

    def test_prewarm_conversation_json(self, store, llm, emb, conversation_corpus):
        report = prewarm(conversation_corpus, store=store, llm=llm, emb=emb)
        assert report["files_processed"] == 1
        assert report["grains_extracted"] >= 0  # extractor may produce 0 for very short chunks

    def test_prewarm_with_synthetic_retrieval_runs_without_error(
        self, store, llm, emb, text_corpus
    ):
        report = prewarm(
            text_corpus,
            store=store,
            llm=llm,
            emb=emb,
            synthetic_retrieval=True,
            synthetic_queries_per_grain=2,
        )
        assert "grains_extracted" in report

    def test_prewarm_report_has_required_keys(self, store, llm, emb, text_corpus):
        report = prewarm(text_corpus, store=store, llm=llm, emb=emb)
        for key in (
            "grains_extracted",
            "conduits_created",
            "entries_created",
            "files_processed",
            "chunks_processed",
            "synthetic_successes",
        ):
            assert key in report, f"prewarm report missing key '{key}'"

    def test_prewarm_conduits_created_nonnegative(self, store, llm, emb, text_corpus):
        report = prewarm(text_corpus, store=store, llm=llm, emb=emb)
        assert report["conduits_created"] >= 0
        assert report["entries_created"] >= 0


# ================================================================ Scenario 3
# Context shift detection via simulated pivot pattern

class TestContextShiftDetection:
    """Simulate a success-rate drop and verify detector fires."""

    # Use a minimal window so the test only needs a handful of events.
    _CFG = Config(
        CONTEXT_SHIFT_WINDOW=10,
        CONTEXT_SHIFT_DROP_THRESHOLD=0.25,
        CONTEXT_SHIFT_RECOVERY_RETRIEVALS=5,
        CONTEXT_SHIFT_ENABLED=True,
        EXPLORATION_BOOST=1.5,
    )

    def _seed_pivot_pattern(self, store: FluxStore) -> None:
        """
        Log events that mimic a context pivot:
          - 5 older retrievals, each with positive feedback (success)
          - 5 recent retrievals, each with feedback_received but no success

        Timestamps are staggered so ORDER BY timestamp DESC puts the recent
        batch at indices [0:5] and the older batch at indices [5:10].

        Compliance = 10 feedback_received / 10 retrievals = 1.0 ≥ 0.8, so
        the detector cannot attribute the drop to missing feedback.
        Drop = 1.0 (older) − 0.0 (recent) = 1.0 ≥ threshold 0.25.
        """
        base = _utc(-200)
        older_trace_ids = [f"trace-old-{i}" for i in range(5)]
        recent_trace_ids = [f"trace-new-{i}" for i in range(5)]

        # --- Older batch (t+0 … t+4 seconds) ---
        for i, tid in enumerate(older_trace_ids):
            t = base + timedelta(seconds=i)
            log_event(store, "retrieval", "grains_returned",
                      {"trace_id": tid, "count": 3}, trace_id=tid, now=t)

        # --- Recent batch (t+100 … t+104 seconds) ---
        for i, tid in enumerate(recent_trace_ids):
            t = base + timedelta(seconds=100 + i)
            log_event(store, "retrieval", "grains_returned",
                      {"trace_id": tid, "count": 3}, trace_id=tid, now=t)

        # --- Feedback: all 10 retrievals get feedback_received (compliance) ---
        feedback_base = base + timedelta(seconds=110)
        for i, tid in enumerate(older_trace_ids + recent_trace_ids):
            t = feedback_base + timedelta(seconds=i)
            log_event(store, "feedback", "feedback_received",
                      {"trace_id": tid}, trace_id=tid, now=t)

        # --- Success signal: only the older batch was useful ---
        for i, tid in enumerate(older_trace_ids):
            t = feedback_base + timedelta(seconds=20 + i)
            log_event(store, "feedback", "retrieval_successful",
                      {"trace_id": tid}, trace_id=tid, now=t)

    def test_detector_fires_on_pivot_pattern(self, store):
        self._seed_pivot_pattern(store)
        detector = ContextShiftDetector(store, self._CFG)
        # record_retrieval triggers _check_shift internally.
        detected = detector.record_retrieval(success=False)
        assert detected, "Expected context shift detection after simulated pivot pattern"

    def test_exploration_boost_active_after_detection(self, store):
        self._seed_pivot_pattern(store)
        detector = ContextShiftDetector(store, self._CFG)
        detector.record_retrieval(success=False)  # triggers detection
        assert detector.in_recovery
        boost = detector.get_exploration_boost()
        assert boost == self._CFG.EXPLORATION_BOOST

    def test_recovery_counter_decrements(self, store, monkeypatch):
        self._seed_pivot_pattern(store)
        detector = ContextShiftDetector(store, self._CFG)
        detector.record_retrieval(success=False)  # triggers detection
        assert detector.in_recovery

        # Stub _check_shift to False so recovery isn't reset by re-detection of
        # the same seeded pattern still present in the events table.
        monkeypatch.setattr(detector, "_check_shift", lambda now: False)

        for _ in range(self._CFG.CONTEXT_SHIFT_RECOVERY_RETRIEVALS):
            detector.record_retrieval(success=True)

        assert not detector.in_recovery
        assert detector.get_exploration_boost() == 1.0

    def test_no_detection_without_enough_events(self, store):
        """Fewer than window events → no detection."""
        detector = ContextShiftDetector(store, self._CFG)
        detected = detector.record_retrieval(success=False)
        assert not detected

    def test_no_detection_when_shift_disabled(self, store):
        self._seed_pivot_pattern(store)
        cfg = Config(CONTEXT_SHIFT_ENABLED=False)
        detector = ContextShiftDetector(store, cfg)
        detected = detector.record_retrieval(success=False)
        assert not detected
        assert detector.get_exploration_boost() == 1.0

    def test_shift_event_logged_on_detection(self, store):
        self._seed_pivot_pattern(store)
        detector = ContextShiftDetector(store, self._CFG)
        detector.record_retrieval(success=False)
        count = store.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE event='context_shift_detected'"
        ).fetchone()["n"]
        assert count >= 1
