"""Tests for FluxService booth architecture (§1A.7, §1A.7a)."""
from __future__ import annotations

import dataclasses
import json
import queue
import threading
import time

import pytest

from flux.config import Config
from flux.service import FluxService, _SlidingWindowRateLimiter
from flux.storage import FluxStore

from mocks import MockEmbeddingBackend, MockLLMBackend


@pytest.fixture
def cfg():
    return dataclasses.replace(
        Config(),
        OPERATING_MODE="caller_extracts",
        READ_WORKERS=2,
        MAX_GRAINS_PER_CALL=5,
        MAX_WRITE_QUEUE_DEPTH=10,
        MAX_GRAINS_PER_MINUTE=20,
    )


@pytest.fixture
def store(tmp_path):
    s = FluxStore(tmp_path / "flux.db")
    yield s
    s.close()


@pytest.fixture
def svc(store, cfg):
    s = FluxService(store, MockLLMBackend(), MockEmbeddingBackend(), cfg)
    s.start()
    yield s
    s.stop()


# ---------------------------------------------------------------- rate limiter

class TestSlidingWindowRateLimiter:
    def test_allows_under_limit(self):
        rl = _SlidingWindowRateLimiter(max_per_minute=10)
        assert rl.check_and_record("a", count=5) is True
        assert rl.check_and_record("a", count=5) is True

    def test_blocks_over_limit(self):
        rl = _SlidingWindowRateLimiter(max_per_minute=5)
        assert rl.check_and_record("a", count=5) is True
        assert rl.check_and_record("a", count=1) is False

    def test_independent_per_caller(self):
        rl = _SlidingWindowRateLimiter(max_per_minute=5)
        assert rl.check_and_record("alice", count=5) is True
        assert rl.check_and_record("bob", count=5) is True   # bob has own window

    def test_block_does_not_record(self):
        rl = _SlidingWindowRateLimiter(max_per_minute=3)
        rl.check_and_record("x", count=3)
        assert rl.check_and_record("x", count=1) is False
        assert rl.check_and_record("x", count=1) is False  # still blocked


# ---------------------------------------------------------------- lifecycle

class TestFluxServiceLifecycle:
    def test_start_and_stop(self, store, cfg):
        svc = FluxService(store, MockLLMBackend(), MockEmbeddingBackend(), cfg)
        svc.start()
        assert svc._running is True
        svc.stop()
        assert svc._running is False

    def test_double_start_is_safe(self, svc):
        svc.start()  # already started
        assert svc._running is True

    def test_double_stop_is_safe(self, store, cfg):
        svc = FluxService(store, MockLLMBackend(), MockEmbeddingBackend(), cfg)
        svc.start()
        svc.stop()
        svc.stop()  # idempotent


# ---------------------------------------------------------------- store

class TestFluxServiceStore:
    def test_store_returns_grain_id(self, svc):
        gid = svc.store("Paris is the capital of France", caller_id="test")
        assert isinstance(gid, str)
        assert len(gid) > 0

    def test_store_empty_content_raises(self, svc):
        with pytest.raises(ValueError, match="empty"):
            svc.store("   ", caller_id="test")

    def test_store_rate_limit_raises(self, svc, cfg):
        for _ in range(cfg.MAX_GRAINS_PER_MINUTE):
            svc.store(f"fact {_}", caller_id="heavy")
        with pytest.raises(RuntimeError, match="Rate limit"):
            svc.store("one more", caller_id="heavy")

    def test_store_rate_limit_per_caller(self, svc, cfg):
        for _ in range(cfg.MAX_GRAINS_PER_MINUTE):
            svc.store(f"fact {_}", caller_id="alice")
        # bob is unaffected
        gid = svc.store("bob's fact", caller_id="bob")
        assert gid


# ---------------------------------------------------------------- batch

class TestFluxServiceBatch:
    def test_batch_store_returns_ids(self, svc):
        items = [{"content": f"fact {i}", "provenance": "user_stated"} for i in range(3)]
        ids = svc.store_batch(items, caller_id="test")
        assert len(ids) == 3
        assert all(isinstance(i, str) for i in ids)

    def test_batch_exceeds_cap_raises(self, svc, cfg):
        items = [{"content": f"f{i}"} for i in range(cfg.MAX_GRAINS_PER_CALL + 1)]
        with pytest.raises(ValueError, match="MAX_GRAINS_PER_CALL"):
            svc.store_batch(items, caller_id="test")

    def test_batch_at_cap_succeeds(self, svc, cfg):
        items = [{"content": f"f{i}"} for i in range(cfg.MAX_GRAINS_PER_CALL)]
        ids = svc.store_batch(items, caller_id="test")
        assert len(ids) == cfg.MAX_GRAINS_PER_CALL


# ---------------------------------------------------------------- retrieve

class TestFluxServiceRetrieve:
    def test_retrieve_returns_result(self, svc):
        svc.store("Dogs are mammals", caller_id="test")
        result = svc.retrieve("dogs", caller_id="test")
        assert hasattr(result, "grains")
        assert hasattr(result, "trace_id")
        assert isinstance(result.confidence, float)

    def test_retrieve_empty_store(self, svc):
        result = svc.retrieve("anything", caller_id="test")
        assert result.grains == []

    def test_retrieve_records_caller_id(self, svc, store):
        svc.retrieve("anything", caller_id="agent-retrieve")

        row = store.conn.execute(
            """
            SELECT data FROM events
            WHERE category='retrieval' AND event='grains_returned'
            ORDER BY timestamp DESC LIMIT 1
            """
        ).fetchone()

        assert json.loads(row["data"])["caller_id"] == "agent-retrieve:chat"


# ---------------------------------------------------------------- feedback

class TestFluxServiceFeedback:
    def test_feedback_async_does_not_raise(self, svc):
        svc.store("test fact", caller_id="test")
        result = svc.retrieve("test", caller_id="test")
        if result.grains:
            svc.feedback(result.trace_id, result.grains[0]["id"], True, caller_id="test")
        # non-blocking — no assertion needed, just no crash

    def test_feedback_sync_returns_result(self, svc):
        svc.store("test fact", caller_id="test")
        result = svc.retrieve("test fact", caller_id="test")
        if result.grains:
            fb = svc.feedback_sync(result.trace_id, result.grains[0]["id"], True)
            assert hasattr(fb, "trace_id")
            assert hasattr(fb, "action")

    def test_feedback_sync_records_caller_id(self, svc, store):
        svc.store("test fact", caller_id="test")
        result = svc.retrieve("test fact", caller_id="test")
        if not result.grains:
            pytest.skip("No grains retrieved")

        svc.feedback_sync(
            result.trace_id,
            result.grains[0]["id"],
            True,
            caller_id="agent-feedback",
        )
        row = store.conn.execute(
            """
            SELECT data FROM events
            WHERE category='feedback' AND event='feedback_received'
            ORDER BY timestamp DESC LIMIT 1
            """
        ).fetchone()

        assert json.loads(row["data"])["caller_id"] == "agent-feedback:chat"


# ---------------------------------------------------------------- health and list

class TestFluxServiceHealth:
    def test_health_returns_dict(self, svc):
        h = svc.health()
        assert isinstance(h, dict)
        assert "status" in h

    def test_list_grains_empty(self, svc):
        grains = svc.list_grains()
        assert isinstance(grains, list)

    def test_list_grains_after_store(self, svc):
        svc.store("observable fact", caller_id="test")
        grains = svc.list_grains()
        assert len(grains) >= 1
        assert "id" in grains[0]
        assert "content_snippet" in grains[0]
        assert "status" in grains[0]

    def test_list_grains_invalid_status_raises(self, svc):
        with pytest.raises(ValueError, match="Invalid status"):
            svc.list_grains(status="nonexistent")

    def test_list_grains_status_filter(self, svc):
        svc.store("active grain test", caller_id="test")
        active = svc.list_grains(status="active")
        assert all(g["status"] == "active" for g in active)


# ---------------------------------------------------------------- concurrency

class TestFluxServiceConcurrency:
    def test_concurrent_stores_all_succeed(self, svc):
        results = []
        errors = []

        def do_store(i):
            try:
                gid = svc.store(f"concurrent fact {i}", caller_id=f"caller{i % 3}")
                results.append(gid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_store, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(results) == 6

    def test_concurrent_retrieves_all_succeed(self, svc):
        svc.store("shared fact for retrieval", caller_id="setup")
        results = []

        def do_retrieve():
            r = svc.retrieve("shared fact", caller_id="test")
            results.append(r)

        threads = [threading.Thread(target=do_retrieve) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 4
