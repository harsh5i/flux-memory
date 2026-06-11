"""Tests for store-time hygiene: honest store status, dedup gate, and
health warning alerts (transitions + throttle)."""
from __future__ import annotations

from datetime import timedelta

import pytest
from mocks import MockEmbeddingBackend

import flux.alerts as alerts
from flux.alerts import maybe_send_warning_alert
from flux.config import Config
from flux.graph import utcnow
from flux.health import _upsert_warning, _clear_warning
from flux.retrieval import flux_store_ex


class FailingEmbeddingBackend:
    model_name = "failing-embedding"

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding backend down")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding backend down")


# ------------------------------------------------------------- store status

def test_store_with_embedder_returns_wired(store):
    grain_id, status = flux_store_ex(
        "The sky is blue.", store=store, llm=None, emb=MockEmbeddingBackend(),
    )
    assert status == "stored_wired"
    row = store.conn.execute(
        "SELECT 1 FROM grain_embeddings WHERE grain_id = ?", (grain_id,),
    ).fetchone()
    assert row is not None


def test_store_without_embedder_returns_bare(store):
    grain_id, status = flux_store_ex("The sky is blue.", store=store)
    assert status == "stored_bare"
    row = store.conn.execute(
        "SELECT 1 FROM grain_embeddings WHERE grain_id = ?", (grain_id,),
    ).fetchone()
    assert row is None


def test_store_with_failing_embedder_returns_bare_but_stores(store):
    grain_id, status = flux_store_ex(
        "The sky is blue.", store=store, llm=None, emb=FailingEmbeddingBackend(),
    )
    assert status == "stored_bare"
    row = store.conn.execute(
        "SELECT 1 FROM grains WHERE id = ?", (grain_id,),
    ).fetchone()
    assert row is not None


# ------------------------------------------------------------------- dedup

def test_duplicate_content_returns_existing_grain(store):
    emb = MockEmbeddingBackend()
    first_id, first_status = flux_store_ex(
        "Paris is the capital of France.", store=store, llm=None, emb=emb,
    )
    second_id, second_status = flux_store_ex(
        "Paris is the capital of France.", store=store, llm=None, emb=emb,
    )
    assert first_status == "stored_wired"
    assert second_status == "duplicate"
    assert second_id == first_id
    count = store.conn.execute("SELECT COUNT(*) FROM grains").fetchone()[0]
    assert count == 1


def test_distinct_content_stores_separately(store):
    emb = MockEmbeddingBackend()
    a, status_a = flux_store_ex(
        "Paris is the capital of France.", store=store, llm=None, emb=emb,
    )
    b, status_b = flux_store_ex(
        "Olive oil smoke point is around 190C.", store=store, llm=None, emb=emb,
    )
    assert a != b
    assert status_a == status_b == "stored_wired"


def test_dedup_disabled_with_threshold_above_one(store):
    emb = MockEmbeddingBackend()
    cfg = Config(DEDUP_SIMILARITY_THRESHOLD=1.01)
    a, _ = flux_store_ex(
        "Paris is the capital of France.", store=store, llm=None, emb=emb, cfg=cfg,
    )
    b, status_b = flux_store_ex(
        "Paris is the capital of France.", store=store, llm=None, emb=emb, cfg=cfg,
    )
    assert a != b
    assert status_b == "stored_wired"


def test_dedup_logs_event(store):
    emb = MockEmbeddingBackend()
    flux_store_ex("Paris is the capital of France.", store=store, llm=None, emb=emb)
    flux_store_ex("Paris is the capital of France.", store=store, llm=None, emb=emb)
    row = store.conn.execute(
        "SELECT 1 FROM events WHERE event = 'grain_deduplicated'",
    ).fetchone()
    assert row is not None


# ------------------------------------------------------------------- alerts

ALERT_CFG = Config(
    ALERTS_ENABLED=True,
    ALERT_TELEGRAM_BOT_TOKEN="test-token",
    ALERT_TELEGRAM_CHAT_ID="12345",
)


@pytest.fixture
def sent(monkeypatch):
    calls: list[str] = []

    def fake_send(cfg, text):
        calls.append(text)
        return True

    monkeypatch.setattr(alerts, "_send_telegram", fake_send)
    return calls


def _warn(store, now, signal="orphan_rate"):
    return _upsert_warning(
        store, signal=signal, value=0.5, severity="WARNING",
        healthy_range="0 - 0.15", suggestion="check bootstrap", now=now,
    )


def test_upsert_warning_transitions(store):
    now = utcnow()
    assert _warn(store, now) == "new"
    assert _warn(store, now) == "ongoing"
    _clear_warning(store, "orphan_rate", now)
    assert _warn(store, now) == "refired"


def test_alert_sent_on_new_warning(store, sent):
    now = utcnow()
    ok = maybe_send_warning_alert(
        store, ALERT_CFG, signal="orphan_rate", severity="WARNING", value=0.5,
        healthy_range="0 - 0.15", suggestion="check bootstrap",
        transition="new", now=now,
    )
    assert ok
    assert len(sent) == 1
    assert "orphan_rate" in sent[0]


def test_no_alert_on_ongoing_warning(store, sent):
    ok = maybe_send_warning_alert(
        store, ALERT_CFG, signal="orphan_rate", severity="WARNING", value=0.5,
        healthy_range="0 - 0.15", suggestion="", transition="ongoing", now=utcnow(),
    )
    assert not ok
    assert sent == []


def test_no_alert_when_disabled(store, sent):
    cfg = Config(ALERT_TELEGRAM_BOT_TOKEN="t", ALERT_TELEGRAM_CHAT_ID="c")
    ok = maybe_send_warning_alert(
        store, cfg, signal="orphan_rate", severity="WARNING", value=0.5,
        healthy_range="0 - 0.15", suggestion="", transition="new", now=utcnow(),
    )
    assert not ok
    assert sent == []


def test_alert_throttled_within_interval(store, sent):
    now = utcnow()
    kwargs = dict(
        signal="orphan_rate", severity="WARNING", value=0.5,
        healthy_range="0 - 0.15", suggestion="",
    )
    assert maybe_send_warning_alert(store, ALERT_CFG, transition="new", now=now, **kwargs)
    # Re-fires 1 hour later: throttled (interval is 6h).
    assert not maybe_send_warning_alert(
        store, ALERT_CFG, transition="refired", now=now + timedelta(hours=1), **kwargs,
    )
    # Re-fires past the interval: sent again.
    assert maybe_send_warning_alert(
        store, ALERT_CFG, transition="refired", now=now + timedelta(hours=7), **kwargs,
    )
    assert len(sent) == 2


def test_alert_failure_does_not_record_sent(store, monkeypatch):
    monkeypatch.setattr(alerts, "_send_telegram", lambda cfg, text: False)
    now = utcnow()
    ok = maybe_send_warning_alert(
        store, ALERT_CFG, signal="orphan_rate", severity="WARNING", value=0.5,
        healthy_range="0 - 0.15", suggestion="", transition="new", now=now,
    )
    assert not ok
    row = store.conn.execute(
        "SELECT 1 FROM meta WHERE key = 'alert_last_sent:orphan_rate'",
    ).fetchone()
    assert row is None
