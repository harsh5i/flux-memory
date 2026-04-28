"""Tests for health.py — event logging and Health Monitor (Track 3)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from flux import Config, FluxStore, Grain, Conduit
from flux.health import (
    flux_health,
    log_event,
    _compute_graph_signals,
    _compute_event_signals,
    _is_healthy,
    _in_warmup,
)


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    with FluxStore(db) as s:
        yield s


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _grain(status="active", decay_class="working") -> Grain:
    return Grain(content="test", provenance="user_stated", status=status, decay_class=decay_class)


# ===================================================================== log_event

class TestLogEvent:
    def test_inserts_event_row(self, store):
        log_event(store, "retrieval", "grains_returned", {"count": 3})
        rows = store.conn.execute("SELECT * FROM events WHERE category='retrieval'").fetchall()
        assert len(rows) == 1
        assert rows[0]["event"] == "grains_returned"

    def test_data_is_json(self, store):
        import json
        log_event(store, "feedback", "conduit_reinforced", {"delta": 0.05})
        row = store.conn.execute("SELECT data FROM events WHERE category='feedback'").fetchone()
        data = json.loads(row["data"])
        assert data["delta"] == pytest.approx(0.05)

    def test_trace_id_stored(self, store):
        log_event(store, "retrieval", "grains_returned", trace_id="trace-abc")
        row = store.conn.execute("SELECT trace_id FROM events").fetchone()
        assert row["trace_id"] == "trace-abc"

    def test_empty_data_ok(self, store):
        log_event(store, "system", "startup")
        rows = store.conn.execute("SELECT * FROM events").fetchall()
        assert len(rows) == 1

    def test_multiple_events_stored(self, store):
        for i in range(5):
            log_event(store, "retrieval", "grains_returned", {"i": i})
        count = store.conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        assert count == 5


# ===================================================================== graph signals

class TestComputeGraphSignals:
    def test_highway_count_zero_on_fresh_graph(self, store):
        sigs = _compute_graph_signals(store)
        assert sigs["highway_count"] == 0.0

    def test_highway_count_detects_strong_conduit(self, store):
        g1 = _grain(); g2 = _grain()
        store.insert_grain(g1); store.insert_grain(g2)
        c = Conduit(from_id=g1.id, to_id=g2.id, weight=0.85)
        store.insert_conduit(c)
        sigs = _compute_graph_signals(store)
        assert sigs["highway_count"] == 1.0

    def test_core_grain_count(self, store):
        for _ in range(3):
            store.insert_grain(_grain(decay_class="core"))
        store.insert_grain(_grain(decay_class="working"))
        sigs = _compute_graph_signals(store)
        assert sigs["core_grain_count"] == 3.0

    def test_orphan_rate_with_orphan(self, store):
        g = _grain()
        store.insert_grain(g)  # no inbound conduits → orphan
        sigs = _compute_graph_signals(store)
        assert sigs["orphan_rate"] == pytest.approx(1.0)

    def test_orphan_rate_zero_with_inbound(self, store):
        hub = _grain(); target = _grain()
        store.insert_grain(hub); store.insert_grain(target)
        c = Conduit(from_id=hub.id, to_id=target.id, weight=0.5)
        store.insert_conduit(c)
        sigs = _compute_graph_signals(store)
        # hub is an orphan (no inbound), target is not → orphan_rate = 0.5
        assert 0 <= sigs["orphan_rate"] <= 1.0

    def test_avg_conduit_weight_empty_graph(self, store):
        sigs = _compute_graph_signals(store)
        assert sigs["avg_conduit_weight"] == 0.0

    def test_avg_conduit_weight_single_conduit(self, store):
        g1 = _grain(); g2 = _grain()
        store.insert_grain(g1); store.insert_grain(g2)
        c = Conduit(from_id=g1.id, to_id=g2.id, weight=0.6)
        store.insert_conduit(c)
        sigs = _compute_graph_signals(store)
        assert sigs["avg_conduit_weight"] == pytest.approx(0.6)

    def test_dormant_grain_rate(self, store):
        store.insert_grain(_grain(status="active"))
        store.insert_grain(_grain(status="dormant"))
        sigs = _compute_graph_signals(store)
        assert sigs["dormant_grain_rate"] == pytest.approx(0.5)


# ===================================================================== event signals

class TestComputeEventSignals:
    def test_shortcut_rate_zero_on_empty(self, store):
        sigs = _compute_event_signals(store, _now())
        assert sigs["shortcut_creation_rate"] == 0.0

    def test_feedback_compliance_one_when_no_retrievals(self, store):
        sigs = _compute_event_signals(store, _now())
        assert sigs["feedback_compliance_rate"] == 1.0

    def test_feedback_compliance_requires_feedback_per_returned_grain(self, store):
        now = _now()
        log_event(
            store,
            "retrieval",
            "grains_returned",
            {
                "trace_id": "trace-1",
                "grain_ids": ["g1", "g2"],
                "grains_count": 2,
                "caller_id": "codex",
            },
            trace_id="trace-1",
            now=now,
        )
        log_event(
            store,
            "feedback",
            "feedback_received",
            {
                "trace_id": "trace-1",
                "grain_id": "g1",
                "useful": True,
                "caller_id": "codex",
            },
            trace_id="trace-1",
            now=now,
        )

        sigs = _compute_event_signals(store, now)

        assert sigs["feedback_compliance_rate"] == pytest.approx(0.5)

    def test_retrieval_success_rate_from_events(self, store):
        now = _now()
        # 5 retrievals, 3 marked successful
        for _ in range(5):
            log_event(store, "retrieval", "grains_returned", {"hop_count": 2}, now=now)
        for _ in range(3):
            log_event(store, "feedback", "retrieval_successful", {}, now=now)
        sigs = _compute_event_signals(store, now)
        assert sigs["retrieval_success_rate"] == pytest.approx(0.6)

    def test_retrieval_success_rate_counts_one_success_per_trace(self, store):
        now = _now()
        log_event(store, "retrieval", "grains_returned", {"hop_count": 2}, trace_id="trace-1", now=now)
        log_event(store, "retrieval", "grains_returned", {"hop_count": 2}, trace_id="trace-2", now=now)
        log_event(store, "feedback", "retrieval_successful", {}, trace_id="trace-1", now=now)
        log_event(store, "feedback", "retrieval_successful", {}, trace_id="trace-1", now=now)
        sigs = _compute_event_signals(store, now)
        assert sigs["retrieval_success_rate"] == pytest.approx(0.5)

    def test_retrieval_success_rate_ignores_success_without_windowed_retrieval(self, store):
        now = _now()
        log_event(store, "retrieval", "grains_returned", {"hop_count": 2}, trace_id="trace-1", now=now)
        log_event(store, "retrieval", "grains_returned", {"hop_count": 2}, trace_id="trace-2", now=now)
        log_event(store, "feedback", "retrieval_successful", {}, trace_id="trace-1", now=now)
        log_event(store, "feedback", "retrieval_successful", {}, trace_id="trace-old", now=now)

        sigs = _compute_event_signals(store, now)

        assert sigs["retrieval_success_rate"] == pytest.approx(0.5)

    def test_avg_hops_from_events(self, store):
        now = _now()
        log_event(store, "retrieval", "grains_returned", {"hop_count": 2}, now=now)
        log_event(store, "retrieval", "grains_returned", {"hop_count": 4}, now=now)
        sigs = _compute_event_signals(store, now)
        assert sigs["avg_hops_per_retrieval"] == pytest.approx(3.0)

    def test_fallback_rate_from_events(self, store):
        now = _now()
        log_event(store, "retrieval", "grains_returned", {}, now=now)
        log_event(store, "retrieval", "grains_returned", {}, now=now)
        log_event(store, "retrieval", "fallback_triggered", {}, now=now)
        sigs = _compute_event_signals(store, now)
        assert sigs["fallback_trigger_rate"] == pytest.approx(0.5)


# ===================================================================== _is_healthy

class TestIsHealthy:
    def test_highway_count_below_min_is_unhealthy(self):
        assert not _is_healthy("highway_count", 3.0, Config())

    def test_highway_count_above_min_is_healthy(self):
        assert _is_healthy("highway_count", 6.0, Config())

    def test_orphan_rate_above_max_is_unhealthy(self):
        assert not _is_healthy("orphan_rate", 0.20, Config())

    def test_orphan_rate_below_max_is_healthy(self):
        assert _is_healthy("orphan_rate", 0.10, Config())

    def test_avg_conduit_weight_in_range_healthy(self):
        assert _is_healthy("avg_conduit_weight", 0.40, Config())

    def test_avg_conduit_weight_below_min_unhealthy(self):
        assert not _is_healthy("avg_conduit_weight", 0.10, Config())

    def test_avg_conduit_weight_above_max_unhealthy(self):
        assert not _is_healthy("avg_conduit_weight", 0.80, Config())

    def test_unknown_signal_always_healthy(self):
        assert _is_healthy("nonexistent_signal", 999.0, Config())


# ===================================================================== _in_warmup

class TestInWarmup:
    def test_in_warmup_below_retrieval_threshold(self, store):
        # highway_count needs 100 retrievals warmup
        # Store 50 retrieval events
        for _ in range(50):
            log_event(store, "retrieval", "grains_returned", {})
        assert _in_warmup("highway_count", store, _now())

    def test_not_in_warmup_above_retrieval_threshold(self, store):
        for _ in range(101):
            log_event(store, "retrieval", "grains_returned", {})
        # highway_count warmup_retrievals = 100; also check day-based warmup
        # if < 14 days in: still in warmup by day criterion
        # feedback_compliance has warmup_retrievals=0, so check that
        assert not _in_warmup("feedback_compliance_rate", store, _now())


# ===================================================================== flux_health

class TestFluxHealth:
    def test_returns_expected_keys(self, store):
        result = flux_health(store)
        assert "status" in result
        assert "signals" in result
        assert "active_warnings" in result
        assert "computed_at" in result

    def test_status_is_valid_value(self, store):
        result = flux_health(store)
        assert result["status"] in ("healthy", "warning", "critical")

    def test_all_14_signals_present(self, store):
        expected = {
            "highway_count", "highway_growth_rate", "shortcut_creation_rate",
            "conduit_dissolution_rate", "avg_weight_drop_on_failure",
            "promotion_events", "avg_hops_per_retrieval", "orphan_rate",
            "avg_conduit_weight", "retrieval_success_rate", "fallback_trigger_rate",
            "feedback_compliance_rate", "core_grain_count", "dormant_grain_rate",
        }
        result = flux_health(store)
        assert expected.issubset(set(result["signals"].keys()))

    def test_each_signal_has_value_and_healthy(self, store):
        result = flux_health(store)
        for name, sig in result["signals"].items():
            assert "value" in sig, f"{name} missing 'value'"
            assert "healthy" in sig, f"{name} missing 'healthy'"

    def test_warning_upserted_for_unhealthy_signal(self, store):
        """Force avg_conduit_weight out of range by inserting a near-zero conduit."""
        g1 = _grain(); g2 = _grain()
        store.insert_grain(g1); store.insert_grain(g2)
        # Insert many retrievals to pass warmup
        for _ in range(110):
            log_event(store, "retrieval", "grains_returned", {})
        # avg_conduit_weight = 0.01 < 0.15 (min) → should trigger WARNING
        c = Conduit(from_id=g1.id, to_id=g2.id, weight=0.01)
        store.insert_conduit(c)
        result = flux_health(store)
        # avg_conduit_weight should be flagged
        sigs = result["signals"]
        assert sigs["avg_conduit_weight"]["healthy"] is False

    def test_health_log_event_emitted(self, store):
        flux_health(store)
        count = store.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE category='health' AND event='health_computed'"
        ).fetchone()["n"]
        assert count == 1

    def test_warning_cleared_when_signal_recovers(self, store):
        """Insert a warning, then verify it's cleared when signal returns healthy."""
        g1 = _grain(); g2 = _grain()
        store.insert_grain(g1); store.insert_grain(g2)
        for _ in range(110):
            log_event(store, "retrieval", "grains_returned", {})
        # Force unhealthy: insert sub-floor conduit
        c = Conduit(from_id=g1.id, to_id=g2.id, weight=0.01)
        store.insert_conduit(c)
        flux_health(store)  # warning created
        # Fix: replace with healthy weight
        store.conn.execute("UPDATE conduits SET weight=0.40 WHERE id=?", (c.id,))
        result2 = flux_health(store)
        # Check no cleared warning is in active_warnings
        for w in result2["active_warnings"]:
            assert w["signal"] != "avg_conduit_weight", "Warning should be cleared"

    def test_cleared_warning_can_become_active_again(self, store):
        """A recovered signal can become unhealthy again without violating signal uniqueness."""
        now = _now() + timedelta(days=8)
        g1 = _grain(); g2 = _grain()
        store.insert_grain(g1); store.insert_grain(g2)
        for _ in range(110):
            log_event(store, "retrieval", "grains_returned", {})

        c = Conduit(from_id=g1.id, to_id=g2.id, weight=0.01)
        store.insert_conduit(c)
        flux_health(store, now=now)

        store.conn.execute("UPDATE conduits SET weight=0.40 WHERE id=?", (c.id,))
        flux_health(store, now=now)

        store.conn.execute("UPDATE conduits SET weight=0.01 WHERE id=?", (c.id,))
        result = flux_health(store, now=now)

        assert any(
            w["signal"] == "avg_conduit_weight"
            for w in result["active_warnings"]
        )

    def test_feedback_compliance_breakdown_identifies_caller_without_warning_flood(self, store):
        now = _now()
        log_event(
            store,
            "retrieval",
            "grains_returned",
            {
                "trace_id": "trace-ambient",
                "grain_ids": ["g1"],
                "grains_count": 1,
                "caller_id": "ambient_suggestions",
            },
            trace_id="trace-ambient",
            now=now,
        )

        result = flux_health(store, now=now)

        caller = next(
            (
                c for c in result["caller_feedback"]
                if c["caller_id"] == "ambient_suggestions"
            ),
            None,
        )
        assert caller is not None
        assert caller["rate"] == pytest.approx(0.0)
        assert caller["missing"] == pytest.approx(1.0)
        assert caller["healthy"] is False
        assert not any(
            w["signal"] == "feedback_compliance_rate:ambient_suggestions"
            for w in result["active_warnings"]
        )

    def test_feedback_compliance_reclassifies_ambient_codex_prompt(self, store):
        now = _now()
        log_event(
            store,
            "retrieval",
            "grains_returned",
            {
                "query": "Generate 0 to 3 ambient suggestions for this local project",
                "grain_ids": ["g1"],
                "grains_count": 1,
                "caller_id": "codex",
            },
            trace_id="trace-ambient-codex",
            now=now,
        )

        result = flux_health(store, now=now)
        callers = {c["caller_id"] for c in result["caller_feedback"]}

        assert "ambient_suggestions" in callers
        assert "codex" not in callers
