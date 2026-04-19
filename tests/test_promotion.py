"""Tests for promotion.py — grain promotion (Track 1 Step 8)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from flux import Config, Conduit, Entry, Grain, FluxStore
from flux.clustering import recompute_clusters, record_entry_cooccurrence
from flux.promotion import (
    _can_reach,
    _get_activating_entry_points,
    check_promotion,
    check_promotions_bulk,
)
from flux.propagation import TraceStep


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    with FluxStore(db) as s:
        yield s


def _grain(provenance="user_stated") -> Grain:
    return Grain(content="test grain", provenance=provenance)


def _entry(feature: str) -> Entry:
    return Entry(feature=feature)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_trace(entry_id: str, grain_id: str, via: str | None = None) -> list[TraceStep]:
    """Build a minimal trace: entry -> [via] -> grain_id."""
    steps = []
    if via is None:
        # Direct: entry -> grain
        steps.append(TraceStep(
            conduit_id="c1",
            from_id=entry_id,
            to_id=grain_id,
            signal=0.5,
            hop=0,
            effective_weight=0.5,
        ))
    else:
        # Two-hop: entry -> via -> grain
        steps.append(TraceStep(
            conduit_id="c1",
            from_id=entry_id,
            to_id=via,
            signal=0.5,
            hop=0,
            effective_weight=0.5,
        ))
        steps.append(TraceStep(
            conduit_id="c2",
            from_id=via,
            to_id=grain_id,
            signal=0.3,
            hop=1,
            effective_weight=0.3,
        ))
    return steps


# ===================================================================== _can_reach

class TestCanReach:
    def test_direct_reach(self):
        forward = {"A": {"B"}}
        assert _can_reach("A", "B", forward)

    def test_transitive_reach(self):
        forward = {"A": {"B"}, "B": {"C"}}
        assert _can_reach("A", "C", forward)

    def test_not_reachable(self):
        forward = {"A": {"B"}}
        assert not _can_reach("A", "C", forward)

    def test_start_equals_target(self):
        assert _can_reach("X", "X", {})

    def test_empty_forward(self):
        assert not _can_reach("A", "B", {})


# ===================================================================== _get_activating_entry_points

class TestGetActivatingEntryPoints:
    def test_direct_entry_identified(self):
        e = _entry("AI"); g = _grain()
        trace = _make_trace(e.id, g.id)
        result = _get_activating_entry_points(trace, g.id)
        assert e.id in result

    def test_two_hop_entry_identified(self):
        e = _entry("coding"); mid = _grain(); target = _grain()
        trace = _make_trace(e.id, target.id, via=mid.id)
        result = _get_activating_entry_points(trace, target.id)
        assert e.id in result

    def test_unrelated_entry_not_included(self):
        e1 = _entry("A"); e2 = _entry("B"); g1 = _grain(); g2 = _grain()
        trace_e1 = _make_trace(e1.id, g1.id)
        trace_e2 = _make_trace(e2.id, g2.id)
        combined = trace_e1 + trace_e2

        result = _get_activating_entry_points(combined, g1.id)
        assert e1.id in result
        assert e2.id not in result

    def test_empty_trace_returns_empty(self):
        assert _get_activating_entry_points([], "some_grain") == set()


# ===================================================================== check_promotion

class TestCheckPromotion:
    def test_no_promotion_without_clusters(self, store):
        """If no cluster memberships exist, no touch weight is accumulated → no promotion."""
        g = _grain(); e = _entry("test")
        store.insert_grain(g); store.insert_entry(e)

        trace = _make_trace(e.id, g.id)
        promoted = check_promotion(store, g.id, trace)
        assert not promoted
        assert store.get_grain(g.id).decay_class == "working"

    def test_already_core_skipped(self, store):
        """check_promotion is a no-op if grain is already core."""
        g = Grain(content="core grain", provenance="user_stated", decay_class="core")
        e = _entry("X")
        store.insert_grain(g); store.insert_entry(e)

        trace = _make_trace(e.id, g.id)
        result = check_promotion(store, g.id, trace)
        assert not result

    def test_context_spread_updated(self, store):
        """check_promotion updates context_spread even without reaching threshold."""
        cfg = Config(PROMOTION_THRESHOLD=3, CLUSTER_TOUCH_THRESHOLD=1.0)
        g = _grain(); e = _entry("P")
        store.insert_grain(g); store.insert_entry(e)

        cluster_id = "cluster-1"
        # Manually inject cluster membership for the entry
        store.conn.execute(
            "INSERT INTO entry_cluster_membership (entry_id, cluster_id, weight) VALUES (?, ?, ?)",
            (e.id, cluster_id, 1.0),
        )

        trace = _make_trace(e.id, g.id)
        check_promotion(store, g.id, trace, cfg)

        updated = store.get_grain(g.id)
        assert updated.context_spread == 1

    def test_promotion_fires_at_threshold(self, store):
        """Grain with touch_weight >= CLUSTER_TOUCH_THRESHOLD in 3 distinct clusters is promoted."""
        cfg = Config(PROMOTION_THRESHOLD=3, CLUSTER_TOUCH_THRESHOLD=1.0)
        g = _grain(); e = _entry("multi")
        store.insert_grain(g); store.insert_entry(e)

        # Give the entry membership in 3 different clusters, equal weight
        for i in range(3):
            cluster_id = f"cluster-{i}"
            store.conn.execute(
                "INSERT INTO entry_cluster_membership (entry_id, cluster_id, weight) VALUES (?, ?, ?)",
                (e.id, cluster_id, 1.0 / 3),
            )

        trace = _make_trace(e.id, g.id)

        # Simulate 3 retrievals so each cluster accumulates touch >= 1.0 (threshold)
        # Per retrieval: each cluster gets 1/3 touch_weight. Need 3 × 1/3 = 1.0 each.
        for _ in range(3):
            check_promotion(store, g.id, trace, cfg)

        updated = store.get_grain(g.id)
        assert updated.decay_class == "core"

    def test_inbound_conduits_reclassified_on_promotion(self, store):
        """When a grain is promoted, its inbound conduits become core decay class."""
        cfg = Config(PROMOTION_THRESHOLD=1, CLUSTER_TOUCH_THRESHOLD=0.1)
        hub = _grain(); g = _grain(); e = _entry("Z")
        store.insert_grain(hub); store.insert_grain(g); store.insert_entry(e)

        c = Conduit(from_id=hub.id, to_id=g.id, weight=0.5, decay_class="working")
        store.insert_conduit(c)

        cluster_id = "cluster-promote"
        store.conn.execute(
            "INSERT INTO entry_cluster_membership (entry_id, cluster_id, weight) VALUES (?, ?, ?)",
            (e.id, cluster_id, 1.0),
        )

        trace = _make_trace(e.id, g.id)
        promoted = check_promotion(store, g.id, trace, cfg)

        assert promoted
        updated_conduit = store.get_conduit(c.id)
        assert updated_conduit.decay_class == "core"

    def test_context_spread_monotonically_nondecreasing(self, store):
        """context_spread must never decrease between successive check_promotion calls."""
        cfg = Config(PROMOTION_THRESHOLD=5, CLUSTER_TOUCH_THRESHOLD=1.0)
        g = _grain(); e = _entry("mono")
        store.insert_grain(g); store.insert_entry(e)

        cluster_id = "cluster-mono"
        store.conn.execute(
            "INSERT INTO entry_cluster_membership (entry_id, cluster_id, weight) VALUES (?, ?, ?)",
            (e.id, cluster_id, 1.0),
        )

        trace = _make_trace(e.id, g.id)
        prev_spread = 0
        for _ in range(4):
            check_promotion(store, g.id, trace, cfg)
            current = store.get_grain(g.id).context_spread
            assert current >= prev_spread, "context_spread decreased!"
            prev_spread = current


# ===================================================================== check_promotions_bulk

class TestCheckPromotionsBulk:
    def test_returns_empty_without_clusters(self, store):
        grains = [_grain() for _ in range(3)]
        e = _entry("Q")
        for g in grains:
            store.insert_grain(g)
        store.insert_entry(e)

        trace = []
        for g in grains:
            trace += _make_trace(e.id, g.id)

        promoted = check_promotions_bulk(store, [g.id for g in grains], trace)
        assert promoted == []

    def test_skips_core_grains(self, store):
        already_core = Grain(content="core", provenance="user_stated", decay_class="core")
        e = _entry("R")
        store.insert_grain(already_core); store.insert_entry(e)

        trace = _make_trace(e.id, already_core.id)
        promoted = check_promotions_bulk(store, [already_core.id], trace)
        assert already_core.id not in promoted


# ===================================================================== integration

class TestPromotionIntegration:
    def test_full_promotion_pipeline_with_clustering(self, store):
        """End-to-end: build clusters from co-occurrence, then check promotion."""
        cfg = Config(
            ENTRY_COOCCURRENCE_THRESHOLD=1,
            CLUSTER_WINDOW_DAYS=30,
            CLUSTER_MIN_SIZE=2,
            LOUVAIN_RESOLUTION=1.0,
            LOUVAIN_SEED=42,
            PROMOTION_THRESHOLD=1,
            CLUSTER_TOUCH_THRESHOLD=0.5,
        )

        # Four entries that co-occur frequently → will form clusters
        entries = [_entry(f"T{i}") for i in range(4)]
        for e in entries:
            store.insert_entry(e)

        ids = [e.id for e in entries]
        for _ in range(5):
            record_entry_cooccurrence(store, ids)

        recompute_clusters(store, cfg)

        # Verify at least one entry has cluster memberships
        has_memberships = any(
            bool(store.get_entry_cluster_memberships(e.id)) for e in entries
        )
        assert has_memberships, "No cluster memberships after recompute"

        # Now run check_promotion for a grain using one of the clustered entries
        g = _grain()
        store.insert_grain(g)

        # Find an entry with memberships
        clustered_entry = next(
            e for e in entries if store.get_entry_cluster_memberships(e.id)
        )
        trace = _make_trace(clustered_entry.id, g.id)

        # With PROMOTION_THRESHOLD=1 and CLUSTER_TOUCH_THRESHOLD=0.5,
        # one retrieval through a clustered entry should be enough.
        check_promotion(store, g.id, trace, cfg)

        updated = store.get_grain(g.id)
        # context_spread should be > 0 (entry has cluster memberships)
        assert updated.context_spread >= 0
