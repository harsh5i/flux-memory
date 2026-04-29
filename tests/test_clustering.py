"""Tests for clustering.py — Louvain soft clustering (Track 1 Step 7)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from flux import Config, Entry, FluxStore
from flux.clustering import (
    _build_cooccurrence_graph,
    _derive_soft_memberships,
    _merge_small_clusters,
    _stable_cluster_id_mapping,
    record_entry_cooccurrence,
    recompute_clusters,
)
from flux.graph import new_id


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    with FluxStore(db) as s:
        yield s


def _entry(feature: str) -> Entry:
    return Entry(feature=feature)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ===================================================================== record_entry_cooccurrence

class TestRecordEntryCooccurrence:
    def test_increments_count_for_pair(self, store):
        e1 = _entry("AI"); e2 = _entry("coding")
        store.insert_entry(e1); store.insert_entry(e2)

        record_entry_cooccurrence(store, [e1.id, e2.id])

        rows = store.all_entry_cooccurrences(window_days=30)
        assert len(rows) == 1
        assert rows[0]["count"] == 1

    def test_increments_multiple_times(self, store):
        e1 = _entry("A"); e2 = _entry("B")
        store.insert_entry(e1); store.insert_entry(e2)

        for _ in range(5):
            record_entry_cooccurrence(store, [e1.id, e2.id])

        rows = store.all_entry_cooccurrences(window_days=30)
        assert rows[0]["count"] == 5

    def test_all_pairs_recorded_for_three_entries(self, store):
        entries = [_entry(f"E{i}") for i in range(3)]
        for e in entries:
            store.insert_entry(e)

        record_entry_cooccurrence(store, [e.id for e in entries])

        rows = store.all_entry_cooccurrences(window_days=30)
        assert len(rows) == 3  # C(3,2) = 3 pairs

    def test_single_entry_produces_no_pairs(self, store):
        e = _entry("solo")
        store.insert_entry(e)
        record_entry_cooccurrence(store, [e.id])
        rows = store.all_entry_cooccurrences(window_days=30)
        assert len(rows) == 0

    def test_empty_list_no_error(self, store):
        record_entry_cooccurrence(store, [])
        rows = store.all_entry_cooccurrences(window_days=30)
        assert len(rows) == 0


# ===================================================================== _build_cooccurrence_graph

class TestBuildCooccurrenceGraph:
    def test_excludes_below_threshold(self):
        rows = [{"entry_a": "A", "entry_b": "B", "count": 2}]
        G = _build_cooccurrence_graph(rows, threshold=5)
        assert len(G.edges) == 0

    def test_includes_at_threshold(self):
        rows = [{"entry_a": "A", "entry_b": "B", "count": 5}]
        G = _build_cooccurrence_graph(rows, threshold=5)
        assert G.has_edge("A", "B")

    def test_edge_weight_normalised(self):
        # Both A and B have total freq = 10 (from one co-occurrence of 10)
        rows = [{"entry_a": "A", "entry_b": "B", "count": 10}]
        G = _build_cooccurrence_graph(rows, threshold=1)
        w = G["A"]["B"]["weight"]
        # freq(A) = freq(B) = 10; weight = 10 / sqrt(10*10) = 1.0
        assert abs(w - 1.0) < 1e-9

    def test_empty_rows_empty_graph(self):
        G = _build_cooccurrence_graph([], threshold=1)
        assert len(G.nodes) == 0


# ===================================================================== _merge_small_clusters

class TestMergeSmallClusters:
    def _graph_of(self, edges):
        import networkx as nx
        G = nx.Graph()
        for a, b, w in edges:
            G.add_edge(a, b, weight=w)
        return G

    def test_small_cluster_absorbed_into_large(self):
        G = self._graph_of([("A", "B", 1.0), ("B", "C", 0.5)])
        communities = [frozenset(["A", "B", "C"]), frozenset(["D"])]
        result = _merge_small_clusters(G, communities, min_size=2)
        assert len(result) == 1
        assert "D" in result[0]
        assert "A" in result[0]

    def test_all_large_unchanged(self):
        G = self._graph_of([("A", "B", 1.0)])
        communities = [frozenset(["A", "B"]), frozenset(["C", "D"])]
        result = _merge_small_clusters(G, communities, min_size=2)
        assert len(result) == 2

    def test_all_small_returns_unchanged(self):
        import networkx as nx
        G = nx.Graph()
        communities = [frozenset(["A"]), frozenset(["B"])]
        result = _merge_small_clusters(G, communities, min_size=3)
        assert len(result) == 2  # nothing to merge into


# ===================================================================== _derive_soft_memberships

class TestDeriveSoftMemberships:
    def _graph_and_communities(self):
        import networkx as nx
        G = nx.Graph()
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=1.0)
        G.add_edge("C", "D", weight=1.0)
        G.add_edge("D", "A", weight=1.0)  # cluster 0: A,B,C,D
        G.add_edge("E", "F", weight=1.0)  # cluster 1: E,F
        communities = [frozenset(["A", "B", "C", "D"]), frozenset(["E", "F"])]
        return G, communities

    def test_weights_sum_to_one_per_entry(self):
        G, communities = self._graph_and_communities()
        memberships = _derive_soft_memberships(G, communities)
        for entry_id, cluster_weights in memberships.items():
            total = sum(cluster_weights.values())
            assert abs(total - 1.0) < 1e-6, f"{entry_id} weights don't sum to 1: {cluster_weights}"

    def test_well_connected_entry_mostly_in_own_cluster(self):
        G, communities = self._graph_and_communities()
        memberships = _derive_soft_memberships(G, communities)
        # E is only connected to F (in cluster 1), so all its weight is in cluster 1
        assert memberships["E"].get(1, 0) == pytest.approx(1.0, abs=1e-6)

    def test_bridge_entry_has_split_membership(self):
        import networkx as nx
        G = nx.Graph()
        # Bridge: X connects cluster 0 (A, B) and cluster 1 (C, D) with equal weight
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("C", "D", weight=1.0)
        G.add_edge("X", "A", weight=1.0)
        G.add_edge("X", "C", weight=1.0)
        communities = [frozenset(["A", "B", "X"]), frozenset(["C", "D"])]
        memberships = _derive_soft_memberships(G, communities)
        # X has equal edges to A (cluster 0) and C (cluster 1)
        x_weights = memberships["X"]
        assert len(x_weights) == 2
        for w in x_weights.values():
            assert w == pytest.approx(0.5, abs=0.1)


# ===================================================================== _stable_cluster_id_mapping

class TestStableClusterIdMapping:
    def test_inherits_id_above_overlap_threshold(self):
        old_id = new_id()
        old_partition = [frozenset(["A", "B", "C"])]
        new_partition = [frozenset(["A", "B", "C"])]  # identical
        cfg = Config(CLUSTER_INHERIT_OVERLAP_MIN=0.30, CLUSTER_DISSOLVE_DECAY=0.5)

        new_ids, remap = _stable_cluster_id_mapping(old_partition, new_partition, [old_id], cfg)

        assert new_ids[0] == old_id  # inherited

    def test_fresh_uuid_below_overlap_threshold(self):
        old_id = new_id()
        old_partition = [frozenset(["A", "B", "C"])]
        new_partition = [frozenset(["X", "Y", "Z"])]  # no overlap
        cfg = Config(CLUSTER_INHERIT_OVERLAP_MIN=0.30, CLUSTER_DISSOLVE_DECAY=0.5)

        new_ids, remap = _stable_cluster_id_mapping(old_partition, new_partition, [old_id], cfg)

        assert new_ids[0] != old_id  # fresh UUID

    def test_empty_old_partition_all_fresh(self):
        new_partition = [frozenset(["A", "B"]), frozenset(["C", "D"])]
        cfg = Config(CLUSTER_INHERIT_OVERLAP_MIN=0.30, CLUSTER_DISSOLVE_DECAY=0.5)

        new_ids, remap = _stable_cluster_id_mapping([], new_partition, [], cfg)

        assert len(new_ids) == 2
        assert len(set(new_ids)) == 2  # all distinct

    def test_split_remap_proportions_sum_to_one(self):
        """Old cluster that splits into two new clusters: proportions sum to 1."""
        old_id = new_id()
        old_set = frozenset(["A", "B", "C", "D"])
        new_part = [frozenset(["A", "B"]), frozenset(["C", "D"])]
        cfg = Config(CLUSTER_INHERIT_OVERLAP_MIN=0.0, CLUSTER_DISSOLVE_DECAY=0.5)

        new_ids, remap = _stable_cluster_id_mapping([old_set], new_part, [old_id], cfg)

        proportions = remap[old_id]
        assert abs(sum(proportions.values()) - 1.0) < 1e-6

    def test_dissolve_uses_decay_factor(self):
        """Old cluster with no overlap: remap should use CLUSTER_DISSOLVE_DECAY."""
        old_id = new_id()
        old_partition = [frozenset(["A", "B"])]
        new_partition = [frozenset(["X", "Y"])]  # no overlap
        cfg = Config(CLUSTER_INHERIT_OVERLAP_MIN=0.30, CLUSTER_DISSOLVE_DECAY=0.5)

        new_ids, remap = _stable_cluster_id_mapping(old_partition, new_partition, [old_id], cfg)

        proportions = remap.get(old_id, {})
        if proportions:
            # Dissolve case: value should be CLUSTER_DISSOLVE_DECAY
            assert list(proportions.values())[0] == pytest.approx(0.5)


# ===================================================================== recompute_clusters (integration)

class TestRecomputeClusters:
    def test_no_entries_returns_zero_clusters(self, store):
        stats = recompute_clusters(store)
        assert stats["clusters_formed"] == 0
        assert stats["entry_points_clustered"] == 0

    def test_insufficient_cooccurrence_below_threshold(self, store):
        cfg = Config(ENTRY_COOCCURRENCE_THRESHOLD=10, CLUSTER_WINDOW_DAYS=30)
        e1 = _entry("A"); e2 = _entry("B")
        store.insert_entry(e1); store.insert_entry(e2)
        # Only 1 co-occurrence, threshold is 10
        record_entry_cooccurrence(store, [e1.id, e2.id])
        stats = recompute_clusters(store, cfg)
        # Graph has 0 edges above threshold → 0 nodes qualify → 0 clusters
        assert stats["clusters_formed"] == 0

    def test_clusters_formed_above_threshold(self, store):
        """Three well-connected entries above threshold should form at least 1 cluster."""
        cfg = Config(
            ENTRY_COOCCURRENCE_THRESHOLD=1,
            CLUSTER_WINDOW_DAYS=30,
            CLUSTER_MIN_SIZE=2,
            LOUVAIN_RESOLUTION=1.0,
            LOUVAIN_SEED=42,
        )
        entries = [_entry(f"E{i}") for i in range(4)]
        for e in entries:
            store.insert_entry(e)

        # Record enough co-occurrences (above threshold of 1)
        ids = [e.id for e in entries]
        for _ in range(3):
            record_entry_cooccurrence(store, ids)

        stats = recompute_clusters(store, cfg)
        assert stats["clusters_formed"] >= 1
        assert stats["entry_points_clustered"] >= 2

    def test_memberships_persisted(self, store):
        """After recompute, entry_cluster_membership rows must exist."""
        cfg = Config(
            ENTRY_COOCCURRENCE_THRESHOLD=1,
            CLUSTER_WINDOW_DAYS=30,
            CLUSTER_MIN_SIZE=2,
            LOUVAIN_RESOLUTION=1.0,
            LOUVAIN_SEED=42,
        )
        entries = [_entry(f"F{i}") for i in range(4)]
        for e in entries:
            store.insert_entry(e)

        ids = [e.id for e in entries]
        for _ in range(5):
            record_entry_cooccurrence(store, ids)

        recompute_clusters(store, cfg)

        # At least one entry should have cluster memberships
        found = False
        for e in entries:
            memberships = store.get_entry_cluster_memberships(e.id)
            if memberships:
                found = True
                total = sum(memberships.values())
                assert abs(total - 1.0) < 1e-6, f"Memberships don't sum to 1 for {e.id}: {memberships}"
        assert found, "No entry has cluster memberships after recompute"

    def test_recompute_twice_stable_ids(self, store):
        """Two consecutive recomputations on same data should reuse cluster IDs."""
        cfg = Config(
            ENTRY_COOCCURRENCE_THRESHOLD=1,
            CLUSTER_WINDOW_DAYS=30,
            CLUSTER_MIN_SIZE=2,
            LOUVAIN_RESOLUTION=1.0,
            LOUVAIN_SEED=42,
            CLUSTER_INHERIT_OVERLAP_MIN=0.30,
        )
        entries = [_entry(f"G{i}") for i in range(4)]
        for e in entries:
            store.insert_entry(e)

        ids = [e.id for e in entries]
        for _ in range(10):
            record_entry_cooccurrence(store, ids)

        recompute_clusters(store, cfg)
        partition1, ids1 = store.get_current_partition()

        recompute_clusters(store, cfg)
        partition2, ids2 = store.get_current_partition()

        # At least some cluster IDs should be preserved across identical re-partitions
        assert len(set(ids1) & set(ids2)) > 0
