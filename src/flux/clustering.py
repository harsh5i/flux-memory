"""Entry-point soft clustering (Sections 4.9 algorithm, 13.2 remapping).

Entry points that co-occur in queries form clusters. Clusters define the
"context" that grain promotion measures: a grain retrieved from N distinct
clusters earns context_spread N and is promoted when N >= PROMOTION_THRESHOLD.

Public API:

    record_entry_cooccurrence(store, entry_ids, now)
        Called after every retrieval. Increments pairwise co-occurrence counts.

    recompute_clusters(store, cfg, now)
        Full re-partition. Runs at most once per CLUSTER_RECOMPUTE_MIN_INTERVAL_DAYS.
        Callers must enforce the interval; this function always runs when called.

Algorithm (Section 13.2):
  1. Build normalised co-occurrence graph from entry_cooccurrence table.
  2. Run networkx Louvain community detection.
  3. Merge communities smaller than CLUSTER_MIN_SIZE.
  4. Derive soft membership weights (proportion of edge weight to each cluster).
  5. Map new cluster indices to stable UUIDs via best Jaccard overlap.
  6. Remap grain_cluster_touch weights across split/merge/dissolve cases.
  7. Persist new cluster and membership rows atomically.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any

import networkx as nx

from .config import Config, DEFAULT_CONFIG
from .graph import new_id, utcnow
from .storage import FluxStore


# ------------------------------------------------------------------ public API

def record_entry_cooccurrence(
    store: FluxStore,
    entry_ids: list[str],
    now: datetime | None = None,
) -> None:
    """Increment pairwise co-occurrence counts for all pairs in entry_ids.

    Called after every retrieval with the list of entry point IDs that injected
    signal. Pairs are canonicalised (lower_id, higher_id) inside the storage
    layer, so order here does not matter.
    """
    now = now or utcnow()
    ids = list(entry_ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            store.increment_entry_cooccurrence(ids[i], ids[j], now)


def recompute_clusters(
    store: FluxStore,
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
) -> dict:
    """Full cluster recomputation (Section 13.2).

    Returns a stats dict: {"clusters_formed": int, "entry_points_clustered": int}.
    """
    now = now or utcnow()

    # Step 1: Build co-occurrence graph.
    cooc_rows = store.all_entry_cooccurrences(window_days=cfg.CLUSTER_WINDOW_DAYS)
    G = _build_cooccurrence_graph(cooc_rows, cfg.ENTRY_COOCCURRENCE_THRESHOLD)

    if len(G.nodes) < 2:
        return {"clusters_formed": 0, "entry_points_clustered": len(G.nodes)}

    # Step 2: Louvain partition.
    communities: list[frozenset] = list(
        nx.community.louvain_communities(
            G,
            weight="weight",
            resolution=cfg.LOUVAIN_RESOLUTION,
            seed=cfg.LOUVAIN_SEED,
        )
    )

    # Step 3: Merge small communities.
    communities = _merge_small_clusters(G, communities, cfg.CLUSTER_MIN_SIZE)

    # Step 4: Derive soft membership weights (entry_id -> {cluster_index -> float}).
    memberships = _derive_soft_memberships(G, communities)

    # Step 5: Stable ID mapping.
    old_partition, old_cluster_ids = store.get_current_partition()
    new_cluster_ids, touch_remap = _stable_cluster_id_mapping(
        old_partition, communities, old_cluster_ids, cfg
    )

    # Step 6: Remap grain_cluster_touch.
    _remap_grain_cluster_touches(store, touch_remap)

    # Step 7: Persist. Convert cluster-index keys to cluster-ID keys.
    id_memberships: dict[str, dict[str, float]] = {
        entry_id: {new_cluster_ids[idx]: w for idx, w in cluster_weights.items()}
        for entry_id, cluster_weights in memberships.items()
    }
    store.replace_cluster_memberships(id_memberships, new_cluster_ids, communities, now)

    return {
        "clusters_formed": len(communities),
        "entry_points_clustered": len(G.nodes),
    }


# ------------------------------------------------------------------ helpers

def _build_cooccurrence_graph(
    cooc_rows: list[dict],
    threshold: int,
) -> nx.Graph:
    """Build a normalised weighted undirected graph from co-occurrence rows.

    Edge weight = count / sqrt(freq_a * freq_b) so high-frequency entry points
    don't dominate the partition. Edges below threshold are excluded.
    """
    freq: dict[str, float] = defaultdict(float)
    for row in cooc_rows:
        freq[row["entry_a"]] += row["count"]
        freq[row["entry_b"]] += row["count"]

    G: nx.Graph = nx.Graph()
    for row in cooc_rows:
        a, b, count = row["entry_a"], row["entry_b"], row["count"]
        if count < threshold:
            continue
        denom = math.sqrt(freq[a] * freq[b])
        w = count / denom if denom > 0 else 0.0
        if w > 0:
            G.add_edge(a, b, weight=w)

    return G


def _merge_small_clusters(
    G: nx.Graph,
    communities: list[frozenset],
    min_size: int,
) -> list[frozenset]:
    """Absorb communities smaller than min_size into the nearest large one.

    'Nearest' = highest total inter-community edge weight. If all communities
    are small, returns them unchanged (cannot merge into nothing).
    """
    large = [c for c in communities if len(c) >= min_size]
    small = [c for c in communities if len(c) < min_size]

    if not large:
        return communities

    result = list(large)
    for s in small:
        best_idx, best_w = 0, -1.0
        for idx, lc in enumerate(result):
            w = sum(
                G[a][b]["weight"]
                for a in s
                for b in lc
                if G.has_edge(a, b)
            )
            if w > best_w:
                best_idx, best_w = idx, w
        result[best_idx] = result[best_idx] | s

    return [frozenset(c) for c in result]


def _derive_soft_memberships(
    G: nx.Graph,
    communities: list[frozenset],
) -> dict[str, dict[int, float]]:
    """Derive per-entry soft membership weights (Section 13.2, Step 4).

    For entry e, membership in cluster i = (sum of weights to cluster i members)
    / (total weight of e's edges). Weights sum to 1.0 per entry.
    """
    community_of: dict[str, int] = {
        entry_id: idx
        for idx, community in enumerate(communities)
        for entry_id in community
    }

    memberships: dict[str, dict[int, float]] = {}
    for entry_id in G.nodes():
        neighbours = list(G[entry_id])
        total = sum(G[entry_id][n]["weight"] for n in neighbours)

        if total == 0.0 or not neighbours:
            memberships[entry_id] = {community_of.get(entry_id, 0): 1.0}
            continue

        cluster_weights: dict[int, float] = defaultdict(float)
        for n in neighbours:
            cid = community_of.get(n)
            if cid is not None:
                cluster_weights[cid] += G[entry_id][n]["weight"] / total

        total_w = sum(cluster_weights.values())
        if total_w > 0:
            memberships[entry_id] = {cid: w / total_w for cid, w in cluster_weights.items()}
        else:
            memberships[entry_id] = {community_of.get(entry_id, 0): 1.0}

    return memberships


def _stable_cluster_id_mapping(
    old_partition: list[frozenset],
    new_partition: list[frozenset],
    old_cluster_ids: list[str],
    cfg: Config,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Match new clusters to old cluster IDs via best Jaccard overlap (Section 13.2, Step 5).

    Returns:
        new_cluster_ids: list[str] — one UUID per new cluster (inherited or fresh)
        touch_remap:     old_cluster_id -> {new_cluster_id -> proportion}
                         Used by _remap_grain_cluster_touches to redistribute
                         accumulated touch weights across split/merge/dissolve cases.
    """
    # Build Jaccard overlap matrix.
    overlap: dict[tuple[int, int], float] = {}
    for i, old_set in enumerate(old_partition):
        for j, new_set in enumerate(new_partition):
            inter = len(old_set & new_set)
            union = len(old_set | new_set)
            if union > 0:
                overlap[(i, j)] = inter / union

    # Greedy best-match (descending overlap), inheriting old IDs above threshold.
    new_cluster_ids: list[str | None] = [None] * len(new_partition)
    used_old: set[int] = set()
    for (i, j), score in sorted(overlap.items(), key=lambda kv: kv[1], reverse=True):
        if score < cfg.CLUSTER_INHERIT_OVERLAP_MIN:
            break
        if new_cluster_ids[j] is None and i not in used_old:
            new_cluster_ids[j] = old_cluster_ids[i]
            used_old.add(i)

    # Assign fresh UUIDs to unmatched new clusters.
    for j in range(len(new_partition)):
        if new_cluster_ids[j] is None:
            new_cluster_ids[j] = new_id()

    # Build touch_remap covering identity / split / merge / dissolve cases.
    touch_remap: dict[str, dict[str, float]] = {}
    for i, old_set in enumerate(old_partition):
        old_id = old_cluster_ids[i]
        splits: dict[str, float] = {}

        for j, new_set in enumerate(new_partition):
            shared = len(old_set & new_set)
            if shared > 0:
                splits[new_cluster_ids[j]] = shared / len(old_set)

        total = sum(splits.values())
        if total == 0:
            # DISSOLVE: no member overlap with any new cluster.
            best_j = max(
                range(len(new_partition)),
                key=lambda j: overlap.get((i, j), 0.0),
                default=None,
            )
            if best_j is not None:
                splits = {new_cluster_ids[best_j]: cfg.CLUSTER_DISSOLVE_DECAY}
        else:
            # Normalise so proportions sum to 1.0 (handles SPLIT cleanly).
            splits = {cid: p / total for cid, p in splits.items()}

        touch_remap[old_id] = splits

    return new_cluster_ids, touch_remap


def _remap_grain_cluster_touches(
    store: FluxStore,
    touch_remap: dict[str, dict[str, float]],
) -> None:
    """Redistribute grain_cluster_touch weights to new cluster IDs (Section 13.2, Step 6).

    Handles all four cases:
      IDENTITY — old cluster maps 1:1 to new cluster (same ID inherited)
      SPLIT    — touch weight distributed proportionally to shared members
      MERGE    — touch weights from multiple old clusters sum into one new cluster
      DISSOLVE — touch weight transferred at CLUSTER_DISSOLVE_DECAY rate
    """
    if not touch_remap:
        return

    all_touches = store.all_grain_cluster_touches()
    new_touches: dict[tuple[str, str], float] = defaultdict(float)

    for grain_id, old_cluster_id, old_weight in all_touches:
        remap = touch_remap.get(old_cluster_id)
        if remap is None:
            continue  # Old cluster completely dissolved with no successor.
        for new_cluster_id, proportion in remap.items():
            new_touches[(grain_id, new_cluster_id)] += old_weight * proportion

    store.replace_grain_cluster_touches(new_touches)
