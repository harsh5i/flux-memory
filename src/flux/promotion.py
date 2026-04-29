"""Grain promotion — use-driven working→core reclassification (Section 4.9).

A grain starts as 'working' (7-day decay half-life). It earns promotion to
'core' (30-day half-life) by proving it is useful across multiple contexts,
measured by context_spread: the count of distinct entry-point clusters from
which it has been successfully retrieved.

When context_spread reaches PROMOTION_THRESHOLD:
  - grain.decay_class changes from 'working' to 'core'
  - all inbound conduits are reclassified to 'core' decay class

The soft-membership model (Section 13.2) means each retrieval distributes
touch_weight across all clusters the activating entry points belong to,
proportionally to their membership weights. A grain retrieved many times via
entry points strongly in Cluster A accumulates touch mostly in Cluster A
(context_spread stays low). A grain retrieved via entry points spread across
Clusters A, B, C accumulates touch in all three (promoted at spread=3).

Public API:

    check_promotion(store, grain_id, trace, cfg, now) -> bool
        Check and optionally promote a single grain after a successful retrieval.

    check_promotions_bulk(store, grain_ids, trace, cfg, now) -> list[str]
        Check a batch of grains (all retrieved in the same propagation).
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

from .config import Config, DEFAULT_CONFIG
from .graph import utcnow
from .health import log_event
from .propagation import TraceStep
from .storage import FluxStore


def check_promotion(
    store: FluxStore,
    grain_id: str,
    trace: list[TraceStep],
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
    trace_id: str | None = None,
) -> bool:
    """Run the promotion check for one grain after a successful retrieval.

    Accumulates cluster touch weight from this retrieval, updates context_spread,
    and promotes the grain if the threshold is reached.

    Returns True if the grain was promoted in this call.
    """
    now = now or utcnow()
    grain = store.get_grain(grain_id)
    if grain is None or grain.decay_class == "core":
        return False

    # 1. Which entry points injected signal that reached this grain?
    activating_entries = _get_activating_entry_points(trace, grain_id)
    if not activating_entries:
        return False

    # 2-3. Accumulate touch weight into grain_cluster_touch per cluster.
    for entry_id in activating_entries:
        memberships = store.get_entry_cluster_memberships(entry_id)
        for cluster_id, membership_weight in memberships.items():
            store.increment_grain_cluster_touch(grain_id, cluster_id, membership_weight, now)

    # 4. Count clusters above threshold → new context_spread value.
    spread = store.count_clusters_above_threshold(grain_id, cfg.CLUSTER_TOUCH_THRESHOLD)
    store.update_grain_context_spread(grain_id, spread)

    # 5. Promote if threshold reached.
    if spread >= cfg.PROMOTION_THRESHOLD:
        inbound_before = store.conn.execute(
            "SELECT COUNT(*) AS n FROM conduits WHERE to_id = ? AND decay_class != 'core'",
            (grain_id,),
        ).fetchone()
        store.promote_grain_to_core(grain_id)
        store.upgrade_inbound_conduits_to_core(grain_id)
        log_event(store, "feedback", "promotion_triggered", {
            "grain_id": grain_id,
            "context_spread": spread,
            "promotion_threshold": cfg.PROMOTION_THRESHOLD,
            "inbound_conduits_upgraded": int(inbound_before["n"] if inbound_before else 0),
        }, trace_id=trace_id, now=now)
        return True

    return False


def check_promotions_bulk(
    store: FluxStore,
    grain_ids: Iterable[str],
    trace: list[TraceStep],
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
    trace_id: str | None = None,
) -> list[str]:
    """Run promotion checks for a batch of grains from the same retrieval.

    Returns the list of grain IDs promoted in this call.
    """
    now = now or utcnow()
    promoted = []
    for grain_id in grain_ids:
        if check_promotion(store, grain_id, trace, cfg, now=now, trace_id=trace_id):
            promoted.append(grain_id)
    return promoted


# ------------------------------------------------------------------ helpers

def _get_activating_entry_points(
    trace: list[TraceStep],
    grain_id: str,
) -> set[str]:
    """Return the set of entry IDs whose injected signal reached grain_id.

    Entries are nodes with hop==0 steps in the trace. We confirm they reached
    grain_id by BFS-reachability over the trace's forward adjacency map.
    """
    # Forward adjacency from the trace: from_id -> set[to_id].
    forward: dict[str, set[str]] = {}
    for step in trace:
        forward.setdefault(step.from_id, set()).add(step.to_id)

    # Entry IDs = from_ids at hop 0 (conduits directly from an entry point).
    entry_ids = {step.from_id for step in trace if step.hop == 0}

    return {eid for eid in entry_ids if _can_reach(eid, grain_id, forward)}


def _can_reach(start: str, target: str, forward: dict[str, set[str]]) -> bool:
    """BFS reachability on a forward adjacency dict (the retrieval trace)."""
    if start == target:
        return True
    visited = {start}
    queue = [start]
    while queue:
        node = queue.pop(0)
        for nxt in forward.get(node, set()):
            if nxt == target:
                return True
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return False
