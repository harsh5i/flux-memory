"""Temporal decay maintenance — cleanup pass and dormancy expiry (Sections 4.5–4.6).

Two scheduled entry-points:

    cleanup_pass(store, cfg, now)
        Incremental GC: scans conduits untouched for >= CLEANUP_STALE_HOURS,
        deletes those whose effective_weight has fallen below WEIGHT_FLOOR,
        then marks newly-orphaned grains dormant.

    expiry_pass(store, cfg, now)
        Archives grains that have been dormant for >= DORMANCY_LIMIT_DAYS.

Lazy decay (effective_weight in propagation.py) runs inline during retrieval
and is the primary mechanism. These passes are the background cleanup arm:
they garbage-collect edges that will never be traversed again and reclaim
grains that have lost all inbound routes.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from .config import Config, DEFAULT_CONFIG
from .graph import utcnow
from .health import log_event
from .propagation import effective_weight
from .storage import FluxStore


def cleanup_pass(
    store: FluxStore,
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
) -> dict:
    """Incremental conduit GC and orphan detection (Section 4.5).

    Scans at most CLEANUP_BATCH_SIZE conduits that have not been touched for
    CLEANUP_STALE_HOURS. For each, computes the current effective_weight; if it
    is below WEIGHT_FLOOR the conduit is deleted. After deletions, any grain
    whose inbound conduit count dropped to zero is marked dormant.

    Returns a stats dict consumed by the Health Monitor.
    """
    now = now or utcnow()
    stale_cutoff = now - timedelta(hours=cfg.CLEANUP_STALE_HOURS)

    candidates = store.conduits_unused_since(stale_cutoff, limit=cfg.CLEANUP_BATCH_SIZE)

    deleted = 0
    affected_grains: set[str] = set()

    for conduit in candidates:
        w = effective_weight(conduit, cfg, now)
        if w < cfg.WEIGHT_FLOOR:
            affected_grains.add(conduit.to_id)
            store.delete_conduit(conduit.id)
            deleted += 1

    # Incremental orphan detection — only examine grains whose inbound conduit
    # count may have changed in this pass. Full-graph scans are deferred to a
    # separate diagnostic query (Section 12.7).
    newly_dormant = 0
    for grain_id in affected_grains:
        grain = store.get_grain(grain_id)
        if grain is None or grain.status != "active":
            continue
        if store.count_inbound_conduits(grain_id) == 0:
            store.update_grain_status(grain_id, "dormant", dormant_since=now)
            newly_dormant += 1

    stats = {
        "candidates_scanned": len(candidates),
        "conduits_deleted": deleted,
        "grains_marked_dormant": newly_dormant,
    }
    log_event(store, "decay", "cleanup_pass_completed", stats, now=now)
    return stats


def expiry_pass(
    store: FluxStore,
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
) -> dict:
    """Dormancy expiry — archive grains that have been dormant too long (Section 4.6).

    Grains in 'dormant' status that have been dormant for >= DORMANCY_LIMIT_DAYS
    are transitioned to 'archived'. Archived grains are permanently excluded from
    propagation and cannot receive reinforcement.

    A single active retrieval before archival fully reverses dormancy, so the
    dormancy->archived pipeline is entirely reversible up to the archival boundary.

    Returns a stats dict.
    """
    now = now or utcnow()
    limit = timedelta(days=cfg.DORMANCY_LIMIT_DAYS)
    archived = 0

    for grain in store.iter_grains(status="dormant"):
        if grain.dormant_since is None:
            continue
        if (now - grain.dormant_since) >= limit:
            store.update_grain_status(grain.id, "archived")
            archived += 1

    stats = {"grains_archived": archived}
    log_event(store, "decay", "expiry_pass_completed", stats, now=now)
    return stats
