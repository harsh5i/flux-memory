"""Incremental decay cleanup pass (Section 4.5).

Lazy decay itself lives in :func:`flux.propagation.effective_weight` -- the
stored weight is only a snapshot at the last touch, and readers compute the
real-time value on demand. This module is the complementary *write side*: a
bounded background sweep that (a) garbage-collects conduits whose effective
weight has fallen below ``WEIGHT_FLOOR`` and (b) opportunistically marks any
grain that lost its last inbound conduit as dormant.

The sweep is deliberately incremental. A full-graph decay scan is O(E) and
creates avoidable SQLite lock contention on large graphs; this pass only
inspects conduits unused since ``CLEANUP_STALE_HOURS`` and caps work at
``CLEANUP_BATCH_SIZE``. Scheduling is the caller's concern (spec suggests
every ``CLEANUP_INTERVAL_HOURS``) -- :func:`cleanup_pass` is pure and safe to
run on any cadence, including on-demand from tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import DEFAULT_CONFIG, Config
from .graph import utcnow
from .propagation import effective_weight
from .storage import FluxStore


@dataclass
class CleanupResult:
    scanned: int
    deleted_conduits: int
    dormant_grains: int


def cleanup_pass(
    store: FluxStore,
    cfg: Config = DEFAULT_CONFIG,
    *,
    now: datetime | None = None,
) -> CleanupResult:
    """Run one bounded cleanup sweep.

    Selects up to ``CLEANUP_BATCH_SIZE`` conduits that have not been touched
    in the last ``CLEANUP_STALE_HOURS`` (oldest first), deletes those whose
    effective weight is below ``WEIGHT_FLOOR``, then checks each affected
    grain for inbound-conduit orphanhood and marks it dormant if so.

    Grace-period conduits are shielded by the floor baked into
    :func:`effective_weight`, matching the §4.5 contract that newly-created
    knowledge cannot be evicted before it has had a chance to earn a
    retrieval.
    """
    now = now or utcnow()
    cutoff = now - timedelta(hours=cfg.CLEANUP_STALE_HOURS)

    candidates = store.query_conduits_unused_since(cutoff, limit=cfg.CLEANUP_BATCH_SIZE)

    deleted = 0
    dormant = 0
    affected_grains: set[str] = set()

    for conduit in candidates:
        if effective_weight(conduit, cfg, now) < cfg.WEIGHT_FLOOR:
            store.delete_conduit(conduit.id)
            deleted += 1
            # The conduit's ``to_id`` is the one whose inbound count may drop
            # to zero. Bidirectional shortcuts lose both endpoints' coverage,
            # but any grain that still has other inbound edges will survive
            # the orphan check below, so including both is safe and cheaper
            # than branching on direction here.
            affected_grains.add(conduit.to_id)
            if conduit.direction == "bidirectional":
                affected_grains.add(conduit.from_id)

    for grain_id in affected_grains:
        if store.count_inbound_conduits(grain_id) == 0:
            store.mark_dormant(grain_id, now)
            dormant += 1

    return CleanupResult(
        scanned=len(candidates),
        deleted_conduits=deleted,
        dormant_grains=dormant,
    )
