"""Cleanup-pass tests for Flux Memory (Section 4.5).

The lazy-decay half of §4.5 is exercised in test_propagation.py's
``effective_weight_*`` suite; this file only covers the write-side sweep
implemented in ``flux.decay.cleanup_pass``.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from flux.config import Config, DEFAULT_CONFIG
from flux.decay import cleanup_pass
from flux.graph import Conduit, Entry, Grain


# --- helpers ---------------------------------------------------------------
def _grain(store, content="g", provenance="user_stated", status="active"):
    g = Grain(content=content, provenance=provenance, status=status)  # type: ignore[arg-type]
    store.insert_grain(g)
    return g


def _conduit(store, from_id, to_id, **kw):
    c = Conduit(from_id=from_id, to_id=to_id, **kw)
    store.insert_conduit(c)
    return c


NOW = datetime(2026, 4, 19, 12, 0, tzinfo=timezone.utc)
# A timestamp safely outside both the stale cutoff (72h) and the grace
# window (72h), so conduits parked there get no artificial protection from
# either mechanism and the decay math alone governs survival.
LONG_AGO = NOW - timedelta(days=120)


# --- stale candidate selection ---------------------------------------------
def test_cleanup_ignores_fresh_conduits(store):
    """A conduit used inside the stale window is not even a candidate."""
    g1 = _grain(store, "a")
    g2 = _grain(store, "b")
    fresh_last_used = NOW - timedelta(hours=1)  # well inside CLEANUP_STALE_HOURS
    _conduit(store, g1.id, g2.id, weight=0.001,
             created_at=LONG_AGO, last_used=fresh_last_used)

    result = cleanup_pass(store, DEFAULT_CONFIG, now=NOW)

    assert result.scanned == 0
    assert result.deleted_conduits == 0


def test_cleanup_deletes_stale_below_floor_conduit(store):
    """Long-unused, weight below WEIGHT_FLOOR -> deleted."""
    g1 = _grain(store, "a")
    g2 = _grain(store, "b")
    c = _conduit(store, g1.id, g2.id, weight=0.001,
                 created_at=LONG_AGO, last_used=LONG_AGO)

    result = cleanup_pass(store, DEFAULT_CONFIG, now=NOW)

    assert result.deleted_conduits == 1
    assert store.get_conduit(c.id) is None


def test_cleanup_preserves_stale_but_healthy_conduit(store):
    """Stale by time but stored weight decays to something still above floor
    -> keep. (WEIGHT_FLOOR default 0.05; 0.9 decayed over 120 days on working
    half-life still exceeds that by a safe margin.)"""
    g1 = _grain(store, "a")
    g2 = _grain(store, "b")
    c = _conduit(store, g1.id, g2.id, weight=0.9,
                 created_at=LONG_AGO, last_used=NOW - timedelta(hours=80))

    result = cleanup_pass(store, DEFAULT_CONFIG, now=NOW)

    assert result.deleted_conduits == 0
    assert store.get_conduit(c.id) is not None


def test_cleanup_respects_batch_size(store):
    """With BATCH_SIZE=2, only two below-floor stale conduits are processed
    per pass."""
    cfg = replace(DEFAULT_CONFIG, CLEANUP_BATCH_SIZE=2)
    g = _grain(store, "sink")
    for i in range(5):
        src = _grain(store, f"src{i}")
        _conduit(store, src.id, g.id, weight=0.001,
                 created_at=LONG_AGO, last_used=LONG_AGO)

    result = cleanup_pass(store, cfg, now=NOW)

    assert result.scanned == 2
    assert result.deleted_conduits == 2
    # Three stale-below-floor conduits remain for the next pass.
    assert store.count_inbound_conduits(g.id) == 3


# --- grace-period protection -----------------------------------------------
def test_cleanup_spares_grace_period_conduit(store):
    """A brand-new conduit with trivially small weight is stale by the
    last_used cutoff but grace-floor makes effective_weight >= WEIGHT_FLOOR.
    Constructing this case requires last_used outside stale window; a
    freshly-minted conduit in practice never satisfies that, but the
    invariant still holds for pathological initial state."""
    # Construct a conduit that is old-enough-by-last_used but young-by-created_at.
    # This does not arise from normal reinforcement, but guards the floor logic.
    g1 = _grain(store, "a")
    g2 = _grain(store, "b")
    created = NOW - timedelta(hours=1)  # deep inside grace window
    last_used = NOW - timedelta(hours=80)  # beyond stale cutoff
    # created_at cannot be after last_used in real operation, so this is a
    # contrived row: we write it directly to exercise the floor-wins path.
    c = Conduit(
        from_id=g1.id, to_id=g2.id, weight=0.001,
        created_at=created, last_used=last_used,
    )
    store.insert_conduit(c)

    result = cleanup_pass(store, DEFAULT_CONFIG, now=NOW)

    # Grace floor keeps effective_weight >= NEW_CONDUIT_MIN_WEIGHT > WEIGHT_FLOOR.
    assert result.deleted_conduits == 0
    assert store.get_conduit(c.id) is not None


# --- incremental orphan detection ------------------------------------------
def test_cleanup_marks_grain_dormant_when_last_inbound_deleted(store):
    """A grain that loses its only inbound conduit in this pass gets
    mark_dormant. Section 4.5 explicitly bundles this with the cleanup
    pass."""
    src = _grain(store, "src")
    dst = _grain(store, "dst")
    _conduit(store, src.id, dst.id, weight=0.001,
             created_at=LONG_AGO, last_used=LONG_AGO)

    result = cleanup_pass(store, DEFAULT_CONFIG, now=NOW)

    assert result.deleted_conduits == 1
    assert result.dormant_grains == 1
    refreshed = store.get_grain(dst.id)
    assert refreshed.status == "dormant"
    assert refreshed.dormant_since is not None


def test_cleanup_spares_grain_with_surviving_inbound(store):
    """Grain still has another inbound edge after the pass -> stays active."""
    a = _grain(store, "a")
    b = _grain(store, "b")
    dst = _grain(store, "dst")
    _conduit(store, a.id, dst.id, weight=0.001,   # will be deleted
             created_at=LONG_AGO, last_used=LONG_AGO)
    _conduit(store, b.id, dst.id, weight=0.8,      # healthy, survives
             created_at=LONG_AGO, last_used=NOW - timedelta(hours=80))

    result = cleanup_pass(store, DEFAULT_CONFIG, now=NOW)

    assert result.deleted_conduits == 1
    assert result.dormant_grains == 0
    assert store.get_grain(dst.id).status == "active"


def test_cleanup_idempotent_on_clean_graph(store):
    """No stale candidates -> all counters zero, no side effects."""
    _grain(store, "solitary")
    result = cleanup_pass(store, DEFAULT_CONFIG, now=NOW)
    assert result == type(result)(scanned=0, deleted_conduits=0, dormant_grains=0)
