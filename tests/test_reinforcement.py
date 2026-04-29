"""Reinforcement and penalization tests (Section 4.3, 4.4, 7.2)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from flux.config import Config, DEFAULT_CONFIG
from flux.graph import Conduit, Entry, Grain
from flux.propagation import TraceStep, propagate
from flux.reinforcement import penalize, reinforce


# --- helpers ---------------------------------------------------------------
def _grain(store, content="g", provenance="user_stated", status="active"):
    g = Grain(content=content, provenance=provenance, status=status)  # type: ignore[arg-type]
    store.insert_grain(g)
    return g


def _entry(store, feature, affinities=None):
    e = Entry(feature=feature, affinities=affinities or {})
    store.insert_entry(e)
    return e


def _conduit(store, from_id, to_id, weight=0.5, **kw):
    c = Conduit(from_id=from_id, to_id=to_id, weight=weight, **kw)
    store.insert_conduit(c)
    return c


def _trace_step(conduit, signal=0.5, hop=0):
    return TraceStep(
        conduit_id=conduit.id,
        from_id=conduit.from_id,
        to_id=conduit.to_id,
        signal=signal,
        hop=hop,
        effective_weight=conduit.weight,
    )


# --- reinforcement: conduit widening ---------------------------------------
def test_reinforce_widens_conduit_to_successful_grain(store):
    e = _entry(store, "x")
    g = _grain(store, "g")
    c = _conduit(store, e.id, g.id, weight=0.5)

    reinforce(store, [_trace_step(c)], [g.id])

    after = store.get_conduit(c.id)
    # Spec 4.3: weight += LEARNING_RATE * (1 - weight) = 0.5 + 0.05 * 0.5 = 0.525
    assert after.weight == pytest.approx(0.525, rel=1e-3)
    assert after.use_count == c.use_count + 1


def test_reinforce_logs_highway_when_threshold_crossed(store):
    e = _entry(store, "x")
    g = _grain(store, "g")
    c = _conduit(store, e.id, g.id, weight=0.79)

    reinforce(store, [_trace_step(c)], [g.id], trace_id="trace-highway")

    row = store.conn.execute(
        """
        SELECT trace_id, data FROM events
        WHERE category='feedback' AND event='highway_formed'
        """
    ).fetchone()
    assert row is not None
    assert row["trace_id"] == "trace-highway"
    data = json.loads(row["data"])
    assert data["conduit_id"] == c.id
    assert data["previous_weight"] < 0.80 <= data["new_weight"]


def test_reinforce_caps_at_ceiling(store):
    e = _entry(store, "x")
    g = _grain(store, "g")
    c = _conduit(store, e.id, g.id, weight=0.94)

    reinforce(store, [_trace_step(c)], [g.id])

    after = store.get_conduit(c.id)
    assert after.weight <= DEFAULT_CONFIG.WEIGHT_CEILING + 1e-9


def test_reinforce_ignores_conduits_not_leading_to_success(store):
    """Only conduits whose to_id is in successful_grain_ids get widened."""
    e = _entry(store, "x")
    g_good = _grain(store, "good")
    g_other = _grain(store, "other")
    c_good = _conduit(store, e.id, g_good.id, weight=0.5)
    c_other = _conduit(store, e.id, g_other.id, weight=0.5)

    reinforce(store, [_trace_step(c_good), _trace_step(c_other)], [g_good.id])

    assert store.get_conduit(c_good.id).weight > 0.5
    assert store.get_conduit(c_other.id).weight == pytest.approx(0.5)


# --- reinforcement: provenance multiplier (Section 7.2) --------------------
def test_reinforce_provenance_multiplier_scales_rate(store):
    """AI-stated grains reinforce at half the rate of user-stated grains."""
    e = _entry(store, "x")
    g_user = _grain(store, "user", provenance="user_stated")
    g_ai = _grain(store, "ai", provenance="ai_stated")
    c_user = _conduit(store, e.id, g_user.id, weight=0.5)
    c_ai = _conduit(store, e.id, g_ai.id, weight=0.5)

    reinforce(store, [_trace_step(c_user)], [g_user.id])
    reinforce(store, [_trace_step(c_ai)], [g_ai.id])

    delta_user = store.get_conduit(c_user.id).weight - 0.5
    delta_ai = store.get_conduit(c_ai.id).weight - 0.5
    # ai_stated multiplier is 0.5; user_stated is 1.0.
    assert delta_ai == pytest.approx(delta_user * 0.5, rel=1e-3)


# --- reinforcement: co-retrieval + shortcuts ------------------------------
def test_reinforce_increments_co_retrieval_count_for_pair(store):
    e = _entry(store, "x")
    g1 = _grain(store, "g1")
    g2 = _grain(store, "g2")
    c1 = _conduit(store, e.id, g1.id, weight=0.5)
    c2 = _conduit(store, e.id, g2.id, weight=0.5)

    reinforce(store, [_trace_step(c1), _trace_step(c2)], [g1.id, g2.id])

    assert store.get_co_retrieval_count(g1.id, g2.id) == 1


def test_reinforce_canonicalizes_co_retrieval_key(store):
    """Passing the pair in either order must map to the same row."""
    e = _entry(store, "x")
    g1 = _grain(store, "g1")
    g2 = _grain(store, "g2")
    c1 = _conduit(store, e.id, g1.id, weight=0.5)
    c2 = _conduit(store, e.id, g2.id, weight=0.5)

    reinforce(store, [_trace_step(c1), _trace_step(c2)], [g1.id, g2.id])
    reinforce(store, [_trace_step(c1), _trace_step(c2)], [g2.id, g1.id])

    assert store.get_co_retrieval_count(g1.id, g2.id) == 2
    assert store.get_co_retrieval_count(g2.id, g1.id) == 2


def test_reinforce_creates_shortcut_once_threshold_met(store):
    """After SHORTCUT_THRESHOLD co-retrievals, a direct conduit between the
    pair is created at INITIAL_SHORTCUT_WEIGHT."""
    cfg = Config(SHORTCUT_THRESHOLD=2)
    e = _entry(store, "x")
    g1 = _grain(store, "g1")
    g2 = _grain(store, "g2")
    c1 = _conduit(store, e.id, g1.id, weight=0.5)
    c2 = _conduit(store, e.id, g2.id, weight=0.5)

    # First pass: count becomes 1, no shortcut yet.
    reinforce(store, [_trace_step(c1), _trace_step(c2)], [g1.id, g2.id], cfg=cfg)
    assert store.conduit_between(g1.id, g2.id) is None

    # Second pass: count reaches 2 == threshold, shortcut is created.
    reinforce(store, [_trace_step(c1), _trace_step(c2)], [g1.id, g2.id], cfg=cfg)
    shortcut = store.conduit_between(g1.id, g2.id)
    assert shortcut is not None
    assert shortcut.weight == pytest.approx(cfg.INITIAL_SHORTCUT_WEIGHT)
    assert shortcut.direction == "bidirectional"

    row = store.conn.execute(
        """
        SELECT data FROM events
        WHERE category='feedback' AND event='shortcut_created'
        """
    ).fetchone()
    assert row is not None
    data = json.loads(row["data"])
    assert data["conduit_id"] == shortcut.id
    assert data["co_retrieval_count"] == cfg.SHORTCUT_THRESHOLD


def test_reinforce_reinforces_existing_shortcut(store):
    """Co-retrieving a pair that already has a shortcut widens the shortcut."""
    e = _entry(store, "x")
    g1 = _grain(store, "g1")
    g2 = _grain(store, "g2")
    c1 = _conduit(store, e.id, g1.id, weight=0.5)
    c2 = _conduit(store, e.id, g2.id, weight=0.5)
    shortcut = _conduit(store, g1.id, g2.id, weight=0.5, direction="bidirectional")

    reinforce(store, [_trace_step(c1), _trace_step(c2)], [g1.id, g2.id])

    after = store.get_conduit(shortcut.id)
    assert after.weight > 0.5


def test_reinforce_evicts_weakest_edge_when_at_cap(store):
    """A successful pair on a grain already at MAX_EDGES_PER_GRAIN causes the
    weakest existing edge to be evicted, then the shortcut is created."""
    cfg = Config(MAX_EDGES_PER_GRAIN=3, SHORTCUT_THRESHOLD=1)
    e = _entry(store, "x")
    g1 = _grain(store, "g1")
    g2 = _grain(store, "g2")
    filler_a = _grain(store, "fa")
    filler_b = _grain(store, "fb")
    # Three edges on g1: e->g1 (0.8), g1->filler_a (0.6, WEAKEST), g1->filler_b (0.9).
    c_e = _conduit(store, e.id, g1.id, weight=0.8)
    c_weak = _conduit(store, g1.id, filler_a.id, weight=0.1)  # clearly weakest
    _conduit(store, g1.id, filler_b.id, weight=0.9)
    assert store.count_edges(g1.id) == 3

    # g2 is unconstrained.
    c_g2 = _conduit(store, e.id, g2.id, weight=0.5)

    reinforce(store, [_trace_step(c_e), _trace_step(c_g2)], [g1.id, g2.id], cfg=cfg)

    # Weakest edge evicted, shortcut added.
    assert store.get_conduit(c_weak.id) is None
    assert store.conduit_between(g1.id, g2.id) is not None


# --- reinforcement: entry affinities ---------------------------------------
def test_reinforce_sharpens_entry_affinity(store):
    e = _entry(store, "x")
    g = _grain(store, "g")
    c = _conduit(store, e.id, g.id, weight=0.5)

    reinforce(store, [_trace_step(c)], [g.id])

    e_after = store.get_entry(e.id)
    # Base affinity 1.0 * 1.1 = 1.1
    assert e_after.affinities.get(c.id) == pytest.approx(1.1)


def test_reinforce_affinity_caps_at_ceiling(store):
    e = _entry(store, "x", affinities={"placeholder": 1.95})  # will be overwritten
    g = _grain(store, "g")
    c = _conduit(store, e.id, g.id, weight=0.5)
    e.affinities = {c.id: 1.95}
    store.update_entry_affinities(e.id, e.affinities)

    reinforce(store, [_trace_step(c)], [g.id])

    e_after = store.get_entry(e.id)
    assert e_after.affinities[c.id] <= 2.0 + 1e-9


# --- penalization: conduit narrowing ---------------------------------------
def test_penalize_narrows_conduit_to_failed_grain(store):
    e = _entry(store, "x")
    g = _grain(store, "bad")
    c = _conduit(store, e.id, g.id, weight=0.5)

    penalize(store, [_trace_step(c)], [g.id])

    after = store.get_conduit(c.id)
    # Spec 4.4: weight *= DECAY_FACTOR (0.85) = 0.425
    assert after.weight == pytest.approx(0.5 * DEFAULT_CONFIG.DECAY_FACTOR, rel=1e-3)

    row = store.conn.execute(
        """
        SELECT data FROM events
        WHERE category='feedback' AND event='conduit_penalized'
        """
    ).fetchone()
    assert row is not None
    data = json.loads(row["data"])
    assert data["conduit_id"] == c.id
    assert data["weight_drop"] == pytest.approx(0.5 * (1 - DEFAULT_CONFIG.DECAY_FACTOR))
    assert data["deleted"] is False


def test_penalize_deletes_conduit_below_floor(store):
    e = _entry(store, "x")
    g = _grain(store, "bad")
    # 0.06 * 0.85 = 0.051 -> still above 0.05 floor; use 0.05 so 0.05 * 0.85 < 0.05.
    c = _conduit(store, e.id, g.id, weight=0.055)

    penalize(store, [_trace_step(c)], [g.id])

    assert store.get_conduit(c.id) is None


def test_penalize_ignores_conduits_not_leading_to_failure(store):
    e = _entry(store, "x")
    g_bad = _grain(store, "bad")
    g_good = _grain(store, "good")
    c_bad = _conduit(store, e.id, g_bad.id, weight=0.5)
    c_good = _conduit(store, e.id, g_good.id, weight=0.5)

    penalize(store, [_trace_step(c_bad), _trace_step(c_good)], [g_bad.id])

    assert store.get_conduit(c_bad.id).weight < 0.5
    assert store.get_conduit(c_good.id).weight == pytest.approx(0.5)


# --- penalization: entry affinity dampening --------------------------------
def test_penalize_dampens_entry_affinity_to_failed_first_hop(store):
    g = _grain(store, "bad")
    c = Conduit(from_id="placeholder", to_id=g.id, weight=0.5)
    e = Entry(feature="x", affinities={c.id: 1.0})
    store.insert_entry(e)
    c.from_id = e.id
    store.insert_conduit(c)

    penalize(store, [_trace_step(c)], [g.id])

    e_after = store.get_entry(e.id)
    # 1.0 * 0.8 = 0.8
    assert e_after.affinities[c.id] == pytest.approx(0.8)


def test_penalize_affinity_floor(store):
    g = _grain(store, "bad")
    c = Conduit(from_id="placeholder", to_id=g.id, weight=0.5)
    e = Entry(feature="x", affinities={c.id: 0.12})
    store.insert_entry(e)
    c.from_id = e.id
    store.insert_conduit(c)

    penalize(store, [_trace_step(c)], [g.id])

    e_after = store.get_entry(e.id)
    assert e_after.affinities[c.id] >= 0.1  # affinity floor


# --- integration: propagate -> reinforce round-trip ------------------------
def test_reinforce_then_propagate_yields_higher_signal(store):
    """The full loop: propagate, mark useful, reinforce, re-propagate -> stronger."""
    e = _entry(store, "x")
    g = _grain(store, "g")
    _conduit(store, e.id, g.id, weight=0.5)

    first = propagate(store, [e.id])
    reinforce(store, first.trace, [g.id])
    second = propagate(store, [e.id])

    # Entry affinity sharpened AND conduit widened -> strictly higher signal.
    assert second.activated[0][1] > first.activated[0][1]


def test_penalize_then_propagate_yields_lower_signal(store):
    e = _entry(store, "x")
    g = _grain(store, "g")
    _conduit(store, e.id, g.id, weight=0.5)

    first = propagate(store, [e.id])
    penalize(store, first.trace, [g.id])
    second = propagate(store, [e.id])

    if second.activated:
        assert second.activated[0][1] < first.activated[0][1]
    # If penalization dropped signal below threshold, activated is [], which is
    # also strictly less -- accept either case.
