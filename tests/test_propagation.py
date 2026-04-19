"""Signal propagation and lazy-decay tests (Section 4.2, 4.5, 5.1)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from flux.config import Config, DEFAULT_CONFIG
from flux.graph import Conduit, Entry, Grain
from flux.propagation import (
    effective_weight,
    propagate,
    retrieval_confidence,
)


# --- helpers ---------------------------------------------------------------
def _grain(store, content="g", status="active"):
    g = Grain(content=content, provenance="user_stated", status=status)  # type: ignore[arg-type]
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


# --- effective_weight ------------------------------------------------------
def test_effective_weight_no_decay_at_t_zero():
    """A conduit created and last_used now should report full stored weight
    (grace period applies but signal is unchanged)."""
    now = datetime(2026, 4, 19, tzinfo=timezone.utc)
    c = Conduit(from_id="a", to_id="b", weight=0.5, created_at=now, last_used=now)
    assert effective_weight(c, DEFAULT_CONFIG, now) == pytest.approx(0.5)


def test_effective_weight_half_life_working():
    """Working conduit: after 168h unused, weight halves."""
    # Place created_at outside grace (older than 72h) so grace multiplier
    # doesn't inflate the effective half-life.
    created = datetime(2026, 4, 1, tzinfo=timezone.utc)
    last_used = datetime(2026, 4, 10, tzinfo=timezone.utc)
    now = last_used + timedelta(hours=168)
    c = Conduit(from_id="a", to_id="b", weight=0.6, decay_class="working",
                created_at=created, last_used=last_used)
    assert effective_weight(c, DEFAULT_CONFIG, now) == pytest.approx(0.3, rel=1e-3)


def test_effective_weight_grace_period_floor():
    """A brand-new conduit cannot decay below NEW_CONDUIT_MIN_WEIGHT even if
    its stored weight and half-life would otherwise push it to the floor."""
    now = datetime(2026, 4, 19, tzinfo=timezone.utc)
    created = now - timedelta(hours=1)  # still inside grace (< 72h)
    last_used = created
    c = Conduit(from_id="a", to_id="b", weight=0.01, created_at=created, last_used=last_used)
    w = effective_weight(c, DEFAULT_CONFIG, now)
    assert w >= DEFAULT_CONFIG.NEW_CONDUIT_MIN_WEIGHT


def test_effective_weight_ephemeral_decays_fastest():
    """Same elapsed time, ephemeral loses more signal than working."""
    created = datetime(2026, 4, 1, tzinfo=timezone.utc)
    last_used = datetime(2026, 4, 10, tzinfo=timezone.utc)
    now = last_used + timedelta(hours=48)
    w_eph = Conduit(from_id="a", to_id="b", weight=0.8, decay_class="ephemeral",
                    created_at=created, last_used=last_used)
    w_work = Conduit(from_id="c", to_id="d", weight=0.8, decay_class="working",
                     created_at=created, last_used=last_used)
    assert effective_weight(w_eph, DEFAULT_CONFIG, now) < effective_weight(w_work, DEFAULT_CONFIG, now)


# --- propagate: happy path -------------------------------------------------
def test_propagate_single_hop_reaches_grain(store):
    e = _entry(store, "python")
    g = _grain(store, "Python is great")
    _conduit(store, e.id, g.id, weight=0.5)

    result = propagate(store, [e.id])

    assert len(result.activated) == 1
    grain_id, signal = result.activated[0]
    assert grain_id == g.id
    assert signal == pytest.approx(0.5)
    assert len(result.trace) == 1
    assert result.trace[0].hop == 0


def test_propagate_two_hops_on_fresh_graph(store):
    """Section 5.1: a fresh graph (all conduits at INITIAL_WEIGHT_SCALE)
    reliably propagates 2 hops."""
    e = _entry(store, "ai")
    g1 = _grain(store, "hop1")
    g2 = _grain(store, "hop2")
    _conduit(store, e.id, g1.id, weight=DEFAULT_CONFIG.INITIAL_ENTRY_WEIGHT)
    _conduit(store, g1.id, g2.id, weight=DEFAULT_CONFIG.INITIAL_WEIGHT_SCALE)

    result = propagate(store, [e.id])
    ids = [gid for gid, _ in result.activated]
    assert g1.id in ids
    assert g2.id in ids  # 2-hop reach confirmed -- spec Section 5.1 invariant.


def test_propagate_three_hops_die_on_fresh_graph(store):
    """Section 5.1: a fresh graph does NOT reach hop 3. This is intended --
    reinforcement grows useful paths; decay kills unused ones."""
    e = _entry(store, "ai")
    g1 = _grain(store, "h1")
    g2 = _grain(store, "h2")
    g3 = _grain(store, "h3")
    for src, dst in ((e.id, g1.id), (g1.id, g2.id), (g2.id, g3.id)):
        _conduit(store, src, dst, weight=DEFAULT_CONFIG.INITIAL_WEIGHT_SCALE)

    result = propagate(store, [e.id])
    ids = [gid for gid, _ in result.activated]
    assert g3.id not in ids


def test_propagate_respects_max_hops(store):
    """Even on a reinforced graph where signal would otherwise persist,
    MAX_HOPS caps the reach."""
    cfg = Config(MAX_HOPS=2, INITIAL_WEIGHT_SCALE=0.95)  # trick so signal stays strong
    e = _entry(store, "x")
    grains = [_grain(store, f"g{i}") for i in range(5)]
    _conduit(store, e.id, grains[0].id, weight=0.95, decay_class="core")
    for i in range(4):
        _conduit(store, grains[i].id, grains[i + 1].id, weight=0.95, decay_class="core")

    result = propagate(store, [e.id], cfg)
    ids = {gid for gid, _ in result.activated}
    # hop 0 = grains[0], hop 1 = grains[1]; grains[2+] beyond cap.
    assert grains[0].id in ids
    assert grains[1].id in ids
    assert grains[2].id not in ids


def test_propagate_below_threshold_dies(store):
    """A conduit whose effective weight puts initial_signal below threshold
    never enters the frontier."""
    e = _entry(store, "x")
    g = _grain(store, "g")
    # With weight=0.1, initial_signal = 1.0 * 0.1 * 1.0 = 0.1 < 0.15 threshold.
    _conduit(store, e.id, g.id, weight=0.10)

    result = propagate(store, [e.id])
    assert result.activated == []


def test_propagate_excludes_dormant_grain(store):
    e = _entry(store, "x")
    g = _grain(store, "sleepy", status="dormant")
    _conduit(store, e.id, g.id, weight=0.5)

    result = propagate(store, [e.id])
    assert result.activated == []


def test_propagate_excludes_quarantined_grain(store):
    e = _entry(store, "x")
    g = _grain(store, "bad", status="quarantined")
    _conduit(store, e.id, g.id, weight=0.5)

    assert propagate(store, [e.id]).activated == []


def test_propagate_signal_accumulates_across_paths(store):
    """Two entries pointing at the same grain should sum their signal."""
    e1 = _entry(store, "python")
    e2 = _entry(store, "programming")
    g = _grain(store, "Py")
    _conduit(store, e1.id, g.id, weight=0.5)
    _conduit(store, e2.id, g.id, weight=0.5)

    result = propagate(store, [e1.id, e2.id])
    assert len(result.activated) == 1
    assert result.activated[0][1] == pytest.approx(1.0)  # 0.5 + 0.5


def test_propagate_dedupes_visited_conduits(store):
    """A conduit must not be traversed twice even if reachable via two paths."""
    e = _entry(store, "x")
    g1 = _grain(store, "g1")
    g2 = _grain(store, "g2")
    # e -> g1, e -> g2, g1 -> g2 (so g2 has two inbound paths of depth 1 and 2).
    _conduit(store, e.id, g1.id, weight=0.5)
    _conduit(store, e.id, g2.id, weight=0.5)
    _conduit(store, g1.id, g2.id, weight=0.5)

    result = propagate(store, [e.id])
    # Each conduit appears at most once in trace.
    ids = [t.conduit_id for t in result.trace]
    assert len(ids) == len(set(ids))


def test_propagate_entry_affinity_boosts_first_hop(store):
    """Entry.affinities[conduit_id] scales the initial signal on that conduit."""
    g = _grain(store, "g")
    c = Conduit(from_id="placeholder", to_id=g.id, weight=0.5)
    # Use real IDs so they link up.
    e = Entry(feature="x", affinities={c.id: 2.0})
    store.insert_entry(e)
    c.from_id = e.id
    store.insert_conduit(c)

    result = propagate(store, [e.id])
    assert result.activated[0][1] == pytest.approx(0.5 * 2.0)  # 1.0 * 0.5 * 2.0


def test_propagate_exploration_boost_affects_injection_only(store):
    """EXPLORATION_BOOST multiplies the injected signal only, not subsequent hops."""
    e = _entry(store, "x")
    g = _grain(store, "g")
    _conduit(store, e.id, g.id, weight=0.5)

    normal = propagate(store, [e.id])
    boosted = propagate(store, [e.id], exploration_boost=DEFAULT_CONFIG.EXPLORATION_BOOST)
    assert boosted.activated[0][1] == pytest.approx(normal.activated[0][1] * DEFAULT_CONFIG.EXPLORATION_BOOST)


def test_propagate_returns_empty_for_unknown_entry(store):
    assert propagate(store, ["does-not-exist"]).activated == []


def test_propagate_caps_at_top_k(store):
    cfg = Config(TOP_K=2)
    e = _entry(store, "x")
    for i in range(5):
        g = _grain(store, f"g{i}")
        _conduit(store, e.id, g.id, weight=0.5)

    result = propagate(store, [e.id], cfg)
    assert len(result.activated) == 2


# --- retrieval_confidence --------------------------------------------------
def test_confidence_zero_for_empty_activation():
    assert retrieval_confidence([], []) == 0.0


def test_confidence_bounded_0_to_1():
    # Strong signal, good path, single grain -> high confidence but <= 1.0.
    from flux.propagation import TraceStep
    conf = retrieval_confidence(
        [("g1", 1.0)],
        [TraceStep(conduit_id="c", from_id="e", to_id="g1", signal=1.0, hop=0, effective_weight=0.9)],
    )
    assert 0.0 <= conf <= 1.0


def test_confidence_concentration_penalises_diffuse_results():
    """Same total signal spread across many grains should score lower on
    concentration than a single-dominant activation."""
    from flux.propagation import TraceStep
    peaked = retrieval_confidence(
        [("g1", 0.9), ("g2", 0.05), ("g3", 0.05)],
        [TraceStep(conduit_id="c", from_id="e", to_id="g1", signal=0.9, hop=0, effective_weight=0.9)],
    )
    diffuse = retrieval_confidence(
        [("g1", 0.34), ("g2", 0.33), ("g3", 0.33)],
        [TraceStep(conduit_id="c", from_id="e", to_id="g1", signal=0.34, hop=0, effective_weight=0.9)],
    )
    assert peaked > diffuse


# --- bidirectional shortcut propagation (Section 13.8) ---------------------
def test_bidirectional_shortcut_propagates_from_stored_from_side(store):
    """A shortcut stored as (g1→g2, bidirectional) must propagate signal from
    g1 to g2 -- the forward direction, equivalent to a normal forward edge."""
    e = _entry(store, "seed")
    g1 = _grain(store, "g1")
    g2 = _grain(store, "g2")
    _conduit(store, e.id, g1.id, weight=DEFAULT_CONFIG.INITIAL_ENTRY_WEIGHT)
    _conduit(store, g1.id, g2.id,
             weight=DEFAULT_CONFIG.INITIAL_WEIGHT_SCALE,
             direction="bidirectional")

    result = propagate(store, [e.id])
    ids = [gid for gid, _ in result.activated]
    assert g1.id in ids and g2.id in ids


def test_bidirectional_shortcut_propagates_reverse_direction(store):
    """Shortcut stored as (g1→g2, bidirectional) must propagate signal g2→g1
    too -- the shortcut direction property §13.8 specifies."""
    e = _entry(store, "seed")
    g1 = _grain(store, "g1")
    g2 = _grain(store, "g2")
    # Entry reaches g2 directly; shortcut stored as g1→g2 but bidirectional
    # so the reverse hop g2→g1 must fire.
    _conduit(store, e.id, g2.id, weight=DEFAULT_CONFIG.INITIAL_ENTRY_WEIGHT)
    _conduit(store, g1.id, g2.id,
             weight=DEFAULT_CONFIG.INITIAL_WEIGHT_SCALE,
             direction="bidirectional")

    result = propagate(store, [e.id])
    ids = [gid for gid, _ in result.activated]
    assert g2.id in ids
    assert g1.id in ids, "bidirectional shortcut did not propagate in reverse"


def test_forward_only_conduit_does_not_propagate_reverse(store):
    """Plain forward conduits must NOT flow in reverse (entry-point directional
    gate invariant)."""
    e = _entry(store, "seed")
    g1 = _grain(store, "g1")
    g2 = _grain(store, "g2")
    _conduit(store, e.id, g2.id, weight=DEFAULT_CONFIG.INITIAL_ENTRY_WEIGHT)
    _conduit(store, g1.id, g2.id,
             weight=DEFAULT_CONFIG.INITIAL_WEIGHT_SCALE)  # default direction='forward'

    result = propagate(store, [e.id])
    ids = [gid for gid, _ in result.activated]
    assert g2.id in ids
    assert g1.id not in ids


def test_trace_records_traversal_direction_not_storage_order(store):
    """When a bidirectional shortcut (stored g1→g2) is traversed g2→g1, the
    trace step must report from_id=g2, to_id=g1. Reinforcement relies on this
    to know which grain was landed on at each hop."""
    e = _entry(store, "seed")
    g1 = _grain(store, "g1")
    g2 = _grain(store, "g2")
    _conduit(store, e.id, g2.id, weight=DEFAULT_CONFIG.INITIAL_ENTRY_WEIGHT)
    _conduit(store, g1.id, g2.id,
             weight=DEFAULT_CONFIG.INITIAL_WEIGHT_SCALE,
             direction="bidirectional")

    result = propagate(store, [e.id])
    # Find the hop-1 step that activated g1.
    reverse_steps = [t for t in result.trace if t.hop == 1 and t.to_id == g1.id]
    assert len(reverse_steps) == 1
    assert reverse_steps[0].from_id == g2.id
