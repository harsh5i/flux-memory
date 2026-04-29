"""Insert/retrieve round-trips for the core entities."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from flux.graph import Cluster, Conduit, Entry, Grain, Trace


# --------------------------------------------------------------------- grains
def test_insert_and_get_grain_defaults(store):
    g = Grain(content="User prefers Python", provenance="user_stated")
    store.insert_grain(g)

    fetched = store.get_grain(g.id)
    assert fetched is not None
    assert fetched.id == g.id
    assert fetched.content == "User prefers Python"
    assert fetched.provenance == "user_stated"
    assert fetched.decay_class == "working"
    assert fetched.status == "active"
    assert fetched.context_spread == 0
    assert fetched.confidence == 1.0
    assert fetched.dormant_since is None
    assert isinstance(fetched.created_at, datetime)
    assert fetched.created_at.tzinfo is timezone.utc


def test_grain_provenance_is_required(store):
    # provenance is NOT NULL in the schema; the dataclass makes it a required
    # positional kwarg. Smuggling a None through the insert path must fail.
    with pytest.raises((sqlite3.IntegrityError, TypeError)):
        store.insert_grain(
            Grain(content="x", provenance=None)  # type: ignore[arg-type]
        )


def test_grain_provenance_tags_roundtrip(store):
    for prov in ("user_stated", "ai_stated", "ai_inferred", "external_source"):
        g = Grain(content=f"c-{prov}", provenance=prov)  # type: ignore[arg-type]
        store.insert_grain(g)
        assert store.get_grain(g.id).provenance == prov


def test_get_grain_returns_none_for_unknown_id(store):
    assert store.get_grain("does-not-exist") is None


def test_iter_grains_filter_by_status(store):
    store.insert_grain(Grain(content="a", provenance="user_stated"))
    store.insert_grain(Grain(content="b", provenance="user_stated", status="dormant"))
    store.insert_grain(Grain(content="c", provenance="user_stated", status="archived"))

    active = list(store.iter_grains(status="active"))
    assert len(active) == 1 and active[0].content == "a"

    all_ = list(store.iter_grains())
    assert len(all_) == 3


# -------------------------------------------------------------------- conduits
def test_insert_and_get_conduit(store):
    # Insert two grains first so from_id/to_id are meaningful (not required by FK
    # but aligns with real usage).
    g1 = Grain(content="g1", provenance="user_stated")
    g2 = Grain(content="g2", provenance="user_stated")
    store.insert_grain(g1)
    store.insert_grain(g2)

    c = Conduit(from_id=g1.id, to_id=g2.id, weight=0.5, direction="bidirectional")
    store.insert_conduit(c)

    fetched = store.get_conduit(c.id)
    assert fetched is not None
    assert fetched.from_id == g1.id
    assert fetched.to_id == g2.id
    assert fetched.weight == pytest.approx(0.5)
    assert fetched.direction == "bidirectional"
    assert fetched.decay_class == "working"
    assert fetched.use_count == 0
    assert fetched.created_at.tzinfo is timezone.utc
    assert fetched.last_used.tzinfo is timezone.utc


def test_conduit_unique_constraint_on_from_to(store):
    g1 = Grain(content="g1", provenance="user_stated")
    g2 = Grain(content="g2", provenance="user_stated")
    store.insert_grain(g1)
    store.insert_grain(g2)

    store.insert_conduit(Conduit(from_id=g1.id, to_id=g2.id, weight=0.25))
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_conduit(Conduit(from_id=g1.id, to_id=g2.id, weight=0.5))


def test_conduit_created_at_and_last_used_populated(store):
    g1 = Grain(content="g1", provenance="user_stated")
    g2 = Grain(content="g2", provenance="user_stated")
    store.insert_grain(g1)
    store.insert_grain(g2)

    before = datetime.now(timezone.utc) - timedelta(seconds=2)
    c = Conduit(from_id=g1.id, to_id=g2.id)
    store.insert_conduit(c)
    fetched = store.get_conduit(c.id)

    # Stored timestamps must be present and close to now.
    assert fetched.created_at >= before
    assert fetched.last_used >= before


def test_outgoing_and_inbound_conduits(store):
    g1 = Grain(content="g1", provenance="user_stated")
    g2 = Grain(content="g2", provenance="user_stated")
    g3 = Grain(content="g3", provenance="user_stated")
    for g in (g1, g2, g3):
        store.insert_grain(g)
    store.insert_conduit(Conduit(from_id=g1.id, to_id=g2.id))
    store.insert_conduit(Conduit(from_id=g1.id, to_id=g3.id))
    store.insert_conduit(Conduit(from_id=g2.id, to_id=g3.id))

    out_g1 = {c.to_id for c in store.outgoing_conduits(g1.id)}
    assert out_g1 == {g2.id, g3.id}

    in_g3 = {c.from_id for c in store.inbound_conduits(g3.id)}
    assert in_g3 == {g1.id, g2.id}


def test_get_conduit_by_pair(store):
    g1 = Grain(content="g1", provenance="user_stated")
    g2 = Grain(content="g2", provenance="user_stated")
    store.insert_grain(g1)
    store.insert_grain(g2)
    c = Conduit(from_id=g1.id, to_id=g2.id, weight=0.3)
    store.insert_conduit(c)

    fetched = store.get_conduit_by_pair(g1.id, g2.id)
    assert fetched is not None and fetched.id == c.id
    # Directional: (g2, g1) is a different pair and should miss.
    assert store.get_conduit_by_pair(g2.id, g1.id) is None


# --------------------------------------------------------------------- entries
def test_insert_and_get_entry(store):
    e = Entry(feature="AI", affinities={"conduit_123": 1.2})
    store.insert_entry(e)

    fetched = store.get_entry(e.id)
    assert fetched is not None
    assert fetched.feature == "AI"
    assert fetched.affinities == {"conduit_123": 1.2}


def test_entry_feature_is_unique(store):
    store.insert_entry(Entry(feature="AI"))
    # Duplicate inserts are silently ignored (INSERT OR IGNORE) for thread safety.
    store.insert_entry(Entry(feature="AI"))
    count = store.conn.execute("SELECT COUNT(*) FROM entries WHERE feature = 'AI'").fetchone()[0]
    assert count == 1


def test_get_entry_by_feature(store):
    e = Entry(feature="framework")
    store.insert_entry(e)
    assert store.get_entry_by_feature("framework").id == e.id
    assert store.get_entry_by_feature("missing") is None


# -------------------------------------------------------------------- clusters
def test_insert_and_get_cluster(store):
    c = Cluster(size=7)
    store.insert_cluster(c)
    fetched = store.get_cluster(c.id)
    assert fetched is not None
    assert fetched.size == 7
    assert fetched.created_at.tzinfo is timezone.utc


# --------------------------------------------------------------------- traces
def test_insert_and_get_trace(store):
    t = Trace(
        query_text="What framework for my AI project?",
        hop_count=2,
        activated_grain_count=3,
        trace_data='[{"conduit_id":"c1","signal":0.42}]',
    )
    store.insert_trace(t)

    fetched = store.get_trace(t.id)
    assert fetched is not None
    assert fetched.query_text == "What framework for my AI project?"
    assert fetched.hop_count == 2
    assert fetched.activated_grain_count == 3
    assert '"signal":0.42' in fetched.trace_data
    assert fetched.feedback_at is None


# ------------------------------------------------------------------ transactions
def test_transaction_rollback_on_error(store):
    g = Grain(content="keeper", provenance="user_stated")
    store.insert_grain(g)

    with pytest.raises(sqlite3.IntegrityError):
        with store.transaction():
            store.insert_grain(Grain(content="ok", provenance="user_stated"))
            # Duplicate PK -> IntegrityError, which rolls back the whole tx.
            store.insert_grain(Grain(id=g.id, content="dup", provenance="user_stated"))

    # The "ok" grain from inside the tx must not persist.
    rows = store.conn.execute("SELECT COUNT(*) AS n FROM grains").fetchone()
    assert rows["n"] == 1
