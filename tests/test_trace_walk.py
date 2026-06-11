"""Tests for retrieval trace replay data."""
from __future__ import annotations

import json

from flux.graph import Conduit, Entry, Grain, iso, new_id, utcnow
from flux.visualization import trace_walk


def test_trace_walk_resolves_labels_kinds_and_order(store):
    entry = Entry(feature="python")
    g1 = Grain(
        content="Alpha grain content used as the first replay target.",
        provenance="user_stated",
    )
    g2 = Grain(
        content="Beta grain content used as the second replay target.",
        provenance="ai_stated",
    )
    store.insert_entry(entry)
    store.insert_grain(g1)
    store.insert_grain(g2)
    conduit = Conduit(from_id=g1.id, to_id=g2.id, weight=0.7)
    store.insert_conduit(conduit)

    trace_id = new_id()
    created_at = iso(utcnow())
    trace_data = json.dumps([
        {
            "from_id": g1.id,
            "to_id": g2.id,
            "conduit_id": conduit.id,
            "hop": 2,
            "signal": 0.42,
        },
        {
            "from_id": entry.id,
            "to_id": g1.id,
            "conduit_id": "entry-hop",
            "hop": 1,
            "signal": 1.0,
        },
    ])
    store.conn.execute(
        "INSERT INTO traces (id, query_text, created_at, trace_data) "
        "VALUES (?, ?, ?, ?)",
        (trace_id, "python memory", created_at, trace_data),
    )

    data = trace_walk(store, trace_id)

    assert data["trace_id"] == trace_id
    assert data["query"] == "python memory"
    assert data["created_at"] == created_at
    assert [step["hop"] for step in data["steps"]] == [1, 2]
    assert [step["signal"] for step in data["steps"]] == [1.0, 0.42]

    first, second = data["steps"]
    assert first["from"] == {"id": entry.id, "kind": "entry", "label": "python"}
    assert first["to"] == {
        "id": g1.id,
        "kind": "grain",
        "label": g1.content[:90],
    }
    assert second["from"] == {
        "id": g1.id,
        "kind": "grain",
        "label": g1.content[:90],
    }
    assert second["to"] == {
        "id": g2.id,
        "kind": "grain",
        "label": g2.content[:90],
    }


def test_trace_walk_not_found(store):
    assert trace_walk(store, "missing-trace") == {"error": "not found"}
