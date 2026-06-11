from __future__ import annotations

from mocks import MockEmbeddingBackend

from flux.health import log_event
from flux.retrieval import flux_store_ex
from flux.visualization import grain_dossier


def test_grain_dossier_shape_for_stored_grain(store):
    emb = MockEmbeddingBackend()
    content = "Dossier grain keeps the complete detailed content for inspection."
    grain_id, status = flux_store_ex(content, store=store, llm=None, emb=emb)
    assert status == "stored_wired"
    log_event(store, "retrieval", "grains_returned", {"grain_ids": [grain_id]})

    dossier = grain_dossier(store, grain_id)

    assert dossier["grain"] == {
        "id": grain_id,
        "content": content,
        "provenance": "user_stated",
        "decay_class": "working",
        "status": "active",
        "created_at": dossier["grain"]["created_at"],
    }
    assert dossier["degree"] > 0
    assert dossier["retrieval_count"] == 1
    assert isinstance(dossier["conduits"], list)
    assert len(dossier["conduits"]) <= 10
    conduit = dossier["conduits"][0]
    assert set(conduit) == {"other_id", "other_snippet", "weight", "direction"}
    assert len(conduit["other_snippet"]) <= 90
    assert conduit["direction"] in {"incoming", "outgoing", "bidirectional"}
    assert dossier["feedback"] == []


def test_grain_dossier_feedback_history(store):
    emb = MockEmbeddingBackend()
    grain_id, _ = flux_store_ex("Feedback history target grain.", store=store, llm=None, emb=emb)
    log_event(
        store,
        "feedback",
        "feedback_received",
        {
            "grain_id": grain_id,
            "action": "reinforced",
            "effective_signal": 1.25,
        },
        caller_id="codex:chat",
    )

    dossier = grain_dossier(store, grain_id)

    assert dossier["feedback"] == [{
        "timestamp": dossier["feedback"][0]["timestamp"],
        "action": "reinforced",
        "effective_signal": 1.25,
        "caller_id": "codex:chat",
    }]


def test_grain_dossier_not_found(store):
    assert grain_dossier(store, "missing-grain") == {"error": "not found"}
