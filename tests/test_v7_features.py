"""Tests for graded feedback strength, provenance skew detection, the
de-noised highway growth signal, and the RemoteService thin client."""
from __future__ import annotations

import json
from datetime import timedelta

from mocks import MockEmbeddingBackend

import flux.remote as remote
from flux.config import Config
from flux.graph import Conduit, Grain, new_id, utcnow, iso
from flux.health import flux_health, log_event
from flux.remote import RemoteService
from flux.retrieval import flux_feedback, flux_store_ex


# ------------------------------------------------------------ graded strength

_setup_counter = [0]


def _make_trace(store, grain_id: str, conduit: Conduit) -> str:
    trace_id = new_id()
    trace_data = json.dumps([{
        "from_id": conduit.from_id, "to_id": conduit.to_id,
        "conduit_id": conduit.id, "hop": 1, "signal": 1.0,
    }])
    store.conn.execute(
        "INSERT INTO traces (id, query_text, created_at, trace_data) "
        "VALUES (?, ?, ?, ?)",
        (trace_id, "q", iso(utcnow()), trace_data),
    )
    return trace_id


def _setup_feedback_target(store, weight=0.5):
    emb = MockEmbeddingBackend()
    _setup_counter[0] += 1
    n = _setup_counter[0]
    a, _ = flux_store_ex(f"Grain alpha content variant {n}.", store=store,
                         llm=None, emb=emb)
    b, _ = flux_store_ex(f"Grain beta content number {n}, entirely different.",
                         store=store, llm=None, emb=emb)
    conduit = Conduit(from_id=a, to_id=b, weight=weight,
                      created_at=utcnow(), last_used=utcnow())
    store.insert_conduit(conduit)
    trace_id = _make_trace(store, b, conduit)
    return b, conduit, trace_id


def _conduit_weight(store, conduit_id):
    row = store.conn.execute(
        "SELECT weight FROM conduits WHERE id=?", (conduit_id,),
    ).fetchone()
    return row["weight"] if row else None


def test_full_strength_reinforces_more_than_half(store):
    grain, conduit, trace_id = _setup_feedback_target(store)
    flux_feedback(trace_id, grain, True, store=store, strength=1.0)
    full = _conduit_weight(store, conduit.id)

    grain2, conduit2, trace2 = _setup_feedback_target(store)
    flux_feedback(trace2, grain2, True, store=store, strength=0.5)
    half = _conduit_weight(store, conduit2.id)

    assert full > half > 0.5  # both reinforced, full more strongly


def test_half_strength_penalty_is_gentler(store):
    grain, conduit, trace_id = _setup_feedback_target(store)
    flux_feedback(trace_id, grain, False, store=store, strength=1.0)
    full = _conduit_weight(store, conduit.id)

    grain2, conduit2, trace2 = _setup_feedback_target(store)
    flux_feedback(trace2, grain2, False, store=store, strength=0.5)
    half = _conduit_weight(store, conduit2.id)

    assert full < half < 0.5  # both penalized, full more strongly


def test_strength_clamped(store):
    grain, conduit, trace_id = _setup_feedback_target(store)
    result = flux_feedback(trace_id, grain, True, store=store, strength=99.0)
    assert result.action == "reinforced"
    grain2, _, trace2 = _setup_feedback_target(store)
    result2 = flux_feedback(trace2, grain2, True, store=store, strength=0.0)
    assert result2.action == "reinforced"  # clamped to 0.1, direction kept


# ------------------------------------------------------- provenance skew

def test_provenance_skew_warning_fires(store):
    emb = MockEmbeddingBackend()
    for i in range(25):
        flux_store_ex(f"Suspiciously user-stated fact number {i}.",
                      store=store, llm=None, emb=emb,
                      provenance="user_stated", caller_id="olive:chat")
    health = flux_health(store, Config())
    summary = {c["caller_id"]: c for c in health["caller_provenance"]}
    assert summary["olive:chat"]["skewed"] is True
    assert any(w["signal"] == "provenance_skew" for w in health["active_warnings"])


def test_no_skew_with_mixed_provenance(store):
    emb = MockEmbeddingBackend()
    for i in range(30):
        prov = "user_stated" if i % 2 == 0 else "ai_stated"
        flux_store_ex(f"Mixed provenance fact number {i}.",
                      store=store, llm=None, emb=emb,
                      provenance=prov, caller_id="codex:chat")
    health = flux_health(store, Config())
    summary = {c["caller_id"]: c for c in health["caller_provenance"]}
    assert summary["codex:chat"]["skewed"] is False
    assert not any(w["signal"] == "provenance_skew" for w in health["active_warnings"])


# ------------------------------------------------- highway growth de-noising

def test_highway_growth_counts_distinct_lasting_conduits(store):
    now = utcnow()
    lasting = Conduit(from_id="a", to_id="b", weight=0.85,
                      created_at=now, last_used=now)
    store.insert_conduit(lasting)
    # Same conduit re-crosses 5 times; plus 3 events for a conduit that
    # decayed away (not in conduits table any more).
    for _ in range(5):
        log_event(store, "feedback", "highway_formed",
                  {"conduit_id": lasting.id}, now=now)
    for _ in range(3):
        log_event(store, "feedback", "highway_formed",
                  {"conduit_id": "gone-conduit"}, now=now)

    health = flux_health(store, Config())
    assert health["signals"]["highway_growth_rate"]["value"] == 1.0


# ----------------------------------------------------------- remote service

def test_remote_service_maps_store_and_feedback(monkeypatch):
    calls = []

    def fake_request(self, method, path, payload=None, caller_id="default"):
        calls.append((method, path.split("?")[0], payload, caller_id))
        if path.startswith("/store"):
            return {"grain_id": "g1", "status": "duplicate"}
        if path.startswith("/feedback"):
            return {"status": "ok", "trace_id": "t1", "grain_id": "g1",
                    "action": "reinforced", "signal": 1.5}
        if path.startswith("/health"):
            return {"status": "healthy"}
        if path.startswith("/pending_feedback"):
            return {"missing": 0}
        return {}

    monkeypatch.setattr(RemoteService, "_request", fake_request)
    svc = RemoteService(Config())

    assert svc.store_ex("x", "ai_stated", caller_id="c:r") == ("g1", "duplicate")
    fb = svc.feedback_sync("t1", "g1", True, caller_id="c:r", strength=0.5)
    assert fb.action == "reinforced"
    assert fb.effective_signal == 1.5
    assert svc.health()["status"] == "healthy"
    assert svc.pending_feedback("c:r")["missing"] == 0
    # strength forwarded in the POST body
    feedback_call = [c for c in calls if c[1] == "/feedback"][0]
    assert feedback_call[2]["strength"] == 0.5


def test_probe_service_false_when_nothing_listening():
    cfg = Config(REST_PORT=1)  # nothing listens on port 1
    assert remote.probe_service(cfg) is False


# ----------------------------------------------------------- vitals history

def test_flux_health_persists_snapshot(store):
    flux_health(store, Config())
    count = store.conn.execute("SELECT COUNT(*) FROM health_log").fetchone()[0]
    assert count >= 10  # one row per signal
    distinct_ts = store.conn.execute(
        "SELECT COUNT(DISTINCT timestamp) FROM health_log").fetchone()[0]
    assert distinct_ts == 1


def test_health_snapshot_throttled(store):
    flux_health(store, Config())
    first = store.conn.execute("SELECT COUNT(*) FROM health_log").fetchone()[0]
    flux_health(store, Config())  # immediately again -> throttled
    second = store.conn.execute("SELECT COUNT(*) FROM health_log").fetchone()[0]
    assert second == first


def test_vitals_history_shape(store):
    from flux.health import vitals_history
    flux_health(store, Config())
    v = vitals_history(store, hours=24)
    assert "retrieval_success_rate" in v["series"]
    point = v["series"]["retrieval_success_rate"][0]
    assert len(point) == 3  # [timestamp, value, healthy]
    assert "orphan_rate" in v["ranges"]
    assert v["window_hours"] == 24


# ----------------------------------------------------------- chronicle

def test_chronicle_data_shape(store):
    from flux.visualization import chronicle_data
    emb = MockEmbeddingBackend()
    ids = []
    for i in range(5):
        gid, _ = flux_store_ex(f"Chronicle test grain number {i}.",
                               store=store, llm=None, emb=emb)
        ids.append(gid)
    d = chronicle_data(store)
    assert len(d["grains"]) == 5
    assert d["totals"]["all_conduits"] >= d["totals"]["grain_conduits"]
    g = d["grains"][0]
    assert len(g) == len(d["grain_fields"])
    x, y = g[1], g[2]
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    # conduits reference grain indices
    for c in d["conduits"]:
        assert 0 <= c[0] < 5 and 0 <= c[1] < 5
