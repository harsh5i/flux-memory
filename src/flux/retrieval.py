"""High-level Python SDK surface for Flux Memory (Track 4, §13.5).

Three public channels exposed here:

  Write channel:
    flux_store(content, provenance, store, llm, emb, cfg) → grain_id

  Read channel:
    flux_retrieve(query, store, llm, emb, cfg) → RetrievalResult
    flux_feedback(trace_id, grain_id, useful, store, cfg) → FeedbackResult

  Health:
    flux_health is re-exported from health.py for convenience.

Admin channel (flux_purge etc.) lives in admin.py — not exposed here.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import Config, DEFAULT_CONFIG
from .embedding import EmbeddingBackend, vector_fallback
from .expansion import expand_results
from .extraction import decompose_query, extract_and_store_grains, store_atomic_grain
from .graph import Grain, Trace, new_id, utcnow
from .health import log_event, normalize_caller_id, pending_feedback_for_caller
from .llm import LLMBackend
from .promotion import check_promotion
from .propagation import PropagationResult, TraceStep, propagate, retrieval_confidence
from .reinforcement import penalize, reinforce
from .storage import FluxStore

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- result types

@dataclass
class RetrievalResult:
    grains: list[dict]          # [{id, content, provenance, decay_class, score}, ...]
    trace_id: str
    confidence: float
    fallback_triggered: bool
    hop_count: int
    features: list[str]
    expansion_candidates: list[dict] = field(default_factory=list)  # lateral discovery results


@dataclass
class FeedbackResult:
    trace_id: str
    grain_id: str
    useful: bool
    effective_signal: float     # net signal after modulation
    action: str                 # "reinforced" | "penalized" | "skipped"


# ----------------------------------------------------------------- write channel

def flux_store(
    content: str,
    provenance: str = "user_stated",
    *,
    store: FluxStore,
    llm: LLMBackend | None = None,
    emb: EmbeddingBackend | None = None,
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
    caller_id: str = "default",
) -> str:
    """Insert a single grain into the store and create bootstrap conduits.

    Returns the new grain's ID.

    If llm and emb are provided, bootstrap conduits are created via embedding
    similarity the same way extract_and_store_grains does. If they are omitted,
    the grain is stored bare (no conduits, no entry connections).
    """
    now = now or utcnow()
    content = content.strip()
    if not content:
        raise ValueError("flux_store: content must not be empty")

    valid_provenance = {"user_stated", "ai_stated", "ai_inferred", "external_source"}
    if provenance not in valid_provenance:
        raise ValueError(f"flux_store: provenance must be one of {valid_provenance}")

    if llm is not None and emb is not None:
        # Route through extract_and_store_grains using a synthetic "conversation turn"
        # where the user_message IS the content and ai_response is empty.
        grain_ids = extract_and_store_grains(
            user_message=content,
            ai_response="",
            llm=llm,
            embedding_backend=emb,
            store=store,
            cfg=cfg,
            now=now,
        )
        if grain_ids:
            return grain_ids[0]
        # Fallback: the caller already supplied an atomic fact, so store it and
        # still wire it into the graph with embeddings and entry conduits.
        return store_atomic_grain(
            content,
            provenance,
            llm=llm,
            embedding_backend=emb,
            store=store,
            cfg=cfg,
            now=now,
        )

    grain = Grain(content=content, provenance=provenance, created_at=now)
    store.insert_grain(grain)
    log_event(store, "write", "grain_stored", {
        "grain_id": grain.id,
        "provenance": provenance,
        "content_len": len(content),
        "via": "flux_store_direct",
    }, now=now, caller_id=caller_id)
    return grain.id


# ----------------------------------------------------------------- read channel

def flux_retrieve(
    query: str,
    *,
    store: FluxStore,
    llm: LLMBackend,
    emb: EmbeddingBackend,
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
    caller_id: str = "default",
) -> RetrievalResult:
    """Retrieve the top-k grains most relevant to query.

    Steps:
      1. Decompose query into feature keywords → entry IDs.
      2. Propagate signal from entry IDs.
      3. If confidence < FALLBACK_CONFIDENCE_THRESHOLD, call vector fallback.
      4. Store trace to SQLite.
      5. Return RetrievalResult with grains and trace_id.
    """
    now = now or utcnow()
    query = query.strip()
    caller_id = normalize_caller_id(caller_id, query)

    if cfg.FEEDBACK_ENFORCEMENT_ENABLED:
        pending = pending_feedback_for_caller(
            store,
            caller_id,
            now,
            grace_seconds=cfg.FEEDBACK_ENFORCEMENT_GRACE_SECONDS,
            lookback_days=cfg.TRACE_RETENTION_DAYS,
        )
        if pending["missing"] > 0:
            log_event(
                store,
                "feedback",
                "feedback_required_blocked",
                {
                    "caller_id": caller_id,
                    "missing": pending["missing"],
                    "pending_traces": pending["pending_traces"][:5],
                },
                now=now,
                caller_id=caller_id,
            )
            raise RuntimeError(
                "flux_feedback required before next retrieval for caller "
                f"'{caller_id}': {pending['missing']} pending feedback item(s)."
            )

    # 1. Query decomposition → entry IDs + features.
    entry_ids = decompose_query(query, llm, store, cfg=cfg, now=now)
    features = [store.get_entry(eid).feature for eid in entry_ids if store.get_entry(eid)]

    # 2. Propagate from entry IDs.
    result: PropagationResult = propagate(store, entry_ids, cfg=cfg, now=now)

    # 3. Confidence check → optional vector fallback.
    confidence = retrieval_confidence(result.activated, result.trace)
    fallback_triggered = False

    if confidence < cfg.FALLBACK_CONFIDENCE_THRESHOLD:
        merged = vector_fallback(store, query, emb, result.activated, cfg=cfg)
        result = PropagationResult(
            activated=merged,
            trace=result.trace,
        )
        fallback_triggered = True
        log_event(store, "retrieval", "fallback_triggered", {
            "query": query[:200],
            "confidence": confidence,
        }, now=now, caller_id=caller_id)

    # Recompute hop_count from trace.
    hop_count = max((step.hop for step in result.trace), default=0)

    # 4. Persist trace.
    trace_steps_json = json.dumps([
        {
            "conduit_id": step.conduit_id,
            "from_id": step.from_id,
            "to_id": step.to_id,
            "signal": step.signal,
            "hop": step.hop,
        }
        for step in result.trace
    ])
    trace = Trace(
        query_text=query,
        created_at=now,
        hop_count=hop_count,
        activated_grain_count=len(result.activated),
        trace_data=trace_steps_json,
    )
    store.insert_trace(trace)

    # 5. Context expansion (§11.11) — lateral discovery on low-confidence results.
    expansion_candidates = expand_results(store, result.activated, confidence, cfg, now)
    if expansion_candidates:
        log_event(store, "retrieval", "expansion_triggered", {
            "trace_id": trace.id,
            "candidates": len(expansion_candidates),
            "confidence": confidence,
        }, trace_id=trace.id, now=now, caller_id=caller_id)

    # 6. Build response.
    grains_out = []
    for grain_id, score in result.activated:
        g = store.get_grain(grain_id)
        if g is None:
            continue
        grains_out.append({
            "id": g.id,
            "content": g.content,
            "provenance": g.provenance,
            "decay_class": g.decay_class,
            "score": score,
            "source": "propagation",
        })

    log_event(store, "retrieval", "grains_returned", {
        "query": query[:200],
        "features": features,
        "grain_ids": [g["id"] for g in grains_out],
        "grains_count": len(grains_out),
        "hop_count": hop_count,
        "fallback_triggered": fallback_triggered,
        "expansion_count": len(expansion_candidates),
        "trace_id": trace.id,
    }, trace_id=trace.id, now=now, caller_id=caller_id)

    return RetrievalResult(
        grains=grains_out,
        trace_id=trace.id,
        confidence=confidence,
        fallback_triggered=fallback_triggered,
        hop_count=hop_count,
        features=features,
        expansion_candidates=expansion_candidates,
    )


# ----------------------------------------------------------------- feedback

def flux_feedback(
    trace_id: str,
    grain_id: str,
    useful: bool,
    *,
    store: FluxStore,
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
    caller_id: str = "default",
) -> FeedbackResult:
    """Apply multi-signal feedback for one grain from a retrieval trace (§7.1).

    Modulates the base AI-usage signal by:
      - Grain's historical usefulness ratio (trend signal, §7.1 Signal 3)
      - Grain's provenance (trust signal, §7.2)

    Then calls reinforce() or penalize() on the trace conduits.
    """
    now = now or utcnow()
    caller_id = normalize_caller_id(caller_id)

    trace = store.get_trace(trace_id)
    if trace is None:
        logger.warning("flux_feedback: trace %s not found", trace_id)
        return FeedbackResult(trace_id, grain_id, useful, 0.0, "skipped")

    grain = store.get_grain(grain_id)
    if grain is None:
        logger.warning("flux_feedback: grain %s not found", grain_id)
        return FeedbackResult(trace_id, grain_id, useful, 0.0, "skipped")

    # Reconstruct trace steps from stored JSON.
    trace_steps = _decode_trace_steps(trace.trace_data)

    # Multi-signal modulation (§7.1).
    base_signal = 1.0 if useful else -1.0
    usefulness_ratio = _get_usefulness_ratio(store, grain_id, window_days=7)
    trend_modulator = 0.5 + usefulness_ratio          # range 0.5–1.5
    provenance_modulator = cfg.provenance_multiplier(grain.provenance)
    effective_signal = base_signal * trend_modulator * provenance_modulator

    if effective_signal > 0:
        reinforce(store, trace_steps, [grain_id], cfg=cfg, now=now, trace_id=trace_id)
        check_promotion(store, grain_id, trace_steps, cfg=cfg, now=now, trace_id=trace_id)
        action = "reinforced"
    else:
        penalize(store, trace_steps, [grain_id], cfg=cfg, now=now, trace_id=trace_id)
        action = "penalized"

    # Record usefulness event so future ratio queries can use it.
    log_event(store, "feedback", "feedback_received", {
        "trace_id": trace_id,
        "grain_id": grain_id,
        "useful": useful,
        "effective_signal": round(effective_signal, 4),
        "provenance": grain.provenance,
        "usefulness_ratio": round(usefulness_ratio, 4),
        "action": action,
    }, trace_id=trace_id, now=now, caller_id=caller_id)

    if useful:
        log_event(
            store,
            "feedback",
            "retrieval_successful",
            {},
            trace_id=trace_id,
            now=now,
            caller_id=caller_id,
        )

    return FeedbackResult(
        trace_id=trace_id,
        grain_id=grain_id,
        useful=useful,
        effective_signal=effective_signal,
        action=action,
    )


# ----------------------------------------------------------------- helpers

def _decode_trace_steps(trace_data: str) -> list[TraceStep]:
    try:
        items = json.loads(trace_data)
        return [
            TraceStep(
                conduit_id=item["conduit_id"],
                from_id=item["from_id"],
                to_id=item["to_id"],
                signal=item["signal"],
                hop=item["hop"],
            )
            for item in items
        ]
    except Exception:
        return []


def _get_usefulness_ratio(store: FluxStore, grain_id: str, window_days: int = 7) -> float:
    """Compute (useful_count / total_retrieved) for grain over the past window_days."""
    from datetime import timedelta
    cutoff_iso = utcnow() - timedelta(days=window_days)
    from .graph import iso
    cutoff_str = iso(cutoff_iso)

    total_row = store.conn.execute(
        """
        SELECT COUNT(*) AS n FROM events
        WHERE category='feedback' AND event='feedback_received'
          AND json_extract(data, '$.grain_id') = ?
          AND timestamp >= ?
        """,
        (grain_id, cutoff_str),
    ).fetchone()
    total = total_row["n"] if total_row else 0
    if total == 0:
        return 0.5  # neutral prior when no history

    useful_row = store.conn.execute(
        """
        SELECT COUNT(*) AS n FROM events
        WHERE category='feedback' AND event='feedback_received'
          AND json_extract(data, '$.grain_id') = ?
          AND json_extract(data, '$.useful') = 1
          AND timestamp >= ?
        """,
        (grain_id, cutoff_str),
    ).fetchone()
    useful = useful_row["n"] if useful_row else 0
    return useful / total
