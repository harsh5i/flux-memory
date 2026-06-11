"""Epistemic layer: contradiction, confidence, typed links, tombstones.

Most memory systems accumulate contradictions and retrieve both stale and
current truth side by side. Flux instead reasons about *whether what it knows
is still true*:

  - On store, a new grain that is semantically very close to an existing one
    is judged by the LLM: do they contradict, does one supersede the other,
    are they duplicates, or independent? Contradiction/supersession creates a
    typed conduit and lowers the loser's confidence.
  - confidence is a per-grain epistemic trust score (0..1) that evolves —
    raised by corroboration, lowered by contradiction/supersession, and
    decayed for unconfirmed ai_inferred claims with age. It scales retrieval.
  - Conduits carry a relation type (related / contradicts / supersedes /
    supports / caused_by) so the graph encodes *why* things connect.
  - When a grain that mattered is archived, a one-line tombstone grain is left
    so the fact that something was once known survives its details.

The LLM judge runs only when an embedding near-neighbour already exists above
EPISTEMIC_CHECK_SIMILARITY, so the common store path pays nothing.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

from .config import Config, DEFAULT_CONFIG
from .embedding import EmbeddingBackend, EmbeddingIndex
from .graph import Conduit, Grain, parse_iso, utcnow
from .health import log_event
from .llm import LLMBackend
from .storage import FluxStore

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """Two memory statements are semantically similar. Classify their relationship.

A (existing, stored {age}): {a}
B (new): {b}

Reply with ONE word only:
- CONTRADICT  — they assert incompatible facts about the same thing
- SUPERSEDE   — same subject, B is an updated/current version of A (A is now stale)
- DUPLICATE   — they say the same thing
- INDEPENDENT — related topic but not in conflict

Answer:"""

_VALID = {"CONTRADICT", "SUPERSEDE", "DUPLICATE", "INDEPENDENT"}


def _clamp(v: float, lo: float = 0.02, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _set_confidence(store: FluxStore, grain_id: str, value: float) -> None:
    store.conn.execute("UPDATE grains SET confidence = ? WHERE id = ?",
                       (_clamp(value), grain_id))


def _get_confidence(store: FluxStore, grain_id: str) -> float:
    row = store.conn.execute("SELECT confidence FROM grains WHERE id = ?",
                             (grain_id,)).fetchone()
    return float(row["confidence"]) if row and row["confidence"] is not None else 1.0


def effective_confidence(grain_row, cfg: Config, now: datetime) -> float:
    """Confidence used in retrieval scoring: stored confidence, with an extra
    age-based decay for unconfirmed ai_inferred grains (they grow less trusted
    until corroborated)."""
    base = float(grain_row["confidence"]) if grain_row["confidence"] is not None else 1.0
    if grain_row["provenance"] == "ai_inferred":
        try:
            created = parse_iso(grain_row["created_at"])
            age_days = max(0.0, (now - created).total_seconds() / 86400.0)
            half = cfg.AI_INFERRED_CONFIDENCE_HALFLIFE_DAYS
            base *= 0.5 + 0.5 * math.pow(0.5, age_days / half)  # 1.0 → 0.5 over time
        except Exception:
            pass
    return _clamp(base)


def check_on_store(store: FluxStore, grain: Grain, llm: LLMBackend | None,
                   emb: EmbeddingBackend, cfg: Config = DEFAULT_CONFIG,
                   now: datetime | None = None,
                   index: EmbeddingIndex | None = None) -> dict | None:
    """Run the epistemic check for a freshly-stored grain.

    Returns a dict describing the action taken, or None if nothing fired.
    Never raises into the store path.
    """
    if not cfg.EPISTEMIC_CHECK_ENABLED or llm is None:
        return None
    now = now or utcnow()
    try:
        vec = emb.embed(grain.content)
        if index is not None:
            neighbours = index.top_k(vec, 3)
        else:
            from .embedding import load_all_embeddings, top_k_nearest
            gids, matrix = load_all_embeddings(store)
            neighbours = top_k_nearest(vec, gids, matrix, 3)
    except Exception as exc:
        logger.error("epistemic: neighbour lookup failed: %s", exc)
        return None

    for other_id, sim in neighbours:
        if other_id == grain.id or sim < cfg.EPISTEMIC_CHECK_SIMILARITY:
            continue
        other = store.get_grain(other_id)
        if other is None or other.status != "active":
            continue
        verdict = _judge(llm, other, grain, now)
        if verdict in (None, "INDEPENDENT", "DUPLICATE"):
            return None  # dedup already handles duplicates upstream
        return _apply(store, other, grain, verdict, cfg, now)
    return None


def _judge(llm: LLMBackend, existing: Grain, new: Grain, now: datetime) -> str | None:
    try:
        age = _age_phrase(existing.created_at, now)
        raw = llm.complete(_JUDGE_PROMPT.format(
            age=age, a=existing.content[:400], b=new.content[:400])).strip().upper()
    except Exception as exc:
        logger.error("epistemic: judge LLM call failed: %s", exc)
        return None
    for token in _VALID:
        if token in raw:
            return token
    return None


def _apply(store: FluxStore, existing: Grain, new: Grain, verdict: str,
           cfg: Config, now: datetime) -> dict:
    """Create the typed conduit and adjust confidence per verdict."""
    if verdict == "SUPERSEDE":
        relation = "supersedes"          # new supersedes existing
        from_id, to_id = new.id, existing.id
        penalty = cfg.SUPERSEDED_CONFIDENCE_PENALTY
        loser = existing.id
    else:  # CONTRADICT
        relation = "contradicts"
        from_id, to_id = new.id, existing.id
        penalty = cfg.CONTRADICTION_CONFIDENCE_PENALTY
        # the older / lower-provenance grain loses trust
        loser = _weaker_grain(existing, new, cfg)

    try:
        store.insert_conduit(Conduit(
            from_id=from_id, to_id=to_id, weight=0.7,
            created_at=now, last_used=now,
            direction="forward", decay_class="working", relation=relation,
        ))
    except Exception:
        pass

    new_conf = _clamp(_get_confidence(store, loser) * penalty)
    _set_confidence(store, loser, new_conf)
    log_event(store, "feedback", "epistemic_conflict", {
        "relation": relation,
        "existing_grain_id": existing.id,
        "new_grain_id": new.id,
        "loser_grain_id": loser,
        "loser_confidence": round(new_conf, 4),
    }, now=now)
    return {"relation": relation, "loser": loser, "loser_confidence": new_conf}


def _weaker_grain(a: Grain, b: Grain, cfg: Config) -> str:
    """Pick the grain to distrust: lower provenance multiplier, then older."""
    ma = cfg.provenance_multiplier(a.provenance)
    mb = cfg.provenance_multiplier(b.provenance)
    if ma != mb:
        return a.id if ma < mb else b.id
    return a.id if a.created_at <= b.created_at else b.id


def confirm(store: FluxStore, grain_id: str, cfg: Config = DEFAULT_CONFIG) -> None:
    """Raise a grain's confidence — called when corroborated (positive feedback
    or a SUPPORTS relation)."""
    _set_confidence(store, grain_id,
                    _get_confidence(store, grain_id) * cfg.CONFIRMATION_CONFIDENCE_BOOST)


def _age_phrase(created: datetime, now: datetime) -> str:
    days = (now - created).total_seconds() / 86400.0
    if days < 1:
        return "today"
    if days < 30:
        return f"{int(days)} days ago"
    return f"{int(days / 30)} months ago"


# ----------------------------------------------------------------- tombstones

def leave_tombstone(store: FluxStore, grain: Grain, cfg: Config,
                    now: datetime) -> str | None:
    """Before a grain is archived, leave a one-line residue so the fact that it
    was once known survives. Returns the tombstone grain id, or None."""
    if not cfg.TOMBSTONE_ENABLED:
        return None
    degree = store.conn.execute(
        "SELECT COUNT(*) AS n FROM conduits WHERE from_id=? OR to_id=?",
        (grain.id, grain.id),
    ).fetchone()["n"]
    if degree < cfg.TOMBSTONE_MIN_DEGREE:
        return None
    from .graph import new_id
    summary = grain.content.strip().split("\n")[0][:120]
    tomb = Grain(
        id=new_id(),
        content=f"[forgotten {now.strftime('%Y-%m-%d')}] once knew: {summary}",
        provenance="ai_inferred",
        decay_class="working",
        created_at=now,
    )
    store.insert_grain(tomb)
    store.conn.execute("UPDATE grains SET source_tags=? WHERE id=?",
                       ('["tombstone"]', tomb.id))
    _set_confidence(store, tomb.id, 0.4)
    log_event(store, "decay", "tombstone_left", {
        "archived_grain_id": grain.id,
        "tombstone_grain_id": tomb.id,
    }, now=now)
    return tomb.id
