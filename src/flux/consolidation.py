"""Dream cycle: memory consolidation (encode → decay → consolidate).

Flux already encodes (store) and forgets (decay). This module adds the third
phase of biological memory: periodically, dense clumps of semantically-related
working grains are abstracted into ONE higher-level grain, synthesized by the
service's own LLM backend. The sources are linked to the synthesis as evidence
and keep decaying naturally — episodic noise fades, semantic knowledge stays.

Runs inside the Flux daemon (like the decay pass and health tick) — callers
are never involved. Guardrails:
  - candidate detection is pure math (embeddings); the LLM only writes prose
  - the synthesis must embed within DREAM_CENTROID_MIN of the clump centroid,
    proving it actually summarizes the sources (hallucination gate)
  - provenance is ai_inferred: low trust multiplier until feedback earns more
  - at most DREAM_MAX_PER_CYCLE consolidations per cycle, all logged
  - source grains are tagged so they are never consolidated twice
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

import numpy as np

from .config import Config, DEFAULT_CONFIG
from .embedding import EmbeddingBackend, cosine_similarity, load_all_embeddings
from .extraction import store_atomic_grain_ex
from .graph import Conduit, iso, utcnow
from .health import log_event
from .llm import LLMBackend
from .storage import FluxStore

logger = logging.getLogger(__name__)

_CONSOLIDATED_TAG = "consolidated"

_SYNTHESIS_PROMPT = """You are consolidating an AI's memory. Below are {n} related memory \
fragments observed over time. Write ONE clear statement (at most 60 words) that captures \
the durable knowledge they collectively express — the pattern, fact, or conclusion that \
remains true after the individual moments are forgotten.

Rules:
- State only what the fragments support. Never invent names, numbers, or causes.
- No preamble, no quotes, no bullet points — output the single statement only.

Fragments:
{fragments}
"""


def _already_consolidated(tags_json: str | None) -> bool:
    try:
        return _CONSOLIDATED_TAG in (json.loads(tags_json or "[]") or [])
    except (TypeError, ValueError):
        return False


def find_candidates(store: FluxStore, cfg: Config = DEFAULT_CONFIG,
                    now: datetime | None = None) -> list[list[str]]:
    """Find clumps of related working grains worth consolidating. No LLM.

    Greedy: walk active working-class grains older than DREAM_MIN_GRAIN_AGE_HOURS
    that are not already consolidated; group each unclaimed grain with all
    unclaimed neighbours at cosine >= DREAM_SIMILARITY; keep clumps of at least
    DREAM_MIN_CLUSTER. Returns up to DREAM_MAX_PER_CYCLE clumps (grain id lists),
    largest first.
    """
    now = now or utcnow()
    age_cutoff = iso(now - timedelta(hours=cfg.DREAM_MIN_GRAIN_AGE_HOURS))

    rows = store.conn.execute(
        """
        SELECT id, source_tags FROM grains
        WHERE status = 'active' AND decay_class = 'working' AND created_at <= ?
        """,
        (age_cutoff,),
    ).fetchall()
    eligible = {r["id"] for r in rows if not _already_consolidated(r["source_tags"])}
    if len(eligible) < cfg.DREAM_MIN_CLUSTER:
        return []

    grain_ids, matrix = load_all_embeddings(store)
    keep = [i for i, gid in enumerate(grain_ids) if gid in eligible]
    if len(keep) < cfg.DREAM_MIN_CLUSTER:
        return []
    ids = [grain_ids[i] for i in keep]
    m = matrix[keep].astype(np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    m = m / norms

    sims = m @ m.T
    claimed: set[int] = set()
    clumps: list[list[str]] = []
    order = np.argsort(-(sims >= cfg.DREAM_SIMILARITY).sum(axis=1))  # densest first
    for i in order:
        if i in claimed:
            continue
        members = [j for j in np.where(sims[i] >= cfg.DREAM_SIMILARITY)[0]
                   if j not in claimed]
        if len(members) < cfg.DREAM_MIN_CLUSTER:
            continue
        claimed.update(members)
        clumps.append([ids[j] for j in members])
        if len(clumps) >= cfg.DREAM_MAX_PER_CYCLE:
            break
    return clumps


def consolidate_clump(store: FluxStore, grain_ids: list[str],
                      llm: LLMBackend, emb: EmbeddingBackend,
                      cfg: Config = DEFAULT_CONFIG,
                      now: datetime | None = None,
                      index=None) -> str | None:
    """Synthesize one clump into a higher-level grain. Returns its id or None.

    The synthesis is stored through the normal write path (embedding, dedup,
    bootstrap conduits), then wired to every source with evidence conduits.
    Sources are tagged consolidated and left to decay naturally.
    """
    now = now or utcnow()
    rows = store.conn.execute(
        f"SELECT id, content FROM grains WHERE id IN "
        f"({','.join('?' * len(grain_ids))})",
        grain_ids,
    ).fetchall()
    if len(rows) < cfg.DREAM_MIN_CLUSTER:
        return None
    contents = [r["content"] for r in rows]

    fragments = "\n".join(f"- {c[:300]}" for c in contents)
    try:
        synthesis = (llm.complete(_SYNTHESIS_PROMPT.format(
            n=len(contents), fragments=fragments)) or "").strip().strip('"')
    except Exception as exc:
        logger.error("dream: synthesis LLM call failed: %s", exc)
        return None
    if not synthesis or len(synthesis) > 600:
        logger.warning("dream: synthesis rejected (empty or too long)")
        return None

    # Hallucination gate: the synthesis must live where its sources live.
    try:
        synth_vec = emb.embed(synthesis)
        source_vecs = [emb.embed(c) for c in contents]
        centroid = np.mean(np.array(source_vecs, dtype=np.float64), axis=0).tolist()
        closeness = cosine_similarity(synth_vec, centroid)
    except Exception as exc:
        logger.error("dream: gate embedding failed: %s", exc)
        return None
    if closeness < cfg.DREAM_CENTROID_MIN:
        log_event(store, "decay", "consolidation_rejected", {
            "reason": "centroid_gate",
            "closeness": round(closeness, 4),
            "threshold": cfg.DREAM_CENTROID_MIN,
            "sources": len(grain_ids),
        }, now=now)
        return None

    new_id_, status = store_atomic_grain_ex(
        synthesis, "ai_inferred", llm=None, embedding_backend=emb,
        store=store, cfg=cfg, now=now, index=index,
        caller_id="flux:dream",
    )
    if status == "duplicate":
        logger.info("dream: synthesis deduped into existing grain %s", new_id_)

    # Evidence conduits: synthesis <-> every source.
    for gid in grain_ids:
        if gid == new_id_:
            continue
        try:
            store.insert_conduit(Conduit(
                from_id=new_id_, to_id=gid, weight=0.6,
                created_at=now, last_used=now,
                direction="bidirectional", decay_class="working",
            ))
        except Exception:
            pass  # conduit already exists

    # Tag sources so they are never consolidated twice.
    for r in rows:
        try:
            tags = json.loads(store.conn.execute(
                "SELECT source_tags FROM grains WHERE id=?", (r["id"],)
            ).fetchone()["source_tags"] or "[]")
        except (TypeError, ValueError):
            tags = []
        if _CONSOLIDATED_TAG not in tags:
            tags.append(_CONSOLIDATED_TAG)
        store.conn.execute("UPDATE grains SET source_tags=? WHERE id=?",
                           (json.dumps(tags), r["id"]))

    log_event(store, "decay", "grains_consolidated", {
        "synthesis_grain_id": new_id_,
        "synthesis": synthesis[:200],
        "source_count": len(grain_ids),
        "source_ids": grain_ids[:20],
        "centroid_closeness": round(closeness, 4),
        "caller_id": "flux:dream",
    }, now=now)
    return new_id_


def dream_cycle(store: FluxStore, llm: LLMBackend, emb: EmbeddingBackend,
                cfg: Config = DEFAULT_CONFIG, now: datetime | None = None,
                index=None) -> dict:
    """Run one full dream cycle. Returns stats."""
    now = now or utcnow()
    clumps = find_candidates(store, cfg, now=now)
    consolidated = []
    for clump in clumps:
        gid = consolidate_clump(store, clump, llm, emb, cfg=cfg, now=now, index=index)
        if gid:
            consolidated.append(gid)
    log_event(store, "decay", "dream_cycle_completed", {
        "clumps_found": len(clumps),
        "consolidated": len(consolidated),
        "synthesis_ids": consolidated,
    }, now=now)
    logger.info("dream cycle: %d clumps, %d consolidated", len(clumps), len(consolidated))
    return {"clumps_found": len(clumps), "consolidated": consolidated}
