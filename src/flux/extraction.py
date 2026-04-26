"""Query decomposition and grain extraction (Section 4.1, §11.1, Track 2).

Two public functions:

    decompose_query(query, llm, cfg) -> list[str]
        Extracts 2-5 feature keywords from a natural language query. Returns
        existing Entry IDs for known features; creates new Entry rows for new
        features. These Entry IDs are the starting points for signal propagation.

    extract_and_store_grains(user_message, ai_response, llm, embedding, store, cfg)
        Reads the conversation turn, emits atomic grain dicts via the LLM,
        stores each as a Grain, bootstraps conduits from embedding similarity,
        and connects the grain to relevant Entry Points. Called at end of turn.

Both functions are intentionally thin: they delegate to llm.py for inference,
embedding.py for vector ops, and storage.py for persistence.
"""
from __future__ import annotations

import logging
from datetime import datetime

from .config import Config, DEFAULT_CONFIG
from .embedding import EmbeddingBackend, store_embedding, top_k_nearest, load_all_embeddings
from .graph import Conduit, Entry, Grain, new_id, utcnow
from .health import log_event
from .llm import (
    LLMBackend,
    _FEATURE_EXTRACTION_PROMPT,
    _GRAIN_EXTRACTION_PROMPT,
    parse_features,
    parse_grains,
)
from .storage import FluxStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- query decomposition

def decompose_query(
    query: str,
    llm: LLMBackend,
    store: FluxStore,
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
) -> list[str]:
    """Decompose query into feature keyword entry-point IDs (§4.1).

    Calls the LLM to extract 2-5 keywords. For each keyword:
      - If an Entry with that feature already exists, return its ID.
      - Otherwise create a new Entry and return the new ID.

    Returns a list of Entry IDs ready for signal injection.
    """
    now = now or utcnow()
    prompt = _FEATURE_EXTRACTION_PROMPT.format(query=query)
    try:
        raw = llm.complete(prompt)
        features = parse_features(raw)
    except Exception as exc:
        logger.error("decompose_query: LLM call failed: %s", exc)
        features = _fallback_tokenize(query)

    log_event(store, "retrieval", "features_extracted", {
        "query": query[:200],
        "features": features,
    }, now=now)

    entry_ids: list[str] = []
    for feature in features:
        feature = feature.strip().lower()
        if not feature:
            continue
        entry = store.get_entry_by_feature(feature)
        if entry is None:
            entry = Entry(feature=feature)
            store.insert_entry(entry)
            log_event(store, "write", "entry_point_created", {"feature": feature}, now=now)
        entry_ids.append(entry.id)

    return entry_ids


def _fallback_tokenize(query: str) -> list[str]:
    """Stopword-stripped tokenization used when LLM is unavailable."""
    import re
    STOPWORDS = {"the", "a", "an", "is", "are", "was", "for", "of", "to",
                 "in", "on", "at", "by", "and", "or", "but", "help", "me",
                 "my", "i", "you", "we", "they", "it", "that", "this"}
    words = re.findall(r"[a-zA-Z]+", query.lower())
    return [w for w in words if w not in STOPWORDS][:5] or ["query"]


# ---------------------------------------------------------------- grain extraction

def extract_and_store_grains(
    user_message: str,
    ai_response: str,
    llm: LLMBackend,
    embedding_backend: EmbeddingBackend,
    store: FluxStore,
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
) -> list[str]:
    """Extract atomic grains from a conversation turn and store them (§4.7, §11.1).

    Steps:
      1. Call LLM with (user_message + ai_response) to extract grain dicts.
      2. For each grain dict:
         a. Create a Grain and insert it.
         b. Embed the content (one-time) → find k-nearest existing grains.
         c. Create bootstrap conduits to those neighbours.
         d. Extract features from content → create/link Entry Points.
         e. Store the embedding for future fallback.
      3. Return list of new grain IDs.
    """
    now = now or utcnow()
    prompt = _GRAIN_EXTRACTION_PROMPT.format(
        user_message=user_message[:1000],
        ai_response=ai_response[:1000],
    )
    try:
        raw = llm.complete(prompt)
        grain_dicts = parse_grains(raw)
    except Exception as exc:
        logger.error("extract_and_store_grains: LLM call failed: %s", exc)
        return []

    if not grain_dicts:
        return []

    new_grain_ids: list[str] = []
    for gd in grain_dicts:
        content = gd.get("content", "").strip()
        provenance = gd.get("provenance", "ai_stated")
        if not content:
            continue

        grain_id = store_atomic_grain(
            content,
            provenance,
            llm=llm,
            embedding_backend=embedding_backend,
            store=store,
            cfg=cfg,
            now=now,
        )
        new_grain_ids.append(grain_id)

    log_event(store, "write", "bootstrap_conduits_created", {
        "grains_stored": len(new_grain_ids),
    }, now=now)

    return new_grain_ids


def store_atomic_grain(
    content: str,
    provenance: str,
    *,
    llm: LLMBackend,
    embedding_backend: EmbeddingBackend,
    store: FluxStore,
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
) -> str:
    """Store one already-atomic grain and wire it into the graph.

    This is the caller_extracts path: the caller has already decided the memory
    is worth storing, so Flux should still create embeddings and entry conduits
    even when the local LLM cannot perform full grain extraction.
    """
    now = now or utcnow()
    grain = Grain(content=content, provenance=provenance, created_at=now)
    store.insert_grain(grain)
    log_event(store, "write", "grain_stored", {
        "grain_id": grain.id,
        "provenance": provenance,
        "content_len": len(content),
    }, now=now)
    backfill_grain_graph(grain, llm=llm, embedding_backend=embedding_backend,
                         store=store, cfg=cfg, now=now)
    return grain.id


def backfill_grain_graph(
    grain: Grain,
    *,
    llm: LLMBackend,
    embedding_backend: EmbeddingBackend,
    store: FluxStore,
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
) -> dict:
    """Create missing embedding, neighbour conduits, and entry conduits for a grain."""
    now = now or utcnow()
    before_edges = _count_conduits(store)
    embedding_created = False

    row = store.conn.execute(
        "SELECT 1 FROM grain_embeddings WHERE grain_id = ?",
        (grain.id,),
    ).fetchone()

    if row is None:
        existing_grain_ids, existing_matrix = load_all_embeddings(store)
        model_name = getattr(embedding_backend, "model_name", "unknown")
        try:
            embedding = embedding_backend.embed(grain.content)
            embedding_created = True
        except Exception as exc:
            logger.error("backfill_grain_graph: embedding failed for grain %s: %s", grain.id, exc)
        else:
            if existing_grain_ids:
                neighbours = top_k_nearest(embedding, existing_grain_ids, existing_matrix, k=5)
                for neighbour_id, similarity in neighbours:
                    if similarity <= 0:
                        continue
                    weight = similarity * cfg.INITIAL_WEIGHT_SCALE
                    if weight < cfg.WEIGHT_FLOOR:
                        continue
                    conduit = Conduit(
                        from_id=grain.id,
                        to_id=neighbour_id,
                        weight=weight,
                        created_at=now,
                        last_used=now,
                        direction="bidirectional",
                        decay_class="working",
                    )
                    try:
                        store.insert_conduit(conduit)
                    except Exception:
                        pass
            store_embedding(store, grain.id, embedding, model_name, now)

    _connect_to_entries(grain, llm, store, cfg, now)
    after_edges = _count_conduits(store)
    return {
        "grain_id": grain.id,
        "embedding_created": embedding_created,
        "conduits_created": after_edges - before_edges,
    }


def rebuild_missing_graph(
    store: FluxStore,
    llm: LLMBackend,
    embedding_backend: EmbeddingBackend,
    cfg: Config = DEFAULT_CONFIG,
    *,
    limit: int | None = None,
    now: datetime | None = None,
) -> dict:
    """Backfill graph artifacts for active grains missing embeddings or inbound routes."""
    now = now or utcnow()
    scanned = 0
    rebuilt = 0
    embeddings_created = 0
    conduits_created = 0

    for grain in store.iter_grains(status="active"):
        if limit is not None and rebuilt >= limit:
            break
        scanned += 1
        has_embedding = store.conn.execute(
            "SELECT 1 FROM grain_embeddings WHERE grain_id = ?",
            (grain.id,),
        ).fetchone() is not None
        has_inbound = store.count_inbound_conduits(grain.id) > 0
        if has_embedding and has_inbound:
            continue
        stats = backfill_grain_graph(
            grain,
            llm=llm,
            embedding_backend=embedding_backend,
            store=store,
            cfg=cfg,
            now=now,
        )
        rebuilt += 1
        embeddings_created += 1 if stats["embedding_created"] else 0
        conduits_created += stats["conduits_created"]

    log_event(store, "write", "graph_rebuild_completed", {
        "grains_scanned": scanned,
        "grains_rebuilt": rebuilt,
        "embeddings_created": embeddings_created,
        "conduits_created": conduits_created,
    }, now=now)
    return {
        "grains_scanned": scanned,
        "grains_rebuilt": rebuilt,
        "embeddings_created": embeddings_created,
        "conduits_created": conduits_created,
    }


def _count_conduits(store: FluxStore) -> int:
    row = store.conn.execute("SELECT COUNT(*) AS n FROM conduits").fetchone()
    return int(row["n"] if row else 0)


def _connect_to_entries(
    grain: Grain,
    llm: LLMBackend,
    store: FluxStore,
    cfg: Config,
    now: datetime,
) -> None:
    """Create entry-to-grain conduits from features extracted from the grain's content."""
    prompt = _FEATURE_EXTRACTION_PROMPT.format(query=grain.content)
    try:
        raw = llm.complete(prompt)
        features = parse_features(raw)
    except Exception:
        features = _fallback_tokenize(grain.content)

    for feature in features:
        feature = feature.strip().lower()
        if not feature:
            continue
        entry = store.get_entry_by_feature(feature)
        if entry is None:
            entry = Entry(feature=feature)
            store.insert_entry(entry)
        conduit = Conduit(
            from_id=entry.id,
            to_id=grain.id,
            weight=cfg.INITIAL_ENTRY_WEIGHT,
            created_at=now,
            last_used=now,
            direction="forward",
            decay_class="working",
        )
        try:
            store.insert_conduit(conduit)
        except Exception:
            pass  # Entry-grain conduit already exists; skip.
