"""Pre-warming subsystem (Track 6 Step 1, §11.10).

Seeds a fresh Flux graph from existing text sources before going live.
Eliminates cold-start pain for users who already have history.

Supported source types:
  - "text"              — plain .txt files or any readable text file
  - "markdown"          — .md files (Obsidian vault, notes)
  - "conversation_json" — JSON arrays of {role, content} turns

Pipeline per source:
  1. Read file content.
  2. Chunk into units of ≤ chunk_size characters.
  3. Run grain extractor LLM on each chunk (user_message=chunk, ai_response="").
  4. Store grains via the normal write channel (bootstrap conduits created).
  5. Optional synthetic retrieval pass: for each new grain, generate a
     synthetic query from its content, run flux_retrieve, call flux_feedback
     marking the grain useful if it was returned.

Returns a report dict with counts of grains, conduits, entries created.

Invocation:
    from flux.prewarm import prewarm
    report = prewarm(
        sources=[{"path": "~/notes", "type": "markdown"}],
        store=store, llm=llm, emb=emb,
    )

Or via CLI (python -m flux.prewarm --db ... --source ... --type ...).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .config import Config, DEFAULT_CONFIG
from .embedding import EmbeddingBackend
from .graph import utcnow
from .health import log_event
from .llm import LLMBackend
from .retrieval import flux_retrieve, flux_feedback
from .storage import FluxStore

logger = logging.getLogger(__name__)


def prewarm(
    sources: list[dict],
    *,
    store: FluxStore,
    llm: LLMBackend,
    emb: EmbeddingBackend,
    cfg: Config = DEFAULT_CONFIG,
    chunk_size: int = 2000,
    synthetic_retrieval: bool = False,
    synthetic_queries_per_grain: int = 3,
    now=None,
) -> dict:
    """Run the pre-warming pipeline for each source.

    Args:
        sources: list of {"path": str, "type": "text"|"markdown"|"conversation_json"}
        store: open FluxStore
        llm: LLM backend for grain extraction
        emb: embedding backend for bootstrap conduits
        cfg: Config (defaults to DEFAULT_CONFIG)
        chunk_size: max characters per chunk
        synthetic_retrieval: if True, run synthetic retrieval pass after extraction
        synthetic_queries_per_grain: queries per grain in synthetic pass

    Returns:
        dict with: grains_extracted, conduits_before, conduits_after,
                   entries_before, entries_after, files_processed, chunks_processed
    """
    from .extraction import extract_and_store_grains

    now = now or utcnow()

    before_grains = _count(store, "grains")
    before_conduits = _count(store, "conduits")
    before_entries = _count(store, "entries")

    files_processed = 0
    chunks_processed = 0
    new_grain_ids: list[str] = []

    for source in sources:
        raw_path = source.get("path", "")
        source_type = source.get("type", "text")
        path = Path(raw_path).expanduser()

        file_paths = _collect_files(path, source_type)
        for fpath in file_paths:
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.warning("prewarm: cannot read %s — %s", fpath, exc)
                continue

            chunks = _chunk(text, chunk_size, source_type)
            for chunk in chunks:
                if not chunk.strip():
                    continue
                try:
                    ids = extract_and_store_grains(
                        user_message=chunk,
                        ai_response="",
                        llm=llm,
                        embedding_backend=emb,
                        store=store,
                        cfg=cfg,
                        now=now,
                    )
                    new_grain_ids.extend(ids)
                    chunks_processed += 1
                except Exception as exc:
                    logger.warning("prewarm: extraction failed for chunk in %s — %s", fpath, exc)
            files_processed += 1

    # Optional synthetic retrieval pass.
    synthetic_successes = 0
    if synthetic_retrieval and new_grain_ids:
        synthetic_successes = _synthetic_pass(
            store, llm, emb, cfg, new_grain_ids,
            synthetic_queries_per_grain, now,
        )

    after_grains = _count(store, "grains")
    after_conduits = _count(store, "conduits")
    after_entries = _count(store, "entries")

    report = {
        "grains_extracted": after_grains - before_grains,
        "conduits_created": after_conduits - before_conduits,
        "entries_created": after_entries - before_entries,
        "files_processed": files_processed,
        "chunks_processed": chunks_processed,
        "synthetic_successes": synthetic_successes,
    }

    log_event(store, "system", "prewarm_completed", report, now=now)
    logger.info("prewarm complete: %s", report)
    return report


# ----------------------------------------------------------------- internals

def _count(store: FluxStore, table: str) -> int:
    row = store.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row["n"]) if row else 0


def _collect_files(path: Path, source_type: str) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        extensions = {
            "markdown": {".md", ".markdown"},
            "text": {".txt", ".text"},
            "conversation_json": {".json"},
        }.get(source_type, {".txt", ".md"})
        return [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in extensions]
    logger.warning("prewarm: path not found: %s", path)
    return []


def _chunk(text: str, chunk_size: int, source_type: str) -> list[str]:
    if source_type == "conversation_json":
        return _chunk_conversation_json(text, chunk_size)
    if source_type == "markdown":
        return _chunk_by_heading(text, chunk_size)
    return _chunk_by_size(text, chunk_size)


def _chunk_by_size(text: str, size: int) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # If para itself exceeds size, break it by words.
        if len(para) > size:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            words = para.split()
            buf = ""
            for w in words:
                if len(buf) + len(w) + 1 > size and buf:
                    chunks.append(buf.strip())
                    buf = w
                else:
                    buf = (buf + " " + w).strip() if buf else w
            if buf.strip():
                current = buf
        elif len(current) + len(para) + 2 > size and current:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text[:size]]


def _chunk_by_heading(text: str, size: int) -> list[str]:
    sections = re.split(r"(?m)^#{1,3}\s", text)
    result: list[str] = []
    for section in sections:
        if len(section) <= size:
            result.append(section.strip())
        else:
            result.extend(_chunk_by_size(section, size))
    return [s for s in result if s.strip()]


def _chunk_conversation_json(text: str, chunk_size: int) -> list[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _chunk_by_size(text, chunk_size)

    turns: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if content:
                    turns.append(f"{role}: {content}")
    elif isinstance(data, dict):
        # Handle {"messages": [...]} or {"conversation": [...]} wrappers.
        for key in ("messages", "conversation", "turns"):
            if key in data and isinstance(data[key], list):
                return _chunk_conversation_json(json.dumps(data[key]), chunk_size)

    # Merge turns into chunks ≤ chunk_size.
    chunks: list[str] = []
    current = ""
    for turn in turns:
        if len(current) + len(turn) + 1 > chunk_size and current:
            chunks.append(current.strip())
            current = turn
        else:
            current = (current + "\n" + turn).strip() if current else turn
    if current.strip():
        chunks.append(current.strip())
    return chunks or turns[:1]


def _synthetic_pass(
    store: FluxStore,
    llm: LLMBackend,
    emb: EmbeddingBackend,
    cfg: Config,
    grain_ids: list[str],
    queries_per_grain: int,
    now,
) -> int:
    """Run synthetic retrieval on newly inserted grains to pre-shape highways."""
    successes = 0
    for grain_id in grain_ids:
        grain = store.get_grain(grain_id)
        if grain is None:
            continue
        # Generate synthetic queries from the grain's content (first N words).
        words = grain.content.split()
        queries = [
            " ".join(words[: max(3, len(words) // 3)]),
            " ".join(words[len(words) // 3: 2 * len(words) // 3]) or grain.content[:50],
            grain.content[:80],
        ][:queries_per_grain]

        for query in queries:
            if not query.strip():
                continue
            try:
                result = flux_retrieve(query.strip(), store=store, llm=llm, emb=emb, cfg=cfg, now=now)
                for g in result.grains:
                    if g["id"] == grain_id:
                        flux_feedback(result.trace_id, grain_id, True, store=store, cfg=cfg, now=now)
                        successes += 1
                        break
            except Exception as exc:
                logger.debug("prewarm synthetic retrieval error: %s", exc)

    return successes
