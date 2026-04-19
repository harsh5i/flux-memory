"""Admin channel for Flux Memory (§7.6).

NOT exposed through the MCP server. Must be explicitly imported by user-controlled
scripts. All operations require a confirmation_token passed out-of-band.

Functions:
    flux_purge(grain_id, reason, store, token)
    flux_purge_by_content(pattern, store, token, dry_run=True)
    flux_export_grain(grain_id, store, token)
    flux_restore(grain_id, store, token)
"""
from __future__ import annotations

import os
import re
import logging
from datetime import datetime

from .graph import utcnow
from .health import log_event
from .storage import FluxStore

logger = logging.getLogger(__name__)

_ENV_TOKEN_VAR = "FLUX_ADMIN_TOKEN"


def _check_token(token: str | None) -> None:
    expected = os.environ.get(_ENV_TOKEN_VAR)
    if not expected:
        raise PermissionError(
            f"Admin channel requires {_ENV_TOKEN_VAR} to be set in the environment."
        )
    if token != expected:
        raise PermissionError("Admin channel: invalid confirmation_token.")


def flux_purge(
    grain_id: str,
    reason: str,
    *,
    store: FluxStore,
    confirmation_token: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Permanently delete a grain and all its conduits (§7.6).

    Requires a non-empty reason and a valid FLUX_ADMIN_TOKEN environment variable.
    Returns a summary dict.
    """
    _check_token(confirmation_token)
    if not reason or not reason.strip():
        raise ValueError("flux_purge: reason must be a non-empty string")

    now = now or utcnow()
    grain = store.get_grain(grain_id)
    if grain is None:
        raise KeyError(f"flux_purge: grain {grain_id!r} not found")

    # Collect affected entries before deletion (entries that have conduits to this grain).
    all_conduits = store.edges_of(grain_id)
    affected_entry_ids = []
    for c in all_conduits:
        other_id = c.to_id if c.from_id == grain_id else c.from_id
        entry = store.get_entry(other_id)
        if entry is not None:
            affected_entry_ids.append(other_id)

    conduits_removed = len(all_conduits)

    # Transactional delete.
    with store.conn:
        store.conn.execute("DELETE FROM conduits WHERE from_id = ? OR to_id = ?",
                           (grain_id, grain_id))
        store.conn.execute("DELETE FROM grain_cluster_touch WHERE grain_id = ?", (grain_id,))
        store.conn.execute(
            "DELETE FROM co_retrieval_counts WHERE grain_a = ? OR grain_b = ?",
            (grain_id, grain_id),
        )
        store.conn.execute("DELETE FROM grain_embeddings WHERE grain_id = ?", (grain_id,))
        store.conn.execute("DELETE FROM grains WHERE id = ?", (grain_id,))

    log_event(store, "admin", "grain_purged", {
        "grain_id": grain_id,
        "content_snippet": grain.content[:100],
        "reason": reason,
        "conduits_removed": conduits_removed,
        "affected_entries": affected_entry_ids,
    }, now=now)

    return {
        "purged": grain_id,
        "conduits_removed": conduits_removed,
        "affected_entries": affected_entry_ids,
        "timestamp": now.isoformat(),
    }


def flux_purge_by_content(
    content_pattern: str,
    *,
    store: FluxStore,
    confirmation_token: str | None = None,
    dry_run: bool = True,
    now: datetime | None = None,
) -> dict:
    """Search grains by content pattern; optionally purge all matches (§7.6).

    dry_run=True (default): return matches without deleting.
    dry_run=False: purge all matches (requires token + non-empty pattern).
    Batch purges > 10 grains raise ValueError — caller must confirm count first.
    """
    _check_token(confirmation_token)
    now = now or utcnow()

    rows = store.conn.execute("SELECT id, content, provenance, created_at FROM grains").fetchall()
    try:
        pattern_re = re.compile(content_pattern, re.IGNORECASE)
    except re.error:
        pattern_re = None

    matches = []
    for row in rows:
        hit = (content_pattern.lower() in row["content"].lower()) or (
            pattern_re is not None and pattern_re.search(row["content"])
        )
        if hit:
            matches.append({
                "grain_id": row["id"],
                "content": row["content"],
                "provenance": row["provenance"],
                "created_at": row["created_at"],
            })

    if dry_run:
        return {"matches": matches, "purged": False, "purge_count": 0}

    if len(matches) > 10:
        raise ValueError(
            f"flux_purge_by_content: {len(matches)} matches exceed the batch limit of 10. "
            "Provide a narrower pattern or purge individually with flux_purge()."
        )

    for m in matches:
        flux_purge(m["grain_id"], reason=f"batch content purge: {content_pattern[:50]}",
                   store=store, confirmation_token=confirmation_token, now=now)

    return {"matches": matches, "purged": True, "purge_count": len(matches)}


def flux_export_grain(
    grain_id: str,
    *,
    store: FluxStore,
    confirmation_token: str | None = None,
) -> dict:
    """Return full grain metadata for inspection before purge decisions (§7.6)."""
    _check_token(confirmation_token)

    grain = store.get_grain(grain_id)
    if grain is None:
        raise KeyError(f"flux_export_grain: grain {grain_id!r} not found")

    inbound = store.inbound_conduits(grain_id)
    outbound = store.outgoing_conduits(grain_id)

    # Cluster touches.
    touch_rows = store.conn.execute(
        "SELECT cluster_id, touch_weight FROM grain_cluster_touch WHERE grain_id = ?",
        (grain_id,),
    ).fetchall()

    # Recent retrievals (last 20 traces that contain this grain).
    recent_rows = store.conn.execute(
        """
        SELECT id FROM traces
        WHERE trace_data LIKE ?
        ORDER BY created_at DESC LIMIT 20
        """,
        (f'%"{grain_id}"%',),
    ).fetchall()

    return {
        "grain": {
            "id": grain.id,
            "content": grain.content,
            "provenance": grain.provenance,
            "decay_class": grain.decay_class,
            "status": grain.status,
            "created_at": grain.created_at.isoformat(),
            "context_spread": grain.context_spread,
        },
        "inbound_conduits": [
            {"from_id": c.from_id, "weight": c.weight,
             "use_count": c.use_count, "last_used": c.last_used.isoformat()}
            for c in inbound
        ],
        "outbound_conduits": [
            {"to_id": c.to_id, "weight": c.weight,
             "use_count": c.use_count, "last_used": c.last_used.isoformat()}
            for c in outbound
        ],
        "cluster_touches": [
            {"cluster_id": r["cluster_id"], "touch_weight": r["touch_weight"]}
            for r in touch_rows
        ],
        "recent_retrievals": [r["id"] for r in recent_rows],
    }


def flux_restore(
    grain_id: str,
    *,
    store: FluxStore,
    confirmation_token: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Restore a quarantined grain to active status (§7.6)."""
    _check_token(confirmation_token)
    now = now or utcnow()

    grain = store.get_grain(grain_id)
    if grain is None:
        raise KeyError(f"flux_restore: grain {grain_id!r} not found")
    if grain.status == "active":
        raise ValueError(f"flux_restore: grain {grain_id!r} is already active")
    if grain.status == "archived":
        raise ValueError(f"flux_restore: grain {grain_id!r} is archived and cannot be restored")

    previous_status = grain.status
    store.update_grain_status(grain_id, "active")

    log_event(store, "admin", "grain_restored", {
        "grain_id": grain_id,
        "content_snippet": grain.content[:100],
        "previous_status": previous_status,
    }, now=now)

    return {
        "restored": grain_id,
        "previous_status": previous_status,
        "new_status": "active",
    }
