"""Query-time context expansion — lateral discovery (Track 6 Step 2, §11.11).

After normal signal propagation returns top-k activated grains, this module
runs a bounded second-pass scan to surface contextually-related grains that
were NOT reached through primary retrieval.

Expansion fires when:
  - confidence < EXPANSION_CONFIDENCE_THRESHOLD, OR
  - fewer than 2 grains activated

It uses the cluster memberships Flux already maintains — no LLM calls,
no embeddings, one extra DB query.

Lateral candidates are tagged with source="expansion" so the main AI knows
they came from a second pass, not the primary propagation.
"""
from __future__ import annotations

import logging
from datetime import datetime

from .config import Config, DEFAULT_CONFIG
from .graph import utcnow
from .storage import FluxStore

logger = logging.getLogger(__name__)


def expand_results(
    store: FluxStore,
    activated: list[tuple[str, float]],
    confidence: float,
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
) -> list[dict]:
    """Run context expansion if needed and return lateral candidates.

    Returns a list of dicts:
      {"id": grain_id, "content": ..., "provenance": ..., "decay_class": ...,
       "score": float, "source": "expansion"}

    Returns [] if expansion is not needed or finds nothing new.
    """
    if not cfg.EXPANSION_ENABLED:
        return []

    needs_expansion = confidence < cfg.EXPANSION_CONFIDENCE_THRESHOLD or len(activated) < 2
    if not needs_expansion:
        return []

    now = now or utcnow()
    activated_ids = {gid for gid, _ in activated}

    # Collect cluster memberships of all activated grains.
    shared_clusters: dict[str, float] = {}  # cluster_id → max touch_weight
    for grain_id in activated_ids:
        rows = store.conn.execute(
            "SELECT cluster_id, touch_weight FROM grain_cluster_touch WHERE grain_id = ?",
            (grain_id,),
        ).fetchall()
        for row in rows:
            cid = row["cluster_id"]
            tw = row["touch_weight"]
            if tw > shared_clusters.get(cid, 0.0):
                shared_clusters[cid] = tw

    if not shared_clusters:
        logger.debug("expansion: no cluster memberships found for activated grains")
        return []

    # For each shared cluster, find high-weight candidates not already activated.
    candidates: dict[str, float] = {}  # grain_id → best score
    per_cluster_cap = cfg.EXPANSION_CANDIDATES_PER_CLUSTER

    for cluster_id in shared_clusters:
        # Grains in this cluster, sorted by touch_weight desc, excluding already activated.
        rows = store.conn.execute(
            """
            SELECT gct.grain_id, gct.touch_weight, g.status
            FROM grain_cluster_touch gct
            JOIN grains g ON g.id = gct.grain_id
            WHERE gct.cluster_id = ?
              AND g.status = 'active'
            ORDER BY gct.touch_weight DESC
            LIMIT ?
            """,
            (cluster_id, per_cluster_cap + len(activated_ids)),
        ).fetchall()

        added = 0
        for row in rows:
            gid = row["grain_id"]
            if gid in activated_ids or gid in candidates:
                continue
            # Score: cluster touch_weight × cluster's share in activated set
            score = row["touch_weight"] * shared_clusters[cluster_id] * 0.5
            candidates[gid] = max(candidates.get(gid, 0.0), score)
            added += 1
            if added >= per_cluster_cap:
                break

    # Sort and cap at EXPANSION_MAX_CANDIDATES.
    top = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    top = top[:cfg.EXPANSION_MAX_CANDIDATES]

    results: list[dict] = []
    for grain_id, score in top:
        g = store.get_grain(grain_id)
        if g is None:
            continue
        results.append({
            "id": g.id,
            "content": g.content,
            "provenance": g.provenance,
            "decay_class": g.decay_class,
            "score": round(score, 4),
            "source": "expansion",
        })

    logger.debug("expansion: returned %d lateral candidates", len(results))
    return results
