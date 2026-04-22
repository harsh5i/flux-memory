"""SQLite access layer for Flux Memory.

All persistence flows through ``FluxStore``. The schema is in schema.sql; this
module maps rows to/from the dataclasses in graph.py.

WAL mode is a hard requirement (Section 11.2). foreign_keys is enabled as a
defensive default even though the schema does not declare FKs -- future
migrations may add them.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

from .graph import (
    Cluster,
    Conduit,
    Entry,
    Grain,
    Trace,
    iso,
    parse_iso,
    utcnow,
)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class FluxStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "FluxStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------ schema
    def _ensure_schema(self) -> None:
        sql = _SCHEMA_PATH.read_text()
        self.conn.executescript(sql)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Explicit transaction boundary. isolation_level=None means we drive
        BEGIN/COMMIT ourselves; callers that need multi-statement atomicity
        should wrap their work in this context manager."""
        self.conn.execute("BEGIN")
        try:
            yield self.conn
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # ------------------------------------------------------------------ grains
    def insert_grain(self, grain: Grain) -> None:
        self.conn.execute(
            """
            INSERT INTO grains (
                id, content, provenance, confidence, decay_class, status,
                created_at, dormant_since, context_spread
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                grain.id,
                grain.content,
                grain.provenance,
                grain.confidence,
                grain.decay_class,
                grain.status,
                iso(grain.created_at),
                iso(grain.dormant_since),
                grain.context_spread,
            ),
        )

    def get_grain(self, grain_id: str) -> Grain | None:
        row = self.conn.execute(
            "SELECT * FROM grains WHERE id = ?", (grain_id,)
        ).fetchone()
        return _row_to_grain(row) if row else None

    def iter_grains(self, status: str | None = None) -> Iterable[Grain]:
        if status is None:
            rows = self.conn.execute("SELECT * FROM grains")
        else:
            rows = self.conn.execute(
                "SELECT * FROM grains WHERE status = ?", (status,)
            )
        for row in rows:
            yield _row_to_grain(row)

    # ----------------------------------------------------------------- conduits
    def insert_conduit(self, conduit: Conduit) -> None:
        self.conn.execute(
            """
            INSERT INTO conduits (
                id, from_id, to_id, weight, created_at, last_used,
                use_count, direction, decay_class
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conduit.id,
                conduit.from_id,
                conduit.to_id,
                conduit.weight,
                iso(conduit.created_at),
                iso(conduit.last_used),
                conduit.use_count,
                conduit.direction,
                conduit.decay_class,
            ),
        )

    def get_conduit(self, conduit_id: str) -> Conduit | None:
        row = self.conn.execute(
            "SELECT * FROM conduits WHERE id = ?", (conduit_id,)
        ).fetchone()
        return _row_to_conduit(row) if row else None

    def get_conduit_by_pair(self, from_id: str, to_id: str) -> Conduit | None:
        row = self.conn.execute(
            "SELECT * FROM conduits WHERE from_id = ? AND to_id = ?",
            (from_id, to_id),
        ).fetchone()
        return _row_to_conduit(row) if row else None

    def conduit_between(self, a: str, b: str) -> Conduit | None:
        """Directionless lookup: returns the conduit connecting ``a`` and ``b``
        regardless of which endpoint was stored as from/to. Shortcuts are created
        bidirectional but persisted with sorted endpoints, so reinforce-path
        lookups and tests shouldn't have to track that detail."""
        row = self.conn.execute(
            """
            SELECT * FROM conduits
            WHERE (from_id = ? AND to_id = ?) OR (from_id = ? AND to_id = ?)
            LIMIT 1
            """,
            (a, b, b, a),
        ).fetchone()
        return _row_to_conduit(row) if row else None

    def outgoing_conduits(self, from_id: str) -> list[Conduit]:
        rows = self.conn.execute(
            "SELECT * FROM conduits WHERE from_id = ?", (from_id,)
        ).fetchall()
        return [_row_to_conduit(r) for r in rows]

    def propagation_edges_from(self, grain_id: str) -> list[tuple[str, Conduit]]:
        """Edges reachable from ``grain_id`` during signal propagation.

        Returns (next_grain_id, conduit) pairs. Bidirectional conduits are
        yielded from either endpoint — a shortcut stored as (g1→g2, bidirectional)
        surfaces as (g2, conduit) when queried from g1 AND as (g1, conduit)
        when queried from g2. Forward-only conduits yield only (to_id, conduit)
        when queried from their from_id, preserving entry-gate directionality
        per §13.8."""
        rows = self.conn.execute(
            """
            SELECT * FROM conduits
            WHERE from_id = ? OR (to_id = ? AND direction = 'bidirectional')
            """,
            (grain_id, grain_id),
        ).fetchall()
        edges: list[tuple[str, Conduit]] = []
        for r in rows:
            c = _row_to_conduit(r)
            next_id = c.to_id if c.from_id == grain_id else c.from_id
            edges.append((next_id, c))
        return edges

    def inbound_conduits(self, to_id: str) -> list[Conduit]:
        rows = self.conn.execute(
            "SELECT * FROM conduits WHERE to_id = ?", (to_id,)
        ).fetchall()
        return [_row_to_conduit(r) for r in rows]

    def edges_of(self, grain_id: str) -> list[Conduit]:
        """All conduits attached to a grain (incoming + outgoing). Used for the
        MAX_EDGES_PER_GRAIN cap and weakest-edge eviction in Section 4.3."""
        rows = self.conn.execute(
            "SELECT * FROM conduits WHERE from_id = ? OR to_id = ?",
            (grain_id, grain_id),
        ).fetchall()
        return [_row_to_conduit(r) for r in rows]

    def count_edges(self, grain_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM conduits WHERE from_id = ? OR to_id = ?",
            (grain_id, grain_id),
        ).fetchone()
        return int(row["n"])

    def update_conduit_weight(
        self, conduit_id: str, weight: float, last_used, use_count: int | None = None
    ) -> None:
        """Write-time weight update (Section 4.5). Callers are expected to have
        computed the target weight from effective_weight() already."""
        if use_count is None:
            self.conn.execute(
                "UPDATE conduits SET weight = ?, last_used = ? WHERE id = ?",
                (weight, iso(last_used), conduit_id),
            )
        else:
            self.conn.execute(
                "UPDATE conduits SET weight = ?, last_used = ?, use_count = ? WHERE id = ?",
                (weight, iso(last_used), use_count, conduit_id),
            )

    def delete_conduit(self, conduit_id: str) -> None:
        self.conn.execute("DELETE FROM conduits WHERE id = ?", (conduit_id,))

    # ---------------------------------------------------- co-retrieval counts
    def increment_co_retrieval(self, grain_a: str, grain_b: str, delta: int = 1) -> int:
        """Canonicalize (lower, higher) then UPSERT count += delta. Returns new count."""
        a, b = sorted([grain_a, grain_b])
        self.conn.execute(
            """
            INSERT INTO co_retrieval_counts (grain_a, grain_b, count)
            VALUES (?, ?, ?)
            ON CONFLICT(grain_a, grain_b) DO UPDATE SET count = count + excluded.count
            """,
            (a, b, delta),
        )
        row = self.conn.execute(
            "SELECT count FROM co_retrieval_counts WHERE grain_a = ? AND grain_b = ?",
            (a, b),
        ).fetchone()
        return int(row["count"])

    def get_co_retrieval_count(self, grain_a: str, grain_b: str) -> int:
        a, b = sorted([grain_a, grain_b])
        row = self.conn.execute(
            "SELECT count FROM co_retrieval_counts WHERE grain_a = ? AND grain_b = ?",
            (a, b),
        ).fetchone()
        return int(row["count"]) if row else 0

    # ------------------------------------------------------------------ entries
    def insert_entry(self, entry: Entry) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO entries (id, feature, affinities) VALUES (?, ?, ?)",
            (entry.id, entry.feature, json.dumps(entry.affinities)),
        )

    def get_entry(self, entry_id: str) -> Entry | None:
        row = self.conn.execute(
            "SELECT * FROM entries WHERE id = ?", (entry_id,)
        ).fetchone()
        return _row_to_entry(row) if row else None

    def get_entry_by_feature(self, feature: str) -> Entry | None:
        row = self.conn.execute(
            "SELECT * FROM entries WHERE feature = ?", (feature,)
        ).fetchone()
        return _row_to_entry(row) if row else None

    def update_entry_affinities(self, entry_id: str, affinities: dict[str, float]) -> None:
        """Persist the affinity map after reinforce/penalize has sharpened or dampened it."""
        self.conn.execute(
            "UPDATE entries SET affinities = ? WHERE id = ?",
            (json.dumps(affinities), entry_id),
        )

    # ----------------------------------------------------------------- clusters
    def insert_cluster(self, cluster: Cluster) -> None:
        self.conn.execute(
            """
            INSERT INTO clusters (id, size, created_at, last_updated)
            VALUES (?, ?, ?, ?)
            """,
            (cluster.id, cluster.size, iso(cluster.created_at), iso(cluster.last_updated)),
        )

    def get_cluster(self, cluster_id: str) -> Cluster | None:
        row = self.conn.execute(
            "SELECT * FROM clusters WHERE id = ?", (cluster_id,)
        ).fetchone()
        return _row_to_cluster(row) if row else None

    # ------------------------------------------------------------------- traces
    def insert_trace(self, trace: Trace) -> None:
        self.conn.execute(
            """
            INSERT INTO traces (
                id, query_text, created_at, feedback_at,
                hop_count, activated_grain_count, trace_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.id,
                trace.query_text,
                iso(trace.created_at),
                iso(trace.feedback_at),
                trace.hop_count,
                trace.activated_grain_count,
                trace.trace_data,
            ),
        )

    def get_trace(self, trace_id: str) -> Trace | None:
        row = self.conn.execute(
            "SELECT * FROM traces WHERE id = ?", (trace_id,)
        ).fetchone()
        return _row_to_trace(row) if row else None

    # --------------------------------------------------- decay / orphan helpers
    def conduits_unused_since(self, stale_cutoff: datetime, limit: int) -> list[Conduit]:
        """Return conduits whose last_used is before stale_cutoff. Bounded by limit."""
        rows = self.conn.execute(
            "SELECT * FROM conduits WHERE last_used < ? ORDER BY last_used ASC LIMIT ?",
            (iso(stale_cutoff), limit),
        ).fetchall()
        return [_row_to_conduit(r) for r in rows]

    def count_inbound_conduits(self, grain_id: str) -> int:
        """Conduits that reach grain_id: either to_id=grain_id OR a bidirectional
        shortcut where from_id=grain_id (§1A.9 correctness fix)."""
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS n FROM conduits
            WHERE to_id = ?
               OR (from_id = ? AND direction = 'bidirectional')
            """,
            (grain_id, grain_id),
        ).fetchone()
        return int(row["n"])

    def update_grain_status(
        self,
        grain_id: str,
        status: str,
        dormant_since: datetime | None = None,
    ) -> None:
        if dormant_since is not None:
            self.conn.execute(
                "UPDATE grains SET status = ?, dormant_since = ? WHERE id = ?",
                (status, iso(dormant_since), grain_id),
            )
        else:
            self.conn.execute(
                "UPDATE grains SET status = ? WHERE id = ?",
                (status, grain_id),
            )

    # ------------------------------------------------------- entry co-occurrence
    def increment_entry_cooccurrence(
        self, entry_a: str, entry_b: str, now: datetime
    ) -> None:
        """Canonicalized UPSERT on (min, max) pair."""
        a, b = sorted([entry_a, entry_b])
        self.conn.execute(
            """
            INSERT INTO entry_cooccurrence (entry_a, entry_b, count, last_updated)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(entry_a, entry_b) DO UPDATE SET
                count = count + 1,
                last_updated = excluded.last_updated
            """,
            (a, b, iso(now)),
        )

    def all_entry_cooccurrences(self, window_days: int) -> list[dict]:
        """All co-occurrence pairs updated within window_days."""
        cutoff = iso(utcnow() - timedelta(days=window_days))
        rows = self.conn.execute(
            "SELECT entry_a, entry_b, count FROM entry_cooccurrence WHERE last_updated >= ?",
            (cutoff,),
        ).fetchall()
        return [{"entry_a": r["entry_a"], "entry_b": r["entry_b"], "count": r["count"]} for r in rows]

    # ----------------------------------------------------------- clustering I/O
    def get_current_partition(self) -> tuple[list[frozenset], list[str]]:
        """Read the current cluster partition from entry_cluster_membership.

        Returns (partitions, cluster_ids) where partitions[i] is the frozenset of
        entry_ids in cluster_ids[i]. Used by recompute_clusters for stable ID mapping.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT cluster_id FROM entry_cluster_membership"
        ).fetchall()
        cluster_ids = [r["cluster_id"] for r in rows]

        partitions = []
        for cid in cluster_ids:
            entry_rows = self.conn.execute(
                "SELECT entry_id FROM entry_cluster_membership WHERE cluster_id = ?",
                (cid,),
            ).fetchall()
            partitions.append(frozenset(r["entry_id"] for r in entry_rows))

        return partitions, cluster_ids

    def replace_cluster_memberships(
        self,
        id_memberships: dict[str, dict[str, float]],
        new_cluster_ids: list[str],
        communities: list[frozenset],
        now: datetime,
    ) -> None:
        """Atomically replace all cluster and membership rows (Section 13.2, Step 7)."""
        with self.transaction():
            self.conn.execute("DELETE FROM entry_cluster_membership")
            self.conn.execute("DELETE FROM clusters")

            for idx, cluster_id in enumerate(new_cluster_ids):
                community = communities[idx]
                size = sum(
                    1
                    for entry_id, weights in id_memberships.items()
                    if entry_id in community and weights.get(cluster_id, 0.0) > 0.1
                )
                self.conn.execute(
                    "INSERT INTO clusters (id, size, created_at, last_updated) VALUES (?, ?, ?, ?)",
                    (cluster_id, size, iso(now), iso(now)),
                )

            for entry_id, cluster_weights in id_memberships.items():
                for cluster_id, weight in cluster_weights.items():
                    self.conn.execute(
                        "INSERT INTO entry_cluster_membership (entry_id, cluster_id, weight) VALUES (?, ?, ?)",
                        (entry_id, cluster_id, weight),
                    )

    def all_grain_cluster_touches(self) -> list[tuple[str, str, float]]:
        """All (grain_id, cluster_id, touch_weight) rows."""
        rows = self.conn.execute(
            "SELECT grain_id, cluster_id, touch_weight FROM grain_cluster_touch"
        ).fetchall()
        return [(r["grain_id"], r["cluster_id"], r["touch_weight"]) for r in rows]

    def replace_grain_cluster_touches(
        self, new_touches: dict[tuple[str, str], float]
    ) -> None:
        """Atomically replace the entire grain_cluster_touch table (used post-recluster)."""
        with self.transaction():
            self.conn.execute("DELETE FROM grain_cluster_touch")
            for (grain_id, cluster_id), weight in new_touches.items():
                self.conn.execute(
                    "INSERT INTO grain_cluster_touch (grain_id, cluster_id, touch_weight) VALUES (?, ?, ?)",
                    (grain_id, cluster_id, weight),
                )

    # --------------------------------------------------- promotion helpers
    def get_entry_cluster_memberships(self, entry_id: str) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT cluster_id, weight FROM entry_cluster_membership WHERE entry_id = ?",
            (entry_id,),
        ).fetchall()
        return {r["cluster_id"]: r["weight"] for r in rows}

    def increment_grain_cluster_touch(
        self, grain_id: str, cluster_id: str, delta: float, now: datetime
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO grain_cluster_touch (grain_id, cluster_id, touch_weight, last_touched)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(grain_id, cluster_id) DO UPDATE SET
                touch_weight = touch_weight + excluded.touch_weight,
                last_touched = excluded.last_touched
            """,
            (grain_id, cluster_id, delta, iso(now)),
        )

    def count_clusters_above_threshold(self, grain_id: str, min_weight: float) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM grain_cluster_touch WHERE grain_id = ? AND touch_weight >= ?",
            (grain_id, min_weight),
        ).fetchone()
        return int(row["n"])

    def update_grain_context_spread(self, grain_id: str, spread: int) -> None:
        self.conn.execute(
            "UPDATE grains SET context_spread = ? WHERE id = ?",
            (spread, grain_id),
        )

    def promote_grain_to_core(self, grain_id: str) -> None:
        self.conn.execute(
            "UPDATE grains SET decay_class = 'core' WHERE id = ?",
            (grain_id,),
        )

    def upgrade_inbound_conduits_to_core(self, grain_id: str) -> None:
        """Reclassify all inbound conduits (to_id = grain_id) to core decay class."""
        self.conn.execute(
            "UPDATE conduits SET decay_class = 'core' WHERE to_id = ?",
            (grain_id,),
        )


# --------------------------------------------------------------------- mappers
def _row_to_grain(row: sqlite3.Row) -> Grain:
    return Grain(
        id=row["id"],
        content=row["content"],
        provenance=row["provenance"],
        confidence=row["confidence"],
        decay_class=row["decay_class"],
        status=row["status"],
        created_at=parse_iso(row["created_at"]),
        dormant_since=parse_iso(row["dormant_since"]),
        context_spread=row["context_spread"],
    )


def _row_to_conduit(row: sqlite3.Row) -> Conduit:
    return Conduit(
        id=row["id"],
        from_id=row["from_id"],
        to_id=row["to_id"],
        weight=row["weight"],
        created_at=parse_iso(row["created_at"]),
        last_used=parse_iso(row["last_used"]),
        use_count=row["use_count"],
        direction=row["direction"],
        decay_class=row["decay_class"],
    )


def _row_to_entry(row: sqlite3.Row) -> Entry:
    return Entry(
        id=row["id"],
        feature=row["feature"],
        affinities=json.loads(row["affinities"]) if row["affinities"] else {},
    )


def _row_to_cluster(row: sqlite3.Row) -> Cluster:
    return Cluster(
        id=row["id"],
        size=row["size"],
        created_at=parse_iso(row["created_at"]),
        last_updated=parse_iso(row["last_updated"]),
    )


def _row_to_trace(row: sqlite3.Row) -> Trace:
    return Trace(
        id=row["id"],
        query_text=row["query_text"],
        created_at=parse_iso(row["created_at"]),
        feedback_at=parse_iso(row["feedback_at"]),
        hop_count=row["hop_count"] or 0,
        activated_grain_count=row["activated_grain_count"] or 0,
        trace_data=row["trace_data"] or "[]",
    )
