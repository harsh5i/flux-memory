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
)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class FluxStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
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
            "INSERT INTO entries (id, feature, affinities) VALUES (?, ?, ?)",
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
