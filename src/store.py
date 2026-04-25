"""
Store — Persistence layer for Flux Memory.

Uses SQLite for durable storage with JSON serialization for complex objects.
Designed to be fast for the hot path (retrieval) and safe for writes.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

from grain import Grain, DecayClass
from conduit import Conduit, Direction, ConduitType
from entry_point import EntryPoint
from trace import Trace


class FluxStore:
    """
    SQLite-backed storage for Flux Memory.
    
    Tables:
    - grains: id, content, decay_class, created_at, status, context_spread, ...
    - conduits: id, from_id, to_id, weight, last_used, use_count, ...
    - entry_points: id, feature, affinities, ...
    - traces: id, entry_point_ids, hops, result_grain_ids, ...
    """
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS grains (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        decay_class TEXT NOT NULL DEFAULT 'working',
        created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        dormant_since TEXT,
        context_spread INTEGER DEFAULT 0,
        source_tags TEXT DEFAULT '[]'
    );
    
    CREATE TABLE IF NOT EXISTS conduits (
        id TEXT PRIMARY KEY,
        from_id TEXT NOT NULL,
        to_id TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 0.5,
        last_used TEXT NOT NULL,
        use_count INTEGER DEFAULT 0,
        direction TEXT NOT NULL DEFAULT 'forward',
        conduit_type TEXT NOT NULL DEFAULT 'semantic',
        decay_class TEXT NOT NULL DEFAULT 'working',
        created_at TEXT NOT NULL
    );
    
    CREATE INDEX IF NOT EXISTS idx_conduits_from ON conduits(from_id);
    CREATE INDEX IF NOT EXISTS idx_conduits_to ON conduits(to_id);
    
    CREATE TABLE IF NOT EXISTS entry_points (
        id TEXT PRIMARY KEY,
        feature TEXT NOT NULL UNIQUE,
        affinities TEXT DEFAULT '{}',
        level INTEGER DEFAULT 2,
        created_at TEXT NOT NULL,
        use_count INTEGER DEFAULT 0,
        last_used TEXT NOT NULL
    );
    
    CREATE TABLE IF NOT EXISTS traces (
        id TEXT PRIMARY KEY,
        entry_point_ids TEXT DEFAULT '[]',
        hops TEXT DEFAULT '[]',
        result_grain_ids TEXT DEFAULT '[]',
        query TEXT DEFAULT '',
        success INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    );
    
    -- Metadata
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """
    
    def __init__(self, db_path: str = "flux.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)
    
    @contextmanager
    def _conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    # === Grains ===
    
    def save_grain(self, grain: Grain):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO grains
                (id, content, decay_class, created_at, status, dormant_since, context_spread, source_tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                grain.id, grain.content, grain.decay_class.value, grain.created_at.isoformat(),
                grain.status, grain.dormant_since.isoformat() if grain.dormant_since else None,
                grain.context_spread, json.dumps(grain.source_tags)
            ))
    
    def get_grain(self, grain_id: str) -> Optional[Grain]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM grains WHERE id = ?", (grain_id,)).fetchone()
            return self._row_to_grain(row) if row else None
    
    def get_all_grains(self) -> Dict[str, Grain]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM grains WHERE status = 'active'").fetchall()
            return {row["id"]: self._row_to_grain(row) for row in rows}
    
    def _row_to_grain(self, row) -> Grain:
        return Grain(
            id=row["id"],
            content=row["content"],
            decay_class=DecayClass(row["decay_class"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            status=row["status"],
            dormant_since=datetime.fromisoformat(row["dormant_since"]) if row["dormant_since"] else None,
            context_spread=row["context_spread"],
            source_tags=json.loads(row["source_tags"]),
        )
    
    # === Conduits ===
    
    def save_conduit(self, conduit: Conduit):
        conduit_type_val = conduit.conduit_type.value if hasattr(conduit.conduit_type, 'value') else str(conduit.conduit_type)
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO conduits
                (id, from_id, to_id, weight, last_used, use_count, direction, conduit_type, decay_class, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conduit.id, conduit.from_id, conduit.to_id, conduit.weight,
                conduit.last_used.isoformat(), conduit.use_count,
                conduit.direction.value, conduit_type_val, conduit.decay_class, conduit.created_at.isoformat()
            ))
    
    def get_conduit(self, conduit_id: str) -> Optional[Conduit]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM conduits WHERE id = ?", (conduit_id,)).fetchone()
            return self._row_to_conduit(row) if row else None
    
    def get_conduits_by_source(self) -> Dict[str, List[Conduit]]:
        """Get all conduits indexed by source (from_id)."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM conduits").fetchall()
            result: Dict[str, List[Conduit]] = {}
            for row in rows:
                c = self._row_to_conduit(row)
                if row["from_id"] not in result:
                    result[row["from_id"]] = []
                result[row["from_id"]].append(c)
            return result
    
    def get_all_conduits(self) -> Dict[str, Conduit]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM conduits").fetchall()
            return {row["id"]: self._row_to_conduit(row) for row in rows}
    
    def delete_conduit(self, conduit_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM conduits WHERE id = ?", (conduit_id,))
    
    def _row_to_conduit(self, row) -> Conduit:
        ct = row["conduit_type"] if "conduit_type" in row.keys() else "semantic"
        return Conduit(
            id=row["id"],
            from_id=row["from_id"],
            to_id=row["to_id"],
            weight=row["weight"],
            last_used=datetime.fromisoformat(row["last_used"]),
            use_count=row["use_count"],
            direction=Direction(row["direction"]),
            conduit_type=ConduitType(ct) if isinstance(ct, str) else ct,
            decay_class=row["decay_class"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
    
    # === Entry Points ===
    
    def save_entry_point(self, ep: EntryPoint):
        level = getattr(ep, 'level', 2)
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO entry_points
                (id, feature, affinities, level, created_at, use_count, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ep.id, ep.feature, json.dumps(ep.affinities), level,
                ep.created_at.isoformat(), ep.use_count, ep.last_used.isoformat()
            ))
    
    def get_entry_point(self, ep_id: str) -> Optional[EntryPoint]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM entry_points WHERE id = ?", (ep_id,)).fetchone()
            return self._row_to_entry_point(row) if row else None
    
    def get_entry_point_by_feature(self, feature: str) -> Optional[EntryPoint]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM entry_points WHERE feature = ?", (feature,)).fetchone()
            return self._row_to_entry_point(row) if row else None
    
    def get_all_entry_points(self) -> Dict[str, EntryPoint]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM entry_points").fetchall()
            return {row["id"]: self._row_to_entry_point(row) for row in rows}
    
    def _row_to_entry_point(self, row) -> EntryPoint:
        level = row["level"] if "level" in row.keys() else 2
        return EntryPoint(
            id=row["id"],
            feature=row["feature"],
            affinities=json.loads(row["affinities"]),
            level=int(level),
            created_at=datetime.fromisoformat(row["created_at"]),
            use_count=row["use_count"],
            last_used=datetime.fromisoformat(row["last_used"]),
        )
    
    # === Traces ===
    
    def save_trace(self, trace: Trace):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO traces
                (id, entry_point_ids, hops, result_grain_ids, query, success, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                trace.id, json.dumps(trace.entry_point_ids),
                json.dumps([h.to_dict() for h in trace.hops]),
                json.dumps(trace.result_grain_ids), trace.query, int(trace.success),
                trace.created_at.isoformat()
            ))
    
    def get_recent_traces(self, limit: int = 100) -> List[Trace]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM traces ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [Trace.from_dict(json.loads(row["hops"]) or {
                "id": row["id"],
                "entry_point_ids": json.loads(row["entry_point_ids"]),
                "result_grain_ids": json.loads(row["result_grain_ids"]),
                "query": row["query"],
                "success": bool(row["success"]),
                "created_at": row["created_at"],
                "hops": json.loads(row["hops"]),
            }) for row in rows]
    
    # === Stats ===
    
    def get_stats(self) -> dict:
        with self._conn() as conn:
            grains = conn.execute("SELECT COUNT(*) FROM grains WHERE status = 'active'").fetchone()[0]
            conduits = conn.execute("SELECT COUNT(*) FROM conduits").fetchone()[0]
            entry_points = conn.execute("SELECT COUNT(*) FROM entry_points").fetchone()[0]
            traces = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
            core_grains = conn.execute(
                "SELECT COUNT(*) FROM grains WHERE decay_class = 'core'"
            ).fetchone()[0]
            return {
                "grains": grains,
                "core_grains": core_grains,
                "conduits": conduits,
                "entry_points": entry_points,
                "traces": traces,
            }
    
    # === Metadata ===
    
    def get_meta(self, key: str) -> Optional[str]:
        """Get a metadata value by key."""
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None
    
    def set_meta(self, key: str, value: str):
        """Set a metadata value."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, value)
            )