"""
Trace — The recorded path of signal during retrieval.

Traces are the "learning receipt" — they record exactly what path signal
took during a retrieval. Used for:
1. Weight updates (strengthen used paths)
2. Co-retrieval edge creation (connect grains that proved useful together)
3. Analytics (understanding retrieval patterns)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple
import json


@dataclass
class TraceHop:
    """Single hop in a trace."""
    conduit_id: str
    signal_at_hop: float  # Signal strength when traversing this conduit
    from_id: str
    to_id: str
    
    def to_dict(self) -> dict:
        return {
            "conduit_id": self.conduit_id,
            "signal_at_hop": self.signal_at_hop,
            "from_id": self.from_id,
            "to_id": self.to_id,
        }


@dataclass
class Trace:
    """
    Recorded path of signal during retrieval.
    
    Contains:
    - hops: list of (conduit_id, signal_at_hop) pairs
    - entry_points: which entry points started this trace
    - result_grains: which grains were successfully retrieved
    - query: original query (for analytics)
    - success: whether retrieval was useful
    """
    id: str = field(default_factory=lambda: "")
    entry_point_ids: List[str] = field(default_factory=list)
    hops: List[TraceHop] = field(default_factory=list)
    result_grain_ids: List[str] = field(default_factory=list)
    query: str = ""
    success: bool = False  # Did this retrieval help?
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            self.id = f"T-{ts}-{len(self.hops)}"
    
    def add_hop(self, conduit_id: str, signal: float, from_id: str, to_id: str):
        """Add a hop to the trace."""
        self.hops.append(TraceHop(
            conduit_id=conduit_id,
            signal_at_hop=signal,
            from_id=from_id,
            to_id=to_id,
        ))
    
    def add_result(self, grain_id: str):
        """Record a grain that contributed to the result."""
        if grain_id not in self.result_grain_ids:
            self.result_grain_ids.append(grain_id)
    
    def get_conduits_used(self) -> List[str]:
        """Get all conduit IDs used in this trace."""
        return [hop.conduit_id for hop in self.hops]
    
    def get_final_signal_for_grain(self, grain_id: str) -> float:
        """Get the signal strength that reached a specific grain."""
        for hop in reversed(self.hops):
            if hop.to_id == grain_id:
                return hop.signal_at_hop
        return 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "entry_point_ids": self.entry_point_ids,
            "hops": [h.to_dict() for h in self.hops],
            "result_grain_ids": self.result_grain_ids,
            "query": self.query,
            "success": self.success,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        t = cls(
            id=d["id"],
            entry_point_ids=d["entry_point_ids"],
            query=d.get("query", ""),
            success=d.get("success", False),
            created_at=datetime.fromisoformat(d["created_at"]),
        )
        t.result_grain_ids = d.get("result_grain_ids", [])
        t.hops = [TraceHop(**h) for h in d.get("hops", [])]
        return t
    
    def __repr__(self):
        return f"Trace({len(self.hops)} hops, {len(self.result_grain_ids)} results)"