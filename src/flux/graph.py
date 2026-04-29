"""Core Flux Memory data types (Section 3, Section 6.4)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


Provenance = Literal["user_stated", "ai_stated", "ai_inferred", "external_source"]
GrainDecayClass = Literal["working", "core"]
GrainStatus = Literal["active", "dormant", "archived", "quarantined", "pending_deletion"]
ConduitDirection = Literal["forward", "bidirectional"]
ConduitDecayClass = Literal["core", "working", "ephemeral"]


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def parse_iso(s: str | None) -> datetime | None:
    if s is None:
        return None
    # SQLite's datetime('now') emits "YYYY-MM-DD HH:MM:SS" in UTC.
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        # Fall back to fromisoformat for callers that pass ISO-8601 with T/Z.
        return datetime.fromisoformat(s.replace("Z", "+00:00"))


@dataclass
class Grain:
    content: str
    provenance: Provenance
    id: str = field(default_factory=new_id)
    confidence: float = 1.0
    decay_class: GrainDecayClass = "working"
    status: GrainStatus = "active"
    created_at: datetime = field(default_factory=utcnow)
    dormant_since: datetime | None = None
    context_spread: int = 0


@dataclass
class Conduit:
    from_id: str
    to_id: str
    id: str = field(default_factory=new_id)
    weight: float = 0.25
    created_at: datetime = field(default_factory=utcnow)
    last_used: datetime = field(default_factory=utcnow)
    use_count: int = 0
    direction: ConduitDirection = "forward"
    decay_class: ConduitDecayClass = "working"


@dataclass
class Entry:
    feature: str
    id: str = field(default_factory=new_id)
    # affinities: conduit_id -> float multiplier on first-hop signal.
    affinities: dict[str, float] = field(default_factory=dict)


@dataclass
class Cluster:
    id: str = field(default_factory=new_id)
    size: int = 0
    created_at: datetime = field(default_factory=utcnow)
    last_updated: datetime = field(default_factory=utcnow)


@dataclass
class Trace:
    """A retrieval's recorded signal path. trace_data holds the full JSON payload."""
    id: str = field(default_factory=new_id)
    query_text: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    feedback_at: datetime | None = None
    hop_count: int = 0
    activated_grain_count: int = 0
    trace_data: str = "[]"  # JSON-encoded list of {conduit_id, from_id, to_id, signal, hop}
