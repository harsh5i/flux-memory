"""
Grain — Atomic memory unit.

Grains are immutable content containers. They don't store relationships;
conduits do that. A grain knows nothing about how it's connected.

Decay class is determined by context_spread:
- context_spread < 3 → working (7-day half-life)
- context_spread >= 3 → core (30-day half-life)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import hashlib
import json


class DecayClass(Enum):
    WORKING = "working"    # 7-day half-life
    CORE = "core"          # 30-day half-life
    EPHEMERAL = "ephemeral"  # 48h half-life (optional, for session-specific)


@dataclass
class Grain:
    """
    Atomic memory item.
    
    Immutable content + metadata. Never modified after creation.
    """
    content: str
    id: str = field(default_factory=lambda: "")
    decay_class: DecayClass = DecayClass.WORKING
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active | dormant | archived
    dormant_since: Optional[datetime] = None
    context_spread: int = 0  # distinct entry point clusters that led to retrieval
    source_tags: list = field(default_factory=list)  # optional metadata
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from content hash
            h = hashlib.sha256(self.content.encode()).hexdigest()[:12]
            self.id = f"G-{h}"
    
    def promote(self) -> bool:
        """Promote from working to core if context_spread threshold met."""
        if self.decay_class == DecayClass.WORKING and self.context_spread >= 3:
            self.decay_class = DecayClass.CORE
            return True
        return False
    
    def record_retrieval(self, entry_point_cluster: str) -> bool:
        """
        Record that this grain was retrieved via a specific entry point cluster.
        Returns True if this triggers promotion.
        """
        # context_spread counts distinct entry point clusters
        # This is a simplified version; full version tracks clusters
        self.context_spread += 1
        return self.promote()
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "decay_class": self.decay_class.value,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "dormant_since": self.dormant_since.isoformat() if self.dormant_since else None,
            "context_spread": self.context_spread,
            "source_tags": self.source_tags,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Grain":
        return cls(
            id=d["id"],
            content=d["content"],
            decay_class=DecayClass(d["decay_class"]),
            created_at=datetime.fromisoformat(d["created_at"]),
            status=d.get("status", "active"),
            dormant_since=datetime.fromisoformat(d["dormant_since"]) if d.get("dormant_since") else None,
            context_spread=d.get("context_spread", 0),
            source_tags=d.get("source_tags", []),
        )
    
    def __repr__(self):
        decay = self.decay_class.value[0]  # w/c/e
        return f"Grain({self.id}, {decay}, spread={self.context_spread})"