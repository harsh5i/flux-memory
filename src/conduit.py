"""
Conduit — Directional weighted edge between grains.

Conduits are the learning substrate. They:
- Connect grains (or entry points to grains)
- Carry signal during retrieval
- Strengthen when used successfully
- Weaken when they lead to poor results
- Dissolve when weight drops below floor

The weight is "conductance" — how easily signal flows.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import hashlib


class Direction(Enum):
    FORWARD = "forward"        # One-way
    BIDIRECTIONAL = "bidirectional"  # Signal flows both ways


class ConduitType(Enum):
    SEMANTIC = "semantic"           # Content/topic similarity (bootstrap)
    CO_OCCURRENCE = "co-occurrence"   # Frequently retrieved together
    TEMPORAL = "temporal"           # Created in same session/time window
    USER_CONFIRMED = "user-confirmed"  # Feedback-confirmed link
    CATEGORY = "category"           # L1→L2 category routing
    ENTRY_BOOTSTRAP = "entry-bootstrap"  # Entry point → grain initial link


@dataclass
class Conduit:
    """
    Directional weighted edge between grains (or entry point → grain).
    
    Properties:
    - weight: conductance (0.0 to 1.0)
    - last_used: timestamp of last successful traversal
    - use_count: total successful traversals
    - direction: forward or bidirectional
    - conduit_type: edge classification for smarter propagation
    - decay_class: inherited from target grain (core/working/ephemeral)
    """
    from_id: str      # Source grain/entry ID
    to_id: str        # Target grain ID
    weight: float = 0.5
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    direction: Direction = Direction.FORWARD
    conduit_type: ConduitType = ConduitType.SEMANTIC
    decay_class: str = "working"  # inherited from target grain
    id: str = field(default_factory=lambda: "")
    created_at: datetime = field(default_factory=datetime.now)
    
    # Weight bounds
    WEIGHT_MAX: float = 1.0
    WEIGHT_MIN: float = 0.01  # Floor; below this = dissolve
    WEIGHT_FLOOR: float = 0.05  # Minimum for viable path
    
    def __post_init__(self):
        if not self.id:
            h = hashlib.sha256(f"{self.from_id}->{self.to_id}".encode()).hexdigest()[:8]
            self.id = f"C-{h}"
    
    def strengthen(self, delta: float = 0.1) -> float:
        """
        Increase weight after successful traversal.
        Returns new weight.
        """
        self.weight = min(self.WEIGHT_MAX, self.weight + delta)
        self.last_used = datetime.now()
        self.use_count += 1
        return self.weight
    
    def weaken(self, delta: float = 0.05) -> float:
        """
        Decrease weight after poor result or decay.
        Returns new weight.
        """
        self.weight = max(0.0, self.weight - delta)
        return self.weight
    
    def is_viable(self) -> bool:
        """Check if conduit is strong enough to use."""
        return self.weight >= self.WEIGHT_FLOOR
    
    def should_dissolve(self) -> bool:
        """Check if conduit should be removed."""
        return self.weight < self.WEIGHT_MIN
    
    def update_decay_class(self, target_grain_class: str):
        """Inherit decay class from target grain."""
        self.decay_class = target_grain_class
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "from_id": self.from_id,
            "to_id": self.to_id,
            "weight": self.weight,
            "last_used": self.last_used.isoformat(),
            "use_count": self.use_count,
            "direction": self.direction.value,
            "conduit_type": self.conduit_type.value,
            "decay_class": self.decay_class,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Conduit":
        ct = d.get("conduit_type", "semantic")
        return cls(
            id=d["id"],
            from_id=d["from_id"],
            to_id=d["to_id"],
            weight=d["weight"],
            last_used=datetime.fromisoformat(d["last_used"]),
            use_count=d["use_count"],
            direction=Direction(d["direction"]),
            conduit_type=ConduitType(ct) if isinstance(ct, str) else ct,
            decay_class=d["decay_class"],
            created_at=datetime.fromisoformat(d["created_at"]),
        )
    
    def __repr__(self):
        return f"Conduit({self.from_id} → {self.to_id}, w={self.weight:.2f})"