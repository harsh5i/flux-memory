"""
Entry Point — Where query signal enters the fabric.

Entry points are created from query decomposition (keyword extraction).
They develop learned affinities toward proven first-hop conduits.

An entry point represents a "query feature" — a keyword, concept, or
entity that the system has learned to route through.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict
import hashlib


@dataclass
class EntryPoint:
    """
    Where query signal enters the fabric.
    
    Properties:
    - feature: the query feature this responds to (e.g., "VMO2", "deadline")
    - level: 1 = category (broad), 2 = specific (default)
    - affinities: learned bias toward first-hop conduits {conduit_id: weight}
    """
    feature: str
    id: str = field(default_factory=lambda: "")
    level: int = 2  # 1 = category (broad), 2 = specific (default)
    affinities: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            h = hashlib.sha256(self.feature.encode()).hexdigest()[:8]
            self.id = f"E-{h}"
    
    def record_use(self, conduit_id: str, success: bool = True):
        """
        Record that signal traveled through a conduit from this entry point.
        Adjust affinities based on success.
        """
        self.last_used = datetime.now()
        self.use_count += 1
        
        if conduit_id not in self.affinities:
            self.affinities[conduit_id] = 0.5  # Start neutral
        
        if success:
            # Strengthen affinity
            self.affinities[conduit_id] = min(1.0, self.affinities[conduit_id] + 0.1)
        else:
            # Weaken affinity
            self.affinities[conduit_id] = max(0.0, self.affinities[conduit_id] - 0.05)
    
    def get_top_conduits(self, limit: int = 5) -> list:
        """
        Get top conduits by affinity.
        Returns list of (conduit_id, affinity) tuples.
        """
        sorted_affinities = sorted(
            self.affinities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_affinities[:limit]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "feature": self.feature,
            "level": self.level,
            "affinities": self.affinities,
            "created_at": self.created_at.isoformat(),
            "use_count": self.use_count,
            "last_used": self.last_used.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "EntryPoint":
        return cls(
            id=d["id"],
            feature=d["feature"],
            level=d.get("level", 2),
            affinities=d["affinities"],
            created_at=datetime.fromisoformat(d["created_at"]),
            use_count=d["use_count"],
            last_used=datetime.fromisoformat(d["last_used"]),
        )
    
    def __repr__(self):
        top = len([a for a in self.affinities.values() if a > 0.5])
        return f"EntryPoint({self.feature}, {top} strong paths)"