"""Parameter defaults for Flux Memory (Section 5 of the spec).

A single ``Config`` dataclass holds every tunable constant. Section 13.13
says all parameters must be externally configurable; YAML loader lives in
Track 5. Until that lands, tests and callers construct ``Config()`` with
optional overrides to exercise parameter sensitivity.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Propagation ---
    ATTENUATION: float = 0.85
    ACTIVATION_THRESHOLD: float = 0.15
    MAX_HOPS: int = 5
    TOP_K: int = 5

    # --- Learning & decay ---
    LEARNING_RATE: float = 0.05
    DECAY_FACTOR: float = 0.85
    WEIGHT_CEILING: float = 0.95
    WEIGHT_FLOOR: float = 0.05

    HALF_LIFE_CORE_HOURS: float = 720.0      # 30 days
    HALF_LIFE_WORKING_HOURS: float = 168.0   # 7 days
    HALF_LIFE_EPHEMERAL_HOURS: float = 48.0  # 2 days

    # --- Promotion & clustering ---
    PROMOTION_THRESHOLD: int = 3
    CLUSTER_TOUCH_THRESHOLD: float = 1.0
    ENTRY_COOCCURRENCE_THRESHOLD: int = 10
    CLUSTER_WINDOW_DAYS: int = 30
    CLUSTER_MIN_SIZE: int = 3
    LOUVAIN_RESOLUTION: float = 1.0
    LOUVAIN_SEED: int = 42
    CLUSTER_RECOMPUTE_MIN_INTERVAL_DAYS: int = 7
    CLUSTER_INHERIT_OVERLAP_MIN: float = 0.30
    CLUSTER_DISSOLVE_DECAY: float = 0.5

    # --- Trace retention ---
    TRACE_RETENTION_COUNT: int = 10_000
    TRACE_RETENTION_DAYS: int = 30

    # --- Expansion (Track 6) ---
    EXPANSION_CONFIDENCE_THRESHOLD: float = 0.4
    EXPANSION_CANDIDATES_PER_CLUSTER: int = 2
    EXPANSION_MAX_CANDIDATES: int = 3
    EXPANSION_ENABLED: bool = True

    # --- Context shift (Track 6) ---
    CONTEXT_SHIFT_WINDOW: int = 30
    CONTEXT_SHIFT_DROP_THRESHOLD: float = 0.25
    CONTEXT_SHIFT_RECOVERY_RETRIEVALS: int = 50
    CONTEXT_SHIFT_ENABLED: bool = True

    # --- Quarantine & usefulness ---
    USEFULNESS_WINDOW_DAYS: int = 7
    QUARANTINE_USEFULNESS_THRESHOLD: float = 0.2
    QUARANTINE_MIN_RETRIEVALS: int = 10
    QUARANTINE_CORRECTION_COUNT: int = 3
    QUARANTINE_PERIOD_DAYS: int = 30
    CORRECTION_DETECTION_TURNS: int = 3

    # --- Cleanup ---
    CLEANUP_INTERVAL_HOURS: float = 6.0
    CLEANUP_STALE_HOURS: float = 72.0
    CLEANUP_BATCH_SIZE: int = 1000

    # --- New conduit grace ---
    NEW_CONDUIT_GRACE_HOURS: float = 72.0
    NEW_CONDUIT_GRACE_MULTIPLIER: float = 2.0
    NEW_CONDUIT_MIN_WEIGHT: float = 0.10

    # --- Vector fallback (Track 2) ---
    FALLBACK_CONFIDENCE_THRESHOLD: float = 0.25
    VECTOR_FALLBACK_K: int = 10
    VECTOR_FALLBACK_SCALE: float = 0.5

    # --- Initial weights ---
    INITIAL_SHORTCUT_WEIGHT: float = 0.50
    INITIAL_ENTRY_WEIGHT: float = 0.50
    INITIAL_WEIGHT_SCALE: float = 0.50
    SHORTCUT_THRESHOLD: int = 3
    MAX_EDGES_PER_GRAIN: int = 50

    # --- Lifecycle ---
    DORMANCY_LIMIT_DAYS: int = 30
    EXPLORATION_BOOST: float = 1.5

    # --- Admin channel ---
    PURGE_UNDO_WINDOW_HOURS: float = 24.0

    # --- Provenance reinforcement multipliers (Section 7.2) ---
    def provenance_multiplier(self, provenance: str) -> float:
        return {
            "user_stated": 1.0,
            "external_source": 0.9,
            "ai_stated": 0.5,
            "ai_inferred": 0.3,
        }.get(provenance, 0.5)

    def half_life_hours(self, decay_class: str) -> float:
        return {
            "core": self.HALF_LIFE_CORE_HOURS,
            "working": self.HALF_LIFE_WORKING_HOURS,
            "ephemeral": self.HALF_LIFE_EPHEMERAL_HOURS,
        }.get(decay_class, self.HALF_LIFE_WORKING_HOURS)


DEFAULT_CONFIG = Config()
