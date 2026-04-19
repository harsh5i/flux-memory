"""Flux Memory — a self-organizing retrieval fabric for AI memory."""
from .config import DEFAULT_CONFIG, Config
from .graph import (
    Cluster,
    Conduit,
    ConduitDecayClass,
    ConduitDirection,
    Entry,
    Grain,
    GrainDecayClass,
    GrainStatus,
    Provenance,
    Trace,
)
from .propagation import (
    PropagationResult,
    TraceStep,
    effective_weight,
    propagate,
    retrieval_confidence,
)
from .storage import FluxStore
from .decay import cleanup_pass, expiry_pass
from .clustering import record_entry_cooccurrence, recompute_clusters
from .promotion import check_promotion, check_promotions_bulk

__all__ = [
    "Cluster",
    "Conduit",
    "ConduitDecayClass",
    "ConduitDirection",
    "Config",
    "DEFAULT_CONFIG",
    "Entry",
    "FluxStore",
    "Grain",
    "GrainDecayClass",
    "GrainStatus",
    "Provenance",
    "PropagationResult",
    "Trace",
    "TraceStep",
    "check_promotion",
    "check_promotions_bulk",
    "cleanup_pass",
    "effective_weight",
    "expiry_pass",
    "propagate",
    "record_entry_cooccurrence",
    "recompute_clusters",
    "retrieval_confidence",
]
