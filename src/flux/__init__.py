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
    "effective_weight",
    "propagate",
    "retrieval_confidence",
]
