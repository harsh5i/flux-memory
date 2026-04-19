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
from .health import flux_health, log_event
from .llm import MockLLMBackend, OllamaBackend
from .embedding import (
    MockEmbeddingBackend,
    SentenceTransformerBackend,
    cosine_similarity,
    vector_fallback,
)
from .extraction import decompose_query, extract_and_store_grains
from .retrieval import RetrievalResult, FeedbackResult, flux_retrieve, flux_feedback, flux_store
from .visualization import export_graphml, export_json, export_dot, subgraph, cluster_view
from .ops import ConfigWatcher, GracefulShutdown, backup, restore

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
    "MockEmbeddingBackend",
    "MockLLMBackend",
    "OllamaBackend",
    "Provenance",
    "PropagationResult",
    "SentenceTransformerBackend",
    "Trace",
    "TraceStep",
    "check_promotion",
    "check_promotions_bulk",
    "cleanup_pass",
    "cosine_similarity",
    "decompose_query",
    "effective_weight",
    "expiry_pass",
    "extract_and_store_grains",
    "ConfigWatcher",
    "FeedbackResult",
    "GracefulShutdown",
    "RetrievalResult",
    "backup",
    "cluster_view",
    "export_dot",
    "export_graphml",
    "export_json",
    "flux_feedback",
    "flux_health",
    "flux_retrieve",
    "flux_store",
    "log_event",
    "propagate",
    "restore",
    "subgraph",
    "record_entry_cooccurrence",
    "recompute_clusters",
    "retrieval_confidence",
    "vector_fallback",
]
