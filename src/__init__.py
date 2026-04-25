"""
Flux Memory — Self-Organizing Retrieval Fabric
"""

from .grain import Grain, DecayClass
from .conduit import Conduit, Direction
from .entry_point import EntryPoint
from .trace import Trace, TraceHop
from .propagation import SignalEngine, PropagationConfig, RetrievalResult
from .store import FluxStore
from .decay import run_decay_cycle, compute_decay_factor
from .flux import Flux
from .decompose import DECOMPOSITION_PROMPT, QueryDecomposer
from .embedding import EmbeddingBootstrap

__all__ = [
    "Flux",
    "Grain",
    "DecayClass",
    "Conduit", 
    "Direction",
    "EntryPoint",
    "Trace",
    "TraceHop",
    "SignalEngine",
    "PropagationConfig",
    "RetrievalResult",
    "FluxStore",
    "run_decay_cycle",
    "compute_decay_factor",
    "DECOMPOSITION_PROMPT",
    "QueryDecomposer",
    "EmbeddingBootstrap",
]

__version__ = "0.1.0"