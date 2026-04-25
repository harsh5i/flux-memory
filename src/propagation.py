"""
Signal Propagation Engine.

The core retrieval mechanism of Flux Memory.

Signal starts at 1.0 at entry points and attenuates at each hop.
Grains are collected when signal exceeds threshold.

Algorithm:
1. Inject signal at entry points
2. Propagate through conduits (BFS/DFS with attenuation)
3. Collect grains above threshold
4. Record trace for learning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque
from datetime import datetime

from grain import Grain
from conduit import Conduit
from entry_point import EntryPoint
from trace import Trace, TraceHop


@dataclass
class PropagationConfig:
    """Configuration for signal propagation."""
    # Signal attenuation per hop
    attenuation_factor: float = 0.7
    
    # Minimum signal to consider a grain "reached"
    signal_threshold: float = 0.1
    
    # Maximum hops to prevent runaway traversal
    max_hops: int = 10
    
    # Maximum results to return
    max_results: int = 20
    
    # Signal boost for high-affinity conduits from entry points
    affinity_boost: float = 0.1


@dataclass
class RetrievalResult:
    """Result of a signal propagation retrieval."""
    grains: List[Tuple[Grain, float]]  # (grain, final_signal)
    trace: Trace
    total_hops: int
    entry_points_used: List[str]


class SignalEngine:
    """
    Signal propagation engine for Flux Memory retrieval.
    """
    
    def __init__(self, config: PropagationConfig = None):
        self.config = config or PropagationConfig()
    
    def propagate(
        self,
        entry_points: List[EntryPoint],
        conduits_by_source: Dict[str, List[Conduit]],
        grains: Dict[str, Grain],
        query: str = "",
    ) -> RetrievalResult:
        """
        Propagate signal from entry points through the graph.
        
        Args:
            entry_points: Entry points to inject signal
            conduits_by_source: Map of source_id -> list of outgoing conduits
            grains: Map of grain_id -> Grain
            query: Original query (for trace)
        
        Returns:
            RetrievalResult with grains and trace
        """
        trace = Trace(query=query)
        
        # Track signal at each node
        signal_at: Dict[str, float] = {}
        visited: Set[str] = set()
        
        # Priority queue: (negative_signal, node_id, hop_count, from_conduit)
        # Using negative signal for max-heap behavior
        queue = deque()
        
        # --- Two-phase retrieval: L1 categories → L2 specifics → grains ---
        
        # Phase 1: Collect L1 (category) entry points and L2 (specific) entry points
        l1_entry_points = [ep for ep in entry_points if getattr(ep, 'level', 2) == 1]
        l2_entry_points = [ep for ep in entry_points if getattr(ep, 'level', 2) == 2]
        
        # If we have L1 category hits, route signal through their conduits
        if l1_entry_points:
            for l1_ep in l1_entry_points:
                trace.entry_point_ids.append(l1_ep.id)
                # Route from L1 to targets via category conduits (high weight 0.9)
                for c in conduits_by_source.get(l1_ep.id, []):
                    if c.is_viable():
                        target_id = c.to_id
                        initial_signal = 0.5 + (c.weight * self.config.affinity_boost)
                        queue.append((initial_signal, target_id, 0, c.id, l1_ep.id))
                        if target_id not in signal_at:
                            signal_at[target_id] = 0.0
                        signal_at[target_id] = max(signal_at[target_id], initial_signal)
        
        # Phase 2: Inject signal at L2 (specific) entry points
        for ep in l2_entry_points:
            trace.entry_point_ids.append(ep.id)
            
            # Get targets from both affinities AND direct conduits
            # This ensures manually created conduits are also followed
            targets = {}  # conduit_id -> affinity/weight
            
            # From entry point affinities (learned)
            for conduit_id, affinity in ep.get_top_conduits(limit=10):
                targets[conduit_id] = affinity
            
            # From conduits_by_source (includes manually created conduits)
            for c in conduits_by_source.get(ep.id, []):
                if c.is_viable() and c.id not in targets:
                    targets[c.id] = c.weight
            
            for conduit_id, affinity in targets.items():
                # Find the conduit (check both by ID and as target)
                found_conduit = None
                for c in conduits_by_source.get(ep.id, []):
                    if c.id == conduit_id or c.to_id == conduit_id:
                        found_conduit = c
                        break
                
                if found_conduit:
                    target_id = found_conduit.to_id
                else:
                    # conduit_id might be a grain_id from direct conduit
                    target_id = conduit_id
                
                initial_signal = 1.0 + (affinity * self.config.affinity_boost)
                queue.append((initial_signal, target_id, 0, conduit_id, ep.id))
                if target_id not in signal_at:
                    signal_at[target_id] = 0.0
                signal_at[target_id] = max(signal_at[target_id], initial_signal)
        
        # Collect reached grains
        reached_grains: Dict[str, float] = {}
        
        # BFS/DFS traversal
        while queue:
            signal, node_id, hop_count, conduit_id, from_id = queue.popleft()
            
            # Skip if already visited with higher signal
            if node_id in visited and signal <= signal_at.get(node_id, 0):
                continue
            
            # Mark visited
            visited.add(node_id)
            signal_at[node_id] = max(signal_at.get(node_id, 0), signal)
            
            # Record in trace
            trace.add_hop(conduit_id, signal, from_id, node_id)
            
            # If this is a grain and signal exceeds threshold, collect it
            if node_id in grains and signal >= self.config.signal_threshold:
                reached_grains[node_id] = signal
                trace.add_result(node_id)
                
                if len(reached_grains) >= self.config.max_results:
                    break
            
            # Stop if max hops reached
            if hop_count >= self.config.max_hops:
                continue
            
            # Propagate to neighbors
            attenuated = signal * self.config.attenuation_factor
            if attenuated < self.config.signal_threshold:
                continue
            
            for conduit in conduits_by_source.get(node_id, []):
                if not conduit.is_viable():
                    continue
                
                # Apply conduit weight to signal, with type-based boost
                type_multiplier = 1.0
                ct = getattr(conduit, 'conduit_type', None)
                if ct is not None:
                    from conduit import ConduitType
                    if ct == ConduitType.USER_CONFIRMED:
                        type_multiplier = 1.3  # Strongest — confirmed by feedback
                    elif ct == ConduitType.CATEGORY:
                        type_multiplier = 1.2  # Category routing is high-confidence
                    elif ct == ConduitType.CO_OCCURRENCE:
                        type_multiplier = 1.1  # Frequently retrieved together
                    elif ct == ConduitType.ENTRY_BOOTSTRAP:
                        type_multiplier = 0.9  # Initial links are less reliable
                
                next_signal = attenuated * conduit.weight * type_multiplier
                
                if next_signal >= self.config.signal_threshold:
                    queue.append((next_signal, conduit.to_id, hop_count + 1, conduit.id, node_id))
        
        # Build result
        grain_results = []
        for grain_id, signal in sorted(reached_grains.items(), key=lambda x: -x[1]):
            if grain_id in grains:
                grain_results.append((grains[grain_id], signal))
        
        return RetrievalResult(
            grains=grain_results,
            trace=trace,
            total_hops=len(trace.hops),
            entry_points_used=[ep.id for ep in entry_points],
        )
    
    def update_from_trace(
        self,
        trace: Trace,
        conduits: Dict[str, Conduit],
        entry_points: Dict[str, EntryPoint],
        success: bool = True,
    ):
        """
        Update weights based on a trace.
        
        If success: strengthen used conduits, record entry point usage
        If failure: weaken conduits that led to poor results
        """
        trace.success = success
        
        # Update conduit weights
        for hop in trace.hops:
            if hop.conduit_id in conduits:
                conduit = conduits[hop.conduit_id]
                if success:
                    # Strengthen proportionally to signal strength
                    delta = 0.1 * hop.signal_at_hop
                    conduit.strengthen(delta)
                else:
                    conduit.weaken(0.05)
        
        # Update entry point affinities
        for ep_id in trace.entry_point_ids:
            if ep_id in entry_points:
                ep = entry_points[ep_id]
                for hop in trace.hops:
                    if hop.from_id == ep_id or hop.conduit_id.startswith(f"E-"):
                        ep.record_use(hop.conduit_id, success=success)
    
    def __repr__(self):
        return f"SignalEngine(attenuation={self.config.attenuation})"