"""
Flux Memory — Self-Organizing Retrieval Fabric

Main API for storing, retrieving, and learning from memory usage.

Usage:
    flux = Flux(store_path="flux.db")
    
    # Store a memory
    grain = flux.remember("VMO2 project deadline is May 15")
    
    # Retrieve memories
    results = flux.query("What's the VMO2 deadline?")
    
    # Mark retrieval as useful
    flux.mark_useful(results[0])
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
import json

from grain import Grain, DecayClass
from conduit import Conduit, Direction
from entry_point import EntryPoint
from trace import Trace
from propagation import SignalEngine, PropagationConfig, RetrievalResult
from store import FluxStore
from decay import run_decay_cycle
from decompose import QueryDecomposer, extract_features_simple
from embedding import EmbeddingBootstrap


class Flux:
    """
    Main Flux Memory API.
    
    Provides:
    - remember(content): Store a new grain
    - query(text): Retrieve relevant grains via signal propagation
    - feedback(result_id, success): Mark retrieval as useful/not useful
    - decay(): Run decay cycle
    """
    
    def __init__(
        self,
        store_path: str = "flux.db",
        signal_config: PropagationConfig = None,
        use_llm_decompose: bool = True,
        use_embeddings: bool = True,
        llm_model: str = "gemma4:e2b",
        embedding_model: str = "nomic-embed-text",
    ):
        self.store = FluxStore(store_path)
        
        # Query decomposition
        self.decomposer = QueryDecomposer(
            use_llm=use_llm_decompose,
            model=llm_model,
        )
        
        # Embedding bootstrap
        self.use_embeddings = use_embeddings
        self.embeddings = EmbeddingBootstrap(embedding_model=embedding_model) if use_embeddings else None
        
        # Signal propagation engine
        self.signal_engine = SignalEngine(signal_config or PropagationConfig())
        
        # Cache for hot path
        self._grains_cache: Dict[str, Grain] = None
        self._conduits_cache: Dict[str, Conduit] = None
        self._entry_points_cache: Dict[str, EntryPoint] = None
        self._conduits_by_source_cache: Dict[str, List[Conduit]] = None
        
        # Last retrieval (for feedback)
        self._last_result: RetrievalResult = None
        
        # Auto-decay: track last decay run, auto-trigger if overdue
        self._last_decay_at: datetime = None
        self._decay_interval_hours: float = 24.0  # Run decay every 24h
        
        # Auto-backup: track last backup, auto-trigger after N new grains
        self._grains_since_last_backup: int = 0
        self._backup_threshold: int = 5  # Backup every 5 new grains
    
    def _refresh_cache(self):
        """Refresh in-memory caches from store."""
        self._grains_cache = self.store.get_all_grains()
        self._conduits_cache = self.store.get_all_conduits()
        self._entry_points_cache = self.store.get_all_entry_points()
        self._conduits_by_source_cache = self.store.get_conduits_by_source()
    
    def _auto_decay(self):
        """Run decay automatically if 24h have passed since last run."""
        now = datetime.now()
        if self._last_decay_at is None:
            # First run or fresh instance — check DB for last decay time
            meta = self.store.get_meta("last_decay_at")
            if meta:
                try:
                    self._last_decay_at = datetime.fromisoformat(meta)
                except (ValueError, TypeError):
                    self._last_decay_at = None
        
        should_decay = (
            self._last_decay_at is None or
            (now - self._last_decay_at).total_seconds() / 3600 >= self._decay_interval_hours
        )
        
        if should_decay:
            self.decay(now=now)
            self._last_decay_at = now
            self.store.set_meta("last_decay_at", now.isoformat())
    
    def _auto_backup(self):
        """Run backup automatically if threshold grains added since last backup."""
        if self._grains_since_last_backup >= self._backup_threshold:
            self._grains_since_last_backup = 0
            # Run backup in background (non-blocking)
            try:
                from pathlib import Path
                import json
                backup_dir = Path.home() / ".openclaw" / "flux" / "backup"
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_file = backup_dir / "memory_backup.json"
                
                grains = self.store.get_all_grains()
                backup_data = {
                    "timestamp": datetime.now().isoformat(),
                    "grains": [
                        {
                            "id": g.id,
                            "content": g.content,
                            "tags": g.source_tags,
                            "decay_class": g.decay_class.value,
                            "context_spread": g.context_spread,
                            "created_at": g.created_at.isoformat(),
                        }
                        for g in grains.values()
                    ],
                    "stats": self.stats(),
                }
                
                with open(backup_file, "w") as f:
                    json.dump(backup_data, f, indent=2)
            except Exception:
                pass  # Non-critical; don't block operations
    
    def remember(
        self,
        content: str,
        tags: List[str] = None,
        decay_class: DecayClass = DecayClass.WORKING,
    ) -> Grain:
        """
        Store a new grain in Flux Memory.
        
        Creates a grain and bootstraps conduits to existing grains
        via embedding similarity.
        
        Auto-triggers decay (if 24h overdue) and backup (every 5 grains).
        
        Args:
            content: The memory content
            tags: Optional source tags
            decay_class: Initial decay class (usually WORKING; promoted through use)
        
        Returns:
            The created Grain
        """
        # Auto-decay if overdue
        self._auto_decay()
        
        grain = Grain(
            content=content,
            source_tags=tags or [],
            decay_class=decay_class,
        )
        
        self.store.save_grain(grain)
        self._grains_cache = None  # Invalidate cache
        
        # Create entry points for key content words
        features = self.decomposer.decompose(content)
        for feature in features:
            self._ensure_entry_point(feature, grain.id)
        
        # Create bootstrap conduits using embeddings (if available)
        self._bootstrap_conduits(grain)
        
        # Track grains since last backup for auto-backup
        self._grains_since_last_backup += 1
        self._auto_backup()
        
        return grain
    
    def _ensure_entry_point(self, feature: str, initial_grain_id: str = None):
        """Create an entry point if it doesn't exist. Bootstrap missing connections."""
        if self._entry_points_cache is None:
            self._refresh_cache()
        
        # Find existing EP by feature (case-insensitive)
        ep = self.store.get_entry_point_by_feature(feature)
        if ep is None:
            # Also check case-insensitive
            for eid, existing_ep in self._entry_points_cache.items():
                if existing_ep.feature.lower() == feature.lower():
                    ep = existing_ep
                    break
        
        if ep is None:
            # Create new entry point
            ep = EntryPoint(feature=feature)
            
            # If we have an initial grain, create conduit
            if initial_grain_id:
                conduit = Conduit(
                    from_id=ep.id,
                    to_id=initial_grain_id,
                    weight=0.6,  # Initial bias
                )
                self.store.save_conduit(conduit)
                ep.affinities[conduit.id] = 0.6
            
            self.store.save_entry_point(ep)
            self._entry_points_cache = None
        elif not ep.affinities and self._grains_cache:
            # Entry point exists but has NO connections — bootstrap it
            # Find grains whose content contains this feature
            feature_lower = feature.lower()
            for gid, grain in self._grains_cache.items():
                if feature_lower in grain.content.lower():
                    conduit = Conduit(
                        from_id=ep.id,
                        to_id=gid,
                        weight=0.5,
                    )
                    self.store.save_conduit(conduit)
                    ep.affinities[conduit.id] = 0.5
                    if len(ep.affinities) >= 5:  # Cap bootstrap connections
                        break
            if ep.affinities:
                self.store.save_entry_point(ep)
                self._entry_points_cache = None
        
        return ep
    
    def _bootstrap_conduits(self, new_grain: Grain, max_connections: int = 5):
        """
        Create bootstrap conduits for a new grain.
        
        Uses embedding similarity if available, falls back to keyword overlap.
        """
        if self._grains_cache is None:
            self._refresh_cache()
        
        # Try embedding-based bootstrap first
        if self.embeddings:
            connections = self.embeddings.bootstrap_conduits(
                new_grain,
                self._grains_cache,
                max_connections=max_connections,
            )
            
            for target_id, weight in connections:
                # Skip self-loops (grain connecting to itself)
                if target_id == new_grain.id:
                    continue
                
                # Create bidirectional conduit
                c1 = Conduit(
                    from_id=new_grain.id,
                    to_id=target_id,
                    weight=weight,
                    direction=Direction.BIDIRECTIONAL,
                )
                c2 = Conduit(
                    from_id=target_id,
                    to_id=new_grain.id,
                    weight=weight,
                    direction=Direction.BIDIRECTIONAL,
                )
                self.store.save_conduit(c1)
                self.store.save_conduit(c2)
            
            if connections:
                self._conduits_cache = None
                return
        
        # Fallback: keyword overlap
        new_features = set(self.decomposer.decompose(new_grain.content))
        
        candidates: List[Tuple[Grain, int]] = []
        for gid, grain in self._grains_cache.items():
            if gid == new_grain.id:
                continue
            grain_features = set(self.decomposer.decompose(grain.content))
            overlap = len(new_features & grain_features)
            if overlap > 0:
                candidates.append((grain, overlap))
        
        candidates.sort(key=lambda x: -x[1])
        for grain, _ in candidates[:max_connections]:
            c1 = Conduit(
                from_id=new_grain.id,
                to_id=grain.id,
                weight=0.3,
                direction=Direction.BIDIRECTIONAL,
            )
            c2 = Conduit(
                from_id=grain.id,
                to_id=new_grain.id,
                weight=0.3,
                direction=Direction.BIDIRECTIONAL,
            )
            self.store.save_conduit(c1)
            self.store.save_conduit(c2)
        
        self._conduits_cache = None
    
    def query(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[Tuple[Grain, float]]:
        """
        Retrieve relevant grains via signal propagation.
        
        Auto-triggers decay if 24h overdue.
        
        Args:
            query: The query text
            max_results: Maximum grains to return
        
        Returns:
            List of (Grain, signal_strength) tuples
        """
        # Auto-decay if overdue
        self._auto_decay()
        
        if self._grains_cache is None:
            self._refresh_cache()
        
        # Decompose query into features using LLM
        features = self.decomposer.decompose(query)
        
        # Find or create entry points
        entry_points = []
        for feature in features:
            ep = self._ensure_entry_point(feature)
            entry_points.append(ep)
        
        # Run signal propagation
        result = self.signal_engine.propagate(
            entry_points=entry_points,
            conduits_by_source=self._conduits_by_source_cache,
            grains=self._grains_cache,
            query=query,
        )
        
        # Limit results
        result.grains = result.grains[:max_results]
        
        # Store for feedback
        self._last_result = result
        
        # Save trace
        self.store.save_trace(result.trace)
        
        # Update grains' context_spread
        for grain, signal in result.grains:
            grain.record_retrieval("query")  # Simplified; proper version tracks clusters
            self.store.save_grain(grain)
        
        return result.grains
    
    def feedback(self, grain_id: str, success: bool = True) -> dict:
        """
        Provide feedback on a retrieval result.
        
        Args:
            grain_id: ID of the grain that was retrieved
            success: Whether the retrieval was useful
        
        Returns:
            Dict with status and details
        """
        if self._last_result is None:
            return {"status": "no_last_result", "grain_id": grain_id, "feedback": "skipped"}
        
        # Ensure caches are loaded (decay() clears them)
        if self._conduits_cache is None or self._entry_points_cache is None:
            self._refresh_cache()
        
        # Update conduit weights based on trace
        self.signal_engine.update_from_trace(
            trace=self._last_result.trace,
            conduits=self._conduits_cache,
            entry_points=self._entry_points_cache,
            success=success,
        )
        
        # Save updated conduits and entry points
        if success:
            for grain, _ in self._last_result.grains:
                if grain.id == grain_id:
                    grain.record_retrieval("feedback")
                    self.store.save_grain(grain)
                    # Tag conduits that led to this grain as USER_CONFIRMED
                    for hop in self._last_result.trace.hops:
                        if hop.to_id == grain_id:
                            cid = hop.conduit_id
                            if cid in self._conduits_cache:
                                from conduit import ConduitType
                                self._conduits_cache[cid].conduit_type = ConduitType.USER_CONFIRMED
                                self.store.save_conduit(self._conduits_cache[cid])
        
        return {"status": "ok", "grain_id": grain_id, "feedback": "useful" if success else "not_useful"}
    
    def mark_useful(self, grain: Grain):
        """Convenience method to mark a retrieved grain as useful."""
        self.feedback(grain.id, success=True)
    
    def close_loop(self, used_grain_ids: List[str] = None):
        """
        Close the feedback loop after a retrieval.
        
        Provide feedback on all grains from the last search:
        - used_grain_ids: grains actually cited/referenced → useful=True
        - everything else from last result → useful=False
        
        If used_grain_ids is None, all results from last search are marked useful.
        
        If _last_result is stale/empty but grain_ids are provided,
        applies feedback directly to those grains (fallback path).
        
        Args:
            used_grain_ids: List of grain IDs that were actually used
        """
        used_set = set(used_grain_ids) if used_grain_ids else None
        feedbacks = 0
        
        if self._last_result is not None and len(self._last_result.grains) > 0:
            # Normal path: we have the search results
            for grain, signal in self._last_result.grains:
                if used_set is not None:
                    is_useful = grain.id in used_set
                else:
                    # No explicit list → mark all as useful
                    is_useful = True
                
                self.feedback(grain.id, success=is_useful)
                feedbacks += 1
        elif used_set:
            # Fallback: no last_result but we have grain IDs
            # Mark each provided grain as useful directly
            for gid in used_set:
                grain = self._grains_cache.get(gid) if self._grains_cache else None
                if grain is None:
                    self._refresh_cache()
                    grain = self._grains_cache.get(gid)
                if grain:
                    grain.record_retrieval("feedback-positive")
                    self.store.save_grain(grain)
                    feedbacks += 1
        
        # Persist updated conduits and entry points
        if self._conduits_cache:
            for conduit in self._conduits_cache.values():
                self.store.save_conduit(conduit)
        if self._entry_points_cache:
            for ep in self._entry_points_cache.values():
                self.store.save_entry_point(ep)
        
        return {
            "status": "ok",
            "feedbacks": feedbacks,
            "used": len(used_set) if used_set else feedbacks,
            "not_used": max(0, feedbacks - len(used_set)) if used_set else 0,
        }
    
    def decay(self, now: datetime = None):
        """
        Run decay cycle on all conduits and grains.
        
        Should be called periodically (e.g., daily).
        """
        if now is None:
            now = datetime.now()
        
        if self._conduits_cache is None:
            self._refresh_cache()
        
        conduits, grains, to_remove = run_decay_cycle(
            self._conduits_cache,
            self._grains_cache,
            now,
        )
        
        # Save updated items
        for cid in to_remove:
            self.store.delete_conduit(cid)
        
        for conduit in conduits.values():
            self.store.save_conduit(conduit)
        
        for grain in grains.values():
            self.store.save_grain(grain)
        
        # Clear cache
        self._conduits_cache = None
        self._grains_cache = None
        
        return {
            "conduits_removed": len(to_remove),
            "grains_updated": len(grains),
        }
    
    def stats(self) -> dict:
        """Get Flux Memory statistics."""
        return self.store.get_stats()
    
    def self_verify(self, queries: List[dict] = None) -> List[dict]:
        """
        Active learning: query Flux, check results for freshness/accuracy.
        
        Returns a list of verification results:
        - query: the test query
        - top_result: the top grain returned
        - age_days: how old the grain is
        - context_spread: how often it's been retrieved
        - needs_refresh: True if grain is old and never verified
        
        This doesn't do web verification directly (that requires external tools).
        It identifies which grains need verification so the agent can schedule
        web searches for them.
        
        Args:
            queries: List of {query, expected_topic} dicts. If None, auto-generates.
        
        Returns:
            List of verification results with needs_refresh flag.
        """
        if self._grains_cache is None:
            self._refresh_cache()
        
        if queries is None:
            # Auto-generate: pick one core grain per category and test it
            queries = []
            categories_seen = set()
            for gid, grain in self._grains_cache.items():
                # Extract first tag as category
                tags = getattr(grain, 'tags', [])
                if tags:
                    cat = tags[0]
                    if cat not in categories_seen and len(queries) < 5:
                        categories_seen.add(cat)
                        queries.append({
                            "query": grain.content[:50],
                            "expected_grain_id": gid,
                        })
        
        results = []
        now = datetime.now()
        
        for q in queries:
            query_text = q["query"] if isinstance(q, dict) else str(q)
            search_results = self.query(query_text, max_results=3)
            
            if search_results:
                top_grain, signal = search_results[0]
                age_days = (now - top_grain.created_at).total_seconds() / 86400
                needs_refresh = (
                    age_days > 7 and top_grain.context_spread < 2 and
                    top_grain.decay_class == DecayClass.WORKING
                )
                
                results.append({
                    "query": query_text,
                    "top_grain_id": top_grain.id,
                    "top_content_preview": top_grain.content[:80],
                    "signal": signal,
                    "age_days": round(age_days, 1),
                    "context_spread": top_grain.context_spread,
                    "decay_class": top_grain.decay_class.value,
                    "needs_refresh": needs_refresh,
                })
            else:
                results.append({
                    "query": query_text,
                    "top_grain_id": None,
                    "needs_refresh": True,
                    "reason": "no_results",
                })
        
        return results
    
    def __repr__(self):
        stats = self.stats()
        return f"Flux({stats['grains']} grains, {stats['conduits']} conduits)"