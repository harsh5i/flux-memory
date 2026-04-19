# FLUX MEMORY
## Project Documentation: A Self-Organizing Retrieval Fabric for AI Memory

**Version:** 0.5 (Design Specification)
**Status:** Build-ready
**Author:** Harsh + Claude
**Date:** April 2026

**Revision history:**
- v0.1: Initial design specification
- v0.2: Multi-signal feedback, soft clustering, lazy decay, shortcut cap, vector fallback spec
- v0.3: New-knowledge starvation protection, explicit Louvain clustering algorithm, manual grain-purge API, bootstrap-weight protection
- v0.4: Networkx chosen as sole clustering library, explicit cluster split/merge remapping algorithm, admin channel gatekeeper protocol
- v0.5 (current): Simulation removed in favor of direct build with health monitoring as continuous validation

---

## 1. EXECUTIVE SUMMARY

Flux Memory is a novel AI memory mechanism where retrieval is not search, but signal propagation through a self-modifying weighted graph. Every retrieval reshapes the graph. Successful retrievals widen paths. Failed retrievals narrow them. New shortcuts emerge from co-retrieval. Unused paths decay and dissolve.

The core innovation: **the routing fabric between queries and stored memories is itself the learned index.** No embedding similarity search in the retrieval hot path. No reindexing. No separate training loop. The system learns how to find what it stored, purely through the act of being used.

Flux does use a local LLM for query decomposition (extracting feature keywords from queries) and a local embedding model for initial bootstrap of new grains. These are honest dependencies, not hidden. What Flux eliminates is the full-graph embedding search on every retrieval, which is the dominant cost in standard vector-DB memory systems.

**What Flux Memory is NOT:**
- Not a neural network (no backpropagation, no distributed encoding)
- Not a traditional memory DB (no static index, no fixed retrieval)
- Not a knowledge graph (no semantic ontology, no schema)
- Not spreading activation alone (topology itself mutates, not just signal)

**Closest analogy:** Adaptive routing in packet networks, where routers learn which paths deliver packets fastest, except here the "packets" are queries and the "delivery" is relevant memory.

---

## 2. PROBLEM STATEMENT

### 2.1 Current Memory Systems: What They Do

Most AI memory architectures follow a store-and-retrieve pattern:

- **Vector databases**: Embed text as vectors, retrieve by cosine similarity. Static after write.
- **Knowledge graphs**: Store entities and relationships, traverse by schema. Edges are semantic, not learned.
- **Hybrid systems**: Combine vectors, graphs, and keyword search. Multiple retrieval channels run in parallel, results fused.
- **Spreading activation systems**: Propagate signal through a graph. Typically used as one channel among many, not as the core organizing principle.

### 2.2 What They All Share

Every system above treats retrieval as a **read-only operation**. The act of finding a memory does not change the system's ability to find it again. Structure is created at write time and remains static until explicitly rewritten.

### 2.3 The Gap

No existing system implements all three of these simultaneously:

1. Retrieval modifies edge weights (the path that just worked gets reinforced)
2. Failed retrieval weakens paths (the path that led to irrelevant results gets penalized)
3. New edges emerge from co-retrieval success (two memories that proved useful together get directly connected)

Flux Memory fills this gap.

---

## 3. CORE CONCEPTS

### 3.1 Fundamental Units

| Unit | Symbol | What It Is | Lifecycle |
|------|--------|------------|-----------|
| **Grain** | G | Atomic memory item. Immutable content (fact, episode, preference, skill). Has a unique ID and raw content. | Created once. Never modified. Can be archived. |
| **Conduit** | C | Directional weighted edge between two grains (or between an entry point and a grain). Properties: `weight`, `last_used`, `use_count`. | Created at grain insertion (bootstrap) or through co-retrieval (emergent). Strengthened/weakened through use. Dissolved when weight drops below floor. |
| **Entry Point** | E | Where query signal enters the fabric. Created from query decomposition (keyword extraction, feature detection). | Created on first encounter with a new query feature. Develops affinity toward proven first-hop conduits. |
| **Trace** | T | The recorded path signal took during one retrieval. A list of (conduit, signal_at_hop) pairs. The learning receipt. | Created per retrieval. Consumed by the update step. Optionally archived for analytics. |

### 3.2 Key Properties

**Grain properties:**
```
{
  id:              string      // unique identifier
  content:         string      // the actual memory text
  decay_class:     enum        // working | core (starts as working, promoted through use)
  created_at:      timestamp   // when stored
  status:          enum        // active | dormant | archived
  dormant_since:   timestamp | null
  context_spread:  int         // count of distinct entry point clusters that successfully retrieved this grain
}
```

**Grain promotion (use-driven classification):**

All grains start as `working` class (7-day decay). No upfront classification. No LLM call. No manual tagging.

A grain earns promotion to `core` class (30-day decay) by proving it is useful across multiple contexts. The system tracks how many distinct entry point clusters have led to successful retrieval of a grain. When that count crosses a threshold (`PROMOTION_THRESHOLD`, default 3), the grain is promoted.

Example:
```
Week 1: "TSA has 13 Regulations" retrieved via E:VMO2 queries        → context_spread = 1
Week 3: same grain retrieved via E:compliance queries                → context_spread = 2
Week 5: same grain retrieved via E:Lumen queries                     → context_spread = 3 → PROMOTED to core
```

A project-specific grain like "VMO2 deadline is May" only ever gets retrieved via VMO2 context. Its context_spread stays at 1. It remains working class and decays naturally when the project ends.

**The grain doesn't know what it is. It learns what it is by how it's used.**

**Conduit properties:**
```
{
  id:          string      // unique identifier
  from:        string      // source grain/entry ID
  to:          string      // target grain ID
  weight:      float       // 0.0 to 1.0 (conductance)
  last_used:   timestamp   // last time signal flowed through
  use_count:   int         // total successful traversals
  direction:   enum        // forward | bidirectional
  decay_class: enum        // core | working | ephemeral
}
```

**Conduit decay classes:**

| Class | Half-Life | When Assigned |
|-------|-----------|---------------|
| **Core** | 720 hours (~30 days) | Conduits pointing to promoted grains (context_spread >= 3) |
| **Working** | 168 hours (~7 days) | Default for all conduits |
| **Ephemeral** | 48 hours (~2 days) | Conduits to grains explicitly tagged as session-specific (optional manual flag) |

When a grain is promoted from working to core, all its inbound conduits are automatically reclassified to core. The conduit inherits the decay class of the grain it points to.

**Entry Point properties:**
```
{
  id:         string      // unique identifier
  feature:    string      // the query feature this responds to
  affinities: map<conduit_id, float>  // learned bias toward first-hop conduits
}
```

### 3.3 The Signal Model

Signal is a float value that starts at 1.0 when injected at an entry point, and attenuates at each hop:

```
signal_out = signal_in * conduit.weight * attenuation_factor
```

Where `attenuation_factor` is a global constant (e.g., 0.85) that prevents infinite propagation.

A grain **activates** when it receives signal above the activation threshold (e.g., 0.15). Multiple incoming signals at a grain are summed (convergent signal reinforces activation).

---

## 4. ALGORITHMS

### 4.1 Query Decomposition

Transform a natural language query into a set of signal features:

```
Input:  "Help me pick a framework for an AI project"
Output: ["framework", "AI", "project"]
```

**Implementation: Local LLM feature extraction.**

A lightweight instruction-tuned LLM running locally (7B-8B class) extracts features from every query. The model is called with a short prompt asking it to extract 2-5 key concept words from the query. The returned list becomes the set of entry points for signal injection.

This is not optional and not replaceable by simpler alternatives:
- Keyword tokenization (split + stopword removal) fails on natural phrasing variations
- Embedding-based decomposition defeats the purpose of avoiding embeddings at retrieval time
- Manual feature input is only acceptable as a test scaffold, not in production

The local LLM gives robustness to phrasing (both "I prefer Python" and "Python is my favorite" extract the same features), handles intent beyond surface words, has zero API cost, and keeps retrieval latency tractable on typical hardware. Realistic latency on a Mac mini with 16GB RAM and a 7B model: 150-300ms per retrieval, dominated by LLM inference. This is slower than a vector DB's 10-20ms per query on a mature graph but comparable or faster on cold graphs where vector DBs perform expensive full-corpus similarity searches.

Each extracted feature maps to an Entry Point. If no Entry Point exists for a feature, one is created.

The feature extractor is part of the read channel of the interaction protocol. It runs on every retrieval.

### 4.2 Signal Propagation (Retrieval)

```python
def retrieve(query, max_hops=5, threshold=0.15, top_k=5):
    features = decompose(query)
    activated = {}          # grain_id -> total_signal
    trace = []              # list of (conduit, signal) tuples
    visited_conduits = set()
    
    # Step 1: Inject signal at entry points
    frontier = []
    for feature in features:
        entry = get_or_create_entry(feature)
        for conduit in get_outgoing_conduits(entry.id):
            initial_signal = 1.0 * conduit.weight * entry.affinities.get(conduit.id, 1.0)
            frontier.append((conduit.to, initial_signal, 0, conduit))
    
    # Step 2: Propagate
    while frontier:
        grain_id, signal, hop, conduit = frontier.pop(0)  # BFS
        
        if hop >= max_hops or signal < threshold:
            continue
        if conduit.id in visited_conduits:
            continue
        
        visited_conduits.add(conduit.id)
        trace.append((conduit, signal))
        
        # Accumulate signal at grain
        activated[grain_id] = activated.get(grain_id, 0) + signal
        
        # Propagate onward
        for next_conduit in get_outgoing_conduits(grain_id):
            next_signal = signal * next_conduit.weight * ATTENUATION
            if next_signal >= threshold:
                frontier.append((next_conduit.to, next_signal, hop + 1, next_conduit))
    
    # Step 3: Rank and return
    results = sorted(activated.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return results, trace
```

**Complexity:** O(E_active) where E_active is the number of conduits with weight above threshold along reachable paths. In a mature system with highways, this is much smaller than total edges.

**Retrieval confidence computation:**

Several downstream mechanisms (query-time expansion, fallback triggering, health signals) need a single-number confidence score for a retrieval. Confidence is computed from the propagation result:

```python
def retrieval_confidence(activated_grains, trace):
    """Returns a confidence score in [0.0, 1.0] for a retrieval."""
    if not activated_grains:
        return 0.0
    
    # Top signal strength (how strongly the best grain activated)
    top_signal = max(signal for grain_id, signal in activated_grains)
    signal_score = min(top_signal / 1.0, 1.0)  # normalize to 0-1
    
    # Path quality (average weight of conduits traversed to reach top grain)
    conduits_to_top = [c for c, s in trace if c.to_id == activated_grains[0][0]]
    if conduits_to_top:
        avg_weight = sum(c.weight for c in conduits_to_top) / len(conduits_to_top)
        path_score = avg_weight  # already in 0-1
    else:
        path_score = 0.0
    
    # Top-k concentration (higher = more peaked, lower = diffuse)
    if len(activated_grains) > 1:
        top_ratio = activated_grains[0][1] / sum(s for _, s in activated_grains)
        concentration_score = top_ratio  # 0.5+ means top grain dominates
    else:
        concentration_score = 1.0
    
    # Weighted combination
    confidence = (0.5 * signal_score + 0.3 * path_score + 0.2 * concentration_score)
    return min(max(confidence, 0.0), 1.0)
```

Confidence thresholds used elsewhere:
- Below 0.4: query-time context expansion fires (Section 11.11)
- Below 0.25: vector fallback fires
- Below 0.15: retrieval logged as likely failure

### 4.3 Reinforcement (Success)

When retrieved grains are confirmed useful:

```python
def reinforce(trace, successful_grain_ids, learning_rate=0.05):
    # 1. Widen conduits on the successful trace
    for conduit, signal in trace:
        if conduit.to in successful_grain_ids:
            conduit.weight = min(
                conduit.weight + learning_rate * (1.0 - conduit.weight),
                WEIGHT_CEILING  # e.g., 0.95
            )
            conduit.last_used = now()
            conduit.use_count += 1
    
    # 2. Update co-retrieval counts and create shortcuts between co-retrieved successful grains
    successful = list(successful_grain_ids)
    for i in range(len(successful)):
        for j in range(i + 1, len(successful)):
            # Increment co-retrieval count for this pair (regardless of whether shortcut exists)
            increment_co_retrieval_count(successful[i], successful[j])
            
            existing = get_conduit(successful[i], successful[j])
            if existing:
                # Reinforce existing shortcut
                existing.weight = min(
                    existing.weight + learning_rate * (1.0 - existing.weight),
                    WEIGHT_CEILING
                )
            else:
                co_count = get_co_retrieval_count(successful[i], successful[j])
                if co_count >= SHORTCUT_THRESHOLD:  # e.g., 3
                    # Check edge cap before creating
                    if can_add_edge(successful[i]) and can_add_edge(successful[j]):
                        create_conduit(
                            from=successful[i],
                            to=successful[j],
                            weight=INITIAL_SHORTCUT_WEIGHT,
                            direction="bidirectional"
                        )
                    else:
                        # Evict weakest edge on saturated grain, then create
                        evict_weakest_edge(successful[i] if not can_add_edge(successful[i]) else successful[j])
                        create_conduit(
                            from=successful[i],
                            to=successful[j],
                            weight=INITIAL_SHORTCUT_WEIGHT,
                            direction="bidirectional"
                        )
    
    def increment_co_retrieval_count(grain_a, grain_b):
        """Canonicalize pair (lower_id, higher_id), then UPSERT count + 1."""
        a, b = sorted([grain_a, grain_b])
        upsert_co_retrieval(a, b, delta=1)
    
    def can_add_edge(grain_id):
        """True if grain has fewer than MAX_EDGES_PER_GRAIN conduits."""
        return count_all_edges(grain_id) < MAX_EDGES_PER_GRAIN
    
    def evict_weakest_edge(grain_id):
        """Remove the lowest-weight conduit attached to this grain."""
        edges = get_all_edges(grain_id)
        weakest = min(edges, key=lambda c: effective_weight(c))
        delete_conduit(weakest)
    
    # 3. Sharpen entry affinities
    for conduit, signal in trace:
        if conduit.from in entry_points:
            entry = get_entry(conduit.from)
            entry.affinities[conduit.id] = min(
                entry.affinities.get(conduit.id, 1.0) * 1.1,
                2.0  # affinity ceiling
            )
```

### 4.4 Penalization (Failure)

When retrieved grains are confirmed irrelevant:

```python
def penalize(trace, failed_grain_ids, decay_factor=0.85):
    # 1. Narrow conduits on the failed trace
    for conduit, signal in trace:
        if conduit.to in failed_grain_ids:
            conduit.weight *= decay_factor
            if conduit.weight < WEIGHT_FLOOR:  # e.g., 0.05
                delete_conduit(conduit)
    
    # 2. Dampen entry affinities toward failed first-hops
    for conduit, signal in trace:
        if conduit.from in entry_points:
            entry = get_entry(conduit.from)
            if conduit.to in failed_grain_ids:
                entry.affinities[conduit.id] = max(
                    entry.affinities.get(conduit.id, 1.0) * 0.8,
                    0.1  # floor
                )
    
    # 3. Widen propagation radius for next similar query
    for feature in get_features_from_trace(trace):
        set_temporary_exploration_boost(feature, duration=3)  # next 3 queries
```

### 4.5 Temporal Decay (Lazy Evaluation)

Decay is computed lazily: conduit weights stored in the database reflect the weight at last touch. The effective weight at any given moment is computed on read as:

```python
def effective_weight(conduit):
    half_lives = {
        'core': 720,       # 30 days
        'working': 168,    # 7 days
        'ephemeral': 48    # 2 days
    }
    half_life = half_lives.get(conduit.decay_class, 168)
    hours_since_use = (now() - conduit.last_used).total_hours()
    
    # Grace period: newly-created conduits decay at a reduced rate for their
    # first NEW_CONDUIT_GRACE_HOURS, giving new grains time to accumulate
    # retrievals before aggressive decay kicks in.
    hours_since_creation = (now() - conduit.created_at).total_hours()
    if hours_since_creation < NEW_CONDUIT_GRACE_HOURS:  # default: 72 hours
        effective_half_life = half_life * NEW_CONDUIT_GRACE_MULTIPLIER  # default: 2.0
    else:
        effective_half_life = half_life
    
    decay_multiplier = 0.5 ** (hours_since_use / effective_half_life)
    weight = conduit.stored_weight * decay_multiplier
    
    # New-knowledge floor: newly-inserted conduits cannot decay below a minimum
    # floor during their grace period, preventing starvation of rare-but-valuable
    # inferred grains before they earn their first successful retrieval.
    if hours_since_creation < NEW_CONDUIT_GRACE_HOURS:
        weight = max(weight, NEW_CONDUIT_MIN_WEIGHT)  # default: WEIGHT_FLOOR * 2
    
    return weight
```

This is called inline during signal propagation (Section 4.2). No full-graph iteration is required for decay itself.

**Why lazy:** A full-graph decay pass is O(E) where E is the total number of conduits. For graphs of 10,000+ grains with typical connectivity, E reaches hundreds of thousands. Running this every hour creates avoidable load and lock contention on SQLite. Lazy decay pushes the computation into the retrieval path, where it runs only on conduits actually traversed.

**Cleanup pass (incremental, not full-scan):**

A lighter background job runs periodically to garbage-collect conduits whose effective weight has dropped below WEIGHT_FLOOR. This uses an index on `last_used` to find conduits that have not been touched recently, computes their effective weight, and deletes those below floor. This is incremental and bounded in cost:

```python
def cleanup_pass():
    """Runs every CLEANUP_INTERVAL_HOURS. Scans recently-untouched conduits only."""
    stale_cutoff = now() - timedelta(hours=CLEANUP_STALE_HOURS)  # default 72 hours
    candidates = query_conduits_unused_since(stale_cutoff, limit=CLEANUP_BATCH_SIZE)
    
    for conduit in candidates:
        if effective_weight(conduit) < WEIGHT_FLOOR:
            delete_conduit(conduit)
    
    # Incremental orphan detection
    for grain_id in recently_affected_grains():
        if count_inbound_conduits(grain_id) == 0:
            mark_dormant(grain_id)
```

**New parameters for cleanup:**

- `CLEANUP_INTERVAL_HOURS`: 6 (how often to run cleanup)
- `CLEANUP_STALE_HOURS`: 72 (only consider conduits unused for at least this long)
- `CLEANUP_BATCH_SIZE`: 1000 (max conduits processed per run)

**Write-time weight updates:**

When a conduit is touched (reinforced, penalized, or traversed for reading), its effective weight is first computed, then the new weight is written along with `last_used = now()`. This means the stored weight always reflects weight-as-of-last-touch, and lazy evaluation picks up from there.

```python
def touch_conduit(conduit, delta):
    current = effective_weight(conduit)
    new_weight = clamp(current + delta, WEIGHT_FLOOR, WEIGHT_CEILING)
    conduit.stored_weight = new_weight
    conduit.last_used = now()
    conduit.use_count += 1
```

This preserves correctness: any conduit, whether touched once a week or once a year, always returns the right effective weight when read.

### 4.6 Dormancy Expiry

Runs less frequently (e.g., daily):

```python
def expiry_pass(dormancy_limit_days=30):
    for grain in all_dormant_grains():
        if (now() - grain.dormant_since).days > dormancy_limit_days:
            grain.status = "archived"
            # Optionally: move to cold storage, delete from active graph
```

### 4.7 Grain Insertion (Memory Growth)

When a new memory enters:

```python
def insert_grain(content):
    grain = create_grain(
        content=content,
        decay_class='working',    # everything starts as working
        context_spread=0           # no cross-context retrievals yet
    )
    
    # Bootstrap: one-time embedding similarity to find neighbors
    embedding = embed(content)
    neighbors = find_nearest_grains(embedding, k=5)
    
    for neighbor in neighbors:
        similarity = cosine_similarity(embedding, embed(neighbor.content))
        create_conduit(
            from=grain.id,
            to=neighbor.id,
            weight=similarity * INITIAL_WEIGHT_SCALE,
            direction="bidirectional",
            decay_class='working'
        )
    
    # Connect to entry points extracted from content
    features = extract_features(content)
    for feature in features:
        entry = get_or_create_entry(feature)
        create_conduit(
            from=entry.id,
            to=grain.id,
            weight=INITIAL_ENTRY_WEIGHT,
            decay_class='working'
        )
    
    return grain
```

**Key principle:** Embedding is used only once, at insertion, to bootstrap initial conduits. After that, the grain's reachability is determined entirely by use-driven weight evolution. No classification happens at write time. The grain earns its permanence through use.

### 4.8 Vector Fallback

When graph propagation returns poor results, the vector fallback provides a safety net. The embedding model is retained (in memory or on disk) specifically for this path.

**Trigger conditions** (any one fires fallback):
- Propagation returns zero activated grains
- `retrieval_confidence(results, trace) < FALLBACK_CONFIDENCE_THRESHOLD` (default 0.25)
- Top-k activated grains all have decay_class='working' with low weights (graph is at bootstrap state)

**Algorithm:**

```python
def vector_fallback(query_text, existing_results, existing_trace):
    """
    Returns a merged result set combining graph results (if any) with 
    vector-similarity top-k.
    """
    query_embedding = embed(query_text)
    
    # Find nearest grains by cosine similarity over stored embeddings
    # Only consider active (non-dormant, non-quarantined) grains
    vector_candidates = find_nearest_grains(
        query_embedding,
        k=VECTOR_FALLBACK_K,  # default: 10
        filter_status='active'
    )
    
    # Score each candidate: cosine similarity, scaled to match graph signal range
    vector_results = [
        (grain.id, similarity * VECTOR_FALLBACK_SCALE)
        for grain, similarity in vector_candidates
    ]
    
    # Merge policy: union with dedup, highest score wins for duplicates
    merged = {}
    for grain_id, score in existing_results:
        merged[grain_id] = ('graph', score)
    for grain_id, score in vector_results:
        if grain_id not in merged or score > merged[grain_id][1]:
            merged[grain_id] = ('vector', score)
    
    # Sort by score, return top-k overall
    final = sorted(merged.items(), key=lambda x: x[1][1], reverse=True)[:TOP_K]
    
    # Extend trace with vector fallback marker so feedback can distinguish sources
    extend_trace_with_fallback(existing_trace, vector_results)
    
    return [(gid, source_score) for gid, (_, source_score) in final]
```

**Feedback interaction:**

When a grain returned via vector fallback is marked useful by feedback, a bootstrap conduit is created or reinforced between the closest activated entry point and that grain. This lets the graph "learn from the fallback" over time, reducing future fallback dependency.

```python
def on_feedback_useful_fallback(grain_id, query_features):
    """Called when a vector-fallback-returned grain is marked useful."""
    for feature in query_features:
        entry = get_or_create_entry(feature)
        existing = get_conduit(entry.id, grain_id)
        if existing:
            reinforce(existing, LEARNING_RATE)
        else:
            create_conduit(
                from=entry.id, to=grain_id,
                weight=INITIAL_ENTRY_WEIGHT,
                direction='forward',
                decay_class='working'
            )
```

**Parameters:**

- `FALLBACK_CONFIDENCE_THRESHOLD`: 0.25 (confidence below this triggers fallback)
- `VECTOR_FALLBACK_K`: 10 (nearest neighbors to retrieve from vector index)
- `VECTOR_FALLBACK_SCALE`: 0.5 (multiplier to align cosine similarity with graph signal range)

**Latency budget:** Vector fallback adds 50-150ms on a local deployment with 10k-100k grains. Because it only fires on low-confidence retrievals (roughly 5-20% of mature-graph queries, higher during cold start), amortized cost is low.

**Health Monitor signal:** `fallback_trigger_rate` tracks what percentage of retrievals fire fallback. Healthy: <5% in mature graph, <20% during warming. Persistent high rates indicate the graph is not learning effectively.

### 4.9 Grain Promotion (Use-Driven Classification)

Runs as part of the reinforcement step (Section 4.3), after every successful retrieval:

```python
def check_promotion(grain_id, trace):
    grain = get_grain(grain_id)
    if grain.decay_class == 'core':
        return  # already promoted
    
    # Identify which entry points activated this grain during the retrieval.
    # An entry point is "in the trace" if it injected signal that reached the grain.
    entry_points_in_trace = get_activating_entry_points(trace, grain_id)
    
    # For each activating entry point, get its soft cluster membership map:
    #   {cluster_id_A: 0.7, cluster_id_B: 0.2, cluster_id_C: 0.1}
    # Accumulate touch_weight for the grain across all touched clusters.
    for entry_id in entry_points_in_trace:
        memberships = get_cluster_memberships(entry_id)  # dict: cluster_id -> weight
        for cluster_id, membership_weight in memberships.items():
            # Apply touch weight to the grain's per-cluster accumulator
            increment_grain_cluster_touch(
                grain_id=grain_id,
                cluster_id=cluster_id,
                delta=membership_weight
            )
    
    # Count clusters where the grain's accumulated touch weight exceeds minimum
    grain.context_spread = count_clusters_above_threshold(
        grain_id=grain_id,
        min_touch_weight=CLUSTER_TOUCH_THRESHOLD  # default: 1.0
    )
    
    # Promote if retrieved successfully from enough distinct clusters
    if grain.context_spread >= PROMOTION_THRESHOLD:  # default: 3
        grain.decay_class = 'core'
        for conduit in get_inbound_conduits(grain_id):
            conduit.decay_class = 'core'

def get_activating_entry_points(trace, grain_id):
    """Return entry points whose injected signal reached this grain in this trace."""
    # Walk the trace backward from grain_id to find the entry points that started propagation paths ending at grain_id
    reached_entries = set()
    for (conduit, signal) in trace:
        if conduit.to_id == grain_id or path_connects(conduit, grain_id, trace):
            # Find the entry point at the start of the path that contains this conduit
            origin_entry = trace_origin(conduit, trace)
            if origin_entry is not None:
                reached_entries.add(origin_entry)
    return reached_entries

def get_cluster_memberships(entry_id):
    """Return {cluster_id: weight} for an entry point. Weights sum to 1.0."""
    return query_entry_cluster_membership(entry_id)

def increment_grain_cluster_touch(grain_id, cluster_id, delta):
    """Add delta to the grain's touch_weight in the given cluster."""
    update_grain_cluster_touch(grain_id, cluster_id, delta)

def count_clusters_above_threshold(grain_id, min_touch_weight):
    """Return count of clusters where grain's touch_weight >= min_touch_weight."""
    return count_clusters_for_grain(grain_id, min_touch_weight)
```

**Soft membership model.** An entry point can belong to multiple clusters with weighted membership. "Security" might have 0.7 membership in VMO2 cluster, 0.4 in security-architecture cluster, 0.2 in DPDPA cluster. Each retrieval distributes touch-weight across all touched clusters proportional to the entry point's membership. This avoids flickering when cluster boundaries shift.

**Distinct cluster count.** `context_spread` counts clusters where the grain has accumulated enough touch weight (default: 1.0, equivalent to one "full" cluster-aligned retrieval). A grain retrieved 10 times via entry points strongly in Cluster A will have Cluster A touch ≈ 9, other clusters ≈ 1-2. context_spread = 1 (correct). A grain retrieved across entries strongly in Clusters A, B, and C will have all three above threshold. context_spread = 3.

**Why cluster-based, not entry-point-based.** A grain retrieved through four different entry points that all belong strongly to the VMO2 cluster accumulates touch_weight primarily in that one cluster. context_spread = 1. The grain correctly stays working class and decays with VMO2.

A grain retrieved through entry points that belong strongly to three different clusters (VMO2, Lumen DPDPA, security architecture) accumulates touch weight in all three. context_spread = 3. Promotion fires correctly. The grain survives.

**Stability across reclustering.** When clusters are recomputed during the decay pass, cluster IDs may change but the `grain_cluster_touch` accumulator persists. Historical touch weights are re-mapped to new cluster structures using a best-overlap rule (each old cluster's touch weight is assigned to the new cluster with highest member overlap). This prevents promotion from resetting on every reclustering.

---

## 5. SYSTEM PARAMETERS

All tunable constants in one place:

| Parameter | Default | Purpose | Tuning Notes |
|-----------|---------|---------|--------------|
| `ATTENUATION` | 0.85 | Signal loss per hop | Lower = narrower reach. Higher = broader but noisier. |
| `ACTIVATION_THRESHOLD` | 0.15 | Minimum signal to activate a grain | Lower = more results. Higher = more precision. |
| `MAX_HOPS` | 5 | Maximum propagation depth | Higher = deeper reach but slower. |
| `TOP_K` | 5 | Maximum grains returned | Context window budget. |
| `LEARNING_RATE` | 0.05 | Weight increase on success | Higher = faster learning but more volatile. |
| `DECAY_FACTOR` | 0.85 | Weight multiplier on failure | Lower = harsher penalty. |
| `WEIGHT_CEILING` | 0.95 | Maximum conduit weight | Prevents monopoly by single paths. |
| `WEIGHT_FLOOR` | 0.05 | Minimum weight before deletion | Lower = more persistent edges. |
| `HALF_LIFE_CORE` | 720 hours (30 days) | Decay half-life for core conduits (knowledge/skill grains) | Protects stable learning from premature decay. |
| `HALF_LIFE_WORKING` | 168 hours (7 days) | Decay half-life for working conduits (context grains) | Standard project-tempo decay. |
| `HALF_LIFE_EPHEMERAL` | 48 hours (2 days) | Decay half-life for ephemeral conduits (session grains) | Fast cleanup of transient context. |
| `PROMOTION_THRESHOLD` | 3 | Distinct context clusters needed for working-to-core promotion | Lower = easier promotion. Higher = stricter proof of transferability. |
| `CLUSTER_TOUCH_THRESHOLD` | 1.0 | Minimum touch_weight for a cluster to count toward context_spread | Represents roughly one "full" cluster-aligned retrieval. |
| `ENTRY_COOCCURRENCE_THRESHOLD` | 10 | Minimum pairwise co-occurrence count before clustering considers an edge | Lower = more edges feed clustering. Higher = sparser clustering graph. |
| `CLUSTER_WINDOW_DAYS` | 30 | Rolling window for entry point co-occurrence counting | Shorter window = clusters adapt faster to interest changes. |
| `CLUSTER_MIN_SIZE` | 3 | Minimum entry points per cluster; smaller clusters merge with nearest neighbor | Prevents singleton clusters. |
| `LOUVAIN_RESOLUTION` | 1.0 | Louvain resolution parameter. Higher = more, smaller clusters. | Use 0.5 for coarser grouping, 1.5-2.0 for finer. |
| `LOUVAIN_SEED` | 42 | Random seed for Louvain (ensures reproducible clustering) | Any fixed integer works; change only when deliberately re-partitioning. |
| `CLUSTER_RECOMPUTE_MIN_INTERVAL_DAYS` | 7 | Minimum days between cluster recomputations | Prevents flicker from minor co-occurrence shifts. |
| `CLUSTER_INHERIT_OVERLAP_MIN` | 0.30 | Minimum Jaccard overlap for a new cluster to inherit an old cluster's ID | Below this threshold, a fresh UUID is assigned. |
| `CLUSTER_DISSOLVE_DECAY` | 0.5 | Touch weight carryover rate when an old cluster has no strong successor | Preserves some continuity rather than losing evidence entirely. |
| `TRACE_RETENTION_COUNT` | 10000 | Most recent traces kept in hot storage | Higher = more historical data for health metrics. Lower = less storage. |
| `TRACE_RETENTION_DAYS` | 30 | Days of traces kept in hot storage | Whichever of count or days is larger wins. |
| `EXPANSION_CONFIDENCE_THRESHOLD` | 0.4 | Signal strength below which query-time context expansion fires | Lower = expansion rarely fires. Higher = expansion more aggressive. |
| `EXPANSION_CANDIDATES_PER_CLUSTER` | 2 | Max lateral candidates surfaced per shared cluster | Higher = more candidates, more noise. |
| `EXPANSION_MAX_CANDIDATES` | 3 | Hard cap on total lateral candidates per retrieval | Upper bound on added context. |
| `EXPANSION_ENABLED` | true | Toggle for query-time context expansion | Can be disabled if expansion produces too much noise. |
| `CONTEXT_SHIFT_WINDOW` | 30 | Number of recent retrievals used to measure success trajectory | Shorter = faster detection, more false positives. |
| `CONTEXT_SHIFT_DROP_THRESHOLD` | 0.25 | Drop in success rate that triggers context shift detection | Lower = more sensitive. |
| `CONTEXT_SHIFT_RECOVERY_RETRIEVALS` | 50 | Number of retrievals during which elevated exploration applies after shift | Higher = longer recovery period. |
| `CONTEXT_SHIFT_ENABLED` | true | Toggle for context shift detection | |
| `USEFULNESS_WINDOW_DAYS` | 7 | Rolling window for per-grain usefulness ratio tracking | |
| `QUARANTINE_USEFULNESS_THRESHOLD` | 0.2 | Usefulness ratio below which quarantine is triggered (with min retrieval count) | |
| `QUARANTINE_MIN_RETRIEVALS` | 10 | Minimum retrievals before quarantine can trigger via usefulness ratio | Prevents over-quarantine of new grains |
| `QUARANTINE_CORRECTION_COUNT` | 3 | Number of downstream correction signals that trigger quarantine | |
| `QUARANTINE_PERIOD_DAYS` | 30 | Days a grain stays quarantined before permanent archival | |
| `CORRECTION_DETECTION_TURNS` | 3 | Number of user turns after retrieval in which correction signals are considered | |
| `CLEANUP_INTERVAL_HOURS` | 6 | How often the incremental cleanup pass runs | Higher = less background work, slower orphan detection |
| `CLEANUP_STALE_HOURS` | 72 | Only conduits unused for at least this long are considered for cleanup | |
| `CLEANUP_BATCH_SIZE` | 1000 | Maximum conduits scanned per cleanup run | Bounds cleanup cost |
| `NEW_CONDUIT_GRACE_HOURS` | 72 | Grace period during which new conduits decay at reduced rate and have a higher floor | Protects rare-but-valuable inferred grains from premature decay |
| `NEW_CONDUIT_GRACE_MULTIPLIER` | 2.0 | Half-life multiplier during grace period (e.g., 7-day working grain gets effectively 14-day half-life for its first 72 hours) | |
| `NEW_CONDUIT_MIN_WEIGHT` | 0.10 | Minimum effective weight for conduits during grace period | Typically 2x WEIGHT_FLOOR. Prevents starvation. |
| `FALLBACK_CONFIDENCE_THRESHOLD` | 0.25 | Retrieval confidence below which vector fallback fires | |
| `VECTOR_FALLBACK_K` | 10 | Nearest neighbors retrieved from vector index during fallback | |
| `VECTOR_FALLBACK_SCALE` | 0.5 | Multiplier to align cosine similarity with graph signal range | |
| `INITIAL_SHORTCUT_WEIGHT` | 0.50 | Starting weight for emergent shortcuts | Raised from 0.30 to ensure new shortcuts propagate signal. |
| `INITIAL_ENTRY_WEIGHT` | 0.50 | Starting weight for entry-to-grain conduits | Raised from 0.25 to ensure fresh-graph propagation reaches 2 hops (see Section 5.1). |
| `INITIAL_WEIGHT_SCALE` | 0.50 | Multiplier on similarity for bootstrap conduits | Raised from 0.30 for the same reason. |
| `SHORTCUT_THRESHOLD` | 3 | Co-retrieval count before shortcut created | Higher = fewer but more reliable shortcuts. |
| `MAX_EDGES_PER_GRAIN` | 50 | Hard cap on inbound + outbound conduits per grain | Prevents shortcut explosion. When exceeded, weakest conduit is pruned. |
| `DORMANCY_LIMIT_DAYS` | 30 | Days in dormancy before archival | |
| `EXPLORATION_BOOST` | 1.5 | Temporary propagation radius multiplier after failure | Applied as: signal multiplier for frontier computation during post-failure retrievals. See Section 5.2. |

### 5.1 Parameter Math: Fresh-Graph Propagation

The default parameters must permit signal to propagate on a fresh graph (bootstrap-only weights, no reinforcement yet). Otherwise the self-organizing mechanism cannot start.

**Fresh-graph propagation check:**

Signal starts at 1.0. After N hops, signal equals:

```
signal_N = 1.0 × W_entry × (ATTENUATION × W_bootstrap × ATTENUATION)^(N-1)
         = W_entry × (0.85 × W_bootstrap × 0.85)^(N-1)
         = 0.50 × (0.85 × 0.50 × 0.85)^(N-1)
         = 0.50 × (0.3613)^(N-1)
```

| Hops | Signal | Above threshold (0.15)? |
|------|--------|-------------------------|
| 1 | 0.425 | ✓ yes |
| 2 | 0.154 | ✓ just barely |
| 3 | 0.056 | ✗ dies (correct) |

A fresh graph propagates 2 hops reliably. Reinforcement grows useful paths; decay weakens unused ones. Mature graphs propagate further on trained highways.

**Reinforced-path propagation:**

After training, successful paths have weights approaching `WEIGHT_CEILING` (0.95). The same math with trained weights:

```
signal_N = 0.95 × (0.85 × 0.95 × 0.85)^(N-1)
         = 0.95 × (0.687)^(N-1)
```

| Hops | Signal | Above threshold? |
|------|--------|------------------|
| 1 | 0.808 | ✓ |
| 2 | 0.555 | ✓ |
| 3 | 0.381 | ✓ |
| 4 | 0.262 | ✓ |
| 5 | 0.180 | ✓ |
| 6 | 0.124 | ✗ (correctly bounded by MAX_HOPS=5) |

Mature highways reach the full 5-hop range. This is the intended behavior.

### 5.2 EXPLORATION_BOOST Math

When a retrieval fails (returns no useful grains per feedback), the next retrieval from affected entry points applies `EXPLORATION_BOOST` to the signal injection:

```
initial_signal = 1.0 × EXPLORATION_BOOST  (only for boosted retrievals)
```

With default `EXPLORATION_BOOST = 1.5`:
- Hop 1 signal: 0.50 × 0.85 × 1.5 = 0.638 (vs 0.425 without boost)
- Hop 2 signal: 0.638 × 0.50 × 0.85 = 0.271
- Hop 3 signal: 0.271 × 0.50 × 0.85 = 0.115

A boosted retrieval on a fresh graph reaches 2+ hops reliably and brushes 3 hops. This gives failure a chance to find alternatives.

Boost applies for the next N retrievals from the same entry point cluster (default N=3) or until a successful retrieval, whichever comes first.

---

## 6. ARCHITECTURE

### 6.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      FLUX MEMORY SYSTEM                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │   QUERY      │   │   SIGNAL     │   │   FEEDBACK   │    │
│  │   DECOMPOSER │──>│   PROPAGATOR │──>│   PROCESSOR  │    │
│  └──────────────┘   └──────┬───────┘   └──────┬───────┘    │
│                            │                   │            │
│                            ▼                   ▼            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              CONDUCTANCE FABRIC                      │   │
│  │  ┌─────────────────────────────────────────────┐     │   │
│  │  │  GRAPH STORE                                │     │   │
│  │  │  - Grains (nodes)                           │     │   │
│  │  │  - Conduits (weighted directed edges)       │     │   │
│  │  │  - Entry Points (query gates)               │     │   │
│  │  └─────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              MAINTENANCE DAEMONS                     │   │
│  │  - Temporal Decay (periodic)                         │   │
│  │  - Orphan Detection (periodic)                       │   │
│  │  - Dormancy Expiry (daily)                           │   │
│  │  - Stats Collection (optional)                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────┐                                           │
│  │  BOOTSTRAP   │  (Embedding model: used ONLY at insert)   │
│  │  ENGINE      │                                           │
│  └──────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Data Flow

```
WRITE PATH:
  New content → Bootstrap Engine (embed once)
             → Find k-nearest existing grains
             → Create grain (decay_class = working) + initial conduits
             → Extract features → connect to entry points
             → Done. Embedding discarded. No classification needed.

READ PATH:
  Query → Decomposer → features
       → Entry Points injected with signal
       → Signal propagates through conduits (weight * attenuation)
       → Grains above threshold activate
       → Top-k returned + Trace recorded

LEARN PATH:
  User feedback (success/fail per grain)
       → Trace consumed
       → Success: widen conduits, create shortcuts, sharpen entries
       → Success: check promotion (cross-context retrieval → working to core)
       → Failure: narrow conduits, dampen entries, boost exploration
       → Trace archived or discarded

DECAY PATH (background):
  Timer → All conduits lose weight proportional to time since last use
       → Decay rate varies by class: core (30d), working (7d), ephemeral (2d)
       → Sub-floor conduits deleted
       → Orphaned grains marked dormant
       → Expired dormant grains archived
```

### 6.3 Storage Backend Options

Flux Memory is **storage-agnostic**. The mechanism defines behavior, not backend. Options:

| Backend | When to Use | Pros | Cons |
|---------|-------------|------|------|
| **SQLite** | Prototype, single-user, local | Zero infra, fast for small graphs (<100K edges) | No concurrency, limited scale |
| **PostgreSQL** | Production, multi-user | Mature, ACID, good graph queries with recursive CTEs | Heavier setup |
| **Redis + sorted sets** | Low-latency, ephemeral | Sub-ms weight lookups, natural TTL for decay | No persistence guarantees |
| **Native graph database** | If graph queries dominate | Native graph traversal, purpose-built query language | Overkill for the weight-update pattern |
| **In-memory (Python dict)** | Testing, experiments | Fastest possible | Not persistent |

**Recommended for v0:** SQLite with WAL mode. Three tables: `grains`, `conduits`, `entries`.

### 6.4 Schema (SQLite v0)

```sql
CREATE TABLE grains (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    provenance      TEXT NOT NULL,           -- 'user_stated' | 'ai_stated' | 'ai_inferred' | 'external_source'
    confidence      REAL DEFAULT 1.0,        -- source-weighted confidence, 0.0-1.0
    decay_class     TEXT DEFAULT 'working',  -- working | core
    status          TEXT DEFAULT 'active',   -- active | dormant | archived | quarantined
    created_at      TEXT DEFAULT (datetime('now')),
    dormant_since   TEXT,
    context_spread  INTEGER DEFAULT 0        -- distinct context clusters that retrieved this grain
);

CREATE TABLE conduits (
    id          TEXT PRIMARY KEY,
    from_id     TEXT NOT NULL,
    to_id       TEXT NOT NULL,
    weight      REAL DEFAULT 0.25,
    created_at  TEXT DEFAULT (datetime('now')),
    last_used   TEXT DEFAULT (datetime('now')),
    use_count   INTEGER DEFAULT 0,
    direction   TEXT DEFAULT 'forward',  -- forward | bidirectional
    decay_class TEXT DEFAULT 'working',  -- core | working | ephemeral
    UNIQUE(from_id, to_id)
);

CREATE TABLE entries (
    id          TEXT PRIMARY KEY,
    feature     TEXT NOT NULL UNIQUE,
    affinities  TEXT DEFAULT '{}'  -- JSON map of conduit_id -> affinity_float
);

CREATE TABLE entry_cluster_membership (
    entry_id     TEXT NOT NULL,
    cluster_id   TEXT NOT NULL,
    weight       REAL NOT NULL,  -- soft membership weight, per-entry weights sum to 1.0
    PRIMARY KEY (entry_id, cluster_id)
);

CREATE TABLE entry_cooccurrence (
    entry_a      TEXT NOT NULL,
    entry_b      TEXT NOT NULL,
    count        INTEGER DEFAULT 0,
    last_updated TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (entry_a, entry_b)
);

CREATE TABLE clusters (
    id           TEXT PRIMARY KEY,
    size         INTEGER DEFAULT 0,  -- number of entry points with membership > 0.1
    created_at   TEXT DEFAULT (datetime('now')),
    last_updated TEXT DEFAULT (datetime('now'))
);

CREATE TABLE grain_cluster_touch (
    grain_id     TEXT NOT NULL,
    cluster_id   TEXT NOT NULL,
    touch_weight REAL DEFAULT 0.0,  -- accumulated cluster-touch weight from successful retrievals
    last_touched TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (grain_id, cluster_id)
);

CREATE TABLE traces (
    id                    TEXT PRIMARY KEY,
    query_text            TEXT,
    created_at            TEXT DEFAULT (datetime('now')),
    feedback_at           TEXT,
    hop_count             INTEGER,
    activated_grain_count INTEGER,
    trace_data            TEXT  -- JSON: full conduit path + signal values
);

CREATE INDEX idx_traces_created ON traces(created_at);

CREATE TABLE co_retrieval_counts (
    grain_a     TEXT NOT NULL,
    grain_b     TEXT NOT NULL,
    count       INTEGER DEFAULT 0,
    PRIMARY KEY (grain_a, grain_b)
);

-- Indexes for propagation performance
CREATE INDEX idx_conduits_from ON conduits(from_id) WHERE weight >= 0.05;
CREATE INDEX idx_conduits_to ON conduits(to_id);
CREATE INDEX idx_grains_status ON grains(status);
```

---

## 7. FEEDBACK MECHANISM

Feedback is the learning signal that drives Flux's adaptation. Because Flux depends on feedback quality, the mechanism uses multiple independent signals rather than a single confounded one.

### 7.1 Multi-Signal Feedback

Flux combines three feedback signals, each with different reliability characteristics. No single signal is trusted in isolation.

**Signal 1: AI usage (primary)**

When the AI calls `flux_retrieve(query)` and receives grains + trace_id, it reasons using some of those grains. Before the turn ends, the AI calls `flux_feedback(trace_id, grain_id, useful)` for each returned grain: `true` if actually used in the response, `false` if ignored.

This signal is noisy. LLMs do not always trace their own reasoning honestly. The AI may mark a grain as "useful" because it was retrieved, not because it added value. Grok and ChatGPT reviews identified this as the primary weakness. Flux treats AI usage as a weak signal requiring corroboration.

**Signal 2: Downstream correction (implicit, high weight)**

If the user corrects the AI's response, rephrases the question, or expresses dissatisfaction in the next turn, Flux treats this as negative feedback on the grains used in the prior retrieval.

Detection heuristics for correction:
- User response contains explicit negation ("no", "that's wrong", "not what I meant")
- User repeats a similar query with different phrasing within N turns (default: 3)
- User's follow-up sharply shifts topic or expresses frustration

This signal is weighted higher than raw AI usage because user correction is closer to ground truth.

**Signal 3: Grain usefulness ratio (implicit, trend signal)**

Over a rolling window, each grain tracks its useful-to-retrieved ratio: (times marked useful) / (times retrieved). A grain consistently retrieved but never useful is probably a wrong highway. A grain consistently useful when retrieved earns stronger reinforcement.

This ratio is not a per-retrieval signal; it modulates how much weight change each feedback call applies.

**Combining the signals:**

```python
def apply_feedback(trace_id, grain_id, ai_useful):
    grain = get_grain(grain_id)
    trace = get_trace(trace_id)
    
    # Start with the AI usage signal
    base_signal = 1.0 if ai_useful else -1.0
    
    # Modulate by grain's historical usefulness ratio (trend signal)
    usefulness_ratio = get_usefulness_ratio(grain_id, window_days=7)
    trend_modulator = 0.5 + usefulness_ratio  # range: 0.5 to 1.5
    
    # Modulate by grain provenance (trust signal)
    provenance_modulator = {
        'user_stated': 1.0,
        'external_source': 0.9,
        'ai_stated': 0.5,
        'ai_inferred': 0.3
    }[grain.provenance]
    
    effective_signal = base_signal * trend_modulator * provenance_modulator
    
    # Apply to conduits in the trace
    for conduit in trace.conduits_reaching(grain_id):
        if effective_signal > 0:
            reinforce(conduit, LEARNING_RATE * effective_signal)
        else:
            penalize(conduit, DECAY_FACTOR, scale=abs(effective_signal))
    
    # Record in usefulness history
    record_usefulness_event(grain_id, ai_useful)
    
    # Schedule downstream-correction check (applied after next user turn)
    schedule_correction_check(trace_id, grain_id, ttl_turns=3)
```

**Why this design:** AI usage alone is too noisy to trust. Multiplying by grain provenance means AI-stated grains get half the reinforcement of user-stated ones (hallucinations reinforce slower than facts). The usefulness ratio means a grain that is repeatedly retrieved-but-ignored still gets downweighted even if the AI keeps claiming it's useful. The downstream correction signal catches cases where the AI confidently used a bad grain.

### 7.2 Grain Provenance

Every grain stores where it came from. This determines how much reinforcement it can receive and how it decays.

| Provenance | Source | Trust | Reinforcement Multiplier |
|------------|--------|-------|--------------------------|
| `user_stated` | User directly asserted this fact | High | 1.0 |
| `external_source` | Extracted from a cited external document | High | 0.9 |
| `ai_stated` | AI asserted it in a response | Medium | 0.5 |
| `ai_inferred` | AI reasoned or inferred it, not directly stated | Low | 0.3 |

The extractor LLM tags each emitted grain with its provenance based on the conversation context:
- If a grain reflects something the user said in their message, tag `user_stated`
- If the grain reflects something the AI said but which the AI was directly reporting from an external source the user cited, tag `external_source`
- If the grain reflects something the AI asserted as fact in its response, tag `ai_stated`
- If the grain reflects something the AI reasoned or concluded without direct statement, tag `ai_inferred`

**Why this matters:** Hallucinations are almost always `ai_stated` or `ai_inferred`. Giving these grains reduced reinforcement capacity means even if they get marked "useful" by a hallucinating AI, their conduits grow slowly. Meanwhile, user-stated facts reinforce at full strength. Over time, the graph naturally privileges high-provenance grains.

### 7.3 Quarantine Mechanism

When a grain shows strong signs of being a hallucination, it is quarantined rather than allowed to continue accumulating reinforcement.

**Quarantine triggers:**

A grain enters quarantine when any of the following occur:
- Downstream correction signal fires on it 3+ times
- Usefulness ratio drops below 0.2 over 10+ retrievals
- User explicitly flags the grain's content as wrong (rare, if API exposed)

**Quarantine effects:**

- Grain status changes to `quarantined`
- Grain is not returned from future retrievals (skipped during propagation)
- Existing conduits to the grain continue decaying normally; no new reinforcement possible
- After `QUARANTINE_PERIOD_DAYS` (default: 30), the grain is archived permanently

**Recovery:**

A quarantined grain can be restored if a user-stated grain is stored with contradicting content (the extractor can flag this as a correction). This is rare but provides a path out of false quarantine.

**Why not just delete:** Immediate deletion loses audit trail. Quarantine preserves the grain's history for diagnostic queries while preventing further pollution of the graph.

### 7.4 Feedback Quality Effects

| Feedback Quality | Effect on System |
|------------------|-----------------|
| Accurate + provenance-aware | Clean highways form on high-provenance grains, hallucinations quarantined, fast convergence |
| Noisy AI usage, good provenance | Slower convergence, some spurious short-term reinforcement, but provenance prevents hallucinations from becoming core |
| Missing downstream correction detection | Bad grains that happen to get marked useful live longer before quarantine. Not catastrophic. |
| All signals absent | System drifts toward uniform weights. Graph stays at bootstrap state. Fallback to vector retrieval. |
| Adversarial (intentionally wrong feedback) | Weight ceiling + provenance multipliers + quarantine limit damage. Poisoned paths decay if not reinforced. |

The multi-signal design means Flux degrades gracefully when any single signal is unreliable. Full feedback quality requires all three signals working, but partial signal quality still produces partial learning.

### 7.5 Failure Modes Addressed

| Reviewer Concern | How Addressed |
|------------------|---------------|
| "AI used this" confounds retrieval quality with reasoning quality | Multi-signal combination; AI usage is one of three signals, not the sole signal |
| Confirmation bias loop (LLM marks everything useful) | Usefulness ratio catches persistent over-marking; quarantine triggers when ratio drops |
| Self-reinforcing hallucinations | Provenance multiplier caps AI-stated/inferred reinforcement; quarantine removes clear cases |
| Absent feedback | Trend signal allows partial learning from usage patterns even without per-call feedback |
| Adversarial feedback | Weight ceiling + quarantine + low-provenance multiplier limit maximum damage |

### 7.6 Admin Channel (Manual Override)

Automatic mechanisms (quarantine, decay, penalization) handle most bad grains but take time. When a user identifies a specific hallucination or incorrect grain that must not influence future retrievals, the admin channel provides immediate manual override.

The admin channel is not exposed to the main AI by default. It requires an explicit admin mode or is accessible only through a direct CLI/SDK call, not through the MCP interface used by the AI.

**Endpoints:**

```python
def flux_purge(grain_id, reason):
    """
    Permanently delete a grain and all conduits to/from it.
    This is a hard delete. No recovery path.
    
    Args:
        grain_id: The grain to purge
        reason: Free-text reason for audit log (required, non-empty)
    
    Returns:
        {
          'purged': grain_id,
          'conduits_removed': int,
          'affected_entries': [entry_ids_that_lost_conduits],
          'timestamp': iso_datetime
        }
    
    Side effects:
        - Grain removed from grains table
        - All conduits with from_id=grain_id or to_id=grain_id removed
        - grain_cluster_touch entries for this grain removed
        - co_retrieval_counts entries involving this grain removed
        - Event logged with category='admin', event='grain_purged', reason included
        - Entries that become empty (no conduits) are flagged for lazy cleanup
    """
    ...

def flux_purge_by_content(content_pattern, dry_run=True):
    """
    Search grains by content pattern, return matches. With dry_run=False,
    purges all matches. Always requires explicit confirmation for non-dry runs.
    
    Args:
        content_pattern: Substring or regex to match against grain.content
        dry_run: If True (default), return matches without purging
    
    Returns:
        {
          'matches': [{grain_id, content, provenance, created_at}, ...],
          'purged': bool,
          'purge_count': int (0 if dry_run)
        }
    """
    ...

def flux_export_grain(grain_id):
    """
    Return full grain metadata for inspection before purge decisions.
    
    Returns:
        {
          'grain': {id, content, provenance, confidence, decay_class, status,
                    created_at, context_spread},
          'inbound_conduits': [{from_id, from_feature, weight, use_count, last_used}, ...],
          'outbound_conduits': [{to_id, weight, use_count, last_used}, ...],
          'cluster_touches': [{cluster_id, touch_weight, last_touched}, ...],
          'usefulness_history': [{timestamp, useful_count, total_count}, ...],
          'recent_retrievals': [trace_ids] (last 20)
        }
    """
    ...

def flux_restore(grain_id):
    """
    Restore a grain from 'quarantined' status back to 'active'.
    
    Typical use: a grain was wrongly quarantined and the user wants to
    recover it. Restores status but does not restore decayed conduit weights.
    
    Args:
        grain_id: The quarantined grain to restore
    
    Returns:
        {
          'restored': grain_id,
          'previous_status': 'quarantined',
          'new_status': 'active',
          'conduits_reactivated': int
        }
    
    Fails if grain is already active or if grain has been archived.
    """
    ...
```

**Audit:**

Every admin channel call produces a structured event in the event log with:
- Timestamp
- Operation (`grain_purged`, `grain_restored`, etc.)
- Target grain_id and content snippet (first 100 chars)
- Caller identifier (if available)
- Reason (required for purge, optional for restore)
- Affected counts (conduits removed, entries affected)

Audit events are retained indefinitely regardless of `TRACE_RETENTION_COUNT`. This is the one log stream that does not age out, because admin actions need to be reconstructible.

**Safety:**

- `flux_purge` requires a non-empty reason parameter
- `flux_purge_by_content` defaults to `dry_run=True`, requires explicit `dry_run=False` with content-pattern match count shown before commit
- Batch purges above 10 grains require a second confirmation call with the expected count
- All operations are transactional; partial failures roll back

**Gatekeeper Protocol (defense against prompt injection and accidental AI access):**

The admin channel lives on a separate interface from the read/write channels used by the main AI. Even so, a misconfigured deployment or a successful prompt injection could theoretically expose these endpoints to the AI. The gatekeeper protocol hardens against this.

1. **Separate process/endpoint.** Admin endpoints are not part of the same Python SDK surface exposed to the main AI. They live in a separate module (`flux.admin`) that must be explicitly imported by a user-controlled script. The MCP server wrapper does NOT expose admin endpoints.

2. **Out-of-band confirmation token.** Every admin call requires a `confirmation_token` parameter, passed out-of-band (e.g., from a config file or environment variable set by the user, not by the AI). Without the correct token, the call fails. The token should not be readable by the AI under any circumstance.

    ```python
    flux_purge(grain_id, reason, confirmation_token=FLUX_ADMIN_TOKEN)
    ```

    If `FLUX_ADMIN_TOKEN` is undefined or empty, admin operations are disabled entirely.

3. **User-intent verification for high-impact operations.** For `flux_purge_by_content` with `dry_run=False` and any batch purge above 10 grains, the system requires a second call within 60 seconds that confirms the exact count of grains to be affected. The AI cannot fabricate this count without first seeing the dry-run output, which only the user has.

4. **Rate limiting.** Admin operations are rate-limited to 50 per hour per process. A runaway prompt injection attempting mass purges will trip the rate limit long before causing catastrophic damage.

5. **Audit breadcrumb.** Every admin call writes to an immutable audit log. If the AI ever does gain access, the audit log reveals the intrusion path for forensic analysis.

6. **Purge undo window.** `flux_purge` does not actually delete grains for `PURGE_UNDO_WINDOW_HOURS` (default 24). Purged grains are marked `status='pending_deletion'` and excluded from retrievals but remain recoverable via `flux_restore`. After the window passes, a background job permanently deletes them. This provides a safety net against mistakes.

**When to use admin channel vs natural mechanisms:**

- User identifies a specific wrong fact → `flux_purge` immediately
- Grain was quarantined by mistake → `flux_restore`
- Need to review before deciding → `flux_export_grain`, then decide
- General low quality → let decay and quarantine handle it automatically
- Adversarial pollution detected → `flux_purge_by_content` with dry-run first

---

## 8. OPERATIONAL CHARACTERISTICS

### 8.1 How It Gets Faster Over Time

| Stage | Avg Hops to Resolve | Why |
|-------|---------------------|-----|
| Cold start (0-50 retrievals) | 4-5 hops | All conduits at bootstrap weight, broad propagation |
| Warming (50-500 retrievals) | 3-4 hops | Successful paths strengthening, some shortcuts forming |
| Mature (500+ retrievals) | 1-2 hops | Highways + shortcuts dominate for frequent patterns |

### 8.2 How It Avoids Sprawl

| Control Mechanism | What It Does |
|-------------------|--------------|
| Weight ceiling (0.95) | No single path becomes infinitely dominant |
| Temporal decay | All conduits lose weight over time. Only actively-used ones survive. |
| Shortcut budget | Shortcuts require co-retrieval count >= threshold. Prevents spurious links. |
| Conduit floor (0.05) | Sub-floor conduits are deleted, keeping the graph lean |
| Dormancy + expiry | Unreachable grains eventually archived |
| Propagation budget | max_hops + threshold + attenuation naturally limit how much of the graph is touched per query |

### 8.3 Efficiency Profile

| Operation | Cost | Notes |
|-----------|------|-------|
| Insert grain | O(k) embedding lookups + O(k) conduit creates | One-time. k = number of bootstrap neighbors (default 5). |
| Retrieve | O(E_active) | Only edges above threshold on reachable paths. Mature system: much less than total edges. |
| Reinforce | O(len(trace)) | Linear in trace length. Typically 3-10 conduits. |
| Penalize | O(len(trace)) | Same. |
| Decay pass | O(E_total) | Full scan. Run infrequently (hourly/daily). |
| Orphan check | O(G_total) | Full scan. Run infrequently. |

---

## 9. POSITIONING: FLUX MEMORY vs EXISTING MEMORY PARADIGMS

This section describes how Flux Memory differs from conventional memory architectures. Existing approaches fall into three broad paradigms: vector-based retrieval (embedding + cosine similarity), graph-based retrieval (entity-relationship traversal), and hybrid systems that combine the two with metadata filters.

Flux Memory is a distinctive recombination of ideas from adjacent fields: Hebbian learning (successful co-activations strengthen connections), spreading activation cognitive models (signal propagates outward from query features), adaptive routing in packet networks (paths that deliver quickly get reinforced), and decay-based forgetting (unused paths dissolve over time). Individually these ideas are not new. The combination applied as a memory substrate for AI systems is.

Flux is best described as an adaptive routing graph with vector bootstrap and vector fallback. Calling it a "fourth paradigm" overstates novelty. Calling it "just another RAG variant" understates the operational differences.

The following dimensions clarify what Flux Memory does differently.

### 9.1 Dimensions

**Dimension 1: Storage**

Conventional systems store each memory as a document plus a vector plus optional metadata. Documents are independent. No relationships exist between them at the storage level.

Flux Memory stores content and routing intelligence separately. Grains hold plain text. Conduits hold weighted directed edges between grains. The content is static. The routing map between grains is adaptive. Vectors are used once at insertion to bootstrap initial conduits, then discarded.

**Dimension 2: Retrieval Mechanism**

Conventional retrieval: embed the query, optionally filter by metadata, compute similarity between the query vector and every document vector in scope, return top-k. Cost is proportional to documents in scope. Every query pays the same cost regardless of history.

Flux retrieval: decompose the query into features, inject signal at entry points, let signal propagate through weighted conduits. Grains receiving signal above threshold activate. Cost is proportional to active conduits on reachable paths. In a mature system with highways, a frequently asked query resolves in 1-2 hops. Novel queries spread broadly. Flux's cost drops with use for repeated patterns.

Conventional retrieval is single-hop. Flux is multi-hop: signal can flow through intermediate grains to reach destinations that don't directly match the query words but are reachable through proven associations.

**Dimension 3: Does Retrieval Change the System?**

Conventional systems: No. Every search is read-only. The system before and after a search is identical.

Flux Memory: Yes. Every retrieval is a read-write operation. The Trace is recorded. After feedback, successful paths widen, failed paths narrow, shortcuts may be created between co-retrieved grains, entry point affinities adjust, and grains may be promoted from working to core class. The graph on day 30 looks nothing like day 1.

**Dimension 4: Does Failure Teach Anything?**

Conventional systems: No. There is no feedback channel. Irrelevant results keep appearing because the mathematical similarity between query and document hasn't changed.

Flux Memory: Yes. Three things happen on failure: the conduit that led to the bad grain loses weight, the entry point develops resistance toward that direction, and the system temporarily widens its propagation radius on the next similar query. After repeated failures, the bad path dissolves. Learning from failure is deliberately faster than learning from success.

**Dimension 5: Do Shortcuts Emerge From Use?**

Conventional systems: No. Connections are either absent (vector DBs) or pre-built from schemas or naming conventions (graph DBs).

Flux Memory: Yes. When two grains are co-retrieved and both confirmed useful, the system tracks the co-occurrence. After the count crosses a threshold, a direct bidirectional conduit is created. The association is learned from behavior, not from labels or semantic similarity.

**Dimension 6: Embeddings at Retrieval Time**

Conventional vector systems: Every query requires an embedding model call. Floor latency per query (20-200ms depending on model and hosting). Changing embedding models requires re-embedding every stored document.

Flux Memory: Zero embedding calls during normal retrieval. Query decomposition is string operations. Entry point lookup is a hash table. Signal propagation is graph traversal. The embedding model is only needed at grain insertion (once, then discarded) and as a rare fallback when propagation returns zero results.

**Dimension 7: Forgetting**

Conventional systems: Never forget. Every document stays permanently at full weight. The only removal mechanism is manual deletion. Stale context accumulates and dilutes search quality over time.

Flux Memory: Forgetting is automatic, gradual, and reversible. Three layers operate: conduit decay, dormancy, and archival. At any point before archival, a single successful retrieval can reverse the process. The system forgets selectively: grains that prove useful across multiple contexts get promoted to core class and decay 4x slower.

**Dimension 8: Cold Start**

Conventional systems: No cold start problem. Works at full strength from memory one because similarity is a mathematical operation that needs no history.

Flux Memory: Cold start is real. On day one, all conduits sit at bootstrap weight, signal spreads broadly and noisily. The vector fallback mitigates this. For frequent query patterns, cold start effectively ends within 15-30 retrievals. The crossover point where Flux Memory surpasses conventional systems occurs around 50-200 retrievals.

**Dimension 9: Organisation**

Conventional systems either have no organisation (flat vector store) or pre-defined organisation (schema, taxonomy, user-defined hierarchy). In both cases, structure is set at design time and stays static unless manually rebuilt.

Flux Memory has no predefined structure. Organisation emerges from use. Grains that are frequently co-retrieved cluster together naturally through conduit reinforcement. The topology is continuously discovered, not imposed.

### 9.2 Summary Matrix

| Dimension | Conventional Systems | Flux Memory |
|-----------|----------------------|-------------|
| Storage | Documents + vectors + metadata | Grains (text) + Conduits (weighted edges) |
| Retrieval | Similarity scan, O(n) per query | Signal propagation, O(active edges) per query |
| Retrieval changes system? | Never | Every time |
| Failure teaches? | No feedback channel | Path narrowing + exploration boost |
| Shortcuts emerge? | No | Yes (co-retrieval creates edges) |
| Embedding at retrieval? | Usually required | Never (except rare fallback) |
| Forgetting | Manual deletion only | Automatic, gradual, reversible, selective |
| Cold start | None (instant full capability) | Real (mitigated by vector fallback) |
| Organisation | Pre-defined or none | Emergent from use |

### 9.3 Where Flux Memory Wins

- Repeated query patterns resolve in 1-2 hops vs full index scan
- Evolving user interests: decay deprioritizes stale paths automatically
- No reindexing needed, ever
- No runtime dependency on embedding model
- Failure actively improves future retrieval
- Knowledge that proves transferable earns its own permanence
- No manual cleanup of stale memories
- Sub-10ms retrieval in mature systems

### 9.4 Where Conventional Systems Win

- Day-one quality equals day-365 quality (no cold start)
- No feedback loop required (works even without any user signal)
- Simpler mental model
- Proven benchmark performance on standard evaluation datasets
- Zero learning curve for the user

### 9.5 Where Neither Paradigm Solves the Problem

- **Lateral discovery**: surprising connections between memories that are semantically distant but contextually relevant. Neither cosine similarity nor conduit propagation finds "you hated event-driven architecture" when you're designing a notification system. This is an open problem held for later exploration.

### 9.6 Feature Matrix

| Feature | Vector DB | Knowledge Graph | Hybrid Systems | Spreading Activation | **Flux Memory** |
|---------|-----------|-----------------|----------------|----------------------|-----------------|
| Retrieval modifies structure | No | No | No | Partial (Hebbian) | **Yes (core mechanic)** |
| Failed retrieval penalizes | No | No | No | No | **Yes** |
| Shortcuts from co-retrieval | No | No | No | No | **Yes** |
| Embedding at retrieval | Always | No | Usually | Yes (pre-filter) | **No (after bootstrap)** |
| Topology evolves through use | No | No | No | Weights only | **Weights + topology** |
| Automatic forgetting | No | No | Some | No | **Yes (class-based decay)** |
| Cold start handling | Good | N/A | Good | Good | **Graceful degradation to vector fallback** |
| Use-driven promotion | No | No | No | No | **Yes (context_spread)** |

---

## 10. RISKS AND MITIGATIONS

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Popularity bias**: Heavily-used grains dominate, rare-but-relevant ones unreachable | High | Weight ceiling (0.95). Exploration boost after failure. Periodic random exploration. |
| **Feedback quality**: Noisy implicit feedback creates spurious highways | Medium | Weight explicit feedback 3x over implicit. Shortcut threshold prevents single-incident shortcuts. |
| **Interest drift**: User pivots, old highways mislead | Medium | Temporal decay handles this over days/weeks. Optional "context shift" signal for immediate reset. |
| **Cold start latency**: First queries are slow | Low | Bootstrap with embedding similarity gives reasonable results from day 1. |
| **Graph corruption**: Bug causes mass weight inflation/deflation | High | Weight ceiling + floor as hard limits. Snapshot/backup before bulk operations. Integrity checks in decay pass. |
| **Scale**: Very large graphs (>1M grains) may have expensive propagation | Medium | Bounded propagation (max_hops, threshold). Index on conduit weights. Partitioning by domain. |
| **Circular reinforcement**: Same wrong answer keeps getting reinforced | Medium | Decay erodes even highways over time. Exploration boost introduces alternatives. Explicit feedback overrides implicit. |

---

## 11. BUILD SPECIFICATION

Flux Memory is built as a single deployable system. No phased feature holdbacks. No MVP compromises. All core mechanisms are present at deployment. Validation happens continuously in production through the Health Monitor (Section 12).

### 11.1 Components to Build

**Graph engine (core)**
- Grain, conduit, and entry point data structures
- Signal propagation algorithm
- Reinforcement, penalization, shortcut creation, promotion logic
- Temporal decay daemon (runs on schedule, e.g., hourly)
- Orphan detection and dormancy expiry

**Storage layer**
- SQLite with WAL mode for local single-user deployment
- Schema as defined in Section 6.4
- Indexes as defined in Section 6.4
- Backup/snapshot capability before bulk operations

**Bootstrap engine**
- Local embedding model (sentence-transformers class)
- One-time embedding at grain insertion
- Nearest-neighbor bootstrap conduit creation
- Vector fallback handler for failed propagations

**Read channel**
- Local feature extractor LLM (7B-8B instruction-tuned)
- Query decomposition into feature list
- `flux_retrieve(query)` endpoint returning grains + trace_id
- `flux_feedback(trace_id, grain_id, useful)` endpoint

**Write channel**
- `flux_store(grain_content)` endpoint
- Grain insertion with bootstrap conduit creation

**Admin channel** (Section 7.6)
- `flux_purge(grain_id, reason)` — hard delete a specific grain and all conduits to/from it
- `flux_purge_by_content(content_pattern)` — search and purge by content match (requires user confirmation)
- `flux_export_grain(grain_id)` — return full grain history for manual inspection before purge
- `flux_restore(grain_id)` — restore a quarantined grain to active status

**Extractor**
- Local extraction LLM that reads (user_query + AI_response) transcripts
- Emits atomic grains via `flux_store`
- Triggered at end of each conversation turn

**Pre-warming subsystem** (one-time bootstrap)
- Ingests existing memory sources before live deployment
- Chunks and extracts grains from historical content
- Optional synthetic retrieval pass to pre-shape highways
- See Section 11.10

**Query-time context expansion** (lateral discovery)
- Bounded second-pass scan after primary retrieval
- Triggers on low-confidence results
- Surfaces grains from shared clusters not reached by primary propagation
- See Section 11.11

**Context shift detection**
- Monitors retrieval success trajectory
- Detects pivots in user interests
- Accelerates recovery via elevated exploration and decay on stale paths
- See Section 11.12

**Health Monitor**
- Continuous metrics computation (Section 12)
- Event log of all operations
- `flux_health()` endpoint returning current state
- Configurable thresholds that trigger warnings when crossed

**Logging subsystem**
- Structured event logging (see Section 11.5)
- Log rotation, configurable retention
- Separate streams for operational events, health metrics, diagnostic traces

**Visualization subsystem**
- Graph export to standard formats for external rendering
- Programmatic access to graph state for dashboards
- See Section 11.6

**Configuration**
- External YAML config file for all parameters (Section 5)
- Hot-reload support where feasible
- Clear documentation of mid-run change semantics (Section 13.13)

**Interface**
- Python SDK for direct import
- MCP server wrapper for agent integration
- Separation of read and write channels (Section 13.5)

### 11.2 Technology Choices

| Component | Choice |
|-----------|--------|
| Language | Python |
| Storage | SQLite (WAL mode) |
| Embedding model | sentence-transformers (local, 384-dim class) |
| Feature extractor LLM | Local 7B-8B instruction-tuned model |
| Grain extractor LLM | Local 7B-8B instruction-tuned model (can share infrastructure with feature extractor) |
| Config format | YAML |
| Inter-process | Python SDK (direct) + MCP server (external agents) |
| Logging | structured JSON events to SQLite and rotating log file |
| Testing | pytest for unit/integration tests |
| Visualization export | GraphML + JSON for external tools |

Local LLM inference runs on whatever inference stack is available on the deployment machine. The feature extractor and grain extractor can be the same model instance; their prompts differ.

### 11.3 Dependencies

**Runtime dependencies:**

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Core language |
| SQLite | 3.35+ | Storage backend (needs UPSERT support) |
| sentence-transformers | 2.x | Local embedding model |
| numpy | latest | Vector operations for fallback |
| pyyaml | latest | Config file parsing |
| pydantic | 2.x | Schema validation |
| networkx | 3.x | Graph construction and Louvain community detection (via `networkx.community.louvain_communities`) |
| local LLM runtime | any | Inference for feature/grain extractors (implementation-dependent) |

**MCP server dependency:**

| Dependency | Purpose |
|------------|---------|
| mcp SDK | Model Context Protocol server implementation |

**Test-time dependencies:**

| Dependency | Purpose |
|------------|---------|
| pytest | Test framework |
| pytest-asyncio | Async test support |
| hypothesis | Property-based testing for graph invariants |

**Hardware requirements (local deployment):**

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 16 GB (for 7B LLM + embedding + graph in memory) | 32 GB |
| Disk | 10 GB for LLM weights, 1 GB for graph + logs | 50 GB headroom |
| CPU | 4 cores | 8+ cores for parallel decay passes |
| GPU | Not required | Optional for faster LLM inference |

**OS support:** macOS, Linux. Windows possible but untested.

### 11.4 Known Limitations

Explicit limitations that users and builders must understand:

**Retrieval limitations:**
- Cold start: first ~50 retrievals produce lower-quality results than a pure vector DB would. The vector fallback mitigates but does not eliminate this.
- Unreachable grains: grains whose bootstrap conduits all decayed and were never retrieved can only be reached via vector fallback. If the fallback is disabled or fails, they become functionally lost until a query happens to re-establish a connection.
- Novel queries: queries using vocabulary never seen before will trigger new entry points with zero reinforcement history. Results will be broad and noisy until feedback shapes them.

**Feedback limitations:**
- System requires accurate feedback from the calling AI. Noisy or adversarial feedback degrades learning quality. No mechanism distinguishes honest mistakes from deliberate misreporting.
- If the AI consistently fails to call `flux_feedback`, the graph cannot learn. The feedback compliance rate metric surfaces this, but cannot fix it.

**Scale limitations:**
- Tested design target: up to 100,000 grains per graph.
- Beyond that, SQLite performance on single-file writes may become a bottleneck. Partitioning strategies are not specified in this design.
- Propagation complexity is bounded by max_hops and attenuation but can still become expensive on heavily-interconnected graphs with low attenuation settings.

**Deployment limitations:**
- Single-user only. Multi-user isolation is in the Annexure (A5).
- No distributed deployment. Graph lives on a single machine.
- No concurrent write support; writes are serialized through SQLite.

**LLM dependency:**
- Feature extractor LLM quality determines upstream retrieval quality. If the LLM produces bad features, even a perfect graph returns wrong results.
- Grain extractor LLM quality determines what enters the graph. Poor extraction produces noisy grains that take time to decay.
- Neither LLM can be replaced by deterministic tokenization without significant quality loss.

**Lateral discovery:**
- Even with query-time context expansion (Section 11.11), Flux may miss deep cross-domain connections where activated grains share no obvious semantic link with the unreachable target. The expansion layer handles moderate cases, not deep surprises.

### 11.5 Logging Specification

All operations emit structured events. Logs serve three purposes: debugging, audit trail, and Health Monitor input.

**Event categories:**

| Category | Events |
|----------|--------|
| Retrieval | query_received, features_extracted, signal_propagated, grains_returned |
| Feedback | feedback_received, conduit_reinforced, conduit_penalized, shortcut_created, promotion_triggered |
| Write | grain_stored, bootstrap_conduits_created, entry_point_created |
| Decay | decay_pass_started, conduit_decayed, conduit_deleted, grain_dormant, grain_archived |
| Cluster | cluster_recomputed, cluster_merged, cluster_split |
| Health | metric_computed, warning_raised, warning_cleared |
| System | startup, shutdown, config_reloaded, backup_created |

**Event structure:**

```json
{
  "timestamp": "2026-04-19T14:32:05.142Z",
  "category": "retrieval",
  "event": "grains_returned",
  "trace_id": "t_abc123",
  "data": {
    "query": "...",
    "features": ["..."],
    "grains_count": 5,
    "hop_count": 2,
    "fallback_triggered": false
  }
}
```

**Storage:**
- Events write to both SQLite (for queryable history and Health Monitor) and a rotating JSON log file (for debugging and external analysis).
- JSON log rotates at 100 MB per file, retains last 10 files.
- SQLite event table is pruned according to `TRACE_RETENTION_COUNT` and `TRACE_RETENTION_DAYS`.

**Log levels:**
- DEBUG: detailed signal propagation steps, weight changes per conduit
- INFO: standard events (retrieval, feedback, grain stored)
- WARNING: Health Monitor warnings, unexpected states
- ERROR: failures in any component

Default log level: INFO. Configurable per category.

### 11.6 Visualization Specification

Flux Memory is natively a graph. Visualization is a first-class concern. The system provides programmatic access and export formats, rendering is handled by external tools.

**Built-in export formats:**

| Format | Purpose |
|--------|---------|
| GraphML | For rendering in external graph tools |
| JSON (node-link format) | For web-based dashboards (D3.js, Cytoscape.js, vis.js) |
| DOT (Graphviz) | For static image generation |

**What a visualization shows:**

Nodes (grains):
- Size: proportional to total inbound signal received over a window
- Color: working class (grey) vs core class (gold)
- Opacity: active (full) / dormant (faded)
- Shape: round for grains, diamond for entry points

Edges (conduits):
- Thickness: proportional to weight
- Color: recently used (green) / aging (yellow) / near floor (red)
- Style: solid for earned conduits, dashed for bootstrap, dotted for emergent shortcuts

Clusters:
- Shown as soft groupings via force-directed layout
- Optional cluster boundaries drawn from the `clusters` table

**Dashboard component (deliverable):**

A minimal web dashboard is included in the build, showing:
- Current graph state (filterable view)
- Health Monitor signals with historical trends
- Active warnings
- Recent traces
- Parameter editor with config hot-reload

Dashboard is a thin web UI layer. It consumes JSON from the visualization export and the `flux_health()` endpoint. It does not require a separate backend.

**Programmatic API:**

```python
flux.graph.export(format="graphml")              # returns serialized graph
flux.graph.subgraph(entry_points=["AI"])         # focused export
flux.graph.timeline(grain_id, window="30d")      # weight evolution over time
flux.graph.cluster_view()                        # current cluster assignments
```

### 11.7 Development Plan

Flux is a single build, not phased. The development order below is sequencing for parallel work, not feature gating. Validation happens continuously in production through the Health Monitor (Section 12), not as a pre-build gate.

**Track 1: Core graph engine** (foundational, blocks other work)
1. Data structures and SQLite schema
2. Signal propagation algorithm
3. Reinforcement and penalization logic
4. Shortcut creation
5. Decay daemon
6. Orphan detection and dormancy
7. Cluster computation
8. Promotion logic

**Track 2: LLM integration** (parallel to Track 1 after data structures land)
1. Feature extractor LLM integration
2. Grain extractor LLM integration
3. Prompt engineering for both
4. Embedding model integration
5. Vector fallback implementation

**Track 3: Observability** (starts after Track 1 step 3)
1. Event logging infrastructure
2. Health Monitor signals
3. Warning system
4. flux_health() endpoint
5. Diagnostic query library

**Track 4: Interface** (starts after Track 1 step 4)
1. Python SDK surface (flux_store, flux_retrieve, flux_feedback, flux_health)
2. MCP server wrapper
3. Visualization export
4. Dashboard UI

**Track 5: Configuration and ops** (cross-cutting, develops alongside other tracks)
1. YAML config loader
2. Parameter hot-reload
3. Backup and restore
4. Graceful shutdown / resume

**Track 6: Advanced features** (requires Tracks 1 and 2 complete)
1. Pre-warming subsystem: source readers, chunking, synthetic retrieval pass
2. Query-time context expansion: cluster-based lateral candidate surfacing
3. Context shift detection: trajectory monitoring + elevated exploration

**Integration checkpoint:**
End-to-end test: a real conversation flows through query decomposition, retrieval (including expansion if triggered), feedback, and extraction, producing measurable graph changes and a clean health report. Pre-warming tested independently by seeding a graph from a sample corpus and verifying the warmed state. Context shift detection tested by simulating a pivot pattern.

### 11.8 Testing Strategy

Three layers of testing, all continuous (no phase gating).

**Unit tests:**
- Every algorithm (propagation, reinforcement, penalization, promotion, decay, clustering) tested in isolation
- Edge cases: empty graph, single grain, disconnected components, weights at ceiling/floor
- Deterministic with seeded inputs

**Integration tests:**
- End-to-end flows: insert → retrieve → feedback → verify graph state
- Cross-component: feedback triggers promotion triggers conduit reclassification
- Multi-step scenarios: simulate 100 retrievals and verify expected health state
- Fallback path: force propagation failure, verify vector fallback fires correctly

**Behavior tests (property-based, using hypothesis):**
- Invariants: weights always within [WEIGHT_FLOOR, WEIGHT_CEILING]
- Invariants: context_spread monotonically non-decreasing until promotion
- Invariants: decay always reduces weight, never increases
- Invariants: shortcut threshold never produces shortcuts with co-retrieval count below threshold
- Randomized query sequences produce stable graph states

**Load tests:**
- 10,000 grain insertions, measure latency
- 10,000 retrievals on mature graph, measure latency and memory
- 100,000 grains scale test, verify no degradation below target
- Concurrent read scenarios (writes are serialized by SQLite)

**Health-driven validation (continuous in production):**
- All health signals from Section 12.1 apply as real-time validation
- System does not need "test mode" because health monitoring IS the test
- Post-deployment, the Health Monitor replaces the test suite as the ongoing correctness check

**Test data:**
- Synthetic grain sets for unit tests (hand-crafted)
- Realistic conversation transcripts for integration tests
- Adversarial inputs: malformed queries, contradictory feedback, extreme parameter values

### 11.9 Deployment Model

Single-user, local-first deployment. Flux runs on the same machine as the main AI. No cloud dependency. No network calls in the hot path. The vector fallback uses the local embedding model.

Multi-user deployments are out of scope for the initial build and deferred to the Annexure (A5).

### 11.10 Pre-Warming (Initial Bootstrap from Existing Memory Sources)

Flux can be seeded with existing memory sources before going live, eliminating cold start pain for users who already have history to draw from. This is part of the initial build, not an optional add-on.

**Sources supported:**
- Conversation exports (JSON, markdown, or plain text)
- Obsidian vault or other note corpus
- Any text corpus representing past context

**Pre-warming pipeline:**

1. Source ingestion: read files from configured paths
2. Chunking: split each source into conversation-sized or topic-sized units (configurable)
3. Extraction: the grain extractor LLM processes each unit, emitting atomic grains
4. Storage: each grain enters via the normal write channel (`flux_store`), triggering bootstrap conduit creation
5. Synthetic retrieval pass (optional): for each extracted grain, generate a few synthetic queries about it, run retrieval, mark the grain as useful. This pre-shapes highways based on what was historically relevant.
6. Report: summary of grains extracted, conduits created, entry points formed

**Configuration:**

```yaml
prewarming:
  enabled: true
  sources:
    - path: ~/Documents/ObsidianVault
      type: obsidian
    - path: ~/exports/conversations
      type: conversation_json
  chunk_size: 2000_chars  # or by-paragraph, by-section
  synthetic_retrieval_pass: true
  synthetic_queries_per_grain: 3
```

**Invocation:** `flux prewarm` command. Runs once, before the system goes live. Safe to re-run (idempotent on grain content deduplication).

**Scope limits:** Pre-warming handles text-based sources. Binary formats (PDFs with images, audio, video) require external preprocessing.

### 11.11 Query-Time Context Expansion (Lateral Discovery Mechanism)

After normal signal propagation returns a set of activated grains, a bounded second-pass scan looks for contextually-related grains that were not reached through the primary retrieval. This is the lightweight lateral discovery mechanism. It is part of the build.

**How it works:**

1. Primary retrieval runs normally, returning top-k activated grains
2. If retrieval confidence is above a threshold AND the top-k contains at least 2 activated grains, context expansion is skipped (the primary result is strong enough)
3. If retrieval confidence is below threshold OR returns fewer than 2 grains, expansion fires:
   - Take the activated grains
   - Identify their shared cluster memberships (from the clustering data Flux already maintains)
   - For each shared cluster, pull up to N grains from that cluster that were NOT activated but have high conduit weight to members that WERE activated
   - Return these as "lateral candidates"
4. Lateral candidates are marked distinctly in the result set, so the main AI knows they came from expansion
5. Feedback on lateral candidates follows normal rules, with one adjustment: if a lateral candidate was marked useful, a new direct conduit forms between it and the grain it was reached through (co-retrieval shortcut mechanism)

**Cost:**

Bounded. Expansion touches only cluster members of already-activated grains, not the whole graph. Additional latency is typically one additional query to the graph (no LLM calls, no embeddings).

**Configuration:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `EXPANSION_CONFIDENCE_THRESHOLD` | 0.4 | Signal strength below which expansion fires |
| `EXPANSION_CANDIDATES_PER_CLUSTER` | 2 | Max lateral candidates surfaced per shared cluster |
| `EXPANSION_MAX_CANDIDATES` | 3 | Hard cap on lateral candidates total |
| `EXPANSION_ENABLED` | true | Toggle the entire mechanism |

**Limitation:** This handles moderate-distance lateral discovery (grains in clusters that share entry points with activated clusters). It does not handle deep cross-domain surprises where no cluster overlap exists. That remains an open problem.

### 11.12 Context Shift Detection

When user interests pivot significantly, old highways can persist and mislead retrieval. Natural decay handles this in 1-4 weeks. Context shift detection accelerates recovery.

**Mechanism:**

The Health Monitor already tracks retrieval success rate. Context shift detection adds a related signal: **retrieval success trajectory over a short rolling window**.

If the retrieval success rate drops by more than `CONTEXT_SHIFT_DROP_THRESHOLD` over `CONTEXT_SHIFT_WINDOW` retrievals, while feedback compliance remains healthy (ruling out the AI just not calling feedback), a context shift event is detected.

**On detection:**

1. A warning is logged (severity: INFO)
2. For the next N retrievals (default: 50), exploration boost is elevated beyond the normal post-failure value
3. Decay is applied more aggressively to conduits that have received recent "not useful" feedback, accelerating their dissolution
4. No existing weights are zeroed out. No drastic resets. The system just leans harder into the natural recovery mechanisms.

**Configuration:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `CONTEXT_SHIFT_WINDOW` | 30 | Number of recent retrievals to measure success trajectory |
| `CONTEXT_SHIFT_DROP_THRESHOLD` | 0.25 | Drop in success rate (e.g., 0.80 → 0.55) that triggers detection |
| `CONTEXT_SHIFT_RECOVERY_RETRIEVALS` | 50 | Number of retrievals during which elevated exploration applies |
| `CONTEXT_SHIFT_ENABLED` | true | Toggle the mechanism |

**Why automatic:** Natural decay alone takes weeks for a user who pivots between projects every few months. Detection surfaces the pivot in days and speeds recovery without requiring the AI or user to explicitly signal it.

**What this is NOT:** This is not a full reset. The system does not forget everything. Old knowledge that proves still useful (promoted core grains) remains intact. Only project-level working conduits that are no longer being reinforced get pushed toward dissolution faster.


## 12. HEALTH MONITOR

The Health Monitor is a first-class component of Flux Memory, not an optional add-on. It runs continuously, maintaining performance logs and verifying that every self-organizing mechanism is working correctly.

Because Flux Memory has multiple mechanisms operating simultaneously (reinforcement, penalization, shortcut creation, decay, promotion), any one of them could silently break. The graph might look busy while actually being dysfunctional. The Health Monitor catches this by continuously measuring the signals that prove each mechanism is alive.

### 12.1 Health Signals

Each signal has a **measurement window**, a **healthy range**, and a **warning threshold**. When a signal stays outside its healthy range for longer than its warning dwell time, the Health Monitor records a warning.

**Measurement windows:**
- Short window: last 100 retrievals
- Medium window: last 24 hours
- Long window: last 7 days

**Warning dwell time** (how long a signal must be unhealthy before a warning fires): 3 consecutive measurements unless otherwise specified.

| Signal | Measures | Window | Healthy Range | Warning Threshold | Diagnostic Suggestion |
|---|---|---|---|---|---|
| **Highway count** | Conduits with weight ≥ 0.80 | Long | ≥ 5 after first 100 retrievals; growing trend expected | < 5 for 7 days after warmup | Reinforcement may be broken. Check LEARNING_RATE. |
| **Highway growth rate** | New highways per week | Long | 1-20 per week in warming phase; stabilizing in mature phase | 0 for 14 days after warmup | Queries may not be repeating enough to form highways. |
| **Shortcut creation rate** | New shortcuts per 100 retrievals | Short | 0.5 - 5 | 0 for 500 consecutive retrievals after warmup; or > 10 | Zero: co-retrieval not triggering. High: SHORTCUT_THRESHOLD may be too low, creating noise. |
| **Conduit dissolution rate** | Conduits deleted per decay pass | Medium | > 0 in mature graph | 0 for 7 consecutive decay passes after warmup | Decay may not be firing. Check half-life parameters and decay daemon. |
| **Avg weight drop on failure** | Mean weight reduction on "not useful" feedback | Short | 0.10 - 0.20 per incident | < 0.05 | Penalization not strong enough. Check DECAY_FACTOR. |
| **Promotion events** | Grains promoted to core per week | Long | > 0 in weeks where cross-context queries occur | 0 for 4 consecutive weeks when cross-context queries are present | Cluster logic or PROMOTION_THRESHOLD may be wrong. |
| **Avg hops per retrieval** | Mean conduit traversals per retrieval | Short | < 3 in mature graph; < 5 in warming; < 7 in cold start | > 5 in mature graph (after 500 retrievals) | Highways not forming for common queries. Review reinforcement. |
| **Orphan rate** | % of active grains with zero inbound conduits | Medium | < 10% | > 15% for 3 days | Decay may be too aggressive, or bootstrap may be failing. |
| **Avg conduit weight** | Mean weight across all conduits | Medium | 0.25 - 0.60 | < 0.15 (system dying) or > 0.75 (no discrimination) | Check learning/decay balance. |
| **Retrieval success rate** | % of retrievals where AI marked at least one grain useful | Short | > 70% after warmup | < 50% for 3 days | Feature extractor producing bad features, or graph malformed. |
| **Fallback trigger rate** | % of retrievals hitting vector fallback | Short | < 20% in warming; < 5% in mature | > 20% in mature graph | Graph not learning effectively. Check reinforcement and promotion. |
| **Feedback compliance rate** | % of retrievals followed by feedback calls | Short | > 90% | < 80% for 1 day | Main AI is not calling flux_feedback reliably. Prompt engineering issue. |
| **Core grain count** | Total grains with decay_class = core | Long | Growing in active use | Shrinking for 30+ days | Promotion not firing, or aggressive decay on core class. |
| **Dormant grain rate** | % of active grains in dormant status | Long | 5% - 25% (healthy churn) | > 40% | Too many grains being orphaned. Check bootstrap and reinforcement. |

### 12.2 Warning Levels

Warnings have three severity levels based on which mechanisms are affected:

**INFO**: A signal is outside healthy range but mechanisms are still functional. Example: shortcut rate slightly low. Logged, surfaced in `flux_health()`, no action required.

**WARNING**: A mechanism appears broken or malfunctioning. Example: zero highways after warmup. Logged, surfaced prominently, diagnostic suggestion included. System continues operating.

**CRITICAL**: Multiple mechanisms broken, or data integrity concern. Example: avg conduit weight collapsing below 0.15, orphan rate above 40%, and retrieval success below 50% simultaneously. Logged with full diagnostic dump. System continues operating but surfaces a strong recommendation to pause feedback ingestion until investigated.

### 12.3 Warmup Period

Many health signals require a warmup period before their thresholds apply. The warmup is defined as:

- 100 retrievals for short-window signals
- 7 days of use for medium-window signals
- 14 days of use for long-window signals

During warmup, signals are tracked but warnings are suppressed. This prevents false alarms on a young graph.

### 12.4 How It Works

Every operation in Flux emits a structured event to the event log. Events include: retrieval calls, feedback calls, grain insertions, conduit weight changes, promotion events, decay pass results, and dormancy transitions.

A background process reads these events and computes rolling metrics. Metrics are stored in a `health_log` table alongside the graph data. Each metric has a healthy range defined in config. When a metric drifts outside its healthy range, a warning is logged (non-blocking) and surfaced through the health endpoint.

### 12.5 Health Endpoint

```
flux_health() → {
  status: "healthy" | "warning" | "critical",
  signals: {
    highways: {count: 14, trend: "growing", healthy: true},
    highway_growth_rate: {per_week: 3, healthy: true},
    shortcuts: {rate_per_100: 2.3, healthy: true},
    dissolution: {conduits_deleted_last_24h: 8, healthy: true},
    penalization: {avg_weight_drop: 0.14, healthy: true},
    promotion: {grains_promoted_this_week: 3, healthy: true},
    avg_hops: {value: 2.1, healthy: true},
    orphan_rate: {pct: 6.8, healthy: true},
    avg_weight: {value: 0.47, healthy: true},
    retrieval_success: {pct: 78, healthy: true},
    fallback_rate: {pct: 4.2, trend: "decreasing", healthy: true},
    feedback_compliance: {pct: 94, healthy: true},
    core_grain_count: {value: 42, trend: "growing", healthy: true},
    dormant_rate: {pct: 12, healthy: true}
  },
  active_warnings: [
    // Example warning entry:
    // {
    //   signal: "fallback_rate",
    //   severity: "WARNING",
    //   current_value: 23,
    //   healthy_range: "< 20% in warming, < 5% in mature",
    //   duration_unhealthy: "4 days",
    //   suggestion: "Graph not learning effectively. Check reinforcement logs."
    // }
  ]
}
```

Callable anytime by the AI, by a dashboard, or by monitoring scripts. The response shows current health state plus all active warnings.

### 12.6 Warning Record Format

Each warning is recorded with:
- Signal name
- Current value vs healthy range
- Severity level (INFO / WARNING / CRITICAL)
- Duration outside healthy range
- Measurement window that produced it
- Suggested diagnostic action

Warnings do not halt operation. Flux continues running. They make drift visible immediately instead of months later.

### 12.7 Diagnostic Queries

```sql
-- Top 10 highways (strongest conduits)
SELECT from_id, to_id, weight, use_count
FROM conduits ORDER BY weight DESC LIMIT 10;

-- Orphaned grains (no inbound conduits)
SELECT g.id, g.content FROM grains g
WHERE g.status = 'active'
AND NOT EXISTS (SELECT 1 FROM conduits c WHERE c.to_id = g.id);

-- Most connected grains (hubs)
SELECT to_id, COUNT(*) as inbound, AVG(weight) as avg_weight
FROM conduits GROUP BY to_id ORDER BY inbound DESC LIMIT 10;

-- Decay candidates (low weight, old)
SELECT id, from_id, to_id, weight, last_used
FROM conduits WHERE weight < 0.1
ORDER BY last_used ASC LIMIT 20;

-- Entry point effectiveness
SELECT e.feature, COUNT(c.id) as conduits, AVG(c.weight) as avg_weight
FROM entries e JOIN conduits c ON c.from_id = e.id
GROUP BY e.feature ORDER BY avg_weight DESC;

-- Promoted grains (core class, earned through cross-context use)
SELECT g.id, g.content, g.context_spread, g.decay_class
FROM grains g WHERE g.decay_class = 'core'
ORDER BY g.context_spread DESC;

-- Promotion candidates (approaching threshold)
SELECT g.id, g.content, g.context_spread
FROM grains g WHERE g.decay_class = 'working' AND g.context_spread >= 2
ORDER BY g.context_spread DESC;

-- Fallback dependency (grains only reachable via fallback)
SELECT g.id, g.content FROM grains g
WHERE g.status = 'active'
AND NOT EXISTS (
    SELECT 1 FROM conduits c WHERE c.to_id = g.id AND c.weight >= 0.15
);
```

---

## 13. DECIDED DESIGN

These are the decisions that define the core Flux Memory system. Anything not here is deferred to the Annexure.

### 13.1 Grain Model

- All grains start as `working` class
- Grains are promoted to `core` class when `context_spread >= 3` (retrieved successfully from 3+ distinct entry point clusters)
- Grain content is atomic English text, one fact per grain
- No LLM classification at write time
- Raw conversations are NOT stored as grains. An extraction step produces grains from conversations.
- Every grain carries a provenance tag (`user_stated`, `external_source`, `ai_stated`, `ai_inferred`) assigned at extraction time
- Provenance modulates reinforcement capacity: high-trust grains reinforce at full rate, low-trust grains at reduced rate (0.3x for `ai_inferred`)
- Grains may enter `quarantined` status if feedback signals indicate they are likely hallucinated or consistently unhelpful

### 13.2 Entry Point Clustering (Soft Membership)

An entry point cluster is a group of entry points that frequently co-activate in the same queries. Clusters are formed by tracking pairwise co-occurrence between entry points over a rolling window.

**Soft membership model:**

Each entry point has a **weighted membership** in multiple clusters. An entry point like `E:security` might have:
- 0.7 membership in the VMO2 cluster
- 0.4 membership in the security-architecture cluster
- 0.2 membership in the DPDPA cluster

Membership weights sum to 1.0 across clusters for each entry point (normalized).

**Why soft membership:** "Security" genuinely belongs to multiple contexts. Forcing it into one cluster causes membership flickering as user interests shift, which breaks promotion. Soft membership lets entry points participate in all contexts they actually appear in, proportional to their co-occurrence patterns.

**Clustering algorithm: Louvain community detection with soft membership derivation.**

Louvain is chosen over alternatives (K-Means, spectral clustering, label propagation) because:
- It produces variable-count clusters (not fixed K), matching real user contexts
- It is computationally cheap: O(n log n) for n entry points, in practice runs in seconds for graphs up to 10k entry points
- It is stable: small changes in input produce small changes in output
- It is widely implemented; we use `networkx.community.louvain_communities` as the canonical implementation

**Step-by-step algorithm:**

```python
def recompute_clusters():
    """
    Recomputes entry point cluster memberships.
    Runs during the cleanup pass (Section 4.5), not on every retrieval.
    """
    # Step 1: Build co-occurrence graph
    G = build_cooccurrence_graph()
    # G is an undirected weighted graph:
    #   nodes = active entry points (with at least N retrievals in window)
    #   edge weights = pairwise co-occurrence count over CLUSTER_WINDOW_DAYS,
    #                  normalized by individual entry point frequencies:
    #                  edge(a,b) = cooccur(a,b) / sqrt(freq(a) * freq(b))
    # Only edges above ENTRY_COOCCURRENCE_THRESHOLD are kept.
    
    # Step 2: Run Louvain on the graph to get hard partition
    partition = networkx.community.louvain_communities(
        G,
        weight='weight',
        resolution=LOUVAIN_RESOLUTION,  # default: 1.0
        seed=LOUVAIN_SEED               # default: 42 for reproducibility
    )
    # partition is a list of sets, each set containing entry_ids in one community
    
    # Step 3: Merge clusters smaller than CLUSTER_MIN_SIZE
    partition = merge_small_clusters(partition, CLUSTER_MIN_SIZE)
    
    # Step 4: Derive soft membership weights
    # For each entry point, membership in a cluster = proportion of its
    # co-occurrence edges that go to that cluster.
    memberships = {}  # entry_id -> {cluster_id -> weight}
    for entry_id in G.nodes():
        total_edge_weight = sum(G[entry_id][n]['weight'] for n in G[entry_id])
        if total_edge_weight == 0:
            # Singleton entry point, assign 1.0 to its own cluster
            memberships[entry_id] = {entry_id: 1.0}
            continue
        
        cluster_weights = {}
        for cluster_id, cluster_set in enumerate(partition):
            weight_to_cluster = sum(
                G[entry_id][n]['weight']
                for n in G[entry_id]
                if n in cluster_set
            )
            if weight_to_cluster > 0:
                cluster_weights[cluster_id] = weight_to_cluster / total_edge_weight
        
        # Normalize to sum to 1.0 (should already be close)
        total = sum(cluster_weights.values())
        memberships[entry_id] = {
            cid: w/total for cid, w in cluster_weights.items()
        }
    
    # Step 5: Map old cluster IDs to new cluster IDs (stability)
    # Use best-overlap matching: each new cluster inherits the ID of the old
    # cluster with which it shares the most members. This preserves cluster
    # identity across recomputations, which preserves grain_cluster_touch values.
    new_cluster_ids, touch_remap = stable_cluster_id_mapping(
        old_partition, partition, old_cluster_ids
    )
    
    # Step 6: Remap grain_cluster_touch entries using the touch_remap table.
    # This handles splits, merges, and identity preservation in one pass.
    remap_grain_cluster_touches(touch_remap)
    
    # Step 7: Persist new memberships to entry_cluster_membership table
    replace_cluster_memberships(memberships, new_cluster_ids)


def stable_cluster_id_mapping(old_partition, new_partition, old_cluster_ids):
    """
    Matches new clusters to old cluster IDs using the Jaccard overlap rule,
    then builds a touch_remap table that specifies how each grain's
    accumulated touch weight in old clusters should be redistributed to
    new clusters.
    
    Returns:
        new_cluster_ids: list mapping new_cluster_index -> cluster_id (UUID)
        touch_remap: dict mapping old_cluster_id -> {new_cluster_id: proportion}
    """
    # Build overlap matrix: M[i][j] = Jaccard(old_partition[i], new_partition[j])
    overlap = {}
    for i, old_set in enumerate(old_partition):
        for j, new_set in enumerate(new_partition):
            intersection = len(old_set & new_set)
            union = len(old_set | new_set)
            if union > 0:
                overlap[(i, j)] = intersection / union
    
    # Greedy matching: for each new cluster, inherit ID of best-matching old cluster
    # if overlap exceeds CLUSTER_INHERIT_OVERLAP_MIN; otherwise assign fresh UUID.
    new_cluster_ids = [None] * len(new_partition)
    used_old_ids = set()
    
    # Sort (old, new) pairs by overlap descending, assign greedily
    pairs = sorted(overlap.items(), key=lambda kv: kv[1], reverse=True)
    for (i, j), score in pairs:
        if score < CLUSTER_INHERIT_OVERLAP_MIN:  # default 0.30
            continue
        if new_cluster_ids[j] is None and i not in used_old_ids:
            new_cluster_ids[j] = old_cluster_ids[i]
            used_old_ids.add(i)
    
    # New clusters with no matching old cluster get fresh UUIDs
    for j in range(len(new_partition)):
        if new_cluster_ids[j] is None:
            new_cluster_ids[j] = generate_uuid()
    
    # Build touch_remap: for each OLD cluster, compute how its touch weight
    # should redistribute to NEW clusters. This handles all four cases:
    #
    #   1. IDENTITY: old cluster X becomes new cluster X unchanged
    #      → all touch goes to same cluster_id
    #
    #   2. SPLIT: old cluster X splits into new clusters Y and Z
    #      → touch weight redistributed proportionally to |members_shared_with_Y|
    #        and |members_shared_with_Z|
    #
    #   3. MERGE: old clusters X and W merge into new cluster Y
    #      → X's touch and W's touch both go to Y (additive)
    #
    #   4. DISSOLVE: old cluster X has no overlap with any new cluster
    #      → touch weight carried over to nearest new cluster (max Jaccard)
    #        at reduced rate (multiplied by CLUSTER_DISSOLVE_DECAY = 0.5)
    touch_remap = {}
    for i, old_set in enumerate(old_partition):
        old_id = old_cluster_ids[i]
        # Find all new clusters this old cluster maps into
        splits = {}  # new_cluster_id -> proportion
        total_preserved = 0
        for j, new_set in enumerate(new_partition):
            shared = len(old_set & new_set)
            if shared > 0:
                proportion = shared / len(old_set)
                splits[new_cluster_ids[j]] = proportion
                total_preserved += proportion
        
        if total_preserved == 0:
            # DISSOLVE case: find nearest new cluster and assign decayed touch
            best_j, best_score = None, 0
            for j, new_set in enumerate(new_partition):
                score = overlap.get((i, j), 0)
                if score > best_score:
                    best_score, best_j = score, j
            if best_j is not None:
                splits[new_cluster_ids[best_j]] = CLUSTER_DISSOLVE_DECAY  # 0.5
            # If nothing matches at all, touch weight is lost (extinct cluster)
        else:
            # Normalize proportions to sum to 1.0 (handles SPLIT case cleanly)
            splits = {cid: p / total_preserved for cid, p in splits.items()}
        
        touch_remap[old_id] = splits
    
    return new_cluster_ids, touch_remap


def remap_grain_cluster_touches(touch_remap):
    """
    For each (grain_id, old_cluster_id, touch_weight) row in grain_cluster_touch,
    redistribute touch_weight across new clusters per touch_remap.
    
    MERGE case: multiple old clusters map to same new cluster → touch weights add.
    SPLIT case: one old cluster maps to multiple new clusters → touch weight splits.
    IDENTITY case: one-to-one mapping → touch weight moves unchanged.
    DISSOLVE case: old cluster has no successor → touch decays via CLUSTER_DISSOLVE_DECAY.
    """
    new_touches = defaultdict(float)  # (grain_id, new_cluster_id) -> weight
    
    for (grain_id, old_cluster_id, old_weight) in query_all_grain_cluster_touches():
        if old_cluster_id not in touch_remap:
            continue  # Old cluster completely dissolved with no successor
        for new_cluster_id, proportion in touch_remap[old_cluster_id].items():
            new_touches[(grain_id, new_cluster_id)] += old_weight * proportion
    
    # Replace grain_cluster_touch table atomically
    replace_grain_cluster_touches(new_touches)
```

**Stability safeguards:**

- **Seeded randomness:** Louvain has a small random component; using a fixed seed produces reproducible partitions across identical inputs.
- **Best-overlap ID mapping:** New clusters inherit IDs from old clusters when Jaccard overlap exceeds `CLUSTER_INHERIT_OVERLAP_MIN` (default 0.30). Below that threshold, a fresh UUID is assigned (genuine new cluster).
- **Touch-weight redistribution on splits and merges:**
    - When an old cluster splits into N new clusters, its touch weight redistributes proportionally to each successor based on shared membership
    - When M old clusters merge into one new cluster, their touch weights sum additively into the new cluster
    - When an old cluster dissolves with no strong successor, its touch weight transfers to the nearest new cluster at a decayed rate (multiplied by `CLUSTER_DISSOLVE_DECAY`, default 0.5)
- **Minimum recomputation interval:** Re-clustering runs at most once per `CLUSTER_RECOMPUTE_MIN_INTERVAL_DAYS` (7 days default), preventing flicker from minor shifts.

**Parameters for clustering remapping:**

- `CLUSTER_INHERIT_OVERLAP_MIN`: 0.30 (minimum Jaccard overlap for a new cluster to inherit an old cluster's ID)
- `CLUSTER_DISSOLVE_DECAY`: 0.5 (touch weight carryover rate when an old cluster has no strong successor)

**Invariants:**
- Cluster membership weights always sum to 1.0 per entry point
- Re-clustering never zeros out historical `context_spread`; accumulated touch map is remapped, not reset
- Historical retrievals are replayed against the new cluster structure only at recomputation boundaries (not continuously)

### 13.3 Decay Model

- Two decay classes: working (7-day half-life), core (30-day half-life)
- Conduit decay class is inherited from the grain it points to
- When a grain is promoted, all inbound conduits are reclassified

### 13.4 Embedding Usage

- Bootstrap only at grain insertion (one embedding per grain, then discarded)
- Vector fallback when signal propagation returns zero or low-confidence results
- Never used as the primary retrieval mechanism

### 13.5 Interaction Protocol

Three separate channels:

- **Write channel:** `flux_store(grain_content)` — inserts a grain
- **Read channel:** `flux_retrieve(query)` → returns grains + trace_id. `flux_feedback(trace_id, grain_id, useful)` — applies reinforcement or penalization based on the trace.
- **Admin channel:** `flux_purge`, `flux_purge_by_content`, `flux_export_grain`, `flux_restore` — manual overrides for user-identified issues. Not exposed to the main AI by default; accessible via CLI/SDK for user-directed operations. See Section 7.6.

Feedback lives on the read channel because it operates on a retrieval trace.

### 13.6 Ingestion Model

A separate extractor LLM reads conversations and emits atomic grains via the write channel. Flux does not process raw conversations itself. The extractor is external to Flux.

### 13.7 Signal Aggregation

When a grain receives signal from multiple conduits, signals are summed. Convergent evidence reinforces activation.

### 13.8 Shortcut Direction

Shortcuts created from co-retrieval are bidirectional by default. Entry-to-grain conduits remain directional (entry points are one-way gates).

### 13.9 Isolation

Single graph per deployment. Single-user deployments only. Multi-user isolation and shared world knowledge are out of scope and deferred to the Annexure.

### 13.10 Query Decomposition

Every retrieval query is processed by a local instruction-tuned LLM (7B-8B class) that extracts 2-5 feature words representing the key concepts in the query. Those features become the entry points where signal is injected. This runs on every retrieval. It is part of the read channel.

This is not optional and not replaceable by keyword tokenization, embedding decomposition, or manual feature entry. The local LLM is the only approach that handles natural phrasing robustly while keeping Flux free of runtime embedding dependency.

### 13.11 End-to-End Interaction Flow

The user never interacts with Flux directly. The main AI is the sole interface.

```
1. User sends query to main AI
2. Main AI calls flux_retrieve(user_query)
   → Feature extractor LLM decomposes query
   → Signal propagates, top-k grains returned with trace_id
3. Main AI reasons using retrieved grains + its own capabilities
4. Main AI responds to user
5. Main AI calls flux_feedback(trace_id, grain_id, useful) for each retrieved grain
   → Based on whether the AI actually used the grain in its response
6. Extractor LLM reads (user_query + AI response) and emits atomic grains
   → Each grain written via flux_store
7. New grains are now in the graph for future retrievals
```

Steps 5 and 6 are both required and serve different purposes. Step 5 updates weights on existing conduits based on retrieval usefulness. Step 6 adds new knowledge to the graph.

### 13.12 Hallucination Containment

Flux does not attempt to filter hallucinated grains at extraction. Hallucination containment is handled by external mechanisms (AI system prompts, response validation, fact-checking layers). Flux's natural dynamics provide secondary protection: hallucinated grains that are never retrieved usefully will decay and dissolve.

### 13.13 Parameter Configurability

All system parameters listed in Section 5 are configurable via an external config file. None are hardcoded. Changing parameters takes effect on the next relevant operation:
- Retrieval parameters (ATTENUATION, ACTIVATION_THRESHOLD, MAX_HOPS, TOP_K) apply on the next query
- Learning parameters (LEARNING_RATE, DECAY_FACTOR) apply on the next feedback
- Decay parameters (HALF_LIFE_*) apply on the next decay pass
- Weight bounds (WEIGHT_CEILING, WEIGHT_FLOOR) apply on the next weight update; raising WEIGHT_FLOOR mid-run will trigger a cleanup wave deleting conduits that fall below the new floor
- Promotion parameters (PROMOTION_THRESHOLD) apply on the next feedback

### 13.14 Trace Retention

Traces record the full signal path of each retrieval: which conduits were traversed, what signal each grain received, which grains were marked useful by feedback. They serve two purposes: applying feedback to the graph, and feeding the Health Monitor with historical data.

- Hot traces (most recent `TRACE_RETENTION_COUNT` or last `TRACE_RETENTION_DAYS`, whichever is larger) are kept in full detail, queryable.
- Older traces age out. Their metrics (hop count, activated grain count, success rate) are folded into daily aggregates in the health log. Raw trace data is then deleted.
- Feedback operates only on hot traces. A trace that has aged out cannot receive late feedback.

This bounds storage while preserving the Health Monitor's ability to compute trend metrics over the recent past.

### 13.15 Pre-Warming

The initial build supports bootstrapping Flux from existing memory sources (conversation exports, notes, text corpora) via a one-time `flux prewarm` operation. Pre-warming extracts grains through the normal extractor LLM, creates bootstrap conduits through the normal mechanism, and optionally runs synthetic retrieval passes to pre-shape highways. Pre-warming is configurable and skippable for users with no prior memory sources. See Section 11.10.

### 13.16 Query-Time Context Expansion

When primary signal propagation returns low-confidence results, a bounded second-pass scan surfaces lateral candidates from clusters shared with activated grains. Lateral candidates are marked distinctly in the result set. Useful lateral candidates trigger shortcut creation through the normal co-retrieval mechanism. Expansion is enabled by default and adds at most one additional graph query per retrieval, no LLM calls. See Section 11.11.

### 13.17 Context Shift Detection

The system monitors retrieval success rate trajectory. When success drops significantly over a short window (while feedback compliance remains healthy), a context shift is detected. Response: elevated exploration boost and accelerated decay on recently-failed paths for the next N retrievals. No existing weights are zeroed. Promoted core grains are unaffected. See Section 11.12.

---

## 14. EXAMPLE: FULL LIFECYCLE

### Scenario: Personal coding assistant memory

**Step 1: Insert grains (Day 1)**

```
G1: "User prefers Python"         → bootstrap conduits to G5 (similarity)
G2: "User works in consulting"    → bootstrap conduits to G5 (weak)
G3: "User likes dark themes"      → bootstrap conduits to G1 (weak)
G4: "User studies philosophy"     → bootstrap conduits to G2 (weak)
G5: "User builds AI architectures"→ bootstrap conduits to G1, G4
```

Initial graph (all conduits at ~0.25):
```
E:coding ──0.25──> G1
E:AI     ──0.25──> G5
E:AI     ──0.25──> G4
E:consulting ──0.25──> G2
E:design ──0.25──> G3
G1 ──0.20──> G5  (bootstrap similarity)
G5 ──0.18──> G4  (bootstrap similarity)
```

**Step 2: First query (Day 1)**

Query: "What framework for my AI project?"
Features: [AI, framework, project]

Signal propagation:
```
E:AI ──0.25──> G5 (signal: 0.25) ✅ activates
E:AI ──0.25──> G4 (signal: 0.25) ✅ activates
G5   ──0.20──> G1 (signal: 0.25 * 0.20 * 0.85 = 0.04) ✗ below threshold
```

Returns: G5, G4
User feedback: G5 useful ✅, G4 irrelevant ❌

Update:
```
E:AI ──> G5: weight 0.25 → 0.29 (reinforced)
E:AI ──> G4: weight 0.25 → 0.21 (penalized)
```

**Step 3: Second query (Day 2)**

Query: "Best Python library for neural networks?"
Features: [Python, neural networks, library]

Signal propagation:
```
E:coding ──0.25──> G1 (signal: 0.25) ✅
E:AI     ──0.29──> G5 (signal: 0.29) ✅   (weight already boosted!)
E:AI     ──0.21──> G4 (signal: 0.21) ✅   (weaker now)
```

Returns: G5, G1, G4
User feedback: G5 useful ✅, G1 useful ✅, G4 irrelevant ❌

Update:
```
E:AI ──> G5: 0.29 → 0.33
E:AI ──> G4: 0.21 → 0.18
E:coding ──> G1: 0.25 → 0.29
G1 ↔ G5: shortcut candidate (co-retrieval count = 1, need 3 more)
```

**Step 4: After 10 similar queries (Day 5)**

```
E:AI ──> G5: weight now 0.72 (HIGHWAY forming)
E:AI ──> G4: weight now 0.08 (nearly dissolved)
E:coding ──> G1: weight now 0.65
G1 ↔ G5: SHORTCUT CREATED (co-retrieval count hit threshold)
           weight: 0.30
```

**Step 5: Query after highway formation (Day 6)**

Query: "AI coding setup?"
Features: [AI, coding]

```
E:AI     ──0.72──> G5 (signal: 0.72) ✅ INSTANT
E:coding ──0.65──> G1 (signal: 0.65) ✅ INSTANT
G1       ──0.30──> G5 (convergent signal adds +0.17 to G5)
G5       ──0.30──> G1 (convergent signal adds +0.18 to G1)
```

**Resolution: 1 hop. Both grains light up immediately.** Compare to Day 1 when G1 couldn't even activate.

**Step 6: Decay over disuse (Day 30)**

User hasn't asked about consulting in weeks:
```
E:consulting ──> G2: weight 0.25 → 0.09 (decayed)
G2: no inbound conduits above floor → STATUS: DORMANT
```

G2 still exists. Its content is intact. But no signal can reach it anymore. It has been functionally forgotten.

---

## 15. GLOSSARY

| Term | Definition |
|------|-----------|
| **Grain** | Atomic memory item. The content node. Immutable after creation. Starts as working class, can be promoted to core through use. |
| **Working Grain** | Default state. Decays with 7-day half-life. Has not yet proven cross-context value. |
| **Core Grain** | Promoted grain. Decays with 30-day half-life. Earned promotion by being successfully retrieved from 3+ distinct contexts. |
| **Promotion** | When a working grain's context_spread reaches the threshold, it becomes core. No LLM, no manual tagging. The grain earns permanence by proving it transfers across contexts. |
| **Context Spread** | Count of distinct entry point clusters that successfully retrieved a grain. The metric that drives promotion. |
| **Conduit** | Weighted directed edge. The routing intelligence. Mutable through use. Has a decay class. |
| **Decay Class** | Controls how fast a conduit decays. Core (30 days) or Working (7 days). Inherited from the grain it points to. Updated when a grain is promoted. |
| **Entry Point** | Query feature gate. Where signal enters the fabric. |
| **Entry Point Cluster** | A group of entry points that tend to co-activate in the same query context (e.g., E:VMO2 + E:bid = one cluster). Used to measure context_spread. |
| **Trace** | Recorded path of one retrieval. The learning receipt. |
| **Highway** | Conduit with weight near ceiling (>0.8). Formed by repeated success. |
| **Shortcut** | Conduit created between co-retrieved grains. Emergent, not pre-tagged. |
| **Activation** | When signal at a grain exceeds threshold. The grain "lights up". |
| **Conductance** | The weight of a conduit. Higher = more signal passes through. |
| **Dormant** | Grain with no inbound conduits. Exists but unreachable by propagation. |
| **Bootstrap** | One-time embedding similarity check at grain insertion. The only time embeddings are used. |
| **Fabric** | The entire dynamic graph of grains + conduits + entries. The living routing layer. |
| **Propagation** | Signal flowing from entry points through conduits to grains. The retrieval act. |

---

## 16. PRIOR ART ACKNOWLEDGMENT

Flux Memory draws conceptual lineage from:

- **Spreading activation** (Collins & Loftus, 1975): Signal propagation through semantic networks. Flux Memory extends this by making the topology itself mutable.
- **Hebbian learning** ("neurons that fire together wire together"): The co-retrieval shortcut mechanism. Flux Memory applies this to a content-addressed store, not a neural network.
- **Adaptive routing** (Q-routing, Boyan & Littman 1994): Routers learning optimal packet paths through reinforcement. Flux Memory adapts this from network routing to memory retrieval.
- **ACT-R** (Anderson, 1993): Cognitive architecture with base-level activation and spreading activation for memory retrieval.

None of these combine all three of: mutable topology + retrieval-driven learning + decay-based forgetting in a single system designed for AI agent memory.

---

## 17. LICENSE AND CONTRIBUTION

**Status:** Pre-implementation design document.
**IP:** Original mechanism design by Harsh + Claude, April 2026.
**Next step:** Build per Section 11.

---

## ANNEXURE: DEFERRED IDEAS

Ideas explored during design but not part of the core build. Documented here for potential future revisit. None of these are required for Flux Memory to function. They are optional enhancements that may be evaluated if specific problems emerge during live use.

### A1. Pre-Defined Metadata Layer

**Idea:** Combine Flux Memory's adaptive routing with a pre-defined organisational hierarchy (e.g., explicit project/person/topic categories). Pre-defined metadata narrows which entry points receive signal; adaptive conduits learn within that narrowed space.

**Potential benefit:** Immediate human-readable organisation from day one plus adaptive retrieval that improves with use. Reduces cold start impact because metadata filtering helps even before highways form.

**Why deferred:** Flux Memory should prove its mechanism works standalone before introducing an external structural layer. Mixing the two from the start would make it unclear which layer is doing the work. Revisit if the pure Flux approach shows clear organisational limitations at scale.

### A2. Symbolic Codex Integration

**Idea:** Incorporate the author's existing Symbolic Codex system into Flux Memory. Two integration points were considered:
- **Entry points as symbolic intent categories** (e.g., `CHOICE_*`, `PREF_*`, `FACT_*`) instead of raw keywords, making entry points more robust to phrasing variations
- **Grains carrying codex labels as metadata** for clustering and visualization, while keeping grain content in English

**Why deferred:** Codex solves a different problem (token compression for AI-to-AI communication) at a different layer (representation) than Flux (retrieval mechanism). Adding Codex before the core mechanism is validated introduces schema drift risk, symbol collision risk, and extraction complexity without a clear benefit to the core retrieval loop. The core Flux mechanism produces a readable English graph, which is easier to debug, visualize, and reason about during early validation. Revisit if token cost of grain injection into LLM context becomes a measurable bottleneck.

### A3. Cross-Phrasing Robustness (Embedding-Free)

**Problem:** Flux's keyword-based entry points are fragile to synonyms. "Python preference" and "likes Python" may decompose to different entry points. Embeddings handle this naturally but are something we chose to avoid at runtime.

**Candidate approaches:**
- Maintain a lightweight synonym map (manually or LLM-curated) at the entry point layer
- Merge entry points that consistently co-activate on similar queries
- Use stemming/lemmatization during query decomposition

**Why deferred:** The feature extractor LLM already handles synonym normalization at extraction time. The vector fallback provides secondary mitigation. Revisit only if logs show frequent fallback triggers on phrasing-variation queries despite the LLM's handling.

### A4. Embedding Drift

**Problem:** If the embedding model used at bootstrap is updated, old bootstrap conduits may be misaligned with new embeddings (relevant only for the fallback path).

**Mitigation:** Low priority because bootstrap conduits get overwritten by use-driven weights over time. For very long-lived systems, a periodic re-bootstrap of dormant grains with a newer embedding model could help.

### A5. Shared World Knowledge Graph (Multi-User)

**Idea:** In multi-user deployments, maintain a shared "world knowledge" graph (e.g., "TSA is the UK Telecommunications Security Act") plus per-user private overlays. Users benefit from shared learning while keeping personal context isolated.

**Why deferred:** The initial build is single-user. Multi-user isolation is resolved as "separate graphs per user." Shared layer is an exploration for later if multi-user deployments become a requirement.

---

*End of document.*
