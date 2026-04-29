"""Signal propagation (Section 4.2) and lazy decay (Section 4.5).

Retrieval is not search -- it's a BFS signal injection at entry points, attenuated
per hop and bounded by ACTIVATION_THRESHOLD and MAX_HOPS. Every conduit weight
read during propagation is filtered through effective_weight() so lazy decay is
applied inline without a background pass.

Query decomposition (features -> entry IDs) lives in Track 2. This module takes
entry IDs as input and returns activated grains + a trace for reinforcement.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from .config import Config, DEFAULT_CONFIG
from .graph import Conduit, utcnow
from .storage import FluxStore


# --------------------------------------------------------------------- types
@dataclass(frozen=True)
class TraceStep:
    """One conduit traversal during propagation. The reinforcement and
    penalization passes walk this list to update weights."""
    conduit_id: str
    from_id: str
    to_id: str
    signal: float
    hop: int
    effective_weight: float


@dataclass
class PropagationResult:
    activated: list[tuple[str, float]] = field(default_factory=list)  # (grain_id, signal), sorted desc
    trace: list[TraceStep] = field(default_factory=list)
    confidence: float = 0.0


# ----------------------------------------------------------------- lazy decay
def effective_weight(
    conduit: Conduit,
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
    *,
    apply_grace_floor: bool = True,
) -> float:
    """Stored weight filtered by half-life decay (Section 4.5).

    Grace period: conduits younger than NEW_CONDUIT_GRACE_HOURS decay at a
    reduced rate AND have an elevated floor, protecting new grains from
    starving before they earn their first useful retrieval.

    ``apply_grace_floor=False`` returns the raw time-decayed weight. Callers
    implementing explicit negative feedback (penalize) use this so the grace
    floor doesn't shield a freshly-created-but-wrong conduit from removal.
    The slower-decay half-life multiplier still applies -- only the hard
    floor is skipped."""
    now = now or utcnow()

    half_life = cfg.half_life_hours(conduit.decay_class)
    hours_since_use = (now - conduit.last_used).total_seconds() / 3600.0
    hours_since_creation = (now - conduit.created_at).total_seconds() / 3600.0

    in_grace = hours_since_creation < cfg.NEW_CONDUIT_GRACE_HOURS
    half_life_eff = half_life * (cfg.NEW_CONDUIT_GRACE_MULTIPLIER if in_grace else 1.0)

    decay = 0.5 ** (hours_since_use / half_life_eff) if half_life_eff > 0 else 1.0
    weight = conduit.weight * decay

    if in_grace and apply_grace_floor:
        weight = max(weight, cfg.NEW_CONDUIT_MIN_WEIGHT)

    return weight


# ----------------------------------------------------------------- propagation
def propagate(
    store: FluxStore,
    entry_ids: Iterable[str],
    cfg: Config = DEFAULT_CONFIG,
    *,
    exploration_boost: float = 1.0,
    now: datetime | None = None,
) -> PropagationResult:
    """BFS signal propagation from a set of entry points (Section 4.2).

    ``exploration_boost`` is applied to the injected signal only; it does not
    compound through hops (Section 5.2). Default 1.0 is the untampered path."""
    now = now or utcnow()
    activated: dict[str, float] = {}
    trace: list[TraceStep] = []
    visited: set[str] = set()

    # Queue item: (landed_grain, signal, hop, conduit-that-led-here, upstream_id)
    # upstream_id is the entry/grain the conduit fired out of in this traversal.
    # For reverse-traversed bidirectional shortcuts upstream_id differs from
    # conduit.from_id, so we keep it explicit rather than inferring later.
    frontier: deque[tuple[str, float, int, Conduit, str]] = deque()

    # Step 1: Inject signal at every entry's outgoing conduit.
    for entry_id in entry_ids:
        entry = store.get_entry(entry_id)
        if entry is None:
            continue
        for conduit in store.outgoing_conduits(entry_id):
            w = effective_weight(conduit, cfg, now)
            affinity = entry.affinities.get(conduit.id, 1.0)
            initial = 1.0 * exploration_boost * w * affinity
            if initial >= cfg.ACTIVATION_THRESHOLD:
                frontier.append((conduit.to_id, initial, 0, conduit, entry_id))

    # Step 2: BFS. Attenuate each hop, cap at MAX_HOPS and threshold, dedupe by conduit.
    while frontier:
        grain_id, signal, hop, conduit, upstream_id = frontier.popleft()

        if hop >= cfg.MAX_HOPS or signal < cfg.ACTIVATION_THRESHOLD:
            continue
        if conduit.id in visited:
            continue
        visited.add(conduit.id)

        grain = store.get_grain(grain_id)
        if grain is None or grain.status != "active":
            # Dormant/archived/quarantined grains are skipped from both
            # activation and onward propagation (Section 8 / line 1064 of spec).
            continue

        activated[grain_id] = activated.get(grain_id, 0.0) + signal
        w_at_entry = effective_weight(conduit, cfg, now)
        # Trace records the TRAVERSAL direction (upstream → landed), not the
        # conduit's storage orientation. Reinforcement/penalization read
        # step.to_id as "grain that was activated at this hop" and step.from_id
        # as "what fed signal into it" -- both must reflect how signal actually
        # moved through this graph walk, especially when a bidirectional
        # shortcut was traversed against its stored from/to.
        trace.append(TraceStep(
            conduit_id=conduit.id,
            from_id=upstream_id,
            to_id=grain_id,
            signal=signal,
            hop=hop,
            effective_weight=w_at_entry,
        ))

        # Propagate onward. Bidirectional conduits fire from either endpoint
        # (§13.8), so ask storage for the true downstream neighbour rather
        # than trusting conduit.to_id, which may point back at ``grain_id``
        # for a reverse-traversed shortcut. Attenuation applies once per hop
        # transition either way.
        for next_grain_id, next_conduit in store.propagation_edges_from(grain_id):
            next_w = effective_weight(next_conduit, cfg, now)
            next_signal = signal * next_w * cfg.ATTENUATION
            if next_signal >= cfg.ACTIVATION_THRESHOLD and next_conduit.id not in visited:
                frontier.append((next_grain_id, next_signal, hop + 1, next_conduit, grain_id))

    # Step 3: Rank and cap at TOP_K.
    ranked = sorted(activated.items(), key=lambda kv: kv[1], reverse=True)[: cfg.TOP_K]

    return PropagationResult(
        activated=ranked,
        trace=trace,
        confidence=retrieval_confidence(ranked, trace),
    )


# ----------------------------------------------------------------- confidence
def retrieval_confidence(
    activated: list[tuple[str, float]],
    trace: list[TraceStep],
) -> float:
    """Section 4.2: signal strength + path quality + top-k concentration."""
    if not activated:
        return 0.0

    top_grain_id, top_signal = activated[0]

    signal_score = min(top_signal / 1.0, 1.0)

    conduits_to_top = [t for t in trace if t.to_id == top_grain_id]
    if conduits_to_top:
        path_score = sum(t.effective_weight for t in conduits_to_top) / len(conduits_to_top)
    else:
        path_score = 0.0

    if len(activated) > 1:
        total = sum(s for _, s in activated)
        concentration_score = top_signal / total if total > 0 else 1.0
    else:
        concentration_score = 1.0

    confidence = 0.5 * signal_score + 0.3 * path_score + 0.2 * concentration_score
    return max(0.0, min(confidence, 1.0))
