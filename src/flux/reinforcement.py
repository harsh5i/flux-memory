"""Reinforcement (Section 4.3) and penalization (Section 4.4).

These are the feedback arms of the signal pipeline. After a retrieval, callers
identify which activated grains were useful vs. irrelevant, and this module:

  reinforce(trace, successful) -> widens conduits to successful grains,
                                   increments co-retrieval counts, creates
                                   shortcuts between co-successful pairs,
                                   sharpens entry affinities.

  penalize(trace, failed)      -> narrows conduits to failed grains, deletes
                                   them if they fall below WEIGHT_FLOOR, and
                                   dampens entry affinities on failed first
                                   hops.

Both passes touch conduits via effective_weight() so lazy decay is applied
before the new weight is written, preserving Section 4.5 correctness.

Provenance multipliers (Section 7.2) scale the effective learning rate per
conduit: user_stated reinforces at 1.0, ai_inferred at 0.3. This is the sole
lever preventing hallucinated grains from compounding like facts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

from .config import Config, DEFAULT_CONFIG
from .graph import Conduit, utcnow
from .propagation import TraceStep, effective_weight
from .storage import FluxStore


# --------------------------------------------------------------------- reinforce
def reinforce(
    store: FluxStore,
    trace: Iterable[TraceStep],
    successful_grain_ids: Iterable[str],
    cfg: Config = DEFAULT_CONFIG,
    *,
    now: datetime | None = None,
) -> None:
    """Widen conduits on the successful trace, upsert co-retrieval counts,
    create/reinforce shortcuts between successful pairs, sharpen affinities."""
    now = now or utcnow()
    successful = set(successful_grain_ids)
    trace_list = list(trace)

    # 1. Widen conduits that led to a successful grain. Use provenance of the
    #    destination grain as the multiplier (Section 7.2).
    for step in trace_list:
        if step.to_id not in successful:
            continue
        conduit = store.get_conduit(step.conduit_id)
        if conduit is None:
            continue
        target = store.get_grain(step.to_id)
        multiplier = cfg.provenance_multiplier(target.provenance) if target else 1.0
        _widen(store, conduit, cfg, now, multiplier=multiplier)

    # 2. Co-retrieval counts and shortcut creation between every successful pair.
    successful_list = sorted(successful)  # stable iteration
    for i in range(len(successful_list)):
        for j in range(i + 1, len(successful_list)):
            a, b = successful_list[i], successful_list[j]
            count = store.increment_co_retrieval(a, b, delta=1)
            existing = store.conduit_between(a, b)
            if existing is not None:
                _widen(store, existing, cfg, now)
            elif count >= cfg.SHORTCUT_THRESHOLD:
                _create_shortcut(store, a, b, cfg, now)

    # 3. Sharpen entry affinities on successful first hops. Entries are
    #    identified by the trace step's from_id when hop == 0.
    for step in trace_list:
        if step.hop != 0 or step.to_id not in successful:
            continue
        entry = store.get_entry(step.from_id)
        if entry is None:
            continue
        current = entry.affinities.get(step.conduit_id, 1.0)
        entry.affinities[step.conduit_id] = min(current * 1.1, 2.0)
        store.update_entry_affinities(entry.id, entry.affinities)


# --------------------------------------------------------------------- penalize
def penalize(
    store: FluxStore,
    trace: Iterable[TraceStep],
    failed_grain_ids: Iterable[str],
    cfg: Config = DEFAULT_CONFIG,
    *,
    now: datetime | None = None,
) -> None:
    """Narrow conduits to failed grains; delete if weight falls below floor.
    Dampen entry affinities on failed first hops."""
    now = now or utcnow()
    failed = set(failed_grain_ids)

    for step in trace:
        if step.to_id not in failed:
            continue
        conduit = store.get_conduit(step.conduit_id)
        if conduit is None:
            continue

        # Penalize operates on the time-decayed weight but skips the grace
        # floor. Grace is about protecting new conduits from time-decay
        # starvation, not from explicit negative feedback.
        current = effective_weight(conduit, cfg, now, apply_grace_floor=False)
        new_weight = current * cfg.DECAY_FACTOR
        if new_weight < cfg.WEIGHT_FLOOR:
            store.delete_conduit(conduit.id)
        else:
            store.update_conduit_weight(conduit.id, new_weight, now)

        if step.hop == 0:
            entry = store.get_entry(step.from_id)
            if entry is not None:
                current_aff = entry.affinities.get(step.conduit_id, 1.0)
                entry.affinities[step.conduit_id] = max(current_aff * 0.8, 0.1)
                store.update_entry_affinities(entry.id, entry.affinities)


# --------------------------------------------------------------------- helpers
def _widen(
    store: FluxStore,
    conduit: Conduit,
    cfg: Config,
    now: datetime,
    *,
    multiplier: float = 1.0,
) -> None:
    """weight += LEARNING_RATE * multiplier * (1 - weight), clamped by ceiling.
    Applies lazy decay first so the delta is relative to the current
    effective weight, not the stale stored weight."""
    current = effective_weight(conduit, cfg, now)
    delta = cfg.LEARNING_RATE * multiplier * (1.0 - current)
    new_weight = min(current + delta, cfg.WEIGHT_CEILING)
    store.update_conduit_weight(
        conduit.id, new_weight, now, use_count=conduit.use_count + 1
    )


def _create_shortcut(
    store: FluxStore,
    grain_a: str,
    grain_b: str,
    cfg: Config,
    now: datetime,
) -> None:
    """Create a bidirectional shortcut. Enforce MAX_EDGES_PER_GRAIN by evicting
    the weakest edge on any saturated endpoint first (Section 4.3)."""
    for grain_id in (grain_a, grain_b):
        if store.count_edges(grain_id) >= cfg.MAX_EDGES_PER_GRAIN:
            _evict_weakest(store, grain_id, cfg, now)

    shortcut = Conduit(
        from_id=grain_a,
        to_id=grain_b,
        weight=cfg.INITIAL_SHORTCUT_WEIGHT,
        created_at=now,
        last_used=now,
        direction="bidirectional",
    )
    store.insert_conduit(shortcut)


def _evict_weakest(
    store: FluxStore, grain_id: str, cfg: Config, now: datetime
) -> None:
    edges = store.edges_of(grain_id)
    if not edges:
        return
    weakest = min(edges, key=lambda c: effective_weight(c, cfg, now))
    store.delete_conduit(weakest.id)
