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

import math
import time
from datetime import datetime, timedelta
from typing import Iterable

from .config import Config, DEFAULT_CONFIG
from .graph import Conduit, utcnow
from .health import log_event
from .propagation import TraceStep, effective_weight
from .storage import FluxStore

HIGHWAY_WEIGHT_THRESHOLD = 0.80

# Window for sampling retrieval activity when computing the dynamic threshold.
# 7 days balances responsiveness vs. statistical stability — short enough that
# the threshold tracks current usage, long enough that day-of-week variance
# doesn't whipsaw it.
_THRESHOLD_WINDOW_HOURS = 168.0
_THRESHOLD_CACHE_TTL_SECONDS = 300.0  # recompute at most every 5 minutes
_threshold_cache: dict[str, tuple[float, int]] = {}


def compute_shortcut_threshold(
    store: FluxStore,
    cfg: Config = DEFAULT_CONFIG,
    *,
    now: datetime | None = None,
    cache_key: str = "default",
) -> int:
    """Dynamic shortcut threshold scaled to graph size and retrieval activity.

    A shortcut should represent a co-occurrence pattern that's significantly
    above what random pairing would produce. Under random selection, the
    expected number of times a specific pair co-occurs in R retrievals
    (top-k = k, graph size = N) is::

        E_random = R * k * (k - 1) / (N * (N - 1))

    We require an observed count at least 3-sigma above this expectation
    (Poisson approximation), so the threshold is::

        ceil(E_random + 3 * sqrt(E_random))

    Floored at 2 (no shortcut on a single co-occurrence — that's noise by
    construction). The result is cached for ``_THRESHOLD_CACHE_TTL_SECONDS``
    to avoid recomputing on every reinforce() call.

    Falls back to ``cfg.SHORTCUT_THRESHOLD`` if the underlying queries fail
    (e.g., missing tables on a fresh DB).
    """
    cached = _threshold_cache.get(cache_key)
    monotonic = time.monotonic()
    if cached and (monotonic - cached[0]) < _THRESHOLD_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        n_row = store.conn.execute(
            "SELECT COUNT(*) AS n FROM grains WHERE status='active'"
        ).fetchone()
        n = int(n_row["n"]) if n_row else 0
        if n < 2:
            value = max(2, int(cfg.SHORTCUT_THRESHOLD))
            _threshold_cache[cache_key] = (monotonic, value)
            return value

        now_dt = now or utcnow()
        cutoff = (
            now_dt - timedelta(hours=_THRESHOLD_WINDOW_HOURS)
        ).strftime("%Y-%m-%dT%H:%M:%S")
        r_row = store.conn.execute(
            """
            SELECT COUNT(*) AS r FROM events
            WHERE category='retrieval' AND event='grains_returned'
              AND timestamp >= ?
            """,
            (cutoff,),
        ).fetchone()
        r = int(r_row["r"]) if r_row else 0
        k = max(2, int(cfg.TOP_K))

        e_random = r * k * (k - 1) / max(1, n * (n - 1))
        threshold = max(2, math.ceil(e_random + 3.0 * math.sqrt(e_random)))
    except Exception:
        threshold = max(2, int(cfg.SHORTCUT_THRESHOLD))

    _threshold_cache[cache_key] = (monotonic, threshold)
    return threshold


# ----------------------------------------------------------- pair_with_priors
def pair_useful_with_priors(
    store: FluxStore,
    new_grain_id: str,
    prior_grain_ids: Iterable[str],
    cfg: Config = DEFAULT_CONFIG,
    *,
    now: datetime | None = None,
    trace_id: str | None = None,
) -> int:
    """Increment co-retrieval and form/widen conduits between a newly-useful
    grain and other useful grains from the same trace.

    flux_feedback receives one grain at a time, so reinforce()'s built-in
    pair-loop never has more than one element to pair. This helper closes that
    gap by walking the priors-from-same-trace and applying the same logic
    (co-retrieval count + widen-or-shortcut) to each pair.

    Returns the number of pairs processed.
    """
    now = now or utcnow()
    pairs = 0
    for prior in prior_grain_ids:
        if prior == new_grain_id:
            continue
        a, b = sorted((new_grain_id, prior))
        count = store.increment_co_retrieval(a, b, delta=1)
        existing = store.conduit_between(a, b)
        if existing is not None:
            _widen(store, existing, cfg, now, trace_id=trace_id)
        elif count >= cfg.SHORTCUT_THRESHOLD:
            _create_shortcut(store, a, b, cfg, now, co_count=count, trace_id=trace_id)
        pairs += 1
    return pairs


# --------------------------------------------------------------------- reinforce
def reinforce(
    store: FluxStore,
    trace: Iterable[TraceStep],
    successful_grain_ids: Iterable[str],
    cfg: Config = DEFAULT_CONFIG,
    *,
    now: datetime | None = None,
    trace_id: str | None = None,
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
        _widen(store, conduit, cfg, now, multiplier=multiplier, trace_id=trace_id)

    # 2. Co-retrieval counts and shortcut creation between every successful pair.
    successful_list = sorted(successful)  # stable iteration
    for i in range(len(successful_list)):
        for j in range(i + 1, len(successful_list)):
            a, b = successful_list[i], successful_list[j]
            count = store.increment_co_retrieval(a, b, delta=1)
            existing = store.conduit_between(a, b)
            if existing is not None:
                _widen(store, existing, cfg, now, trace_id=trace_id)
            elif count >= cfg.SHORTCUT_THRESHOLD:
                _create_shortcut(store, a, b, cfg, now, co_count=count, trace_id=trace_id)

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
    trace_id: str | None = None,
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
        weight_drop = max(current - new_weight, 0.0)
        if new_weight < cfg.WEIGHT_FLOOR:
            store.delete_conduit(conduit.id)
            deleted = True
        else:
            store.update_conduit_weight(conduit.id, new_weight, now)
            deleted = False

        log_event(store, "feedback", "conduit_penalized", {
            "conduit_id": conduit.id,
            "from_id": conduit.from_id,
            "to_id": conduit.to_id,
            "previous_weight": round(current, 6),
            "new_weight": round(new_weight, 6),
            "weight_drop": round(weight_drop, 6),
            "deleted": deleted,
        }, trace_id=trace_id, now=now)

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
    trace_id: str | None = None,
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
    log_event(store, "feedback", "conduit_reinforced", {
        "conduit_id": conduit.id,
        "from_id": conduit.from_id,
        "to_id": conduit.to_id,
        "previous_weight": round(current, 6),
        "new_weight": round(new_weight, 6),
        "delta": round(new_weight - current, 6),
        "multiplier": round(multiplier, 6),
        "use_count": conduit.use_count + 1,
    }, trace_id=trace_id, now=now)

    if current < HIGHWAY_WEIGHT_THRESHOLD <= new_weight:
        log_event(store, "feedback", "highway_formed", {
            "conduit_id": conduit.id,
            "from_id": conduit.from_id,
            "to_id": conduit.to_id,
            "previous_weight": round(current, 6),
            "new_weight": round(new_weight, 6),
            "threshold": HIGHWAY_WEIGHT_THRESHOLD,
        }, trace_id=trace_id, now=now)


def _create_shortcut(
    store: FluxStore,
    grain_a: str,
    grain_b: str,
    cfg: Config,
    now: datetime,
    *,
    co_count: int,
    trace_id: str | None = None,
) -> None:
    """Create a bidirectional shortcut. Enforce MAX_EDGES_PER_GRAIN by evicting
    the weakest edge on any saturated endpoint first (Section 4.3)."""
    # Invariant (spec line 1750): never create a shortcut below threshold.
    # The caller is reinforce(), which guards this with ``count >= threshold``,
    # but this is the one place shortcuts originate -- verifying locally keeps
    # the invariant tight against refactors that add new creation call sites.
    co_count = store.get_co_retrieval_count(grain_a, grain_b)
    assert co_count >= cfg.SHORTCUT_THRESHOLD, (
        f"shortcut invariant violated: co_retrieval={co_count} < "
        f"SHORTCUT_THRESHOLD={cfg.SHORTCUT_THRESHOLD}"
    )

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
    log_event(store, "feedback", "shortcut_created", {
        "conduit_id": shortcut.id,
        "from_id": shortcut.from_id,
        "to_id": shortcut.to_id,
        "weight": round(shortcut.weight, 6),
        "co_retrieval_count": co_count,
        "threshold": cfg.SHORTCUT_THRESHOLD,
    }, trace_id=trace_id, now=now)


def _evict_weakest(
    store: FluxStore, grain_id: str, cfg: Config, now: datetime
) -> None:
    edges = store.edges_of(grain_id)
    if not edges:
        return
    weakest = min(edges, key=lambda c: effective_weight(c, cfg, now))
    store.delete_conduit(weakest.id)
