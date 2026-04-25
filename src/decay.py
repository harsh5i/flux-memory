"""
Decay — Time-based weight adjustment and pruning.

Decay is how Flux Memory handles "forgetting" — conduits and grains
naturally lose weight/prominence over time unless reinforced.

Half-lives:
- Core: 720h (~30 days)
- Working: 168h (~7 days)  
- Ephemeral: 48h (~2 days)
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from grain import Grain, DecayClass
from conduit import Conduit


DECAY_HALF_LIVES = {
    "core": 720,      # hours
    "working": 168,   # hours
    "ephemeral": 48,  # hours
}


def compute_decay_factor(
    last_used: datetime,
    now: datetime,
    decay_class: str,
) -> float:
    """
    Compute decay factor based on time since last use.
    
    Uses exponential decay: factor = 0.5^(hours_elapsed / half_life)
    
    Returns a multiplier (0.0 to 1.0) to apply to weight.
    """
    hours_elapsed = (now - last_used).total_seconds() / 3600
    half_life = DECAY_HALF_LIVES.get(decay_class, 168)  # Default to working
    
    if hours_elapsed <= 0:
        return 1.0
    
    factor = 0.5 ** (hours_elapsed / half_life)
    return factor


def apply_decay_to_conduit(conduit: Conduit, now: datetime) -> Conduit:
    """
    Apply time-based decay to a conduit's weight.
    
    Returns the conduit (modified in place for efficiency).
    """
    factor = compute_decay_factor(
        conduit.last_used,
        now,
        conduit.decay_class,
    )
    
    conduit.weight = conduit.weight * factor
    return conduit


def apply_decay_to_grain(grain: Grain, now: datetime) -> Tuple[Grain, bool]:
    """
    Apply time-based decay to a grain.
    
    Grain content doesn't decay, but its status can change:
    - If dormant too long in working class → archived
    
    Returns (grain, should_archive).
    """
    if grain.status != "active":
        return grain, False
    
    # Check if working-class grain should go dormant
    if grain.decay_class == DecayClass.WORKING:
        # If not used in 7 days, check if should become dormant
        hours_since_creation = (now - grain.created_at).total_seconds() / 3600
        if hours_since_creation > 168 and grain.context_spread < 3:
            # Not promoted, old enough, low context spread
            if grain.dormant_since is None:
                grain.dormant_since = now
                grain.status = "dormant"
    
    # Check if dormant grain should be archived
    if grain.status == "dormant" and grain.dormant_since:
        hours_dormant = (now - grain.dormant_since).total_seconds() / 3600
        if hours_dormant > DECAY_HALF_LIVES.get(grain.decay_class.value, 168):
            return grain, True  # Should archive
    
    return grain, False


def run_decay_cycle(
    conduits: Dict[str, Conduit],
    grains: Dict[str, Grain],
    now: datetime = None,
) -> Tuple[Dict[str, Conduit], Dict[str, Grain], List[str]]:
    """
    Run a full decay cycle across all conduits and grains.
    
    Returns:
    - updated_conduits: conduits with decayed weights (some may need removal)
    - updated_grains: grains with updated status
    - conduit_ids_to_remove: IDs of conduits that dropped below floor
    """
    if now is None:
        now = datetime.now()
    
    conduit_ids_to_remove = []
    updated_conduits = {}
    updated_grains = {}
    
    for cid, conduit in conduits.items():
        apply_decay_to_conduit(conduit, now)
        if conduit.should_dissolve():
            conduit_ids_to_remove.append(cid)
        else:
            updated_conduits[cid] = conduit
    
    for gid, grain in grains.items():
        grain, should_archive = apply_decay_to_grain(grain, now)
        if should_archive:
            grain.status = "archived"
        updated_grains[gid] = grain
    
    return updated_conduits, updated_grains, conduit_ids_to_remove