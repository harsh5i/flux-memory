"""Config dataclass sanity checks (Section 5)."""
from __future__ import annotations

import pytest

from flux.config import DEFAULT_CONFIG, Config


def test_defaults_match_spec_table():
    c = DEFAULT_CONFIG
    # Propagation defaults are the spec's fresh-graph math (Section 5.1).
    assert c.ATTENUATION == 0.85
    assert c.ACTIVATION_THRESHOLD == 0.15
    assert c.MAX_HOPS == 5
    assert c.TOP_K == 5
    assert c.WEIGHT_CEILING == 0.95
    assert c.WEIGHT_FLOOR == 0.05
    # Fresh-graph bootstrap values raised per Section 5.1 so 2 hops still fire.
    assert c.INITIAL_ENTRY_WEIGHT == 0.50
    assert c.INITIAL_SHORTCUT_WEIGHT == 0.50
    assert c.INITIAL_WEIGHT_SCALE == 0.50


def test_config_is_frozen():
    with pytest.raises(Exception):  # FrozenInstanceError subclasses AttributeError
        DEFAULT_CONFIG.ATTENUATION = 0.5  # type: ignore[misc]


def test_overrides_work():
    c = Config(ATTENUATION=0.9, MAX_HOPS=3)
    assert c.ATTENUATION == 0.9
    assert c.MAX_HOPS == 3
    # Other defaults preserved.
    assert c.ACTIVATION_THRESHOLD == 0.15


def test_provenance_multiplier_table():
    c = DEFAULT_CONFIG
    assert c.provenance_multiplier("user_stated") == 1.0
    assert c.provenance_multiplier("external_source") == 0.9
    assert c.provenance_multiplier("ai_stated") == 0.5
    assert c.provenance_multiplier("ai_inferred") == 0.3
    # Unknown tags fall back to the cautious middle value.
    assert c.provenance_multiplier("unknown_source") == 0.5


def test_half_life_by_decay_class():
    c = DEFAULT_CONFIG
    assert c.half_life_hours("core") == 720.0
    assert c.half_life_hours("working") == 168.0
    assert c.half_life_hours("ephemeral") == 48.0
    # Unknown class defaults to working.
    assert c.half_life_hours("anything_else") == 168.0


def test_fresh_graph_two_hop_propagation_math():
    """The spec (Section 5.1) proves 2 hops reach threshold on a fresh graph.
    Encode that invariant here so tuning can't silently break it."""
    c = DEFAULT_CONFIG
    w_entry = c.INITIAL_ENTRY_WEIGHT
    w_bootstrap = c.INITIAL_WEIGHT_SCALE
    hop_factor = c.ATTENUATION * w_bootstrap * c.ATTENUATION
    signal_hop2 = w_entry * hop_factor
    assert signal_hop2 > c.ACTIVATION_THRESHOLD, (
        f"2-hop fresh-graph signal {signal_hop2:.4f} fell below threshold "
        f"{c.ACTIVATION_THRESHOLD}; Section 5.1 invariant broken."
    )
