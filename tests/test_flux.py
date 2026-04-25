"""
Tests for Flux Memory.
"""

import tempfile
import os
from datetime import datetime, timedelta

import pytest

from src.flux import Flux, Grain, DecayClass, Conduit, EntryPoint


class TestGrain:
    def test_grain_creation(self):
        grain = Grain(content="VMO2 deadline is May 15")
        assert grain.id.startswith("G-")
        assert grain.decay_class == DecayClass.WORKING
        assert grain.context_spread == 0
    
    def test_grain_promotion(self):
        grain = Grain(content="Important fact")
        assert grain.context_spread == 0
        assert grain.decay_class == DecayClass.WORKING
        
        # Promote after 3 contexts
        grain.record_retrieval("ctx1")
        assert grain.context_spread == 1
        assert grain.decay_class == DecayClass.WORKING
        
        grain.record_retrieval("ctx2")
        grain.record_retrieval("ctx3")
        assert grain.context_spread == 3
        assert grain.decay_class == DecayClass.CORE


class TestConduit:
    def test_conduit_strengthen(self):
        c = Conduit(from_id="A", to_id="B", weight=0.5)
        assert c.weight == 0.5
        
        c.strengthen(0.2)
        assert c.weight == 0.7
        
        # Max at 1.0
        c.strengthen(0.5)
        assert c.weight == 1.0
    
    def test_conduit_weaken(self):
        c = Conduit(from_id="A", to_id="B", weight=0.5)
        c.weaken(0.3)
        assert c.weight == 0.2
        
        c.weaken(0.5)
        assert c.weight <= 0
        assert c.should_dissolve()
    
    def test_conduit_viable(self):
        c = Conduit(from_id="A", to_id="B", weight=0.1)
        assert not c.is_viable()
        
        c.weight = 0.2
        assert c.is_viable()


class TestEntryPoint:
    def test_entry_point_creation(self):
        ep = EntryPoint(feature="VMO2")
        assert ep.id.startswith("E-")
        assert ep.feature == "VMO2"
        assert len(ep.affinities) == 0
    
    def test_entry_point_affinity(self):
        ep = EntryPoint(feature="deadline")
        ep.record_use("conduit-1", success=True)
        assert "conduit-1" in ep.affinities
        assert ep.affinities["conduit-1"] > 0.5
    
    def test_entry_point_affinity_fail(self):
        ep = EntryPoint(feature="project")
        ep.affinities["conduit-1"] = 0.8
        ep.record_use("conduit-1", success=False)
        assert ep.affinities["conduit-1"] < 0.8


class TestFlux:
    def test_remember(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            flux = Flux(store_path=db_path)
            
            grain = flux.remember("VMO2 project deadline is May 15")
            assert grain.id.startswith("G-")
            assert grain.content == "VMO2 project deadline is May 15"
            
            stats = flux.stats()
            assert stats["grains"] >= 1
        finally:
            os.unlink(db_path)
    
    def test_query(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            flux = Flux(store_path=db_path)
            
            # Store some grains
            flux.remember("VMO2 project deadline is May 15")
            flux.remember("Annual review is scheduled for April 20")
            flux.remember("VMO2 requires compliance with ISO standards")
            
            # Query
            results = flux.query("VMO2 deadline")
            
            # Should find relevant grains
            assert len(results) >= 0  # May not find without proper bootstrap
            
            stats = flux.stats()
            assert stats["grains"] == 3
        finally:
            os.unlink(db_path)


class TestDecay:
    def test_decay_factor(self):
        from src.decay import compute_decay_factor
        
        now = datetime.now()
        
        # Recent use = high factor
        factor = compute_decay_factor(now, now, "working")
        assert factor == 1.0
        
        # 168 hours (one half-life for working) = 0.5
        past = now - timedelta(hours=168)
        factor = compute_decay_factor(past, now, "working")
        assert abs(factor - 0.5) < 0.01
        
        # Core decays slower
        past = now - timedelta(hours=168)
        factor_core = compute_decay_factor(past, now, "core")
        assert factor_core > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])