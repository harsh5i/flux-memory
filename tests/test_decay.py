"""Tests for decay.py — cleanup_pass and expiry_pass (Track 1 Steps 5-6)."""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from flux import Config, Conduit, Grain, FluxStore
from flux.decay import cleanup_pass, expiry_pass


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    with FluxStore(db) as s:
        yield s


def _grain(provenance="user_stated", status="active") -> Grain:
    return Grain(content="test grain", provenance=provenance, status=status)


def _conduit(from_id: str, to_id: str, weight: float = 0.5, last_used_hours_ago: float = 0.0) -> Conduit:
    last_used = datetime.now(timezone.utc) - timedelta(hours=last_used_hours_ago)
    return Conduit(from_id=from_id, to_id=to_id, weight=weight, last_used=last_used)


# ===================================================================== cleanup_pass

class TestCleanupPass:
    def test_deletes_sub_floor_stale_conduit(self, store):
        """A conduit with effective_weight < WEIGHT_FLOOR after decay must be deleted."""
        cfg = Config(
            WEIGHT_FLOOR=0.05,
            CLEANUP_STALE_HOURS=1,
            CLEANUP_BATCH_SIZE=100,
            NEW_CONDUIT_GRACE_HOURS=0,  # disable grace so decay applies immediately
        )
        g1 = _grain(); g2 = _grain()
        store.insert_grain(g1); store.insert_grain(g2)

        # weight=0.04 < WEIGHT_FLOOR=0.05, stale for 2 hours
        c = Conduit(
            from_id=g1.id,
            to_id=g2.id,
            weight=0.04,
            last_used=datetime.now(timezone.utc) - timedelta(hours=2),
            decay_class="working",
        )
        store.insert_conduit(c)

        stats = cleanup_pass(store, cfg)

        assert stats["conduits_deleted"] == 1
        assert store.get_conduit(c.id) is None

    def test_keeps_fresh_conduit(self, store):
        """A conduit touched recently (within CLEANUP_STALE_HOURS) is not even a candidate."""
        cfg = Config(CLEANUP_STALE_HOURS=72, CLEANUP_BATCH_SIZE=100)
        g1 = _grain(); g2 = _grain()
        store.insert_grain(g1); store.insert_grain(g2)

        c = Conduit(from_id=g1.id, to_id=g2.id, weight=0.5)  # last_used = now
        store.insert_conduit(c)

        stats = cleanup_pass(store, cfg)

        assert stats["conduits_deleted"] == 0
        assert store.get_conduit(c.id) is not None

    def test_keeps_stale_but_strong_conduit(self, store):
        """A stale conduit whose effective_weight >= WEIGHT_FLOOR must be kept."""
        cfg = Config(
            WEIGHT_FLOOR=0.05,
            CLEANUP_STALE_HOURS=1,
            CLEANUP_BATCH_SIZE=100,
            HALF_LIFE_WORKING_HOURS=168,  # 7-day half-life
            NEW_CONDUIT_GRACE_HOURS=0,
        )
        g1 = _grain(); g2 = _grain()
        store.insert_grain(g1); store.insert_grain(g2)

        # weight=0.9, stale for 2 hours — decays barely, stays well above floor
        c = Conduit(
            from_id=g1.id, to_id=g2.id, weight=0.9,
            last_used=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        store.insert_conduit(c)

        stats = cleanup_pass(store, cfg)

        assert stats["conduits_deleted"] == 0

    def test_marks_orphaned_grain_dormant(self, store):
        """If deleting a conduit leaves its destination grain with no inbound conduits,
        that grain should be marked dormant."""
        cfg = Config(
            WEIGHT_FLOOR=0.05,
            CLEANUP_STALE_HOURS=1,
            CLEANUP_BATCH_SIZE=100,
            NEW_CONDUIT_GRACE_HOURS=0,
        )
        g1 = _grain(); g2 = _grain()
        store.insert_grain(g1); store.insert_grain(g2)

        # Only inbound conduit to g2, just below floor, stale
        c = Conduit(
            from_id=g1.id, to_id=g2.id, weight=0.03,
            last_used=datetime.now(timezone.utc) - timedelta(hours=2),
            decay_class="working",
        )
        store.insert_conduit(c)

        stats = cleanup_pass(store, cfg)

        assert stats["grains_marked_dormant"] == 1
        g2_updated = store.get_grain(g2.id)
        assert g2_updated.status == "dormant"
        assert g2_updated.dormant_since is not None

    def test_does_not_mark_dormant_when_other_inbound_exists(self, store):
        """Grain with multiple inbound conduits stays active when only one is deleted."""
        cfg = Config(
            WEIGHT_FLOOR=0.05,
            CLEANUP_STALE_HOURS=1,
            CLEANUP_BATCH_SIZE=100,
            NEW_CONDUIT_GRACE_HOURS=0,
        )
        g1 = _grain(); g2 = _grain(); g3 = _grain()
        store.insert_grain(g1); store.insert_grain(g2); store.insert_grain(g3)

        bad = Conduit(
            from_id=g1.id, to_id=g3.id, weight=0.03,
            last_used=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        good = Conduit(from_id=g2.id, to_id=g3.id, weight=0.7)  # recent, strong
        store.insert_conduit(bad)
        store.insert_conduit(good)

        stats = cleanup_pass(store, cfg)

        assert stats["grains_marked_dormant"] == 0
        g3_updated = store.get_grain(g3.id)
        assert g3_updated.status == "active"

    def test_respects_batch_size(self, store):
        """cleanup_pass processes at most CLEANUP_BATCH_SIZE candidates."""
        cfg = Config(
            WEIGHT_FLOOR=0.05,
            CLEANUP_STALE_HOURS=1,
            CLEANUP_BATCH_SIZE=2,
            NEW_CONDUIT_GRACE_HOURS=0,
        )
        # Create hub grains
        hub = _grain(); store.insert_grain(hub)
        grains = [_grain() for _ in range(5)]
        for g in grains:
            store.insert_grain(g)

        for g in grains:
            c = Conduit(
                from_id=hub.id, to_id=g.id, weight=0.03,
                last_used=datetime.now(timezone.utc) - timedelta(hours=2),
            )
            store.insert_conduit(c)

        stats = cleanup_pass(store, cfg)

        # Batch size caps candidates scanned, not necessarily deleted
        assert stats["candidates_scanned"] <= 2

    def test_skips_already_dormant_and_archived_grains(self, store):
        """Grains already in dormant/archived/quarantined status are not re-processed."""
        cfg = Config(WEIGHT_FLOOR=0.05, CLEANUP_STALE_HOURS=1, CLEANUP_BATCH_SIZE=100, NEW_CONDUIT_GRACE_HOURS=0)
        dormant_grain = Grain(content="dormant", provenance="user_stated", status="dormant")
        hub = _grain()
        store.insert_grain(dormant_grain); store.insert_grain(hub)

        c = Conduit(
            from_id=hub.id, to_id=dormant_grain.id, weight=0.03,
            last_used=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        store.insert_conduit(c)

        stats = cleanup_pass(store, cfg)
        # Conduit deleted (weight < floor), but grain already dormant, so dormant count stays 0
        assert stats["grains_marked_dormant"] == 0

    def test_returns_correct_stats_structure(self, store):
        stats = cleanup_pass(store)
        assert "candidates_scanned" in stats
        assert "conduits_deleted" in stats
        assert "grains_marked_dormant" in stats

    def test_weight_invariant_after_cleanup(self, store):
        """Any conduit surviving cleanup_pass has effective_weight >= WEIGHT_FLOOR."""
        from flux.propagation import effective_weight

        cfg = Config(WEIGHT_FLOOR=0.05, CLEANUP_STALE_HOURS=1, CLEANUP_BATCH_SIZE=100, NEW_CONDUIT_GRACE_HOURS=0)
        g1 = _grain(); g2 = _grain(); g3 = _grain()
        store.insert_grain(g1); store.insert_grain(g2); store.insert_grain(g3)

        stale_time = datetime.now(timezone.utc) - timedelta(hours=2)
        conduits = [
            Conduit(from_id=g1.id, to_id=g2.id, weight=0.9, last_used=stale_time),  # survives
            Conduit(from_id=g1.id, to_id=g3.id, weight=0.02, last_used=stale_time), # deleted
        ]
        for c in conduits:
            store.insert_conduit(c)

        cleanup_pass(store, cfg)

        for c_orig in conduits:
            c = store.get_conduit(c_orig.id)
            if c is not None:
                assert effective_weight(c, cfg) >= cfg.WEIGHT_FLOOR


# ===================================================================== expiry_pass

class TestExpiryPass:
    def test_archives_long_dormant_grain(self, store):
        """Grains dormant longer than DORMANCY_LIMIT_DAYS are archived."""
        cfg = Config(DORMANCY_LIMIT_DAYS=30)
        g = Grain(
            content="old dormant", provenance="user_stated", status="dormant",
            dormant_since=datetime.now(timezone.utc) - timedelta(days=31),
        )
        store.insert_grain(g)

        stats = expiry_pass(store, cfg)

        assert stats["grains_archived"] == 1
        updated = store.get_grain(g.id)
        assert updated.status == "archived"

    def test_does_not_archive_recently_dormant_grain(self, store):
        """Grains dormant for fewer than DORMANCY_LIMIT_DAYS are left alone."""
        cfg = Config(DORMANCY_LIMIT_DAYS=30)
        g = Grain(
            content="recent dormant", provenance="user_stated", status="dormant",
            dormant_since=datetime.now(timezone.utc) - timedelta(days=10),
        )
        store.insert_grain(g)

        stats = expiry_pass(store, cfg)

        assert stats["grains_archived"] == 0
        assert store.get_grain(g.id).status == "dormant"

    def test_skips_active_grains(self, store):
        g = _grain()
        store.insert_grain(g)
        stats = expiry_pass(store)
        assert stats["grains_archived"] == 0

    def test_skips_dormant_with_no_dormant_since(self, store):
        """Dormant grain with NULL dormant_since should not be archived (defensive)."""
        g = Grain(content="bad dormant", provenance="user_stated", status="dormant", dormant_since=None)
        store.insert_grain(g)
        stats = expiry_pass(store)
        assert stats["grains_archived"] == 0

    def test_multiple_grains_correct_count(self, store):
        cfg = Config(DORMANCY_LIMIT_DAYS=30)
        old = Grain(
            content="old", provenance="user_stated", status="dormant",
            dormant_since=datetime.now(timezone.utc) - timedelta(days=40),
        )
        young = Grain(
            content="young", provenance="user_stated", status="dormant",
            dormant_since=datetime.now(timezone.utc) - timedelta(days=5),
        )
        active = _grain()
        store.insert_grain(old); store.insert_grain(young); store.insert_grain(active)

        stats = expiry_pass(store, cfg)

        assert stats["grains_archived"] == 1
        assert store.get_grain(old.id).status == "archived"
        assert store.get_grain(young.id).status == "dormant"
        assert store.get_grain(active.id).status == "active"

    def test_returns_correct_stats_structure(self, store):
        stats = expiry_pass(store)
        assert "grains_archived" in stats


# ===================================================================== integration

class TestDecayIntegration:
    def test_cleanup_then_expiry_pipeline(self, store):
        """Full decay pipeline: cleanup removes conduit → grain goes dormant → expiry archives it."""
        cfg = Config(
            WEIGHT_FLOOR=0.05,
            CLEANUP_STALE_HOURS=1,
            CLEANUP_BATCH_SIZE=100,
            NEW_CONDUIT_GRACE_HOURS=0,
            DORMANCY_LIMIT_DAYS=0,  # archive immediately on expiry
        )
        hub = _grain(); target = _grain()
        store.insert_grain(hub); store.insert_grain(target)

        c = Conduit(
            from_id=hub.id, to_id=target.id, weight=0.03,
            last_used=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        store.insert_conduit(c)

        # Step 1: cleanup removes conduit and marks target dormant
        cleanup_stats = cleanup_pass(store, cfg)
        assert cleanup_stats["conduits_deleted"] == 1
        assert cleanup_stats["grains_marked_dormant"] == 1

        # Step 2: expiry archives the dormant grain (DORMANCY_LIMIT_DAYS=0 → immediate)
        expiry_stats = expiry_pass(store, cfg)
        assert expiry_stats["grains_archived"] == 1
        assert store.get_grain(target.id).status == "archived"


# ===================================================================== §1B.5 bidirectional orphan fix

class TestBidirectionalOrphanFix:
    """§1A.9 / §1B.5: A grain reachable only via a bidirectional shortcut must NOT go dormant."""

    def test_grain_reachable_via_bidirectional_shortcut_not_dormant(self, store):
        cfg = Config(
            WEIGHT_FLOOR=0.05,
            CLEANUP_STALE_HOURS=1,
            CLEANUP_BATCH_SIZE=100,
            NEW_CONDUIT_GRACE_HOURS=0,
        )
        # g1 and g2 are connected by a bidirectional shortcut stored as (from=g1, to=g2).
        # From g2's perspective: no to_id=g2 conduit exists, but (from_id=g1, to_id=g2,
        # direction=bidirectional) means g2 IS reachable. The old bug would mark g2 dormant.
        g1 = Grain(content="hub grain", provenance="user_stated")
        g2 = Grain(content="target via bidirectional", provenance="user_stated")
        store.insert_grain(g1)
        store.insert_grain(g2)

        # Bidirectional shortcut: stored as (from=g1, to=g2, direction=bidirectional).
        shortcut = Conduit(
            from_id=g1.id, to_id=g2.id, weight=0.6,
            direction="bidirectional",
            last_used=datetime.now(timezone.utc),
        )
        store.insert_conduit(shortcut)

        # Run cleanup (nothing should decay — conduit is fresh).
        cleanup_pass(store, cfg)

        # g2 must remain active because count_inbound_conduits correctly counts
        # the bidirectional shortcut stored with from_id=g1.
        g2_after = store.get_grain(g2.id)
        assert g2_after.status == "active", (
            "grain reachable via bidirectional shortcut was wrongly marked dormant"
        )

    def test_grain_with_only_forward_conduit_still_goes_dormant_when_decayed(self, store):
        """Ensure the fix didn't break the normal dormancy path for truly isolated grains."""
        cfg = Config(
            WEIGHT_FLOOR=0.05,
            CLEANUP_STALE_HOURS=1,
            CLEANUP_BATCH_SIZE=100,
            NEW_CONDUIT_GRACE_HOURS=0,
        )
        hub = Grain(content="hub", provenance="user_stated")
        isolated = Grain(content="isolated", provenance="user_stated")
        store.insert_grain(hub)
        store.insert_grain(isolated)

        # A decayed forward conduit — isolated has no inbound after deletion.
        c = Conduit(
            from_id=hub.id, to_id=isolated.id, weight=0.01,
            last_used=datetime.now(timezone.utc) - timedelta(hours=5),
        )
        store.insert_conduit(c)

        stats = cleanup_pass(store, cfg)
        assert stats["conduits_deleted"] >= 1
        assert store.get_grain(isolated.id).status == "dormant"
