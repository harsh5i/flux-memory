"""Tests for Track 5 Steps 2–4 — ConfigWatcher, backup/restore, GracefulShutdown."""
from __future__ import annotations

import time
import threading
import pytest
from pathlib import Path

from flux import FluxStore, Config
from flux.ops import ConfigWatcher, backup, restore, GracefulShutdown


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    with FluxStore(db) as s:
        yield s, db


# ===================================================================== ConfigWatcher

class TestConfigWatcher:
    def test_fires_callback_on_change(self, tmp_path):
        yaml_path = tmp_path / "flux.yaml"
        yaml_path.write_text("ATTENUATION: 0.80\n")
        received: list[Config] = []
        watcher = ConfigWatcher(yaml_path, on_reload=received.append, poll_interval_seconds=0.1)
        watcher.start()
        try:
            time.sleep(0.15)  # let first poll pass
            yaml_path.write_text("ATTENUATION: 0.70\n")
            time.sleep(0.3)   # wait for change detection
        finally:
            watcher.stop()
        assert len(received) >= 1
        assert received[-1].ATTENUATION == pytest.approx(0.70)

    def test_does_not_fire_on_unchanged_file(self, tmp_path):
        yaml_path = tmp_path / "flux.yaml"
        yaml_path.write_text("ATTENUATION: 0.80\n")
        received: list[Config] = []
        watcher = ConfigWatcher(yaml_path, on_reload=received.append, poll_interval_seconds=0.1)
        watcher.start()
        try:
            time.sleep(0.35)  # several polls without changing
        finally:
            watcher.stop()
        assert len(received) == 0

    def test_invalid_yaml_keeps_old_config(self, tmp_path):
        yaml_path = tmp_path / "flux.yaml"
        yaml_path.write_text("ATTENUATION: 0.80\n")
        received: list[Config] = []
        watcher = ConfigWatcher(yaml_path, on_reload=received.append, poll_interval_seconds=0.1)
        watcher.start()
        try:
            time.sleep(0.15)
            yaml_path.write_text("- invalid\n- yaml\n- list\n")  # non-mapping → raises ValueError
            time.sleep(0.3)
        finally:
            watcher.stop()
        assert len(received) == 0  # callback not called on invalid YAML

    def test_start_stop_idempotent(self, tmp_path):
        yaml_path = tmp_path / "flux.yaml"
        yaml_path.write_text("ATTENUATION: 0.75\n")
        watcher = ConfigWatcher(yaml_path, on_reload=lambda c: None, poll_interval_seconds=0.1)
        watcher.start()
        watcher.start()  # second start is no-op
        watcher.stop()
        watcher.stop()   # second stop is no-op


# ===================================================================== backup / restore

class TestBackupRestore:
    def test_backup_creates_file(self, store, tmp_path):
        s, db_path = store
        dest = tmp_path / "backups" / "flux_backup.db"
        result = backup(s, dest)
        assert result == dest
        assert dest.exists()
        assert dest.stat().st_size > 0

    def test_backup_event_logged(self, store, tmp_path):
        s, db_path = store
        dest = tmp_path / "flux_backup.db"
        backup(s, dest)
        count = s.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE event='backup_created'"
        ).fetchone()["n"]
        assert count == 1

    def test_backup_is_valid_sqlite(self, store, tmp_path):
        import sqlite3
        s, db_path = store
        dest = tmp_path / "flux_backup.db"
        backup(s, dest)
        conn = sqlite3.connect(str(dest))
        try:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            names = {r[0] for r in tables}
            assert "grains" in names
        finally:
            conn.close()

    def test_restore_copies_backup_to_target(self, tmp_path):
        db_path = tmp_path / "flux.db"
        backup_path = tmp_path / "flux_backup.db"
        # Create source DB.
        with FluxStore(db_path) as s:
            backup(s, backup_path)
        # Now "restore" to a new path.
        target = tmp_path / "flux_restored.db"
        restore(backup_path, target)
        assert target.exists()

    def test_restore_missing_backup_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            restore(tmp_path / "nonexistent.db", tmp_path / "target.db")

    def test_restore_creates_safety_copy(self, tmp_path):
        db_path = tmp_path / "flux.db"
        backup_path = tmp_path / "backup.db"
        with FluxStore(db_path) as s:
            backup(s, backup_path)
        restore(backup_path, db_path)
        assert db_path.with_suffix(".pre_restore").exists()


# ===================================================================== GracefulShutdown

class TestGracefulShutdown:
    def test_startup_event_logged(self, store):
        s, _ = store
        with GracefulShutdown(s):
            pass
        count = s.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE event='startup'"
        ).fetchone()["n"]
        assert count == 1

    def test_shutdown_event_logged(self, store):
        s, _ = store
        with GracefulShutdown(s):
            pass
        count = s.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE event='shutdown'"
        ).fetchone()["n"]
        assert count == 1

    def test_shutdown_requested_defaults_false(self, store):
        s, _ = store
        with GracefulShutdown(s) as gs:
            assert gs.shutdown_requested is False

    def test_context_exits_cleanly_on_normal_use(self, store):
        s, _ = store
        with GracefulShutdown(s) as gs:
            gs.shutdown_requested = True  # simulate graceful exit request
        # No exception raised.
