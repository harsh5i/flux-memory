"""Operational utilities for Flux Memory (Track 5 Steps 2–4).

Track 5 Step 2 — Parameter hot-reload:
    ConfigWatcher — watches a YAML file and calls a callback when it changes.

Track 5 Step 3 — Backup and restore:
    backup(store, path)   — safe SQLite backup via VACUUM INTO / backup API.
    restore(backup_path, target_path) — replace a database with a backup.

Track 5 Step 4 — Graceful shutdown / resume:
    GracefulShutdown — context manager that catches SIGINT/SIGTERM, flushes
    pending writes, and logs a 'shutdown' system event before exiting.

§13.13 notes that parameter changes take effect on the next relevant operation;
hot-reload here swaps the Config object in the caller's namespace via callback.
"""
from __future__ import annotations

import logging
import os
import shutil
import signal
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable

from .config import Config

logger = logging.getLogger(__name__)


# --------------------------------------------------------- Track 5 Step 2: hot-reload

class ConfigWatcher:
    """Poll a YAML config file for changes and fire a callback on reload.

    Usage:
        watcher = ConfigWatcher("flux.yaml", on_reload=lambda cfg: store.cfg = cfg)
        watcher.start()
        # … run your app …
        watcher.stop()

    The watcher runs in a daemon thread (poll_interval_seconds, default 5 s).
    The callback is called with the newly loaded Config object.
    If the YAML is malformed, a warning is logged and the old config is kept.
    """

    def __init__(
        self,
        yaml_path: str | Path,
        on_reload: Callable[[Config], None],
        poll_interval_seconds: float = 5.0,
    ) -> None:
        self._path = Path(yaml_path)
        self._callback = on_reload
        self._interval = poll_interval_seconds
        self._last_mtime: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._last_mtime = self._mtime()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("ConfigWatcher started for %s (interval=%ss)", self._path, self._interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1)
            self._thread = None

    def _mtime(self) -> float:
        try:
            return self._path.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            mtime = self._mtime()
            if mtime != self._last_mtime and mtime != 0.0:
                self._last_mtime = mtime
                try:
                    cfg = Config.from_yaml(self._path)
                    logger.info("ConfigWatcher: config reloaded from %s", self._path)
                    self._callback(cfg)
                except Exception as exc:
                    logger.warning("ConfigWatcher: reload failed (%s), keeping old config", exc)


# --------------------------------------------------------- Track 5 Step 3: backup/restore

def backup(store, dest_path: str | Path, *, now=None) -> Path:
    """Create a safe backup of the Flux SQLite database.

    Uses SQLite's online backup API so the backup is consistent even if the
    source DB is being written to concurrently.

    Returns the Path of the created backup file.
    """
    from .graph import utcnow
    from .health import log_event

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    backup_conn = sqlite3.connect(str(dest))
    try:
        store.conn.backup(backup_conn)
    finally:
        backup_conn.close()

    now = now or utcnow()
    log_event(store, "system", "backup_created", {
        "dest": str(dest),
        "size_bytes": dest.stat().st_size,
    }, now=now)
    logger.info("Backup written to %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def restore(backup_path: str | Path, target_path: str | Path) -> None:
    """Replace target_path with a backup copy.

    The target database must NOT be open when this is called. It is the
    caller's responsibility to close any open FluxStore before calling restore.

    A safety copy of the current target is made at <target>.pre_restore before
    overwriting, in case the restore itself needs to be undone.
    """
    src = Path(backup_path)
    dst = Path(target_path)

    if not src.exists():
        raise FileNotFoundError(f"restore: backup not found at {src}")

    # Safety copy.
    if dst.exists():
        safety = dst.with_suffix(".pre_restore")
        shutil.copy2(dst, safety)
        logger.info("Safety copy of current DB written to %s", safety)

    shutil.copy2(src, dst)
    logger.info("Restored %s → %s", src, dst)


# --------------------------------------------------------- Track 5 Step 4: graceful shutdown

class GracefulShutdown:
    """Context manager that intercepts SIGINT/SIGTERM and logs a clean shutdown event.

    Usage:
        with GracefulShutdown(store) as gs:
            # run your application loop
            while not gs.shutdown_requested:
                process_next_turn()

    On signal, sets gs.shutdown_requested = True and logs a 'shutdown' event.
    When the with-block exits normally (or after the signal), logs the event
    and calls store.conn.commit() to flush any pending writes.
    """

    def __init__(self, store, cfg=None) -> None:
        from .config import DEFAULT_CONFIG
        self._store = store
        self._cfg = cfg or DEFAULT_CONFIG
        self.shutdown_requested = False
        self._old_sigint: signal.Handlers = signal.SIG_DFL
        self._old_sigterm: signal.Handlers = signal.SIG_DFL

    def __enter__(self) -> "GracefulShutdown":
        self._old_sigint = signal.signal(signal.SIGINT, self._handler)
        try:
            self._old_sigterm = signal.signal(signal.SIGTERM, self._handler)
        except (OSError, ValueError):
            pass  # SIGTERM not available on all platforms (e.g., Windows threads)
        from .health import log_event
        from .graph import utcnow
        log_event(self._store, "system", "startup", {}, now=utcnow())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        from .health import log_event
        from .graph import utcnow
        now = utcnow()
        log_event(self._store, "system", "shutdown", {
            "clean": exc_type is None,
        }, now=now)
        try:
            self._store.conn.commit()
        except Exception:
            pass
        # Restore original handlers.
        signal.signal(signal.SIGINT, self._old_sigint)
        try:
            signal.signal(signal.SIGTERM, self._old_sigterm)
        except (OSError, ValueError):
            pass
        return False  # don't suppress exceptions

    def _handler(self, signum, frame) -> None:
        logger.info("GracefulShutdown: received signal %d, shutting down…", signum)
        self.shutdown_requested = True
