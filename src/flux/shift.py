"""Context shift detection (Track 6 Step 3, §11.12).

Monitors retrieval success trajectory over a short rolling window. When the
success rate drops more than CONTEXT_SHIFT_DROP_THRESHOLD in
CONTEXT_SHIFT_WINDOW retrievals (and feedback compliance is healthy, ruling
out missing feedback as the cause), a context shift is detected.

On detection:
  1. A warning is logged (severity INFO).
  2. A system event is emitted so the health monitor can surface it.
  3. An in-memory counter tracks how many more retrievals should run with
     elevated exploration (CONTEXT_SHIFT_RECOVERY_RETRIEVALS).
  4. Returns the exploration_boost multiplier to use for the next retrieval.

Callers (flux_retrieve) check get_exploration_boost() on each call.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from .config import Config, DEFAULT_CONFIG
from .graph import utcnow
from .health import log_event
from .storage import FluxStore

logger = logging.getLogger(__name__)


class ContextShiftDetector:
    """Stateful detector; one instance per running FluxStore session.

    Usage:
        detector = ContextShiftDetector(store, cfg)
        # before each retrieval:
        boost = detector.get_exploration_boost()
        # after each retrieval + feedback:
        detector.record_retrieval(success=True/False)
    """

    def __init__(self, store: FluxStore, cfg: Config = DEFAULT_CONFIG) -> None:
        self._store = store
        self._cfg = cfg
        self._recovery_remaining: int = 0

    def record_retrieval(
        self,
        success: bool,
        now: datetime | None = None,
    ) -> bool:
        """Record a retrieval outcome and detect context shift if enabled.

        Returns True if a context shift was just detected.
        """
        if not self._cfg.CONTEXT_SHIFT_ENABLED:
            return False

        if self._recovery_remaining > 0:
            self._recovery_remaining -= 1

        now = now or utcnow()
        detected = self._check_shift(now)
        if detected:
            self._recovery_remaining = self._cfg.CONTEXT_SHIFT_RECOVERY_RETRIEVALS
        return detected

    def get_exploration_boost(self) -> float:
        """Return the exploration boost multiplier for the next retrieval."""
        if not self._cfg.CONTEXT_SHIFT_ENABLED:
            return 1.0
        if self._recovery_remaining > 0:
            return self._cfg.EXPLORATION_BOOST
        return 1.0

    @property
    def in_recovery(self) -> bool:
        return self._recovery_remaining > 0

    def _check_shift(self, now: datetime) -> bool:
        """Return True if a context shift is detected right now."""
        cfg = self._cfg
        window = cfg.CONTEXT_SHIFT_WINDOW

        # Load the last (window * 2) retrievals to measure two halves.
        rows = self._store.conn.execute(
            """
            SELECT data FROM events
            WHERE category='retrieval' AND event='grains_returned'
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (window * 2,),
        ).fetchall()

        if len(rows) < window:
            return False  # not enough data

        # Recent half vs older half.
        import json
        recent_rows = rows[:window // 2]
        older_rows = rows[window // 2: window]

        def success_rate_for(event_rows) -> float | None:
            total = len(event_rows)
            if total == 0:
                return None
            # Check how many have a matching feedback_received event with useful=true
            # by looking at trace_ids embedded in the events data.
            useful = 0
            for row in event_rows:
                try:
                    data = json.loads(row["data"])
                    trace_id = data.get("trace_id")
                    if trace_id:
                        fb_row = self._store.conn.execute(
                            """
                            SELECT COUNT(*) AS n FROM events
                            WHERE category='feedback' AND event='retrieval_successful'
                              AND trace_id = ?
                            """,
                            (trace_id,),
                        ).fetchone()
                        if fb_row and fb_row["n"] > 0:
                            useful += 1
                except Exception:
                    pass
            return useful / total

        recent_rate = success_rate_for(recent_rows)
        older_rate = success_rate_for(older_rows)

        if recent_rate is None or older_rate is None:
            return False

        drop = older_rate - recent_rate
        if drop < cfg.CONTEXT_SHIFT_DROP_THRESHOLD:
            return False

        # Rule out missing feedback: check compliance rate.
        compliance_row = self._store.conn.execute(
            """
            SELECT
              COUNT(*) AS total_retrievals,
              (SELECT COUNT(*) FROM events
               WHERE category='feedback' AND event='feedback_received'
                 AND timestamp >= (SELECT MIN(timestamp) FROM (
                   SELECT timestamp FROM events
                   WHERE category='retrieval' AND event='grains_returned'
                   ORDER BY timestamp DESC LIMIT ?
                 ))) AS feedback_count
            FROM (
              SELECT timestamp FROM events
              WHERE category='retrieval' AND event='grains_returned'
              ORDER BY timestamp DESC LIMIT ?
            )
            """,
            (window, window),
        ).fetchone()

        if compliance_row:
            total_r = compliance_row["total_retrievals"] or 0
            feedback_c = compliance_row["feedback_count"] or 0
            compliance = feedback_c / total_r if total_r > 0 else 1.0
            if compliance < 0.8:
                # Missing feedback — can't blame a context shift.
                return False

        logger.info(
            "Context shift detected: success rate dropped %.2f → %.2f (drop=%.2f)",
            older_rate, recent_rate, drop,
        )
        log_event(self._store, "system", "context_shift_detected", {
            "older_success_rate": round(older_rate, 3),
            "recent_success_rate": round(recent_rate, 3),
            "drop": round(drop, 3),
            "recovery_retrievals": cfg.CONTEXT_SHIFT_RECOVERY_RETRIEVALS,
        }, now=now)
        return True
