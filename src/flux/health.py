"""Health Monitor (Section 12) and event logging (Section 11.5).

Two public surfaces:

    log_event(store, category, event, data, trace_id, now)
        Write one structured event to the events table and optionally to a
        rotating JSON log file. Every module in Flux calls this at its natural
        operation boundaries.

    flux_health(store, cfg, now) -> dict
        Compute all 14 health signals, compare against thresholds, update the
        warnings table, and return the status dict per §12.5.

Health signals (§12.1):
  Graph-computable (no event history needed):
    highway_count, orphan_rate, avg_conduit_weight,
    core_grain_count, dormant_grain_rate

  Event-log-driven (require event history):
    highway_growth_rate, shortcut_creation_rate, conduit_dissolution_rate,
    avg_weight_drop_on_failure, promotion_events, avg_hops_per_retrieval,
    retrieval_success_rate, fallback_trigger_rate, feedback_compliance_rate

Warmup suppression (§12.3):
  Warnings are suppressed during the warmup period:
  - Short-window signals: first 100 retrievals
  - Medium-window signals: first 7 days
  - Long-window signals: first 14 days
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import Config, DEFAULT_CONFIG
from .graph import new_id, utcnow
from .storage import FluxStore

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------- constants

_HEALTHY_RANGES: dict[str, dict] = {
    "highway_count": {
        "window": "long",
        "min": 5,
        "max": None,
        "warmup_retrievals": 100,
        "severity": "WARNING",
        "suggestion": "Reinforcement may be broken. Check LEARNING_RATE.",
    },
    "highway_growth_rate": {
        "window": "long",
        "min": 0.1,
        "max": 20,
        "warmup_days": 14,
        "severity": "WARNING",
        "suggestion": "Queries may not be repeating enough to form highways.",
    },
    "shortcut_creation_rate": {
        "window": "short",
        "min": 0.5,
        "max": 10,
        "warmup_retrievals": 100,
        "severity": "INFO",
        "suggestion": "Zero: co-retrieval not triggering. High: SHORTCUT_THRESHOLD may be too low.",
    },
    "conduit_dissolution_rate": {
        "window": "medium",
        "min": 0.001,
        "max": None,
        "warmup_days": 7,
        "severity": "WARNING",
        "suggestion": "Decay may not be firing. Check half-life parameters and decay daemon.",
    },
    "avg_weight_drop_on_failure": {
        "window": "short",
        "min": 0.05,
        "max": 0.20,
        "warmup_retrievals": 100,
        "severity": "INFO",
        "suggestion": "Penalization not strong enough. Check DECAY_FACTOR.",
    },
    "promotion_events": {
        "window": "long",
        "min": 0.001,
        "max": None,
        "warmup_days": 14,
        "severity": "WARNING",
        "suggestion": "Cluster logic or PROMOTION_THRESHOLD may be wrong.",
    },
    "avg_hops_per_retrieval": {
        "window": "short",
        "min": 0,
        "max": 5,
        "warmup_retrievals": 500,
        "severity": "WARNING",
        "suggestion": "Highways not forming for common queries. Review reinforcement.",
    },
    "orphan_rate": {
        "window": "medium",
        "min": 0,
        "max": 0.15,
        "warmup_days": 7,
        "severity": "WARNING",
        "suggestion": "Decay may be too aggressive, or bootstrap may be failing.",
    },
    "avg_conduit_weight": {
        "window": "medium",
        "min": 0.15,
        "max": 0.75,
        "warmup_days": 7,
        "severity": "WARNING",
        "suggestion": "Check learning/decay balance. System may be dying or over-concentrated.",
    },
    "retrieval_success_rate": {
        "window": "short",
        "min": 0.50,
        "max": None,
        "warmup_retrievals": 100,
        "severity": "WARNING",
        "suggestion": "Feature extractor producing bad features, or graph malformed.",
    },
    "fallback_trigger_rate": {
        "window": "short",
        "min": 0,
        "max": 0.20,
        "warmup_retrievals": 100,
        "severity": "WARNING",
        "suggestion": "Graph not learning effectively. Check reinforcement and promotion.",
    },
    "feedback_compliance_rate": {
        "window": "short",
        "min": 0.80,
        "max": None,
        "warmup_retrievals": 0,  # no warmup; compliance should be high from day 1
        "severity": "WARNING",
        "suggestion": "One or more callers are not calling flux_feedback once per returned grain.",
    },
    "core_grain_count": {
        "window": "long",
        "min": 0,
        "max": None,
        "warmup_days": 14,
        "severity": "INFO",
        "suggestion": "Promotion not firing, or aggressive decay on core class.",
    },
    "dormant_grain_rate": {
        "window": "long",
        "min": 0.05,
        "max": 0.40,
        "warmup_days": 14,
        "severity": "WARNING",
        "suggestion": "Too many grains being orphaned. Check bootstrap and reinforcement.",
    },
}


# ---------------------------------------------------------------- event logging

_CALLER_ROLE_LABELS = {
    "chat": "Chat",
    "memory_writer": "Memory Writer",
    "background_lookup": "Background Lookup",
    "system": "System",
    "admin": "Admin",
    "test": "Test",
    "legacy": "Legacy",
    "other": "Other",
}
_VALID_CALLER_ROLES = set(_CALLER_ROLE_LABELS)
_ROLE_ALIASES = {
    "memory_writer": "memory_writer",
    "memory_writing": "memory_writer",
    "memory": "memory_writer",
    "writer": "memory_writer",
    "background_lookup": "background_lookup",
    "background": "background_lookup",
    "ambient": "background_lookup",
    "ambient_suggestions": "background_lookup",
    "suggestions": "background_lookup",
}


def _slug(value: Any, default: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def _caller_role(value: Any, default: str = "chat") -> str:
    role = _slug(value, default).replace("-", "_")
    role = _ROLE_ALIASES.get(role, role)
    return role if role in _VALID_CALLER_ROLES else "other"


def _display_client(client: str) -> str:
    parts = [p for p in re.split(r"[-_:.]+", client) if p]
    return " ".join(part.capitalize() for part in parts) if parts else "Unknown"


def compose_caller_id(
    client: str | None = None,
    role: str | None = None,
    *,
    fallback: str | None = None,
) -> str:
    """Return the portable caller_id form: arbitrary client + controlled role."""
    if client or role:
        resolved_client = _slug(client or fallback, "unknown")
        resolved_role = _caller_role(role, "chat")
        return f"{resolved_client}:{resolved_role}"
    return str(fallback or "unknown").strip() or "unknown"


def caller_identity(
    caller_id: str | None,
    query: str | None = None,
    *,
    client: str | None = None,
    role: str | None = None,
) -> dict[str, str]:
    """Resolve legacy and portable caller attribution into client/role fields."""
    query_l = str(query or "").strip().lower()

    if client or role:
        client_id, role_id = compose_caller_id(client, role, fallback=caller_id).split(":", 1)
    else:
        raw = str(caller_id or "").strip()
        raw_l = raw.lower()
        ambient_prompt = "ambient suggestions" in query_l
        memory_prompt = "memory writing agent" in query_l

        if (
            raw_l == "ambient_suggestions"
            or raw_l.startswith("codex_ambient")
            or (raw_l in {"codex", "default", "unknown", "anonymous", ""} and ambient_prompt)
        ):
            client_id, role_id = "codex", "background_lookup"
        elif (
            raw_l == "memory_writing_agent"
            or (raw_l in {"codex", "default", "unknown", "anonymous", ""} and memory_prompt)
        ):
            client_id, role_id = "codex", "memory_writer"
        elif raw_l in {"", "default", "unknown"}:
            client_id, role_id = "unknown", "legacy"
        elif raw_l == "anonymous":
            client_id, role_id = "anonymous", "legacy"
        elif ":" in raw:
            raw_client, raw_role = raw.split(":", 1)
            client_id = _slug(raw_client, "unknown")
            role_id = _caller_role(raw_role, "chat")
        else:
            client_id = _slug(raw, "unknown")
            role_id = "chat"

    caller = f"{client_id}:{role_id}"
    return {
        "caller_id": caller,
        "client": client_id,
        "role": role_id,
        "display_name": f"{_display_client(client_id)} / {_CALLER_ROLE_LABELS[role_id]}",
    }

def log_event(
    store: FluxStore,
    category: str,
    event: str,
    data: dict[str, Any] | None = None,
    trace_id: str | None = None,
    now: datetime | None = None,
    caller_id: str | None = None,
) -> None:
    """Emit a structured event to the events table (§11.5).

    category is one of: retrieval | feedback | write | decay | cluster |
                         health | system | admin
    """
    now = now or utcnow()
    event_id = new_id()
    payload = dict(data or {})
    if caller_id is not None and "caller_id" not in payload:
        payload["caller_id"] = caller_id
    if "caller_id" in payload or "caller_client" in payload or "caller_role" in payload:
        identity = caller_identity(
            payload.get("caller_id"),
            payload.get("query"),
            client=payload.get("caller_client"),
            role=payload.get("caller_role"),
        )
        payload["caller_id"] = identity["caller_id"]
        payload["caller_client"] = identity["client"]
        payload["caller_role"] = identity["role"]
        payload["caller_display_name"] = identity["display_name"]
    store.conn.execute(
        """
        INSERT INTO events (id, timestamp, category, event, trace_id, data)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            now.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
            category,
            event,
            trace_id,
            json.dumps(payload),
        ),
    )
    logger.debug("flux event %s/%s %s", category, event, payload)


def setup_file_logger(log_dir: str | Path, level: int = logging.INFO) -> None:
    """Configure the rotating JSON log file (§11.5).

    Rotates at 100 MB, retains last 10 files. Optional — the events table
    is the primary durable store; the log file is for external tooling.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_dir / "flux.jsonl",
        maxBytes=100 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    flux_logger = logging.getLogger("flux")
    flux_logger.addHandler(handler)
    flux_logger.setLevel(level)


# ---------------------------------------------------------------- signal queries

def _count_retrievals(store: FluxStore, window_hours: float, now: datetime) -> int:
    cutoff = (now - timedelta(hours=window_hours)).strftime("%Y-%m-%dT%H:%M:%S")
    row = store.conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE category='retrieval' AND event='grains_returned' AND timestamp >= ?",
        (cutoff,),
    ).fetchone()
    return int(row["n"]) if row else 0


def _total_retrievals(store: FluxStore) -> int:
    row = store.conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE category='retrieval' AND event='grains_returned'"
    ).fetchone()
    return int(row["n"]) if row else 0


def _days_since_first_event(store: FluxStore, now: datetime) -> float:
    row = store.conn.execute("SELECT MIN(timestamp) AS t FROM events").fetchone()
    if not row or not row["t"]:
        return 0.0
    from .graph import parse_iso
    first = parse_iso(row["t"].replace("Z", "+00:00").replace("T", " ").split(".")[0])
    if first is None:
        return 0.0
    return (now - first).total_seconds() / 86400.0


def _compute_graph_signals(store: FluxStore) -> dict[str, float]:
    """Signals computable directly from the live graph (no event history needed)."""
    signals: dict[str, float] = {}

    # Highway count: conduits with stored weight >= 0.80 (lazy decay not applied
    # here for performance; health monitor is approximate by design).
    row = store.conn.execute(
        "SELECT COUNT(*) AS n FROM conduits WHERE weight >= 0.80"
    ).fetchone()
    signals["highway_count"] = float(row["n"] if row else 0)

    # Core grain count.
    row = store.conn.execute(
        "SELECT COUNT(*) AS n FROM grains WHERE decay_class='core'"
    ).fetchone()
    signals["core_grain_count"] = float(row["n"] if row else 0)

    # Orphan rate: % active grains with zero inbound conduits.
    active_row = store.conn.execute(
        "SELECT COUNT(*) AS n FROM grains WHERE status='active'"
    ).fetchone()
    active_total = int(active_row["n"]) if active_row else 0

    if active_total > 0:
        orphan_row = store.conn.execute(
            """
            SELECT COUNT(*) AS n FROM grains g
            WHERE g.status='active'
            AND NOT EXISTS (SELECT 1 FROM conduits c WHERE c.to_id = g.id)
            """
        ).fetchone()
        signals["orphan_rate"] = float(orphan_row["n"] if orphan_row else 0) / active_total
    else:
        signals["orphan_rate"] = 0.0

    # Avg conduit weight.
    w_row = store.conn.execute("SELECT AVG(weight) AS w FROM conduits").fetchone()
    signals["avg_conduit_weight"] = float(w_row["w"] or 0.0) if w_row else 0.0

    # Dormant grain rate: dormant / (active + dormant).
    dormant_row = store.conn.execute(
        "SELECT COUNT(*) AS n FROM grains WHERE status='dormant'"
    ).fetchone()
    dormant_n = int(dormant_row["n"]) if dormant_row else 0
    denom = active_total + dormant_n
    signals["dormant_grain_rate"] = dormant_n / denom if denom > 0 else 0.0

    return signals


def _event_payload(row: Any) -> dict[str, Any]:
    try:
        return json.loads(row["data"] or "{}")
    except Exception:
        return {}


def normalize_caller_id(caller_id: str | None, query: str | None = None) -> str:
    return caller_identity(caller_id, query)["caller_id"]


def _caller_id_from_payload(payload: dict[str, Any]) -> str:
    return caller_identity(
        payload.get("caller_id"),
        payload.get("query"),
        client=payload.get("caller_client"),
        role=payload.get("caller_role"),
    )["caller_id"]


def _caller_identity_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    return caller_identity(
        payload.get("caller_id"),
        payload.get("query"),
        client=payload.get("caller_client"),
        role=payload.get("caller_role"),
    )


def _parse_event_timestamp(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(timestamp.split(".")[0], "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return None


def _expected_feedback_count(payload: dict[str, Any], has_trace_id: bool) -> int:
    grain_ids = payload.get("grain_ids")
    if isinstance(grain_ids, list):
        return len([gid for gid in grain_ids if gid])

    for key in ("grains_count", "count"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            continue

    # Legacy traced retrieval events predate grain_ids/grains_count. Count them
    # as one expected feedback item so old clients still affect compliance.
    return 1 if has_trace_id else 0


def _received_feedback_count(
    retrieval_payload: dict[str, Any],
    feedback_payload: dict[str, Any],
    expected: int,
) -> int:
    returned_grain_ids = retrieval_payload.get("grain_ids")
    if isinstance(returned_grain_ids, list):
        returned = {str(gid) for gid in returned_grain_ids if gid}
        matched = len(returned & feedback_payload["grain_ids"])
        return min(
            matched + min(
                int(feedback_payload["anonymous_count"]),
                max(expected - matched, 0),
            ),
            expected,
        )
    return min(
        len(feedback_payload["grain_ids"]) + int(feedback_payload["anonymous_count"]),
        expected,
    )


def _feedback_by_trace_since(
    store: FluxStore,
    cutoff: str,
) -> dict[str, dict[str, Any]]:
    feedback_rows = store.conn.execute(
        """
        SELECT trace_id, data
        FROM events
        WHERE category='feedback'
          AND event='feedback_received'
          AND timestamp>=?
        """,
        (cutoff,),
    ).fetchall()

    feedback_by_trace: dict[str, dict[str, Any]] = {}
    for row in feedback_rows:
        trace_id = row["trace_id"]
        if not trace_id:
            continue
        payload = _event_payload(row)
        entry = feedback_by_trace.setdefault(trace_id, {
            "grain_ids": set(),
            "anonymous_count": 0,
        })
        grain_id = payload.get("grain_id")
        if grain_id:
            entry["grain_ids"].add(str(grain_id))
        else:
            entry["anonymous_count"] += 1
    return feedback_by_trace


def pending_feedback_for_caller(
    store: FluxStore,
    caller_id: str,
    now: datetime | None = None,
    *,
    grace_seconds: float = 60.0,
    max_block_seconds: float | None = None,
    lookback_days: int = 30,
) -> dict[str, Any]:
    now = now or utcnow()
    cutoff_dt = now - timedelta(days=lookback_days)
    cutoff = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S")
    target_caller = normalize_caller_id(caller_id)

    retrieval_rows = store.conn.execute(
        """
        SELECT timestamp, trace_id, data
        FROM events
        WHERE category='retrieval'
          AND event='grains_returned'
          AND timestamp>=?
        """,
        (cutoff,),
    ).fetchall()
    feedback_by_trace = _feedback_by_trace_since(store, cutoff)

    pending_traces: list[dict[str, Any]] = []
    expected_total = 0
    received_total = 0

    for row in retrieval_rows:
        trace_id = row["trace_id"]
        if not trace_id:
            continue
        payload = _event_payload(row)
        if _caller_id_from_payload(payload) != target_caller:
            continue
        event_time = _parse_event_timestamp(row["timestamp"])
        if event_time is None:
            continue
        age_seconds = (now - event_time).total_seconds()
        if age_seconds < grace_seconds:
            continue
        if max_block_seconds is not None and age_seconds > max_block_seconds:
            continue

        expected = _expected_feedback_count(payload, has_trace_id=True)
        if expected <= 0:
            continue

        received_payload = feedback_by_trace.get(trace_id, {
            "grain_ids": set(),
            "anonymous_count": 0,
        })
        received = _received_feedback_count(payload, received_payload, expected)
        missing = max(expected - received, 0)
        if missing <= 0:
            continue

        grain_ids = [
            str(gid)
            for gid in payload.get("grain_ids", [])
            if gid
        ] if isinstance(payload.get("grain_ids"), list) else []
        received_grain_ids = received_payload["grain_ids"]
        pending_grain_ids = [
            gid for gid in grain_ids
            if gid not in received_grain_ids
        ][:missing]

        expected_total += expected
        received_total += received
        pending_traces.append({
            "trace_id": trace_id,
            "expected": expected,
            "received": received,
            "missing": missing,
            "age_seconds": round(age_seconds, 3),
            "grain_ids": grain_ids,
            "pending_grain_ids": pending_grain_ids,
        })

    missing_total = sum(item["missing"] for item in pending_traces)
    return {
        "caller_id": target_caller,
        "pending_traces": pending_traces,
        "expected": expected_total,
        "received": received_total,
        "missing": missing_total,
    }


def _feedback_compliance_by_caller(
    store: FluxStore,
    cutoff: str,
) -> dict[str, dict[str, float]]:
    retrieval_rows = store.conn.execute(
        """
        SELECT trace_id, data
        FROM events
        WHERE category='retrieval'
          AND event='grains_returned'
          AND timestamp>=?
        """,
        (cutoff,),
    ).fetchall()

    feedback_by_trace = _feedback_by_trace_since(store, cutoff)
    untraced_feedback_row = store.conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM events
        WHERE category='feedback'
          AND event='feedback_received'
          AND timestamp>=?
          AND (trace_id IS NULL OR trace_id = '')
        """,
        (cutoff,),
    ).fetchone()
    untraced_feedback_count = int(untraced_feedback_row["n"] or 0) if untraced_feedback_row else 0

    stats: dict[str, dict[str, float]] = {}
    for row in retrieval_rows:
        trace_id = row["trace_id"]
        payload = _event_payload(row)
        identity = _caller_identity_from_payload(payload)
        caller_id = identity["caller_id"]
        entry = stats.setdefault(caller_id, {
            "expected": 0.0,
            "received": 0.0,
            "missing": 0.0,
            "retrievals": 0.0,
            "client": identity["client"],
            "role": identity["role"],
            "display_name": identity["display_name"],
        })
        entry["retrievals"] += 1

        expected = _expected_feedback_count(payload, has_trace_id=bool(trace_id))
        if expected <= 0:
            continue

        if not trace_id:
            received = min(untraced_feedback_count, expected)
            untraced_feedback_count = max(untraced_feedback_count - expected, 0)
        else:
            received_payload = feedback_by_trace.get(trace_id, {
                "grain_ids": set(),
                "anonymous_count": 0,
            })
            received = _received_feedback_count(payload, received_payload, expected)

        entry["expected"] += expected
        entry["received"] += received
        entry["missing"] += max(expected - received, 0)

    for entry in stats.values():
        expected = entry["expected"]
        entry["rate"] = (entry["received"] / expected) if expected > 0 else 1.0

    return stats


def _feedback_compliance_rate(
    store: FluxStore,
    cutoff: str,
) -> float:
    stats = _feedback_compliance_by_caller(store, cutoff)
    expected = sum(item["expected"] for item in stats.values())
    if expected <= 0:
        return 1.0
    received = sum(item["received"] for item in stats.values())
    return min(received / expected, 1.0)


def _caller_feedback_summary(
    store: FluxStore,
    cutoff: str,
) -> list[dict[str, Any]]:
    stats = _feedback_compliance_by_caller(store, cutoff)
    summary = []
    for caller_id, item in stats.items():
        rate = float(item.get("rate", 1.0))
        summary.append({
            "caller_id": caller_id,
            "client": item["client"],
            "role": item["role"],
            "display_name": item["display_name"],
            "rate": rate,
            "healthy": rate >= 0.80,
            "expected": item["expected"],
            "received": item["received"],
            "missing": item["missing"],
            "retrievals": item["retrievals"],
        })
    return sorted(summary, key=lambda c: (c["healthy"], c["rate"], c["caller_id"]))


def _compute_event_signals(store: FluxStore, now: datetime) -> dict[str, float]:
    """Signals derived from the event log."""
    signals: dict[str, float] = {}
    SHORT = 24.0   # hours (last 100 retrievals proxy: use 24h)
    MEDIUM = 24.0  # hours
    LONG = 168.0   # hours (7 days)

    cutoff_short = (now - timedelta(hours=SHORT)).strftime("%Y-%m-%dT%H:%M:%S")
    cutoff_medium = (now - timedelta(hours=MEDIUM)).strftime("%Y-%m-%dT%H:%M:%S")
    cutoff_long = (now - timedelta(hours=LONG)).strftime("%Y-%m-%dT%H:%M:%S")

    def _count(category: str, event: str, cutoff: str) -> int:
        r = store.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE category=? AND event=? AND timestamp>=?",
            (category, event, cutoff),
        ).fetchone()
        return int(r["n"]) if r else 0

    def _count_trace_scoped(category: str, event: str, cutoff: str) -> int:
        """Count one event per trace when trace_id exists, plus legacy untraced events."""
        r = store.conn.execute(
            """
            SELECT
              COUNT(DISTINCT NULLIF(trace_id, '')) AS traced,
              SUM(CASE WHEN trace_id IS NULL OR trace_id = '' THEN 1 ELSE 0 END) AS untraced
            FROM events
            WHERE category=? AND event=? AND timestamp>=?
            """,
            (category, event, cutoff),
        ).fetchone()
        if not r:
            return 0
        return int(r["traced"] or 0) + int(r["untraced"] or 0)

    def _count_successful_retrievals(cutoff: str) -> int:
        """Count retrievals that received at least one useful feedback event."""
        traced = store.conn.execute(
            """
            SELECT COUNT(DISTINCT r.trace_id) AS n
            FROM events r
            WHERE r.category='retrieval'
              AND r.event='grains_returned'
              AND r.timestamp>=?
              AND r.trace_id IS NOT NULL
              AND r.trace_id <> ''
              AND EXISTS (
                SELECT 1
                FROM events f
                WHERE f.category='feedback'
                  AND f.event='retrieval_successful'
                  AND f.timestamp>=?
                  AND f.trace_id = r.trace_id
              )
            """,
            (cutoff, cutoff),
        ).fetchone()
        untraced_retrievals = store.conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM events
            WHERE category='retrieval'
              AND event='grains_returned'
              AND timestamp>=?
              AND (trace_id IS NULL OR trace_id = '')
            """,
            (cutoff,),
        ).fetchone()
        untraced_successes = store.conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM events
            WHERE category='feedback'
              AND event='retrieval_successful'
              AND timestamp>=?
              AND (trace_id IS NULL OR trace_id = '')
            """,
            (cutoff,),
        ).fetchone()
        legacy_successes = min(
            int(untraced_retrievals["n"] or 0) if untraced_retrievals else 0,
            int(untraced_successes["n"] or 0) if untraced_successes else 0,
        )
        return (int(traced["n"] or 0) if traced else 0) + legacy_successes

    # Highway growth rate: new conduits that crossed weight 0.80 (proxied by
    # reinforce events that set weight above 0.80, logged in feedback events).
    hgr = store.conn.execute(
        """
        SELECT COUNT(*) AS n FROM events
        WHERE category='feedback' AND event='highway_formed' AND timestamp >= ?
        """,
        (cutoff_long,),
    ).fetchone()
    signals["highway_growth_rate"] = float(hgr["n"] if hgr else 0)

    # Shortcut creation rate: shortcuts created per 100 retrievals.
    shortcuts = _count("feedback", "shortcut_created", cutoff_short)
    retrievals_short = _count("retrieval", "grains_returned", cutoff_short) or 1
    signals["shortcut_creation_rate"] = (shortcuts / retrievals_short) * 100

    # Conduit dissolution rate: conduits deleted per decay pass in last 24h.
    dissolution = store.conn.execute(
        """
        SELECT SUM(CAST(json_extract(data, '$.conduits_deleted') AS REAL)) AS s
        FROM events WHERE category='decay' AND event='cleanup_pass_completed' AND timestamp >= ?
        """,
        (cutoff_medium,),
    ).fetchone()
    signals["conduit_dissolution_rate"] = float(dissolution["s"] or 0.0) if dissolution else 0.0

    # Avg weight drop on failure: mean absolute weight change from penalize events.
    drop = store.conn.execute(
        """
        SELECT AVG(CAST(json_extract(data, '$.weight_drop') AS REAL)) AS a
        FROM events WHERE category='feedback' AND event='conduit_penalized' AND timestamp >= ?
        """,
        (cutoff_short,),
    ).fetchone()
    signals["avg_weight_drop_on_failure"] = float(drop["a"] or 0.0) if drop else 0.0

    # Promotion events per week.
    signals["promotion_events"] = float(_count("feedback", "promotion_triggered", cutoff_long))

    # Avg hops per retrieval.
    hops = store.conn.execute(
        """
        SELECT AVG(CAST(json_extract(data, '$.hop_count') AS REAL)) AS a
        FROM events WHERE category='retrieval' AND event='grains_returned' AND timestamp >= ?
        """,
        (cutoff_short,),
    ).fetchone()
    signals["avg_hops_per_retrieval"] = float(hops["a"] or 0.0) if hops else 0.0

    # Retrieval success rate: % where at least one grain was marked useful.
    ret_total = _count_trace_scoped("retrieval", "grains_returned", cutoff_short)
    ret_success = _count_successful_retrievals(cutoff_short)
    signals["retrieval_success_rate"] = min(ret_success / ret_total, 1.0) if ret_total > 0 else 0.0

    # Fallback trigger rate.
    fallbacks = _count("retrieval", "fallback_triggered", cutoff_short)
    signals["fallback_trigger_rate"] = fallbacks / ret_total if ret_total > 0 else 0.0

    # Feedback compliance: % of returned grains that received feedback.
    signals["feedback_compliance_rate"] = _feedback_compliance_rate(store, cutoff_short)

    return signals


# ---------------------------------------------------------------- health checks

def _is_healthy(signal: str, value: float, cfg: Config) -> bool:
    spec = _HEALTHY_RANGES.get(signal)
    if spec is None:
        return True
    lo = spec.get("min")
    hi = spec.get("max")
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def _healthy_range_str(signal: str) -> str:
    spec = _HEALTHY_RANGES.get(signal, {})
    lo, hi = spec.get("min"), spec.get("max")
    if lo is not None and hi is not None:
        return f"{lo} – {hi}"
    if lo is not None:
        return f">= {lo}"
    if hi is not None:
        return f"<= {hi}"
    return "any"


def _in_warmup(signal: str, store: FluxStore, now: datetime) -> bool:
    """Return True if the signal is still in its warmup period (§12.3)."""
    spec = _HEALTHY_RANGES.get(signal, {})
    warmup_retrievals = spec.get("warmup_retrievals")
    warmup_days = spec.get("warmup_days")
    if warmup_retrievals and _total_retrievals(store) < warmup_retrievals:
        return True
    if warmup_days and _days_since_first_event(store, now) < warmup_days:
        return True
    return False


def _upsert_warning(
    store: FluxStore,
    signal: str,
    value: float,
    severity: str,
    healthy_range: str,
    suggestion: str,
    now: datetime,
) -> None:
    existing = store.conn.execute(
        "SELECT id FROM warnings WHERE signal = ?",
        (signal,),
    ).fetchone()
    ts = now.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    if existing:
        store.conn.execute(
            """
            UPDATE warnings
            SET current_value=?, severity=?, healthy_range=?, last_seen=?,
                suggestion=?, cleared_at=NULL
            WHERE id=?
            """,
            (value, severity, healthy_range, ts, suggestion, existing["id"]),
        )
    else:
        store.conn.execute(
            """
            INSERT INTO warnings (id, signal, severity, current_value, healthy_range,
                                  first_seen, last_seen, suggestion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (new_id(), signal, severity, value, healthy_range, ts, ts, suggestion),
        )


def _clear_warning(store: FluxStore, signal: str, now: datetime) -> None:
    ts = now.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    store.conn.execute(
        "UPDATE warnings SET cleared_at=? WHERE signal=? AND cleared_at IS NULL",
        (ts, signal),
    )


def _get_active_warnings(store: FluxStore) -> list[dict]:
    rows = store.conn.execute(
        "SELECT * FROM warnings WHERE cleared_at IS NULL ORDER BY severity DESC, first_seen ASC"
    ).fetchall()
    result = []
    for r in rows:
        result.append({
            "signal": r["signal"],
            "severity": r["severity"],
            "current_value": r["current_value"],
            "healthy_range": r["healthy_range"],
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "suggestion": r["suggestion"],
        })
    return result


def _sync_feedback_caller_warnings(store: FluxStore, now: datetime) -> None:
    """Clear legacy per-caller warning rows.

    Caller attribution is returned in the health payload as ``caller_feedback``.
    Keeping each caller as a normal active warning floods the warning count and
    makes the dashboard look worse without adding diagnostic value.
    """
    prefix = "feedback_compliance_rate:"
    existing = store.conn.execute(
        """
        SELECT signal FROM warnings
        WHERE signal LIKE ? AND cleared_at IS NULL
        """,
        (f"{prefix}%",),
    ).fetchall()
    for row in existing:
        _clear_warning(store, row["signal"], now)


def _overall_status(warnings: list[dict]) -> str:
    if any(w["severity"] == "CRITICAL" for w in warnings):
        return "critical"
    if any(w["severity"] == "WARNING" for w in warnings):
        return "warning"
    return "healthy"


# ------------------------------------------------------------------ public API

def flux_health(
    store: FluxStore,
    cfg: Config = DEFAULT_CONFIG,
    now: datetime | None = None,
) -> dict:
    """Compute health signals and return the status dict (§12.5).

    Signals are computed, compared to healthy ranges, warnings upserted or
    cleared, and the full dict returned. Suitable for calling by the main AI,
    dashboards, or monitoring scripts.
    """
    now = now or utcnow()

    graph_signals = _compute_graph_signals(store)
    event_signals = _compute_event_signals(store, now)
    all_signals = {**graph_signals, **event_signals}
    cutoff_short = (now - timedelta(hours=24.0)).strftime("%Y-%m-%dT%H:%M:%S")

    signal_results: dict[str, dict] = {}
    for name, value in all_signals.items():
        healthy = _is_healthy(name, value, cfg)
        signal_results[name] = {"value": value, "healthy": healthy}

        spec = _HEALTHY_RANGES.get(name, {})
        if not healthy and not _in_warmup(name, store, now):
            _upsert_warning(
                store,
                signal=name,
                value=value,
                severity=spec.get("severity", "WARNING"),
                healthy_range=_healthy_range_str(name),
                suggestion=spec.get("suggestion", ""),
                now=now,
            )
        else:
            _clear_warning(store, name, now)

    _sync_feedback_caller_warnings(store, now)

    active_warnings = _get_active_warnings(store)
    status = _overall_status(active_warnings)

    log_event(store, "health", "health_computed", {
        "status": status,
        "warning_count": len(active_warnings),
    }, now=now)

    return {
        "status": status,
        "signals": signal_results,
        "active_warnings": active_warnings,
        "caller_feedback": _caller_feedback_summary(store, cutoff_short),
        "computed_at": now.isoformat(),
    }
