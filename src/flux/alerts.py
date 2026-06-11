"""Push notifications for health warnings.

When a health signal first goes unhealthy (or re-fires after clearing), Flux
can push a message to a Telegram chat so warnings are seen instead of sitting
in the warnings table until someone asks.

Design constraints:
  - Alerts must NEVER break health computation: every failure is logged and
    swallowed.
  - Throttled per signal via the meta table (ALERT_MIN_INTERVAL_HOURS), so
    multiple processes computing health concurrently do not spam.
  - Credentials live in the instance flux.yaml (ALERT_TELEGRAM_BOT_TOKEN,
    ALERT_TELEGRAM_CHAT_ID), never in the repo config.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timedelta

from .config import Config
from .storage import FluxStore

logger = logging.getLogger(__name__)

_META_KEY_PREFIX = "alert_last_sent:"
_TELEGRAM_TIMEOUT_SECONDS = 5.0


def _ensure_meta_table(store: FluxStore) -> None:
    store.conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
    )


def maybe_send_warning_alert(
    store: FluxStore,
    cfg: Config,
    *,
    signal: str,
    severity: str,
    value: float,
    healthy_range: str,
    suggestion: str,
    transition: str,
    now: datetime,
) -> bool:
    """Send a Telegram alert for a warning transition, if configured.

    transition: "new" (first time unhealthy), "refired" (unhealthy again after
    clearing), or "ongoing" (still unhealthy — never alerts).

    Returns True if an alert was sent.
    """
    if not cfg.ALERTS_ENABLED:
        return False
    if transition not in ("new", "refired"):
        return False
    if not cfg.ALERT_TELEGRAM_BOT_TOKEN or not cfg.ALERT_TELEGRAM_CHAT_ID:
        logger.warning("alerts: ALERTS_ENABLED but Telegram credentials missing")
        return False
    if _throttled(store, signal, cfg, now):
        return False

    fired = "re-fired" if transition == "refired" else "fired"
    text = (
        f"⚠️ Flux health warning {fired}\n"
        f"signal: {signal}\n"
        f"severity: {severity}\n"
        f"value: {value:.4g} (healthy: {healthy_range})\n"
        f"{suggestion}"
    )
    if not _send_telegram(cfg, text):
        return False

    _record_sent(store, signal, now)
    return True


def _throttled(store: FluxStore, signal: str, cfg: Config, now: datetime) -> bool:
    _ensure_meta_table(store)
    row = store.conn.execute(
        "SELECT value FROM meta WHERE key = ?",
        (_META_KEY_PREFIX + signal,),
    ).fetchone()
    if row is None:
        return False
    try:
        last_sent = datetime.fromisoformat(row["value"])
    except (ValueError, TypeError):
        return False
    return now - last_sent < timedelta(hours=cfg.ALERT_MIN_INTERVAL_HOURS)


def _record_sent(store: FluxStore, signal: str, now: datetime) -> None:
    store.conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (_META_KEY_PREFIX + signal, now.isoformat()),
    )


def _send_telegram(cfg: Config, text: str) -> bool:
    url = f"https://api.telegram.org/bot{cfg.ALERT_TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id": cfg.ALERT_TELEGRAM_CHAT_ID,
        "text": text,
    }).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_TELEGRAM_TIMEOUT_SECONDS) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                logger.error("alerts: Telegram returned HTTP %s", resp.status)
            return ok
    except Exception as exc:
        logger.error("alerts: Telegram send failed: %s", exc)
        return False
