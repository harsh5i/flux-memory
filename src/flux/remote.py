"""Thin HTTP client that lets `flux mcp` proxy to a running Flux service.

When the long-running service (`flux start`) is up, MCP server processes
should NOT open the SQLite database themselves — concurrent writers holding
transactions have repeatedly blocked maintenance and caused lock contention.
RemoteService implements the same surface FluxService exposes to the MCP
dispatch layer, backed by the service's REST API, so the daemon stays the
single writer.

stdlib urllib only — no new dependencies in the MCP path.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from .config import Config, DEFAULT_CONFIG
from .retrieval import FeedbackResult, RetrievalResult

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS = 1.5
_REQUEST_TIMEOUT_SECONDS = 60.0


def service_base_url(cfg: Config) -> str:
    host = cfg.REST_HOST if cfg.REST_HOST != "0.0.0.0" else "127.0.0.1"
    return f"http://{host}:{cfg.REST_PORT}"


def probe_service(cfg: Config = DEFAULT_CONFIG) -> bool:
    """Return True if a Flux REST service is answering on the configured port."""
    try:
        with urllib.request.urlopen(
            service_base_url(cfg) + "/", timeout=_PROBE_TIMEOUT_SECONDS
        ) as resp:
            if resp.status != 200:
                return False
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("name") == "Flux Memory"
    except Exception:
        return False


class RemoteService:
    """FluxService-compatible facade over the REST API (single-writer daemon)."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG) -> None:
        self._base = service_base_url(cfg)

    # ------------------------------------------------------------- plumbing

    def _request(self, method: str, path: str, payload: dict | None = None,
                 caller_id: str = "default") -> dict:
        url = self._base + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url, data=data, method=method,
            headers={
                "Content-Type": "application/json",
                "X-Caller-Id": caller_id,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except Exception:
                pass
            if exc.code == 429:
                raise RuntimeError(detail or "Rate limit exceeded")
            if exc.code == 503:
                raise RuntimeError(detail or "Flux write queue is full")
            if exc.code == 422:
                raise ValueError(detail or "Invalid request")
            raise RuntimeError(f"Flux service error {exc.code}: {detail}")

    # ----------------------------------------------------------- write path

    def store_ex(self, content: str, provenance: str = "ai_stated",
                 caller_id: str = "default") -> tuple[str, str]:
        body = self._request("POST", "/store", {
            "content": content, "provenance": provenance,
        }, caller_id=caller_id)
        return body["grain_id"], body["status"]

    def store(self, content: str, provenance: str = "ai_stated",
              caller_id: str = "default") -> str:
        return self.store_ex(content, provenance, caller_id=caller_id)[0]

    # ------------------------------------------------------------ read path

    def retrieve(self, query: str, caller_id: str = "default") -> RetrievalResult:
        body = self._request("POST", "/retrieve", {"query": query},
                             caller_id=caller_id)
        return RetrievalResult(
            grains=body.get("grains", []),
            trace_id=body.get("trace_id", ""),
            confidence=body.get("confidence", 0.0),
            fallback_triggered=body.get("fallback_triggered", False),
            hop_count=body.get("hop_count", 0),
            features=body.get("features", []),
            expansion_candidates=body.get("expansion_candidates", []),
        )

    def feedback_sync(self, trace_id: str, grain_id: str, useful: bool,
                      caller_id: str = "default",
                      strength: float = 1.0) -> FeedbackResult:
        body = self._request("POST", "/feedback", {
            "trace_id": trace_id, "grain_id": grain_id,
            "useful": useful, "strength": strength,
        }, caller_id=caller_id)
        return FeedbackResult(
            trace_id=body.get("trace_id", trace_id),
            grain_id=body.get("grain_id", grain_id),
            useful=useful,
            effective_signal=body.get("signal", 0.0),
            action=body.get("action", "unknown"),
        )

    # -------------------------------------------------------- observability

    def health(self) -> dict:
        return self._request("GET", "/health")

    def list_grains(self, status: str | None = None, limit: int = 50) -> list[dict]:
        params = {"limit": str(limit)}
        if status:
            params["status"] = status
        body = self._request("GET", "/grains?" + urllib.parse.urlencode(params))
        return body.get("grains", [])

    def pending_feedback(self, target_caller_id: str) -> dict:
        params = urllib.parse.urlencode({"target_caller_id": target_caller_id})
        return self._request("GET", "/pending_feedback?" + params)
