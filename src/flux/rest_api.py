"""REST API for Flux Memory (§1A.4).

Exposes all public operations over HTTP using FastAPI.
Runs alongside the dashboard and Python SDK. MCP clients launch the stdio
MCP server separately with `flux mcp --name <instance>`.

Endpoints:
  POST /store              store a grain (or batch)
  POST /retrieve           retrieve grains for a query
  POST /feedback           apply feedback (async-queued)
  GET  /health             current health report
  GET  /grains             list grains, optional ?status= filter

All requests accept caller attribution headers for per-caller tracking.

Usage (programmatic):
    from flux.rest_api import build_app
    app = build_app(service)
    # run with: uvicorn flux.rest_api:app --port 7465

Usage (via flux CLI):
    flux start  # launches this at cfg.REST_PORT
"""
import logging
from typing import Any, List

from .config import Config, DEFAULT_CONFIG
from .health import compose_caller_id
from .service import FluxService

logger = logging.getLogger(__name__)

_CALLER_HEADER = "X-Caller-Id"
_CALLER_CLIENT_HEADER = "X-Flux-Client"
_CALLER_ROLE_HEADER = "X-Flux-Role"
_DEFAULT_CALLER = "anonymous"

try:
    from fastapi import FastAPI, HTTPException, Request
    from pydantic import BaseModel

    class CallerMixin(BaseModel):
        # Caller identity may arrive in the body instead of headers —
        # several integrations (e.g. the Mirror daemon) send it this way.
        caller_id: str | None = None
        client: str | None = None
        role: str | None = None

    class StoreRequest(CallerMixin):
        content: str
        provenance: str = "ai_stated"

    class StoreBatchRequest(CallerMixin):
        items: List[dict]

    class RetrieveRequest(CallerMixin):
        query: str

    class FeedbackRequest(CallerMixin):
        trace_id: str
        grain_id: str
        useful: bool
        strength: float = 1.0

    class FeedbackBatchItem(BaseModel):
        trace_id: str
        grain_id: str
        useful: bool
        strength: float = 1.0

    class FeedbackBatchRequest(BaseModel):
        items: List[FeedbackBatchItem]

except ImportError:
    pass  # build_app will raise with a helpful message if called without fastapi


def build_app(service: "FluxService", cfg: "Config" = DEFAULT_CONFIG):
    """Construct and return a FastAPI application wired to the given service."""
    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:
        raise ImportError(
            "REST API requires 'fastapi' and 'uvicorn'. "
            "Install: pip install fastapi uvicorn"
        ) from exc

    app = FastAPI(
        title="Flux Memory",
        description="Self-organizing retrieval fabric for AI memory.",
        version="0.6.2",
    )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _caller(request: Request, body=None) -> str:
        # Precedence: headers > body fields > anonymous.
        client = request.headers.get(_CALLER_CLIENT_HEADER) or getattr(body, "client", None)
        role = request.headers.get(_CALLER_ROLE_HEADER) or getattr(body, "role", None)
        fallback = (request.headers.get(_CALLER_HEADER)
                    or getattr(body, "caller_id", None)
                    or _DEFAULT_CALLER)
        return compose_caller_id(client, role, fallback=fallback)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.get("/")
    def root():
        return {
            "name": "Flux Memory",
            "version": "0.6.2",
            "health": "/health",
            "docs": "/docs",
            "endpoints": {
                "store": "/store",
                "store_batch": "/store/batch",
                "retrieve": "/retrieve",
                "feedback": "/feedback",
                "grains": "/grains",
            },
        }

    @app.post("/store")
    def store(body: StoreRequest, request: Request):
        caller_id = _caller(request, body)
        try:
            grain_id, status = service.store_ex(body.content, body.provenance, caller_id=caller_id)
            return {"grain_id": grain_id, "status": status}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except RuntimeError as exc:
            msg = str(exc)
            if "queue is full" in msg:
                raise HTTPException(status_code=503, detail=msg)
            if "Rate limit" in msg:
                raise HTTPException(status_code=429, detail=msg)
            raise HTTPException(status_code=500, detail=msg)

    @app.post("/store/batch")
    def store_batch(body: StoreBatchRequest, request: Request):
        caller_id = _caller(request, body)
        try:
            ids = service.store_batch(body.items, caller_id=caller_id)
            return {"grain_ids": ids, "count": len(ids)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except RuntimeError as exc:
            msg = str(exc)
            code = 503 if "queue is full" in msg else (429 if "Rate limit" in msg else 500)
            raise HTTPException(status_code=code, detail=msg)

    @app.post("/retrieve")
    def retrieve(body: RetrieveRequest, request: Request):
        caller_id = _caller(request, body)
        try:
            result = service.retrieve(body.query, caller_id=caller_id)
            return {
                "grains": result.grains,
                "trace_id": result.trace_id,
                "confidence": result.confidence,
                "fallback_triggered": result.fallback_triggered,
                "hop_count": result.hop_count,
                "features": result.features,
                "expansion_candidates": result.expansion_candidates,
            }
        except Exception as exc:
            logger.error("retrieve error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/feedback")
    def feedback(body: FeedbackRequest, request: Request):
        caller_id = _caller(request, body)
        result = service.feedback_sync(body.trace_id, body.grain_id, body.useful,
                                       caller_id=caller_id, strength=body.strength)
        return {
            "status": "ok",
            "trace_id": result.trace_id,
            "grain_id": result.grain_id,
            "action": result.action,
            "signal": result.effective_signal,
        }

    @app.post("/feedback/batch")
    def feedback_batch(body: FeedbackBatchRequest, request: Request):
        caller_id = _caller(request, body)
        items = [
            {"trace_id": item.trace_id, "grain_id": item.grain_id,
             "useful": item.useful, "strength": item.strength}
            for item in body.items
        ]
        results = service.feedback_batch_sync(items, caller_id=caller_id)
        return {
            "status": "ok",
            "count": len(results),
            "results": [
                {
                    "trace_id": r.trace_id,
                    "grain_id": r.grain_id,
                    "signal": r.effective_signal,
                }
                for r in results
            ],
        }

    @app.get("/health")
    def health():
        return service.health()

    @app.get("/pending_feedback")
    def pending_feedback(request: Request, target_caller_id: str | None = None):
        from .health import normalize_caller_id
        target = normalize_caller_id(target_caller_id or _caller(request))
        return service.pending_feedback(target)

    @app.get("/grains")
    def list_grains(status: str | None = None, limit: int = 50):
        try:
            return {"grains": service.list_grains(status=status, limit=limit)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    return app
