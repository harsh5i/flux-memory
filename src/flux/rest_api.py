"""REST API for Flux Memory (§1A.4).

Exposes all public operations over HTTP using FastAPI.
Runs alongside the MCP server and Python SDK — same operations, three equal paths.

Endpoints:
  POST /store              store a grain (or batch)
  POST /retrieve           retrieve grains for a query
  POST /feedback           apply feedback (async-queued)
  GET  /health             current health report
  GET  /grains             list grains, optional ?status= filter

All requests accept an optional X-Caller-Id header for per-caller tracking.

Usage (programmatic):
    from flux.rest_api import build_app
    app = build_app(service)
    # run with: uvicorn flux.rest_api:app --port 7465

Usage (via flux CLI):
    flux start  # launches this at cfg.REST_PORT
"""
from __future__ import annotations

import logging
from typing import Any

from .config import Config, DEFAULT_CONFIG
from .service import FluxService

logger = logging.getLogger(__name__)

_CALLER_HEADER = "X-Caller-Id"
_DEFAULT_CALLER = "anonymous"


def build_app(service: FluxService, cfg: Config = DEFAULT_CONFIG):
    """Construct and return a FastAPI application wired to the given service."""
    try:
        from fastapi import FastAPI, Header, HTTPException, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError as exc:
        raise ImportError(
            "REST API requires 'fastapi' and 'uvicorn'. "
            "Install: pip install fastapi uvicorn"
        ) from exc

    app = FastAPI(
        title="Flux Memory",
        description="Self-organizing retrieval fabric for AI memory.",
        version="0.6.0",
    )

    # ------------------------------------------------------------------
    # Request / response models
    # ------------------------------------------------------------------

    class StoreRequest(BaseModel):
        content: str
        provenance: str = "ai_stated"

    class StoreBatchRequest(BaseModel):
        items: list[dict]

    class RetrieveRequest(BaseModel):
        query: str

    class FeedbackRequest(BaseModel):
        trace_id: str
        grain_id: str
        useful: bool

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _caller(request: Request) -> str:
        return request.headers.get(_CALLER_HEADER, _DEFAULT_CALLER)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.post("/store")
    def store(body: StoreRequest, request: Request):
        caller_id = _caller(request)
        try:
            grain_id = service.store(body.content, body.provenance, caller_id=caller_id)
            return {"grain_id": grain_id, "status": "stored"}
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
        caller_id = _caller(request)
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
        caller_id = _caller(request)
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
        caller_id = _caller(request)
        service.feedback(body.trace_id, body.grain_id, body.useful, caller_id=caller_id)
        return {"status": "queued"}

    @app.get("/health")
    def health():
        return service.health()

    @app.get("/grains")
    def list_grains(status: str | None = None, limit: int = 50):
        try:
            return {"grains": service.list_grains(status=status, limit=limit)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    return app
