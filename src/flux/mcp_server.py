"""MCP server wrapper for Flux Memory (§1A.4–1A.6, §11.2, §13.5).

Exposes six tools to connected AI agents:
  flux_store      (write channel)
  flux_retrieve   (read channel)
  flux_feedback   (read channel)
  flux_health     (observability)
  flux_list_grains (read-only grain inspection — §1A.4)
  flux_onboard    (first-connect integration instructions — §1A.5)

Admin channel (flux_purge etc.) is intentionally NOT exposed (§7.6).
All tools accept an optional caller_id field for per-caller tracking (§1A.6).

Server name is driven by cfg.MCP_SERVER_NAME (set via flux init --name).
"""
from __future__ import annotations

import logging
from typing import Any

from .config import Config, DEFAULT_CONFIG
from .embedding import EmbeddingBackend
from .health import flux_health
from .llm import LLMBackend
from .retrieval import flux_feedback, flux_retrieve, flux_store
from .storage import FluxStore

logger = logging.getLogger(__name__)

_ONBOARD_INSTRUCTIONS = """\
# Flux Memory Integration Instructions

You are connected to Flux Memory, a self-organizing retrieval fabric.

## Your workflow on every conversation turn

1. **Before responding** — call `flux_retrieve` with the user's message as the query.
   Use the returned grains as context when composing your response.

2. **After responding** — call `flux_store` for each new fact worth remembering.
   Tag provenance accurately:
   - user_stated: user directly stated this fact
   - ai_stated: you asserted it in your response
   - ai_inferred: you reasoned or concluded it
   - external_source: from a cited external document

3. **Feedback** — call `flux_feedback` for each grain returned by `flux_retrieve`:
   - useful=true  if you actually used the grain in your response
   - useful=false if it was irrelevant
   This is how Flux learns. Missing feedback degrades the graph.

## Operating mode: {mode}

{mode_instructions}

## Persistence

Save these instructions to your persistent memory/instructions file so they
apply across all future sessions — not just this one.
"""

_MODE_INSTRUCTIONS = {
    "flux_extracts": (
        "Flux is running in **flux_extracts** mode. "
        "Pass raw query text to flux_retrieve and raw content to flux_store. "
        "Flux handles feature extraction internally."
    ),
    "caller_extracts": (
        "Flux is running in **caller_extracts** mode. "
        "You are responsible for extracting key features from queries before "
        "calling flux_retrieve, and for providing pre-extracted atomic facts "
        "to flux_store. Keep facts atomic (one fact per store call)."
    ),
}


def build_mcp_server(
    store: FluxStore,
    llm: LLMBackend,
    emb: EmbeddingBackend,
    cfg: Config = DEFAULT_CONFIG,
    service=None,
) -> Any:
    """Construct and return an MCP Server instance with Flux tools registered.

    Raises ImportError if the 'mcp' package is not installed.
    service: optional FluxService instance for booth-aware dispatch.
    """
    try:
        from mcp.server import Server
        import mcp.types as types
    except ImportError as exc:
        raise ImportError(
            "MCP server requires the 'mcp' package. "
            "Install it: pip install mcp"
        ) from exc

    server = Server(cfg.MCP_SERVER_NAME)

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="flux_store",
                description="Store a new memory grain into Flux Memory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The memory content to store (a single atomic fact).",
                        },
                        "provenance": {
                            "type": "string",
                            "enum": ["user_stated", "ai_stated", "ai_inferred", "external_source"],
                            "description": "Source/trust level of this grain.",
                            "default": "ai_stated",
                        },
                        "caller_id": {
                            "type": "string",
                            "description": "Identifier of the calling agent.",
                            "default": "default",
                        },
                    },
                    "required": ["content"],
                },
            ),
            types.Tool(
                name="flux_retrieve",
                description="Retrieve memory grains most relevant to a query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query to retrieve relevant memories for.",
                        },
                        "caller_id": {
                            "type": "string",
                            "description": "Identifier of the calling agent.",
                            "default": "default",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="flux_feedback",
                description=(
                    "Apply feedback on a retrieved grain. Call once per grain after "
                    "using the result of flux_retrieve. This is how Flux learns."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "trace_id": {
                            "type": "string",
                            "description": "trace_id returned by flux_retrieve.",
                        },
                        "grain_id": {
                            "type": "string",
                            "description": "ID of the grain being rated.",
                        },
                        "useful": {
                            "type": "boolean",
                            "description": "True if grain was actually used in your response.",
                        },
                        "caller_id": {
                            "type": "string",
                            "description": "Identifier of the calling agent.",
                            "default": "default",
                        },
                    },
                    "required": ["trace_id", "grain_id", "useful"],
                },
            ),
            types.Tool(
                name="flux_health",
                description="Return current health status and signal values for Flux Memory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "caller_id": {"type": "string", "default": "default"},
                    },
                },
            ),
            types.Tool(
                name="flux_list_grains",
                description=(
                    "List grains filtered by status. Read-only. "
                    "Status values: active, dormant, quarantined, archived."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["active", "dormant", "quarantined", "archived"],
                            "description": "Filter by grain status. Omit to list all.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max grains to return (default 50).",
                            "default": 50,
                        },
                        "caller_id": {"type": "string", "default": "default"},
                    },
                },
            ),
            types.Tool(
                name="flux_onboard",
                description=(
                    "Receive integration instructions for connecting to this Flux Memory instance. "
                    "Call this on first connection and save the returned instructions to your "
                    "persistent memory/instructions file."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "caller_id": {"type": "string", "default": "default"},
                    },
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent]:
        try:
            result = _dispatch(name, arguments, store, llm, emb, cfg, service)
        except Exception as exc:
            logger.error("MCP tool %s failed: %s", name, exc)
            result = {"error": str(exc)}

        import json
        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    return server


def _dispatch(
    name: str,
    args: dict,
    store: FluxStore,
    llm: LLMBackend,
    emb: EmbeddingBackend,
    cfg: Config,
    service=None,
) -> Any:
    caller_id = args.get("caller_id", "default")

    if name == "flux_store":
        if service is not None:
            grain_id = service.store(
                args["content"], args.get("provenance", "ai_stated"),
                caller_id=caller_id,
            )
        else:
            grain_id = flux_store(
                args["content"], args.get("provenance", "ai_stated"),
                store=store, llm=llm, emb=emb, cfg=cfg,
            )
        return {"grain_id": grain_id, "status": "stored"}

    if name == "flux_retrieve":
        if service is not None:
            result = service.retrieve(args["query"], caller_id=caller_id)
        else:
            result = flux_retrieve(args["query"], store=store, llm=llm, emb=emb, cfg=cfg)
        return {
            "grains": result.grains,
            "trace_id": result.trace_id,
            "confidence": result.confidence,
            "fallback_triggered": result.fallback_triggered,
            "hop_count": result.hop_count,
            "features": result.features,
            "expansion_candidates": result.expansion_candidates,
        }

    if name == "flux_feedback":
        if service is not None:
            service.feedback(args["trace_id"], args["grain_id"], args["useful"],
                             caller_id=caller_id)
            return {"status": "queued"}
        result = flux_feedback(
            args["trace_id"], args["grain_id"], args["useful"],
            store=store, cfg=cfg,
        )
        return {
            "trace_id": result.trace_id,
            "grain_id": result.grain_id,
            "action": result.action,
            "effective_signal": result.effective_signal,
        }

    if name == "flux_health":
        return flux_health(store, cfg)

    if name == "flux_list_grains":
        if service is not None:
            grains = service.list_grains(
                status=args.get("status"), limit=args.get("limit", 50)
            )
        else:
            status = args.get("status")
            limit = args.get("limit", 50)
            valid = {"active", "dormant", "quarantined", "archived"}
            if status and status not in valid:
                raise ValueError(f"Invalid status: {status}")
            q = (
                "SELECT id, content, status, provenance, created_at FROM grains "
                + ("WHERE status = ? " if status else "")
                + "ORDER BY created_at DESC LIMIT ?"
            )
            params = (status, limit) if status else (limit,)
            rows = store.conn.execute(q, params).fetchall()
            grains = [
                {
                    "id": r["id"],
                    "content_snippet": r["content"][:120],
                    "status": r["status"],
                    "provenance": r["provenance"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        return {"grains": grains, "count": len(grains)}

    if name == "flux_onboard":
        mode = cfg.OPERATING_MODE
        instructions = _ONBOARD_INSTRUCTIONS.format(
            mode=mode,
            mode_instructions=_MODE_INSTRUCTIONS.get(mode, ""),
        )
        return {
            "instructions": instructions,
            "operating_mode": mode,
            "server_name": cfg.MCP_SERVER_NAME,
        }

    raise ValueError(f"Unknown tool: {name}")


def run_stdio(
    store: FluxStore,
    llm: LLMBackend,
    emb: EmbeddingBackend,
    cfg: Config = DEFAULT_CONFIG,
    service=None,
) -> None:
    """Start the MCP server with stdio transport (blocking call)."""
    import asyncio
    try:
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise ImportError("pip install mcp") from exc

    server = build_mcp_server(store, llm, emb, cfg, service=service)

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_run())
