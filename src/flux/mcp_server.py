"""MCP server wrapper for Flux Memory (Track 4 Step 2, §11.2, §13.5).

Exposes three tools to the main AI agent:
  - flux_store   (write channel)
  - flux_retrieve (read channel)
  - flux_feedback (read channel)
  - flux_health   (observability)

Admin channel (flux_purge etc.) is intentionally NOT exposed here (§7.6,
§13.5). The admin channel lives in flux.admin and must be explicitly
imported by user-controlled scripts, never through MCP.

Usage:
    from flux.mcp_server import build_mcp_server
    mcp = build_mcp_server(store, llm, emb, cfg)
    mcp.run()   # starts stdio transport by default

Requires the 'mcp' package (pip install mcp).
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


def build_mcp_server(
    store: FluxStore,
    llm: LLMBackend,
    emb: EmbeddingBackend,
    cfg: Config = DEFAULT_CONFIG,
) -> Any:
    """Construct and return an MCP Server instance with Flux tools registered.

    Raises ImportError if the 'mcp' package is not installed.
    """
    try:
        from mcp.server import Server
        from mcp.server.models import InitializationOptions
        import mcp.types as types
    except ImportError as exc:
        raise ImportError(
            "MCP server requires the 'mcp' package. "
            "Install it: pip install mcp"
        ) from exc

    server = Server("flux-memory")

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
                    },
                    "required": ["trace_id", "grain_id", "useful"],
                },
            ),
            types.Tool(
                name="flux_health",
                description="Return current health status and signal values for Flux Memory.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent]:
        try:
            result = _dispatch(name, arguments, store, llm, emb, cfg)
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
) -> Any:
    if name == "flux_store":
        grain_id = flux_store(
            args["content"],
            args.get("provenance", "ai_stated"),
            store=store,
            llm=llm,
            emb=emb,
            cfg=cfg,
        )
        return {"grain_id": grain_id, "status": "stored"}

    if name == "flux_retrieve":
        result = flux_retrieve(
            args["query"],
            store=store,
            llm=llm,
            emb=emb,
            cfg=cfg,
        )
        return {
            "grains": result.grains,
            "trace_id": result.trace_id,
            "confidence": result.confidence,
            "fallback_triggered": result.fallback_triggered,
            "hop_count": result.hop_count,
            "features": result.features,
        }

    if name == "flux_feedback":
        result = flux_feedback(
            args["trace_id"],
            args["grain_id"],
            args["useful"],
            store=store,
            cfg=cfg,
        )
        return {
            "trace_id": result.trace_id,
            "grain_id": result.grain_id,
            "action": result.action,
            "effective_signal": result.effective_signal,
        }

    if name == "flux_health":
        return flux_health(store, cfg)

    raise ValueError(f"Unknown tool: {name}")


def run_stdio(
    store: FluxStore,
    llm: LLMBackend,
    emb: EmbeddingBackend,
    cfg: Config = DEFAULT_CONFIG,
) -> None:
    """Start the MCP server with stdio transport (blocking call)."""
    import asyncio
    try:
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise ImportError("pip install mcp") from exc

    server = build_mcp_server(store, llm, emb, cfg)

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_run())
