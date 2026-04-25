"""
Flux Memory MCP Server — Integration with OpenClaw.

An MCP server that exposes Flux Memory as a tool for AI assistants.
"""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
from grain import Grain, DecayClass
from conduit import Conduit, Direction
from entry_point import EntryPoint
from trace import Trace
from propagation import SignalEngine, PropagationConfig, RetrievalResult
from store import FluxStore
from decay import run_decay_cycle
from flux import Flux
from integration import remember

# Initialize MCP server
mcp = FastMCP("flux-memory")

# Global Flux instance
_flux: Optional[Flux] = None


def get_flux(db_path: str = None) -> Flux:
    """Get or create Flux instance."""
    global _flux
    if _flux is None:
        if db_path is None:
            db_path = str(Path.home() / ".openclaw" / "flux" / "flux.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _flux = Flux(
            store_path=db_path,
            use_llm_decompose=True,
            use_embeddings=True,
        )
    return _flux


@mcp.tool()
def flux_remember(content: str, tags: str = "") -> Dict[str, Any]:
    """
    Store a new memory in Flux.
    
    Args:
        content: The memory content to store
        tags: Comma-separated tags (optional)
    
    Returns:
        Info about the created grain
    """
    flux = get_flux()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    grain = flux.remember(content, tags=tag_list)
    
    return {
        "status": "created",
        "grain_id": grain.id,
        "content": grain.content[:100] + "..." if len(grain.content) > 100 else grain.content,
        "decay_class": grain.decay_class.value,
    }


@mcp.tool()
def flux_query(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Query Flux Memory for relevant memories.
    
    Args:
        query: The query text
        max_results: Maximum number of results (default 5)
    
    Returns:
        Retrieved grains with signal strengths
    """
    flux = get_flux()
    results = flux.query(query, max_results=max_results)
    
    return {
        "status": "ok",
        "query": query,
        "results": [
            {
                "grain_id": grain.id,
                "content": grain.content,
                "signal": round(signal, 3),
                "context_spread": grain.context_spread,
                "decay_class": grain.decay_class.value,
            }
            for grain, signal in results
        ],
        "trace_id": flux._last_result.trace.id if flux._last_result else None,
    }


@mcp.tool()
def flux_feedback(grain_id: str, useful: bool = True) -> Dict[str, Any]:
    """
    Provide feedback on a retrieval result.
    
    Args:
        grain_id: ID of the grain that was retrieved
        useful: Whether the result was useful (default True)
    
    Returns:
        Confirmation
    """
    flux = get_flux()
    flux.feedback(grain_id, success=useful)
    
    return {
        "status": "ok",
        "grain_id": grain_id,
        "feedback": "useful" if useful else "not_useful",
    }


@mcp.tool()
def flux_close_loop(used_grain_ids: str = "") -> Dict[str, Any]:
    """
    Close the feedback loop after a retrieval.

    Call this AFTER composing your response. Provide the grain IDs you 
    actually used/cited in your answer. Everything else from the last 
    search gets marked not-useful, which weakens those conduits.

    If used_grain_ids is empty, all results from last search are marked useful.

    Args:
        used_grain_ids: Comma-separated grain IDs that were actually used (e.g. "G-abc123,G-def456")

    Returns:
        Feedback summary
    """
    flux = get_flux()
    
    ids = [i.strip() for i in used_grain_ids.split(",") if i.strip()] if used_grain_ids else None
    result = flux.close_loop(used_grain_ids=ids)
    
    return {
        "status": "ok",
        **result,
    }


@mcp.tool()
def flux_stats() -> Dict[str, Any]:
    """
    Get Flux Memory statistics.
    
    Returns:
        Stats about grains, conduits, entry points, traces
    """
    flux = get_flux()
    stats = flux.stats()
    
    return {
        "status": "ok",
        **stats,
    }


@mcp.tool()
def flux_decay() -> Dict[str, Any]:
    """
    Run decay cycle to prune old/dormant memories.
    
    Returns:
        Info about what was pruned
    """
    flux = get_flux()
    result = flux.decay()
    
    return {
        "status": "ok",
        **result,
    }


@mcp.tool()
def flux_get_grain(grain_id: str) -> Dict[str, Any]:
    """
    Get a specific grain by ID.
    
    Args:
        grain_id: The grain ID
    
    Returns:
        Grain details or error
    """
    flux = get_flux()
    grain = flux.store.get_grain(grain_id)
    
    if grain:
        return {
            "status": "ok",
            "grain": grain.to_dict(),
        }
    
    return {
        "status": "not_found",
        "grain_id": grain_id,
    }


@mcp.tool()
def flux_list_grains(limit: int = 20, decay_class: str = None) -> Dict[str, Any]:
    """
    List grains, optionally filtered.
    
    Args:
        limit: Maximum grains to return (default 20)
        decay_class: Filter by decay class (working/core/archived)
    
    Returns:
        List of grains
    """
    flux = get_flux()
    grains = flux.store.get_all_grains()
    
    # Filter by decay class if specified
    if decay_class:
        grains = {gid: g for gid, g in grains.items() if g.decay_class.value == decay_class}
    
    # Sort by creation date, newest first
    sorted_grains = sorted(grains.values(), key=lambda g: -g.created_at.timestamp())
    limited = sorted_grains[:limit]
    
    return {
        "status": "ok",
        "count": len(limited),
        "total": len(grains),
        "grains": [
            {
                "id": g.id,
                "content": g.content[:100] + "..." if len(g.content) > 100 else g.content,
                "decay_class": g.decay_class.value,
                "context_spread": g.context_spread,
                "created_at": g.created_at.isoformat(),
            }
            for g in limited
        ],
    }


@mcp.resource("flux://stats")
def flux_resource_stats() -> str:
    """Flux statistics as a resource."""
    flux = get_flux()
    stats = flux.stats()
    return json.dumps(stats, indent=2)


@mcp.tool()
def flux_remember_from_memory_md() -> Dict[str, Any]:
    """
    Seed Flux with content from MEMORY.md.
    
    Extracts key facts and patterns, stores them as Flux grains.
    
    Returns:
        Stats about seeding process
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from integration import seed_from_memory_md
    
    result = seed_from_memory_md(get_flux())
    return {
        "status": "ok",
        **result,
    }


@mcp.tool()
def flux_dual_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search Flux memory for relevant content.
    
    This is the primary retrieval tool for remembering past context.
    
    Args:
        query: What to search for
        max_results: Maximum results to return
    
    Returns:
        Relevant memories with signal strengths
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from integration import dual_search
    
    results = dual_search(query, get_flux(), max_results=max_results)
    
    return {
        "status": "ok",
        "query": query,
        "results": [
            {
                "content": r.get("content", str(r)),
                "signal": round(score, 3),
                "source": src,
                "id": r.get("id", "N/A"),
            }
            for r, score, src in results
            if "error" not in r
        ],
    }


@mcp.tool()
def flux_auto_remember(content: str, tags: str = "") -> Dict[str, Any]:
    """
    Automatically remember something in Flux.
    
    Use this when the user shares information worth remembering:
    - Preferences, settings, contact info
    - Decisions, agreements, deadlines
    - Names, locations, context
    - Patterns, insights, lessons
    
    Args:
        content: The fact/memory to remember
        tags: Comma-separated tags (optional)
    
    Returns:
        Info about created grain
    """
    import sys
    import json
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from integration import remember
    
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    result = remember(content, tags=tag_list, flux=get_flux())
    
    # Create backup after every remember
    try:
        backup_dir = Path.home() / ".openclaw" / "flux" / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_file = backup_dir / "memory_backup.json"
        
        flux = get_flux()
        grains = flux.store.get_all_grains()
        
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "grains": [
                {
                    "id": g.id,
                    "content": g.content,
                    "tags": g.source_tags,
                    "decay_class": g.decay_class.value,
                    "context_spread": g.context_spread,
                    "created_at": g.created_at.isoformat(),
                }
                for g in grains.values()
            ],
            "stats": flux.stats(),
        }
        
        with open(backup_file, "w") as f:
            json.dump(backup_data, f, indent=2)
    except Exception as e:
        result["backup_error"] = str(e)
    
    return result


def _create_backup() -> Dict[str, Any]:
    """Create a backup of all Flux grains."""
    import json
    from pathlib import Path
    from datetime import datetime
    
    try:
        backup_dir = Path.home() / ".openclaw" / "flux" / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_file = backup_dir / "memory_backup.json"
        
        flux = get_flux()
        grains = flux.store.get_all_grains()
        
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "grains": [
                {
                    "id": g.id,
                    "content": g.content,
                    "tags": g.source_tags,
                    "decay_class": g.decay_class.value,
                    "context_spread": g.context_spread,
                    "created_at": g.created_at.isoformat(),
                }
                for g in grains.values()
            ],
            "stats": flux.stats(),
        }
        
        with open(backup_file, "w") as f:
            json.dump(backup_data, f, indent=2)
        
        return {"status": "ok", "backup_path": str(backup_file), "grains": len(grains)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
def flux_backup() -> Dict[str, Any]:
    """
    Create a backup of all Flux grains.
    
    Returns:
        Backup status and location
    """
    return _create_backup()


@mcp.tool()
def flux_restore(backup_file: str = "") -> Dict[str, Any]:
    """
    Restore Flux from a backup file.
    
    Args:
        backup_file: Path to backup file (default: latest backup)
    
    Returns:
        Restore status
    """
    import json
    from pathlib import Path
    
    try:
        backup_dir = Path.home() / ".openclaw" / "flux" / "backup"
        
        if backup_file:
            backup_path = Path(backup_file)
        else:
            backup_path = backup_dir / "memory_backup.json"
        
        if not backup_path.exists():
            return {"status": "error", "error": f"Backup not found: {backup_path}"}
        
        with open(backup_path) as f:
            backup = json.load(f)
        
        flux = get_flux()
        restored = 0
        skipped = 0
        
        existing = {g.content[:50] for g in flux.store.get_all_grains().values()}
        
        for grain_data in backup.get("grains", []):
            content = grain_data["content"]
            if content[:50] in existing:
                skipped += 1
                continue
            
            flux.remember(content, tags=grain_data.get("tags", []))
            restored += 1
        
        return {
            "status": "ok",
            "backup_file": str(backup_path),
            "restored": restored,
            "skipped": skipped,
            "timestamp": backup.get("timestamp"),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # Run MCP server
    mcp.run()