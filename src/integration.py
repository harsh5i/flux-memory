"""
Flux Integration — Seed and sync with OpenClaw memory.

This module bridges Flux Memory with OpenClaw's existing memory system:
- Seeds Flux with MEMORY.md content
- Provides dual-path retrieval (Flux + memory_search)
- Auto-remembers significant facts
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Flux Memory
from flux import Flux

# Memory paths
MEMORY_MD = Path.home() / ".openclaw" / "workspace" / "MEMORY.md"
MEMORY_DIR = Path.home() / ".openclaw" / "workspace" / "memory"
FLUX_DB = Path.home() / ".openclaw" / "flux" / "flux.db"


def get_flux() -> Flux:
    """Get or create Flux instance."""
    FLUX_DB.parent.mkdir(parents=True, exist_ok=True)
    return Flux(
        store_path=str(FLUX_DB),
        use_llm_decompose=True,
        use_embeddings=True,
    )


def parse_memory_md(content: str) -> List[Dict]:
    """Parse MEMORY.md into grain-worthy content."""
    grains = []
    
    # Split by sections
    sections = re.split(r'^## (.+)$', content, flags=re.MULTILINE)
    
    current_section = "General"
    for i, chunk in enumerate(sections):
        if i % 2 == 1:
            current_section = chunk.strip()
            continue
        
        # Extract key-value pairs
        for line in chunk.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('---'):
                continue
            
            # Key: Value format
            if line.startswith('- **') and ':**' in line:
                match = re.match(r'- \*\*(.+?):\*\* (.+)', line)
                if match:
                    key, value = match.groups()
                    grains.append({
                        "content": f"{key}: {value}",
                        "section": current_section,
                        "tags": [current_section.lower().replace(' ', '-')],
                    })
            # Bullet points
            elif line.startswith('- ') and len(line) > 5:
                content = line[2:].strip()
                if not content.startswith('*') and len(content) > 10:
                    grains.append({
                        "content": content,
                        "section": current_section,
                        "tags": [current_section.lower().replace(' ', '-')],
                    })
            # Pattern entries (timestamped)
            elif line.startswith('[') and 'Pattern:' in line:
                match = re.match(r'\[([^\]]+)\] (.+)', line)
                if match:
                    timestamp, pattern = match.groups()
                    grains.append({
                        "content": f"Pattern detected {timestamp}: {pattern}",
                        "section": "Auto-Detected Patterns",
                        "tags": ["pattern", "auto-detected"],
                    })
    
    return grains


def seed_flux(flux: Flux, grains: List[Dict]) -> Dict:
    """Seed Flux with parsed grains."""
    results = {
        "total": len(grains),
        "created": 0,
        "skipped": 0,
        "errors": [],
    }
    
    # Get existing grains to avoid duplicates
    existing = flux.store.get_all_grains()
    existing_content_hashes = {g.content[:50] for g in existing.values()}
    
    for grain_data in grains:
        content = grain_data["content"]
        
        # Skip if very similar content exists
        if content[:50] in existing_content_hashes:
            results["skipped"] += 1
            continue
        
        try:
            grain = flux.remember(
                content=content,
                tags=grain_data.get("tags", []),
            )
            results["created"] += 1
        except Exception as e:
            results["errors"].append(str(e))
    
    return results


def seed_from_memory_md(flux: Flux = None) -> Dict:
    """Seed Flux with MEMORY.md content."""
    if flux is None:
        flux = get_flux()
    
    if not MEMORY_MD.exists():
        return {"error": "MEMORY.md not found"}
    
    content = MEMORY_MD.read_text()
    grains = parse_memory_md(content)
    
    return seed_flux(flux, grains)


def seed_from_memory_dir(flux: Flux = None) -> Dict:
    """Seed Flux with daily memory files."""
    if flux is None:
        flux = get_flux()
    
    if not MEMORY_DIR.exists():
        return {"error": "memory/ directory not found"}
    
    results = {
        "files": 0,
        "total": 0,
        "created": 0,
        "skipped": 0,
    }
    
    for md_file in sorted(MEMORY_DIR.glob("*.md")):
        if md_file.name.startswith('.'):
            continue
        
        content = md_file.read_text()
        grains = parse_memory_md(content)
        
        file_result = seed_flux(flux, grains)
        results["files"] += 1
        results["total"] += file_result["total"]
        results["created"] += file_result["created"]
        results["skipped"] += file_result["skipped"]
    
    return results


def dual_search(
    query: str,
    flux: Flux = None,
    max_results: int = 5,
) -> List[Tuple[Dict, float, str]]:
    """
    Search both Flux and memory_search (if available).
    
    Returns list of (result, score, source) tuples.
    """
    if flux is None:
        flux = get_flux()
    
    results = []
    
    # Flux search
    try:
        flux_results = flux.query(query, max_results=max_results)
        for grain, signal in flux_results:
            results.append(({
                "content": grain.content,
                "id": grain.id,
                "context_spread": grain.context_spread,
                "decay_class": grain.decay_class.value,
            }, signal, "flux"))
    except Exception as e:
        results.append(({"error": str(e)}, 0.0, "flux"))
    
    return sorted(results, key=lambda x: -x[1])[:max_results]


def remember(
    content: str,
    tags: List[str] = None,
    flux: Flux = None,
) -> Dict:
    """Remember something in Flux."""
    if flux is None:
        flux = get_flux()
    
    grain = flux.remember(content, tags=tags or [])
    
    return {
        "status": "remembered",
        "grain_id": grain.id,
        "content": grain.content[:100],
        "decay_class": grain.decay_class.value,
    }


if __name__ == "__main__":
    print("Seeding Flux with MEMORY.md...")
    flux = get_flux()
    
    # Seed from MEMORY.md
    result1 = seed_from_memory_md(flux)
    print(f"MEMORY.md: {result1}")
    
    # Seed from memory dir
    result2 = seed_from_memory_dir(flux)
    print(f"memory/: {result2}")
    
    # Show stats
    stats = flux.stats()
    print(f"\nFlux Stats: {stats}")
    
    # Test query
    print("\nTesting query: 'Harsh contact'")
    results = dual_search("Harsh contact", flux)
    for r, score, src in results:
        print(f"  [{src}] {score:.2f}: {r.get('content', r)[:50]}...")