#!/usr/bin/env python3
"""
Flux Memory Graph Visualization Server

Run: python3 visualize_server.py
Open: http://localhost:8765
"""

import json
import math
import random
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from flux import Flux
from grain import DecayClass

DB_PATH = Path.home() / ".openclaw" / "flux" / "flux.db"


def get_graph_data():
    """Extract graph data from Flux for visualization."""
    flux = Flux(str(DB_PATH), use_llm_decompose=False, use_embeddings=False)
    
    grains = flux.store.get_all_grains()
    entry_points = flux.store.get_all_entry_points()
    conduits = flux.store.get_conduits_by_source()
    all_conduits = flux.store.get_all_conduits()
    
    nodes = []
    edges = []
    
    # Position nodes using force-directed layout (simplified)
    node_positions = {}
    
    # Create grain nodes
    grain_list = list(grains.values())
    for i, grain in enumerate(grain_list):
        angle = 2 * math.pi * i / len(grain_list)
        radius = 0.3 + 0.1 * random.random()
        x = 0.5 + radius * math.cos(angle)
        y = 0.5 + radius * math.sin(angle)
        
        node_positions[grain.id] = (x, y)
        nodes.append({
            "id": grain.id,
            "type": "grain",
            "label": grain.content[:30] + "..." if len(grain.content) > 30 else grain.content,
            "content": grain.content,
            "core": grain.decay_class == DecayClass.CORE,
            "context_spread": grain.context_spread,
            "x": x,
            "y": y,
        })
    
    # Create entry point nodes (outer ring)
    entry_list = list(entry_points.values())
    for i, ep in enumerate(entry_list[:50]):  # Limit to 50 for performance
        angle = 2 * math.pi * i / min(len(entry_list), 50)
        x = 0.1 + 0.2 * math.cos(angle)
        y = 0.1 + 0.2 * math.sin(angle)
        
        node_positions[ep.id] = (x, y)
        nodes.append({
            "id": ep.id,
            "type": "entry",
            "label": ep.feature[:20],
            "x": x,
            "y": y,
        })
    
    # Create edges from conduits
    for conduit in all_conduits.values():
        if conduit.from_id in node_positions and conduit.to_id in node_positions:
            edges.append({
                "from": conduit.from_id,
                "to": conduit.to_id,
                "weight": conduit.weight,
            })
    
    return {"nodes": nodes, "edges": edges}


class FluxVizHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def do_GET(self):
        if self.path == "/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(get_graph_data()).encode())
        elif self.path == "/" or self.path == "":
            self.path = "/visualize.html"
            return SimpleHTTPRequestHandler.do_GET(self)
        else:
            return SimpleHTTPRequestHandler.do_GET(self)


def main():
    port = 8765
    print(f"🌿 Flux Memory Visualization")
    print(f"   http://localhost:{port}")
    print(f"   Press Ctrl+C to stop\n")
    
    server = HTTPServer(("localhost", port), FluxVizHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()