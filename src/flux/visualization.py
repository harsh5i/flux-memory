"""Graph visualization export (Track 4 Step 3, §11.6).

Three export formats:
  GraphML — for desktop graph tools (Gephi, yEd)
  JSON node-link — for web dashboards (D3.js, Cytoscape.js, vis.js)
  DOT (Graphviz) — for static image generation

Node types: grain (round) and entry (diamond).
Edge types: bootstrap, earned, shortcut (emergent).

All weights are effective_weight() at export time so the snapshot reflects
the current decayed state rather than the stored value.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from .graph import utcnow
from .propagation import effective_weight
from .storage import FluxStore


def export_graphml(store: FluxStore, now: datetime | None = None) -> str:
    """Return a GraphML-format string of the current graph state."""
    now = now or utcnow()
    nodes, edges = _load_graph(store, now)

    root = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/graphml")
    _add_graphml_keys(root)
    g = ET.SubElement(root, "graph", id="flux", edgedefault="directed")

    for node in nodes:
        n = ET.SubElement(g, "node", id=node["id"])
        for key, val in node.items():
            if key == "id":
                continue
            d = ET.SubElement(n, "data", key=f"d_{key}")
            d.text = str(val)

    for i, edge in enumerate(edges):
        e = ET.SubElement(g, "edge", id=f"e{i}",
                          source=edge["source"], target=edge["target"])
        for key, val in edge.items():
            if key in ("source", "target"):
                continue
            d = ET.SubElement(e, "data", key=f"d_{key}")
            d.text = str(val)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def export_json(store: FluxStore, now: datetime | None = None) -> dict:
    """Return a node-link JSON dict suitable for D3.js / Cytoscape.js / vis.js."""
    now = now or utcnow()
    nodes, edges = _load_graph(store, now)
    return {
        "directed": True,
        "multigraph": False,
        "nodes": [{"id": n["id"], **{k: v for k, v in n.items() if k != "id"}}
                  for n in nodes],
        "links": [{"source": e["source"], "target": e["target"],
                   **{k: v for k, v in e.items() if k not in ("source", "target")}}
                  for e in edges],
    }


def export_dot(store: FluxStore, now: datetime | None = None) -> str:
    """Return a Graphviz DOT string of the current graph state."""
    now = now or utcnow()
    nodes, edges = _load_graph(store, now)

    lines = ["digraph flux {", "  rankdir=LR;"]
    for node in nodes:
        shape = "diamond" if node.get("node_type") == "entry" else "ellipse"
        color = "gold" if node.get("decay_class") == "core" else "lightgrey"
        label = node.get("label", node["id"])[:40].replace('"', '\\"')
        lines.append(f'  "{node["id"]}" [label="{label}" shape={shape} fillcolor="{color}" style=filled];')

    for edge in edges:
        weight = edge.get("effective_weight", 0.5)
        penwidth = max(1.0, weight * 5)
        style = _dot_edge_style(edge.get("edge_type", "earned"))
        lines.append(
            f'  "{edge["source"]}" -> "{edge["target"]}" '
            f'[penwidth={penwidth:.2f} style={style} weight={weight:.3f}];'
        )

    lines.append("}")
    return "\n".join(lines)


def subgraph(store: FluxStore, entry_features: list[str],
             now: datetime | None = None) -> dict:
    """Export only the subgraph reachable from the given entry features (JSON)."""
    now = now or utcnow()

    entry_ids: set[str] = set()
    for feat in entry_features:
        e = store.get_entry_by_feature(feat.lower())
        if e:
            entry_ids.add(e.id)

    # BFS to collect reachable node IDs.
    visited: set[str] = set(entry_ids)
    frontier = list(entry_ids)
    while frontier:
        next_frontier = []
        for nid in frontier:
            for cond in store.outgoing_conduits(nid):
                if cond.to_id not in visited:
                    visited.add(cond.to_id)
                    next_frontier.append(cond.to_id)
        frontier = next_frontier

    full = export_json(store, now)
    nodes = [n for n in full["nodes"] if n["id"] in visited]
    links = [l for l in full["links"] if l["source"] in visited and l["target"] in visited]
    return {"directed": True, "multigraph": False, "nodes": nodes, "links": links}


def cluster_view(store: FluxStore) -> dict:
    """Return current cluster assignments as a JSON dict."""
    partitions, cluster_ids = store.get_current_partition()
    clusters = []
    for i, (cid, members) in enumerate(zip(cluster_ids, partitions)):
        clusters.append({"cluster_id": cid, "members": list(members)})
    return {"clusters": clusters}


# ----------------------------------------------------------------- internals

def _load_graph(store: FluxStore, now: datetime) -> tuple[list[dict], list[dict]]:
    from .propagation import effective_weight as _eff_weight
    from .config import DEFAULT_CONFIG

    nodes: list[dict] = []
    edges: list[dict] = []

    # Grains.
    grain_rows = store.conn.execute("SELECT * FROM grains").fetchall()
    for row in grain_rows:
        nodes.append({
            "id": row["id"],
            "label": row["content"][:60],
            "node_type": "grain",
            "decay_class": row["decay_class"],
            "status": row["status"],
            "provenance": row["provenance"],
            "context_spread": row["context_spread"],
        })

    # Entries.
    entry_rows = store.conn.execute("SELECT * FROM entries").fetchall()
    for row in entry_rows:
        nodes.append({
            "id": row["id"],
            "label": row["feature"],
            "node_type": "entry",
            "feature": row["feature"],
        })

    # Conduits — iterate all via store to get proper Conduit objects.
    all_conduits = store.conn.execute("SELECT id FROM conduits").fetchall()
    for cid_row in all_conduits:
        c = store.get_conduit(cid_row["id"])
        if c is None:
            continue
        ew = _eff_weight(c, DEFAULT_CONFIG, now)
        direction = c.direction or "forward"
        edge_type = "shortcut" if direction == "bidirectional" and c.use_count > 0 else "earned"
        edges.append({
            "source": c.from_id,
            "target": c.to_id,
            "weight": c.weight,
            "effective_weight": round(ew, 4),
            "direction": direction,
            "decay_class": c.decay_class,
            "use_count": c.use_count,
            "edge_type": edge_type,
        })

    return nodes, edges


def _add_graphml_keys(root: ET.Element) -> None:
    for key_id, attr_name, attr_type, for_elem in [
        ("d_label", "label", "string", "node"),
        ("d_node_type", "node_type", "string", "node"),
        ("d_decay_class", "decay_class", "string", "node"),
        ("d_status", "status", "string", "node"),
        ("d_provenance", "provenance", "string", "node"),
        ("d_effective_weight", "effective_weight", "double", "edge"),
        ("d_weight", "weight", "double", "edge"),
        ("d_direction", "direction", "string", "edge"),
        ("d_edge_type", "edge_type", "string", "edge"),
    ]:
        ET.SubElement(root, "key", id=key_id, **{
            "attr.name": attr_name,
            "attr.type": attr_type,
            "for": for_elem,
        })


def _dot_edge_style(edge_type: str) -> str:
    return {"shortcut": "dotted", "bootstrap": "dashed"}.get(edge_type, "solid")
