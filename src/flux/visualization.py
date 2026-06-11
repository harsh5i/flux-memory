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
        "stats": _graph_stats(store),
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


def grain_dossier(store: FluxStore, grain_id: str) -> dict:
    """Return detailed Chronicle inspector data for one grain."""
    row = store.conn.execute(
        """
        SELECT id, content, provenance, decay_class, status, created_at
        FROM grains WHERE id = ?
        """,
        (grain_id,),
    ).fetchone()
    if row is None:
        return {"error": "not found"}

    degree_row = store.conn.execute(
        """
        SELECT COUNT(*) AS n FROM conduits
        WHERE from_id = ? OR to_id = ?
        """,
        (grain_id, grain_id),
    ).fetchone()
    degree = int(degree_row["n"] if degree_row else 0)

    conduit_rows = store.conn.execute(
        """
        SELECT from_id, to_id, weight, direction
        FROM conduits
        WHERE from_id = ? OR to_id = ?
        ORDER BY weight DESC
        LIMIT 10
        """,
        (grain_id, grain_id),
    ).fetchall()
    conduits = []
    for conduit in conduit_rows:
        other_id = conduit["to_id"] if conduit["from_id"] == grain_id else conduit["from_id"]
        direction = conduit["direction"]
        if direction != "bidirectional":
            direction = "outgoing" if conduit["from_id"] == grain_id else "incoming"
        conduits.append({
            "other_id": other_id,
            "other_snippet": _node_snippet(store, other_id),
            "weight": conduit["weight"],
            "direction": direction,
        })

    retrieval_row = store.conn.execute(
        """
        SELECT COUNT(*) AS n FROM events
        WHERE event = 'grains_returned' AND data LIKE ?
        """,
        (f"%{grain_id}%",),
    ).fetchone()
    retrieval_count = int(retrieval_row["n"] if retrieval_row else 0)

    feedback_rows = store.conn.execute(
        """
        SELECT
          timestamp,
          json_extract(data, '$.action') AS action,
          json_extract(data, '$.effective_signal') AS effective_signal,
          json_extract(data, '$.caller_id') AS caller_id
        FROM events
        WHERE event = 'feedback_received'
          AND json_extract(data, '$.grain_id') = ?
        ORDER BY timestamp DESC
        LIMIT 10
        """,
        (grain_id,),
    ).fetchall()
    feedback = [{
        "timestamp": f["timestamp"],
        "action": f["action"],
        "effective_signal": f["effective_signal"],
        "caller_id": f["caller_id"],
    } for f in feedback_rows]

    return {
        "grain": {
            "id": row["id"],
            "content": row["content"],
            "provenance": row["provenance"],
            "decay_class": row["decay_class"],
            "status": row["status"],
            "created_at": row["created_at"],
        },
        "degree": degree,
        "conduits": conduits,
        "retrieval_count": retrieval_count,
        "feedback": feedback,
    }


# ----------------------------------------------------------------- internals

def _node_snippet(store: FluxStore, node_id: str) -> str:
    row = store.conn.execute(
        "SELECT content FROM grains WHERE id = ?",
        (node_id,),
    ).fetchone()
    if row is not None:
        return (row["content"] or "")[:90]
    row = store.conn.execute(
        "SELECT feature FROM entries WHERE id = ?",
        (node_id,),
    ).fetchone()
    if row is not None:
        return (row["feature"] or "")[:90]
    return node_id[:90]

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
            "id": c.id,
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


def _graph_stats(store: FluxStore) -> dict[str, int]:
    def count(sql: str) -> int:
        row = store.conn.execute(sql).fetchone()
        return int(row["n"] if row else 0)

    grains = count("SELECT COUNT(*) AS n FROM grains")
    active_grains = count("SELECT COUNT(*) AS n FROM grains WHERE status='active'")
    entries = count("SELECT COUNT(*) AS n FROM entries")
    conduits = count("SELECT COUNT(*) AS n FROM conduits")
    embeddings = count("SELECT COUNT(*) AS n FROM grain_embeddings")
    dormant_grains = count("SELECT COUNT(*) AS n FROM grains WHERE status='dormant'")
    return {
        "grains": grains,
        "active_grains": active_grains,
        "dormant_grains": dormant_grains,
        "entries": entries,
        "conduits": conduits,
        "embeddings": embeddings,
    }


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


# ------------------------------------------------------------------ chronicle

def chronicle_data(store: FluxStore, max_edges: int = 8000) -> dict:
    """Data for the Chronicle replay view: every active grain positioned by a
    2D PCA projection of its embedding (so spatial closeness = semantic
    closeness), with creation timestamps for time-scrubbed playback, plus
    grain-to-grain conduits ordered by weight.

    Returns compact arrays to keep the payload small:
      grains:   [id, x, y, created_at, provenance, decay_class, degree, snippet]
      conduits: [from_idx, to_idx, weight, created_at]  (indices into grains)
    """
    import numpy as np
    from .embedding import load_all_embeddings

    grain_ids, matrix = load_all_embeddings(store)
    coords: dict[str, tuple[float, float]] = {}
    if len(grain_ids) >= 3 and matrix.size:
        m = matrix.astype(np.float64)
        m -= m.mean(axis=0)
        # PCA via covariance eigendecomposition (384x384 — cheap).
        cov = (m.T @ m) / max(len(grain_ids) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc = eigvecs[:, np.argsort(eigvals)[::-1][:2]]
        xy = m @ pc
        # Normalise to [0, 1] with a small margin.
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        span = np.where((hi - lo) == 0, 1.0, hi - lo)
        xy = (xy - lo) / span
        for gid, (px, py) in zip(grain_ids, xy):
            coords[gid] = (round(float(px), 4), round(float(py), 4))

    rows = store.conn.execute(
        """
        SELECT id, content, created_at, provenance, decay_class
        FROM grains WHERE status = 'active' ORDER BY created_at
        """
    ).fetchall()

    degree: dict[str, int] = {}
    for r in store.conn.execute(
        "SELECT from_id, to_id FROM conduits"
    ).fetchall():
        degree[r["from_id"]] = degree.get(r["from_id"], 0) + 1
        degree[r["to_id"]] = degree.get(r["to_id"], 0) + 1

    grains = []
    index_of: dict[str, int] = {}
    for r in rows:
        if r["id"] not in coords:
            continue  # no embedding (rare post-backfill); skip rather than fake a position
        x, y = coords[r["id"]]
        index_of[r["id"]] = len(grains)
        grains.append([
            r["id"], x, y, r["created_at"], r["provenance"],
            r["decay_class"], degree.get(r["id"], 0),
            (r["content"] or "")[:110],
        ])

    conduit_rows = store.conn.execute(
        """
        SELECT c.from_id, c.to_id, c.weight, c.created_at FROM conduits c
        JOIN grains ga ON ga.id = c.from_id
        JOIN grains gb ON gb.id = c.to_id
        ORDER BY c.weight DESC LIMIT ?
        """,
        (max_edges,),
    ).fetchall()
    conduits = [
        [index_of[c["from_id"]], index_of[c["to_id"]],
         round(c["weight"], 3), c["created_at"]]
        for c in conduit_rows
        if c["from_id"] in index_of and c["to_id"] in index_of
    ]

    # Soft clusters: members from grain_cluster_touch, labels from the
    # cluster's strongest entry features.
    cluster_rows = store.conn.execute(
        """
        SELECT grain_id, cluster_id FROM (
          SELECT t.grain_id, t.cluster_id,
                 ROW_NUMBER() OVER (PARTITION BY t.grain_id
                                    ORDER BY t.touch_weight DESC) AS rn
          FROM grain_cluster_touch t
          JOIN grains g ON g.id = t.grain_id AND g.status = 'active'
          WHERE t.touch_weight > 0
        ) WHERE rn = 1
        """
    ).fetchall()
    members: dict[str, list[int]] = {}
    for r in cluster_rows:
        if r["grain_id"] in index_of:
            members.setdefault(r["cluster_id"], []).append(index_of[r["grain_id"]])
    label_rows = store.conn.execute(
        """
        SELECT m.cluster_id, e.feature, m.weight FROM entry_cluster_membership m
        JOIN entries e ON e.id = m.entry_id
        ORDER BY m.cluster_id, m.weight DESC
        """
    ).fetchall()
    labels: dict[str, list[str]] = {}
    for r in label_rows:
        bucket = labels.setdefault(r["cluster_id"], [])
        if len(bucket) < 3:
            bucket.append(r["feature"])
    clusters = [
        {"label": " · ".join(labels.get(cid, [])[:3]) or "cluster", "members": idxs}
        for cid, idxs in members.items()
        if len(idxs) >= 4
    ]

    totals = store.conn.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM conduits) AS all_conduits,
          (SELECT COUNT(*) FROM conduits c
             JOIN grains ga ON ga.id = c.from_id
             JOIN grains gb ON gb.id = c.to_id) AS grain_conduits,
          (SELECT COUNT(*) FROM conduits WHERE weight >= 0.80) AS highways
        """
    ).fetchone()

    return {
        "grains": grains,
        "conduits": conduits,
        "clusters": clusters,
        "totals": {
            "all_conduits": totals["all_conduits"],
            "grain_conduits": totals["grain_conduits"],
            "highways": totals["highways"],
        },
        "grain_fields": ["id", "x", "y", "created_at", "provenance",
                         "decay_class", "degree", "snippet"],
        "conduit_fields": ["from_idx", "to_idx", "weight", "created_at"],
    }
