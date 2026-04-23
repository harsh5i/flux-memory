"""Minimal web dashboard for Flux Memory (Track 4 Step 4, §11.6).

Serves a single-page dashboard over HTTP. The page fetches:
  GET /api/health     → flux_health() JSON
  GET /api/graph      → export_json() node-link JSON
  GET /api/clusters   → cluster_view() JSON

Usage:
    from flux.dashboard import run_dashboard
    run_dashboard(store, host="127.0.0.1", port=7462)

Or from the command line:
    python -m flux.dashboard --db path/to/flux.db --port 7462
"""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flux Memory Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; }
  header { background: #1e2330; padding: 16px 24px; border-bottom: 1px solid #2d3748; }
  header h1 { font-size: 1.25rem; font-weight: 600; }
  header span { font-size: 0.8rem; color: #718096; margin-left: 12px; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 20px; }
  .card { background: #1e2330; border: 1px solid #2d3748; border-radius: 8px; padding: 16px; }
  .card h2 { font-size: 0.9rem; font-weight: 600; text-transform: uppercase;
              letter-spacing: 0.05em; color: #718096; margin-bottom: 12px; }
  .status-badge { display: inline-block; padding: 4px 12px; border-radius: 999px;
                  font-size: 0.85rem; font-weight: 600; }
  .healthy { background: #22543d; color: #9ae6b4; }
  .warning { background: #744210; color: #fbd38d; }
  .critical { background: #742a2a; color: #feb2b2; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; color: #718096; padding: 4px 8px; border-bottom: 1px solid #2d3748; }
  td { padding: 6px 8px; border-bottom: 1px solid #1a202c; }
  .ok { color: #68d391; }
  .bad { color: #fc8181; }
  #graph-canvas { width: 100%; height: 300px; background: #0f1117;
                  border-radius: 4px; overflow: hidden; position: relative; }
  #graph-canvas svg { width: 100%; height: 100%; }
  .warn-item { padding: 8px; background: #2d3748; border-radius: 4px;
               margin-bottom: 6px; font-size: 0.82rem; }
  .warn-item .sig { font-weight: 600; color: #fbd38d; }
  .refresh { font-size: 0.75rem; color: #4a5568; margin-top: 8px; }
  .stat { display: flex; justify-content: space-between; padding: 4px 0;
          font-size: 0.85rem; border-bottom: 1px solid #1a202c; }
</style>
</head>
<body>
<header>
  <h1>⚡ Flux Memory</h1>
  <span id="computed-at">Loading…</span>
</header>
<main>

<div class="card">
  <h2>Health Status</h2>
  <div id="status-badge" class="status-badge" style="margin-bottom:12px">…</div>
  <table>
    <thead><tr><th>Signal</th><th>Value</th><th>OK?</th></tr></thead>
    <tbody id="signals-table"></tbody>
  </table>
</div>

<div class="card">
  <h2>Active Warnings</h2>
  <div id="warnings-list"><em style="color:#4a5568">None</em></div>
</div>

<div class="card" style="grid-column:1/-1">
  <h2>Graph Overview</h2>
  <div id="graph-canvas"><svg id="graph-svg"></svg></div>
  <p class="refresh" id="graph-stats"></p>
</div>

</main>
<p class="refresh" style="padding:0 20px 16px">Auto-refreshes every 30s.</p>
<script>
async function fetchJSON(url) {
  const r = await fetch(url);
  return r.json();
}

function renderHealth(data) {
  document.getElementById('computed-at').textContent =
    'Last computed: ' + (data.computed_at || '?');

  const badge = document.getElementById('status-badge');
  badge.textContent = (data.status || 'unknown').toUpperCase();
  badge.className = 'status-badge ' + (data.status || 'unknown');

  const tbody = document.getElementById('signals-table');
  tbody.innerHTML = '';
  for (const [name, sig] of Object.entries(data.signals || {})) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${name}</td><td>${(+sig.value).toFixed(3)}</td>
      <td class="${sig.healthy ? 'ok' : 'bad'}">${sig.healthy ? '✓' : '✗'}</td>`;
    tbody.appendChild(tr);
  }

  const wlist = document.getElementById('warnings-list');
  const warns = data.active_warnings || [];
  if (warns.length === 0) {
    wlist.innerHTML = '<em style="color:#4a5568">None</em>';
  } else {
    wlist.innerHTML = warns.map(w =>
      `<div class="warn-item"><span class="sig">${w.signal}</span>: ${w.suggestion}</div>`
    ).join('');
  }
}

function renderGraph(data) {
  const nodes = data.nodes || [];
  const links = data.links || [];
  const stats = document.getElementById('graph-stats');
  stats.textContent = `${nodes.length} nodes · ${links.length} edges`;

  const svg = document.getElementById('graph-svg');
  const W = svg.clientWidth || 800, H = svg.clientHeight || 300;
  svg.innerHTML = '';

  if (nodes.length === 0) {
    svg.innerHTML = '<text x="50%" y="50%" text-anchor="middle" fill="#4a5568">No data</text>';
    return;
  }

  // Minimal force-free layout: place nodes in a circle.
  const pos = {};
  nodes.forEach((n, i) => {
    const angle = (2 * Math.PI * i) / nodes.length;
    pos[n.id] = { x: W/2 + (W*0.38)*Math.cos(angle), y: H/2 + (H*0.38)*Math.sin(angle) };
  });

  // Edges first.
  links.forEach(l => {
    const s = pos[l.source], t = pos[l.target];
    if (!s || !t) return;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    const w = Math.max(0.5, (l.effective_weight || 0.3) * 4);
    const hue = l.effective_weight > 0.5 ? '#68d391' : l.effective_weight > 0.25 ? '#fbd38d' : '#fc8181';
    line.setAttribute('x1', s.x); line.setAttribute('y1', s.y);
    line.setAttribute('x2', t.x); line.setAttribute('y2', t.y);
    line.setAttribute('stroke', hue); line.setAttribute('stroke-width', w);
    line.setAttribute('opacity', '0.6');
    svg.appendChild(line);
  });

  // Nodes.
  nodes.forEach(n => {
    const p = pos[n.id];
    if (!p) return;
    const isEntry = n.node_type === 'entry';
    const fill = isEntry ? '#4299e1' : (n.decay_class === 'core' ? '#d69e2e' : '#718096');
    const r = isEntry ? 6 : 5;
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', p.x); circle.setAttribute('cy', p.y);
    circle.setAttribute('r', r); circle.setAttribute('fill', fill);
    circle.setAttribute('opacity', n.status === 'dormant' ? '0.3' : '0.85');
    circle.innerHTML = `<title>${n.label}</title>`;
    svg.appendChild(circle);
  });
}

async function refresh() {
  try {
    const [health, graph] = await Promise.all([
      fetchJSON('/api/health'),
      fetchJSON('/api/graph'),
    ]);
    renderHealth(health);
    renderGraph(graph);
  } catch (e) {
    console.error('refresh failed', e);
  }
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"""


def run_dashboard(
    store: Any,
    host: str = "127.0.0.1",
    port: int = 7462,
    cfg: Any = None,
) -> None:
    """Start the HTTP dashboard server (blocking). Ctrl+C to stop."""
    from .visualization import export_json, cluster_view
    from .health import flux_health

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.debug(fmt, *args)

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self._send(200, "text/html; charset=utf-8", _DASHBOARD_HTML.encode())
            elif self.path == "/api/health":
                data = flux_health(store, cfg) if cfg else flux_health(store)
                self._send(200, "application/json", json.dumps(data, default=str).encode())
            elif self.path == "/api/graph":
                data = export_json(store)
                self._send(200, "application/json", json.dumps(data, default=str).encode())
            elif self.path == "/api/clusters":
                data = cluster_view(store)
                self._send(200, "application/json", json.dumps(data, default=str).encode())
            else:
                self._send(404, "text/plain", b"Not found")

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    logger.info("Flux dashboard running at http://%s:%d", host, port)
    print(f"Flux Memory dashboard -> http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse
    from .storage import FluxStore

    parser = argparse.ArgumentParser(description="Flux Memory Dashboard")
    parser.add_argument("--db", required=True, help="Path to flux.db")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7462)
    args = parser.parse_args()

    with FluxStore(args.db) as _store:
        run_dashboard(_store, host=args.host, port=args.port)
