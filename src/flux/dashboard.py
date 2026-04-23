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
  :root {
    --bg: #0b0d12;
    --surface: #151a22;
    --surface-2: #1b2230;
    --line: #2a3445;
    --muted: #8a94a7;
    --text: #edf2f7;
    --teal: #2dd4bf;
    --green: #74d99f;
    --amber: #f2b84b;
    --rose: #fb7185;
    --violet: #a78bfa;
    --blue: #60a5fa;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-width: 320px;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 18px 24px;
    background: #11161f;
    border-bottom: 1px solid var(--line);
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand-mark {
    display: grid;
    place-items: center;
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: #1f2a38;
    color: var(--amber);
    font-weight: 800;
  }
  h1 { margin: 0; font-size: 1.05rem; font-weight: 700; letter-spacing: 0; }
  .updated { color: var(--muted); font-size: 0.82rem; margin-top: 2px; }
  .header-actions { display: flex; align-items: center; gap: 10px; }
  button {
    border: 1px solid var(--line);
    background: #19212c;
    color: var(--text);
    border-radius: 8px;
    padding: 8px 10px;
    font: inherit;
    cursor: pointer;
  }
  button:hover { border-color: #3b4a60; background: #202a38; }
  main { padding: 18px; display: grid; gap: 16px; }
  .metrics {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 12px;
  }
  .metric, .panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
  }
  .metric { padding: 14px; min-height: 96px; }
  .metric .label {
    color: var(--muted);
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .metric .value { font-size: 1.7rem; line-height: 1.2; font-weight: 750; margin-top: 8px; }
  .metric .sub { color: var(--muted); font-size: 0.78rem; margin-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .value.good { color: var(--green); }
  .value.warn { color: var(--amber); }
  .value.bad { color: var(--rose); }
  .value.info { color: var(--teal); }
  .grid {
    display: grid;
    grid-template-columns: minmax(360px, 0.92fr) minmax(420px, 1.08fr);
    gap: 16px;
  }
  .panel { overflow: hidden; }
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 16px;
    border-bottom: 1px solid var(--line);
    background: #171d27;
  }
  .panel h2 {
    margin: 0;
    font-size: 0.82rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #b4bed0;
  }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 0.77rem;
    font-weight: 800;
    text-transform: uppercase;
  }
  .badge.healthy { background: rgba(116, 217, 159, 0.14); color: var(--green); }
  .badge.warning { background: rgba(242, 184, 75, 0.16); color: var(--amber); }
  .badge.critical { background: rgba(251, 113, 133, 0.16); color: var(--rose); }
  .panel-body { padding: 14px 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
  th {
    text-align: left;
    color: var(--muted);
    padding: 0 8px 8px;
    border-bottom: 1px solid var(--line);
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  td { padding: 9px 8px; border-bottom: 1px solid rgba(42, 52, 69, 0.52); }
  tr:last-child td { border-bottom: 0; }
  .signal-name { color: #dce6f6; }
  .signal-value { color: #cbd5e1; font-variant-numeric: tabular-nums; }
  .state-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: var(--rose);
  }
  .state-dot.ok { background: var(--green); }
  .warnings { display: grid; gap: 10px; }
  .warn-item {
    border: 1px solid rgba(242, 184, 75, 0.22);
    background: rgba(242, 184, 75, 0.08);
    border-radius: 8px;
    padding: 10px 12px;
  }
  .warn-title { color: var(--amber); font-size: 0.82rem; font-weight: 800; margin-bottom: 4px; }
  .warn-text { color: #d7dee9; font-size: 0.84rem; line-height: 1.4; }
  .muted { color: var(--muted); }
  .graph-panel { min-height: 500px; }
  .graph-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    color: var(--muted);
    font-size: 0.78rem;
  }
  .pill {
    border: 1px solid var(--line);
    background: #121821;
    color: #b8c2d4;
    border-radius: 999px;
    padding: 5px 8px;
  }
  #graph-canvas {
    position: relative;
    height: 430px;
    background: #080b10;
    border-top: 1px solid #101722;
  }
  #graph-svg { width: 100%; height: 100%; display: block; }
  .empty-note {
    position: absolute;
    left: 18px;
    bottom: 16px;
    max-width: min(520px, calc(100% - 36px));
    color: #aab5c7;
    font-size: 0.86rem;
    background: rgba(21, 26, 34, 0.88);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px 12px;
  }
  .legend {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    color: var(--muted);
    font-size: 0.78rem;
  }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 10px; height: 10px; border-radius: 999px; display: inline-block; }
  .span-2 { grid-column: 1 / -1; }
  @media (max-width: 1100px) {
    .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .grid { grid-template-columns: 1fr; }
  }
  @media (max-width: 680px) {
    header { align-items: flex-start; flex-direction: column; padding: 16px; }
    main { padding: 12px; }
    .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .metric .value { font-size: 1.35rem; }
    #graph-canvas { height: 340px; }
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="brand-mark">F</div>
    <div>
      <h1>Flux Memory</h1>
      <div class="updated" id="computed-at">Loading</div>
    </div>
  </div>
  <div class="header-actions">
    <div id="status-badge" class="badge warning">Loading</div>
    <button id="refresh-button" type="button" title="Refresh dashboard">Refresh</button>
  </div>
</header>
<main>
  <section class="metrics" id="metrics"></section>

  <section class="grid">
    <div class="panel">
      <div class="panel-header">
        <h2>Health Signals</h2>
      </div>
      <div class="panel-body">
        <table>
          <thead><tr><th>Signal</th><th>Value</th><th>State</th></tr></thead>
          <tbody id="signals-table"></tbody>
        </table>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <h2>Warnings</h2>
        <span class="pill" id="warning-count">0 active</span>
      </div>
      <div class="panel-body">
        <div id="warnings-list" class="warnings"><span class="muted">None</span></div>
      </div>
    </div>
  </section>

  <section class="panel graph-panel span-2">
    <div class="panel-header">
      <div>
        <h2>Graph Overview</h2>
        <div class="graph-meta" id="graph-stats"></div>
      </div>
      <div class="legend">
        <span><i class="swatch" style="background:var(--teal)"></i>Entry</span>
        <span><i class="swatch" style="background:var(--blue)"></i>Grain</span>
        <span><i class="swatch" style="background:var(--amber)"></i>Core</span>
        <span><i class="swatch" style="background:var(--green)"></i>Strong conduit</span>
      </div>
    </div>
    <div id="graph-canvas">
      <svg id="graph-svg"></svg>
      <div id="empty-note" class="empty-note" hidden></div>
    </div>
  </section>
</main>
<script>
const fmt = new Intl.NumberFormat();

function safeNumber(value, digits = 3) {
  const n = Number(value || 0);
  return Number.isInteger(n) ? fmt.format(n) : n.toFixed(digits);
}

function metricCard(label, value, sub, tone) {
  const el = document.createElement('div');
  el.className = 'metric';
  el.innerHTML = `<div class="label"></div><div class="value ${tone || ''}"></div><div class="sub"></div>`;
  el.children[0].textContent = label;
  el.children[1].textContent = value;
  el.children[2].textContent = sub || '';
  return el;
}

async function fetchJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${url} returned ${r.status}`);
  return r.json();
}

function renderMetrics(health, graph) {
  const stats = graph.stats || {};
  const sig = health.signals || {};
  const metrics = document.getElementById('metrics');
  metrics.innerHTML = '';
  metrics.appendChild(metricCard('Grains', safeNumber(stats.grains), `${safeNumber(stats.active_grains)} active`, 'info'));
  metrics.appendChild(metricCard('Entries', safeNumber(stats.entries), 'feature anchors', ''));
  metrics.appendChild(metricCard('Conduits', safeNumber(stats.conduits), `${safeNumber(sig.avg_conduit_weight?.value)} avg weight`, stats.conduits > 0 ? 'good' : 'warn'));
  metrics.appendChild(metricCard('Embeddings', safeNumber(stats.embeddings), 'vector fallback index', stats.embeddings > 0 ? 'good' : 'warn'));
  metrics.appendChild(metricCard('Retrieval', `${safeNumber((sig.retrieval_success_rate?.value || 0) * 100, 1)}%`, 'success rate', ''));
  metrics.appendChild(metricCard('Fallback', `${safeNumber((sig.fallback_trigger_rate?.value || 0) * 100, 1)}%`, 'trigger rate', (sig.fallback_trigger_rate?.value || 0) > 0.2 ? 'bad' : 'good'));
}

function renderHealth(data) {
  document.getElementById('computed-at').textContent =
    `Last computed ${data.computed_at || 'unknown'}`;

  const badge = document.getElementById('status-badge');
  const status = data.status || 'unknown';
  badge.textContent = status;
  badge.className = `badge ${status}`;

  const tbody = document.getElementById('signals-table');
  tbody.innerHTML = '';
  for (const [name, sig] of Object.entries(data.signals || {})) {
    const tr = document.createElement('tr');
    const nameCell = document.createElement('td');
    const valueCell = document.createElement('td');
    const stateCell = document.createElement('td');
    nameCell.className = 'signal-name';
    valueCell.className = 'signal-value';
    nameCell.textContent = name;
    valueCell.textContent = safeNumber(sig.value);
    const dot = document.createElement('span');
    dot.className = `state-dot ${sig.healthy ? 'ok' : ''}`;
    dot.title = sig.healthy ? 'healthy' : 'warning';
    stateCell.appendChild(dot);
    tr.append(nameCell, valueCell, stateCell);
    tbody.appendChild(tr);
  }

  const wlist = document.getElementById('warnings-list');
  const count = document.getElementById('warning-count');
  const warns = data.active_warnings || [];
  count.textContent = `${warns.length} active`;
  wlist.innerHTML = '';
  if (warns.length === 0) {
    const empty = document.createElement('span');
    empty.className = 'muted';
    empty.textContent = 'None';
    wlist.appendChild(empty);
    return;
  }
  warns.forEach(w => {
    const item = document.createElement('div');
    item.className = 'warn-item';
    const title = document.createElement('div');
    title.className = 'warn-title';
    title.textContent = w.signal || 'warning';
    const text = document.createElement('div');
    text.className = 'warn-text';
    text.textContent = w.suggestion || '';
    item.append(title, text);
    wlist.appendChild(item);
  });
}

function renderGraph(data) {
  const nodes = data.nodes || [];
  const links = data.links || [];
  const stats = data.stats || {};
  const graphStats = document.getElementById('graph-stats');
  graphStats.innerHTML = '';
  [
    `${safeNumber(stats.grains || 0)} grains`,
    `${safeNumber(stats.entries || 0)} entries`,
    `${safeNumber(stats.conduits || 0)} conduits`,
    `${safeNumber(stats.embeddings || 0)} embeddings`
  ].forEach(text => {
    const pill = document.createElement('span');
    pill.className = 'pill';
    pill.textContent = text;
    graphStats.appendChild(pill);
  });

  const svg = document.getElementById('graph-svg');
  const note = document.getElementById('empty-note');
  const W = Math.max(svg.clientWidth || 960, 480);
  const H = Math.max(svg.clientHeight || 430, 300);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = '';
  note.hidden = true;

  if (nodes.length === 0) {
    note.hidden = false;
    note.textContent = 'No graph data is stored yet.';
    return;
  }

  const degree = {};
  links.forEach(l => {
    degree[l.source] = (degree[l.source] || 0) + 1;
    degree[l.target] = (degree[l.target] || 0) + 1;
  });

  const entries = nodes.filter(n => n.node_type === 'entry');
  const grains = nodes.filter(n => n.node_type !== 'entry');
  const pos = {};

  if (links.length === 0) {
    note.hidden = false;
    note.textContent = 'No conduits are stored, so grains are currently isolated.';
    layoutBand(entries, W * 0.22, W * 0.14, H, pos);
    layoutBand(grains, W * 0.67, W * 0.26, H, pos);
  } else {
    layoutLinked(nodes, links, degree, W, H, pos);
  }

  links.forEach(l => {
    const s = pos[l.source], t = pos[l.target];
    if (!s || !t) return;
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const midX = (s.x + t.x) / 2;
    const midY = (s.y + t.y) / 2 - 18;
    const w = Math.max(0.8, (l.effective_weight || 0.3) * 5);
    const color = l.effective_weight > 0.5 ? 'var(--green)' : l.effective_weight > 0.25 ? 'var(--amber)' : 'var(--rose)';
    path.setAttribute('d', `M ${s.x} ${s.y} Q ${midX} ${midY} ${t.x} ${t.y}`);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', color);
    path.setAttribute('stroke-width', w);
    path.setAttribute('opacity', '0.58');
    svg.appendChild(path);
  });

  nodes.forEach(n => {
    const p = pos[n.id];
    if (!p) return;
    const isEntry = n.node_type === 'entry';
    const isCore = n.decay_class === 'core';
    const r = Math.min(12, 4.5 + Math.sqrt(degree[n.id] || 0) * 1.5 + (isEntry ? 1 : 0));
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', p.x);
    circle.setAttribute('cy', p.y);
    circle.setAttribute('r', r);
    circle.setAttribute('fill', isEntry ? 'var(--teal)' : isCore ? 'var(--amber)' : 'var(--blue)');
    circle.setAttribute('stroke', '#0b0d12');
    circle.setAttribute('stroke-width', '1.5');
    circle.setAttribute('opacity', n.status === 'dormant' ? '0.34' : '0.88');
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = n.label || n.id;
    circle.appendChild(title);
    svg.appendChild(circle);
  });
}

function layoutBand(items, centerX, radiusX, height, pos) {
  const rows = Math.max(1, Math.ceil(Math.sqrt(items.length || 1)));
  items.forEach((n, i) => {
    const row = i % rows;
    const col = Math.floor(i / rows);
    const x = centerX + ((col % 6) - 2.5) * Math.max(18, radiusX / 4);
    const y = 54 + row * ((height - 108) / Math.max(rows - 1, 1));
    pos[n.id] = { x, y };
  });
}

function layoutLinked(nodes, links, degree, W, H, pos) {
  const sorted = [...nodes].sort((a, b) => (degree[b.id] || 0) - (degree[a.id] || 0));
  sorted.forEach((n, i) => {
    const ring = i < 10 ? 0 : i < 40 ? 1 : 2;
    const ringIndex = ring === 0 ? i : ring === 1 ? i - 10 : i - 40;
    const ringCount = ring === 0 ? Math.min(10, sorted.length) : ring === 1 ? Math.min(30, sorted.length - 10) : Math.max(1, sorted.length - 40);
    const angle = (2 * Math.PI * ringIndex) / Math.max(ringCount, 1) + ring * 0.26;
    const rx = W * (0.16 + ring * 0.13);
    const ry = H * (0.15 + ring * 0.12);
    pos[n.id] = { x: W / 2 + rx * Math.cos(angle), y: H / 2 + ry * Math.sin(angle) };
  });
}

async function refresh() {
  try {
    const [health, graph] = await Promise.all([
      fetchJSON('/api/health'),
      fetchJSON('/api/graph'),
    ]);
    renderMetrics(health, graph);
    renderHealth(health);
    renderGraph(graph);
  } catch (e) {
    console.error('refresh failed', e);
  }
}

document.getElementById('refresh-button').addEventListener('click', refresh);
refresh();
setInterval(refresh, 15000);
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
