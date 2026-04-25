"""Interactive web dashboard for Flux Memory.

Serves a single-page operational dashboard over HTTP. The page fetches:
  GET /api/health     -> flux_health() JSON
  GET /api/graph      -> export_json() node-link JSON
  GET /api/clusters   -> cluster_view() JSON
  GET /api/events     -> recent structured events
"""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

_DASHBOARD_HTML = r"""<!DOCTYPE html><html lang="en"><head>


<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Flux Memory — Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
  :root {
    --bg: #080a0e;
    --surface: #0f1117;
    --surface2: #161921;
    --border: #1e2330;
    --border2: #252c3a;
    --text: #dde3f0;
    --text-muted: #5a6480;
    --text-dim: #3a4255;
    --cyan: #22d3ee;
    --cyan-dim: rgba(34,211,238,0.12);
    --green: #4ade80;
    --green-dim: rgba(74,222,128,0.12);
    --amber: #fbbf24;
    --amber-dim: rgba(251,191,36,0.12);
    --rose: #f43f5e;
    --rose-dim: rgba(244,63,94,0.12);
    --violet: #a78bfa;
    --violet-dim: rgba(167,139,250,0.12);
    --radius: 6px;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    --font-ui: 'Inter', system-ui, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--font-ui); font-size: 13px; overflow: hidden; }

  /* LAYOUT */
  #app { display: flex; flex-direction: column; height: 100vh; }

  /* TOP BAR */
  #topbar {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 16px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    flex-wrap: wrap;
  }
  #topbar .brand { display: flex; align-items: center; gap: 8px; margin-right: 4px; }
  #topbar .brand svg { flex-shrink: 0; }
  #topbar .brand-name { font-size: 14px; font-weight: 600; letter-spacing: -0.3px; color: var(--text); }
  #topbar .brand-sub { font-size: 11px; color: var(--text-muted); }
  #status-badge {
    display: flex; align-items: center; gap: 5px;
    padding: 3px 9px; border-radius: 20px; font-size: 11px; font-weight: 600;
    letter-spacing: 0.4px; text-transform: uppercase;
  }
  .badge-healthy { background: var(--green-dim); color: var(--green); border: 1px solid rgba(74,222,128,0.3); }
  .badge-warning { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(251,191,36,0.3); }
  .badge-critical { background: var(--rose-dim); color: var(--rose); border: 1px solid rgba(244,63,94,0.3); }
  .pulse { width: 6px; height: 6px; border-radius: 50%; animation: pulse 2s ease-in-out infinite; }
  .pulse-green { background: var(--green); box-shadow: 0 0 0 0 rgba(74,222,128,0.6); }
  .pulse-amber { background: var(--amber); box-shadow: 0 0 0 0 rgba(251,191,36,0.6); }
  .pulse-rose { background: var(--rose); box-shadow: 0 0 0 0 rgba(244,63,94,0.6); }
  @keyframes pulse {
    0%,100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.4); opacity: 0.6; }
  }
  .divider { width: 1px; height: 20px; background: var(--border2); flex-shrink: 0; }
  #metrics-strip { display: flex; gap: 2px; flex: 1; flex-wrap: wrap; }
  .metric-pill {
    display: flex; flex-direction: column; align-items: flex-start;
    padding: 4px 12px; background: var(--surface2); border: 1px solid var(--border);
    border-radius: var(--radius); min-width: 80px; cursor: default;
    transition: border-color 0.15s;
  }
  .metric-pill:hover { border-color: var(--border2); }
  .metric-pill .m-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .metric-pill .m-value { font-size: 16px; font-weight: 700; font-family: var(--font-mono); color: var(--text); line-height: 1.2; margin-top: 1px; }
  .metric-pill .m-value.c-cyan { color: var(--cyan); }
  .metric-pill .m-value.c-green { color: var(--green); }
  .metric-pill .m-value.c-amber { color: var(--amber); }
  .metric-pill .m-value.c-rose { color: var(--rose); }
  #topbar-controls { display: flex; align-items: center; gap: 6px; margin-left: auto; }
  #computed-at { font-size: 10px; color: var(--text-dim); font-family: var(--font-mono); white-space: nowrap; }
  .icon-btn {
    display: flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: var(--radius);
    background: var(--surface2); border: 1px solid var(--border);
    color: var(--text-muted); cursor: pointer; transition: all 0.15s;
  }
  .icon-btn:hover { color: var(--text); border-color: var(--border2); }
  .icon-btn.active { color: var(--cyan); border-color: rgba(34,211,238,0.3); background: var(--cyan-dim); }
  #refresh-interval { font-size: 11px; color: var(--text-muted); }

  /* MAIN AREA */
  #main { display: flex; flex: 1; overflow: hidden; }

  /* LEFT: GRAPH PANEL */
  #graph-panel { flex: 1; display: flex; flex-direction: column; min-width: 0; border-right: 1px solid var(--border); }
  #graph-toolbar {
    display: flex; align-items: center; gap: 8px; padding: 8px 12px;
    background: var(--surface); border-bottom: 1px solid var(--border); flex-shrink: 0; flex-wrap: wrap;
  }
  #search-box {
    display: flex; align-items: center; gap: 6px;
    background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 4px 8px; flex: 1; max-width: 220px;
  }
  #search-box input {
    background: none; border: none; outline: none; color: var(--text);
    font-size: 12px; font-family: var(--font-ui); width: 100%;
  }
  #search-box input::placeholder { color: var(--text-dim); }
  .filter-group { display: flex; align-items: center; gap: 3px; }
  .filter-btn {
    padding: 3px 8px; border-radius: var(--radius); font-size: 11px; font-weight: 500;
    border: 1px solid var(--border); background: var(--surface2); color: var(--text-muted);
    cursor: pointer; transition: all 0.15s;
  }
  .filter-btn.active { background: var(--cyan-dim); border-color: rgba(34,211,238,0.35); color: var(--cyan); }
  .filter-btn.active.violet { background: var(--violet-dim); border-color: rgba(167,139,250,0.35); color: var(--violet); }
  .slider-group { display: flex; align-items: center; gap: 6px; }
  .slider-group label { font-size: 10px; color: var(--text-muted); white-space: nowrap; text-transform: uppercase; letter-spacing: 0.4px; }
  #weight-slider { width: 80px; accent-color: var(--cyan); cursor: pointer; }
  #weight-val { font-size: 11px; font-family: var(--font-mono); color: var(--cyan); min-width: 28px; }
  #graph-container { flex: 1; position: relative; overflow: hidden; }
  #graph-canvas { width: 100%; height: 100%; display: block; cursor: grab; }
  #graph-canvas.dragging { cursor: grabbing; }
  #graph-overlay {
    position: absolute; bottom: 12px; left: 12px;
    display: flex; gap: 8px; flex-wrap: wrap;
  }
  .legend-item { display: flex; align-items: center; gap: 4px; font-size: 10px; color: var(--text-muted); }
  .legend-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  #tooltip {
    position: fixed; pointer-events: none; z-index: 999;
    background: var(--surface); border: 1px solid var(--border2);
    border-radius: var(--radius); padding: 8px 10px; max-width: 260px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5); opacity: 0; transition: opacity 0.1s;
    font-size: 11px; line-height: 1.6;
  }
  #tooltip .tt-title { font-size: 12px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
  #tooltip .tt-row { display: flex; justify-content: space-between; gap: 12px; }
  #tooltip .tt-key { color: var(--text-muted); }
  #tooltip .tt-val { font-family: var(--font-mono); color: var(--text); }
  #graph-stats {
    position: absolute; top: 10px; right: 10px;
    background: rgba(15,17,23,0.8); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 5px 9px; font-size: 10px;
    font-family: var(--font-mono); color: var(--text-muted);
    backdrop-filter: blur(4px);
  }
  #no-data-msg {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 8px;
    color: var(--text-dim); font-size: 13px; display: none;
  }

  /* RIGHT PANEL */
  #right-panel {
    width: 320px; flex-shrink: 0; display: flex; flex-direction: column;
    background: var(--surface); overflow-y: auto; overflow-x: hidden;
  }
  .rpanel-section {
    border-bottom: 1px solid var(--border); flex-shrink: 0;
  }
  .rpanel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; cursor: pointer; user-select: none;
  }
  .rpanel-header-left { display: flex; align-items: center; gap: 6px; }
  .rpanel-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; color: var(--text-muted); }
  .rpanel-count {
    font-size: 10px; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 10px; padding: 1px 6px; font-family: var(--font-mono); color: var(--text-muted);
  }
  .rpanel-count.warn { background: var(--amber-dim); border-color: rgba(251,191,36,0.3); color: var(--amber); }
  .rpanel-body { padding: 0 14px 12px; }
  .rpanel-chevron { color: var(--text-dim); transition: transform 0.2s; }
  .rpanel-chevron.open { transform: rotate(180deg); }

  /* INSPECTOR */
  #inspector-empty {
    padding: 24px 14px; text-align: center; color: var(--text-dim); font-size: 11px; line-height: 1.6;
  }
  .inspector-node-header { display: flex; align-items: flex-start; gap: 8px; margin-bottom: 10px; }
  .inspector-icon { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
  .inspector-label { font-size: 12px; font-weight: 600; color: var(--text); line-height: 1.4; word-break: break-word; }
  .inspector-type { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.4px; margin-top: 2px; }
  .inspector-rows { display: flex; flex-direction: column; gap: 4px; }
  .inspector-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; padding: 3px 0; border-bottom: 1px solid var(--border); }
  .inspector-row:last-child { border-bottom: none; }
  .ir-key { font-size: 11px; color: var(--text-muted); flex-shrink: 0; }
  .ir-val { font-size: 11px; font-family: var(--font-mono); color: var(--text); text-align: right; word-break: break-all; }
  .tag { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: 500; }
  .tag-cyan { background: var(--cyan-dim); color: var(--cyan); }
  .tag-violet { background: var(--violet-dim); color: var(--violet); }
  .tag-green { background: var(--green-dim); color: var(--green); }
  .tag-amber { background: var(--amber-dim); color: var(--amber); }
  .tag-rose { background: var(--rose-dim); color: var(--rose); }
  .tag-dim { background: var(--surface2); color: var(--text-muted); }

  /* WARNINGS */
  .warning-card {
    background: var(--surface2); border: 1px solid rgba(251,191,36,0.2);
    border-radius: var(--radius); padding: 8px 10px; margin-bottom: 6px;
  }
  .warning-card:last-child { margin-bottom: 0; }
  .warning-signal { font-size: 11px; font-weight: 600; color: var(--amber); font-family: var(--font-mono); }
  .warning-msg { font-size: 11px; color: var(--text-muted); margin-top: 3px; line-height: 1.5; }
  .warning-val { font-size: 10px; color: var(--text-dim); margin-top: 4px; font-family: var(--font-mono); }

  /* HEALTH TABLE */
  .health-group { margin-bottom: 10px; }
  .health-group-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-dim); margin-bottom: 4px; padding-bottom: 3px; border-bottom: 1px solid var(--border); }
  .health-row { display: flex; align-items: center; gap: 6px; padding: 3px 0; }
  .health-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
  .health-row-key { flex: 1; font-size: 11px; color: var(--text-muted); }
  .health-row-val { font-size: 11px; font-family: var(--font-mono); color: var(--text); min-width: 44px; text-align: right; }

  /* SCROLLBAR */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

  /* NODE BREATHING */
  @keyframes node-breathe {
    0%, 100% { r: 8; opacity: 0.9; }
    50% { r: 9.5; opacity: 1; }
  }
  @keyframes core-breathe {
    0%, 100% { r: 10; opacity: 0.9; }
    50% { r: 12; opacity: 1; }
  }
  @keyframes glow-pulse {
    0%, 100% { opacity: 0.1; r: 14; }
    50% { opacity: 0.28; r: 18; }
  }
  .node-grain-working { animation: node-breathe 3.5s ease-in-out infinite; }
  .node-grain-core { animation: core-breathe 2.8s ease-in-out infinite; }
  .node-glow { animation: glow-pulse 2.8s ease-in-out infinite; }

  /* LOADING OVERLAY */
  #loading {
    position: fixed; inset: 0; background: var(--bg);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 16px; z-index: 1000;
  }
  .loader-ring {
    width: 40px; height: 40px; border-radius: 50%;
    border: 2px solid var(--border2); border-top-color: var(--cyan);
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #loading .load-text { font-size: 12px; color: var(--text-muted); font-family: var(--font-mono); }

  /* RESPONSIVE */
  @media (max-width: 768px) {
    #main { flex-direction: column; }
    #right-panel { width: 100%; height: 300px; border-top: 1px solid var(--border); }
    #graph-panel { border-right: none; }
  }
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&amp;family=JetBrains+Mono:wght@400;600&amp;display=swap" rel="stylesheet">
</head>
<body>

<div id="loading">
  <div class="loader-ring"></div>
  <div class="load-text">Initialising Flux Memory…</div>
</div>

<div id="tooltip" style="opacity: 0;">
  <div class="tt-title" id="tt-title"></div>
  <div id="tt-body"></div>
</div>

<div id="app">
  <!-- TOP BAR -->
  <div id="topbar">
    <div class="brand">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="#22d3ee" stroke-width="1.5" opacity="0.3"></circle>
        <circle cx="12" cy="12" r="6" stroke="#22d3ee" stroke-width="1.5" opacity="0.6"></circle>
        <circle cx="12" cy="12" r="2.5" fill="#22d3ee"></circle>
        <line x1="12" y1="2" x2="12" y2="6" stroke="#22d3ee" stroke-width="1.5" opacity="0.5" stroke-opacity="0.7"></line>
        <line x1="12" y1="18" x2="12" y2="22" stroke="#22d3ee" stroke-width="1.5" opacity="0.5" stroke-opacity="0.7"></line>
        <line x1="2" y1="12" x2="6" y2="12" stroke="#22d3ee" stroke-width="1.5" opacity="0.5" stroke-opacity="0.7"></line>
        <line x1="18" y1="12" x2="22" y2="12" stroke="#22d3ee" stroke-width="1.5" opacity="0.5" stroke-opacity="0.7"></line>
      </svg>
      <div>
        <div class="brand-name">Flux Memory</div>
        <div class="brand-sub" id="instance-name">test1</div>
      </div>
    </div>
    <div id="status-badge" class="badge-warning">
      <div class="pulse pulse-amber"></div>
      <span id="status-text">WARNING</span>
    </div>
    <div class="divider"></div>
    <div id="metrics-strip">
      <div class="metric-pill"><span class="m-label">Grains</span><span class="m-value c-cyan" id="m-grains">-</span></div>
      <div class="metric-pill"><span class="m-label">Entries</span><span class="m-value" id="m-entries">-</span></div>
      <div class="metric-pill"><span class="m-label">Conduits</span><span class="m-value" id="m-conduits">-</span></div>
      <div class="metric-pill"><span class="m-label">Embeddings</span><span class="m-value" id="m-embed">-</span></div>
      <div class="metric-pill"><span class="m-label">Retrieval</span><span class="m-value c-green" id="m-retrieval">-</span></div>
      <div class="metric-pill"><span class="m-label">Fallback</span><span class="m-value c-rose" id="m-fallback">-</span></div>
      <div class="metric-pill"><span class="m-label">Feedback</span><span class="m-value c-rose" id="m-feedback">-</span></div>
    </div>
    <div class="divider"></div>
    <div id="topbar-controls">
      <span id="computed-at">loading</span>
      <button class="icon-btn" id="refresh-btn" title="Refresh now" onclick="fetchAll()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
      </button>
      <button class="icon-btn" id="autorefresh-btn" title="Toggle auto-refresh" onclick="toggleAutoRefresh()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
      </button>
      <span id="refresh-interval" style="display: none;">30s</span>
    </div>
  </div>

  <!-- MAIN -->
  <div id="main">

    <!-- GRAPH PANEL -->
    <div id="graph-panel">
      <div id="graph-toolbar">
        <div id="search-box">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#5a6480" stroke-width="2.5"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65" stroke-opacity="0.7"></line></svg>
          <input type="text" id="search-input" placeholder="Search nodes…" oninput="onSearch(this.value)">
        </div>
        <div class="filter-group">
          <span style="font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:.4px">Type</span>
          <button class="filter-btn active" id="f-grain" onclick="toggleFilter('grain')">Grains</button>
          <button class="filter-btn violet active" id="f-entry" onclick="toggleFilter('entry')">Entries</button>
        </div>
        <div class="filter-group">
          <span style="font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:.4px">Status</span>
          <button class="filter-btn active" id="f-active" onclick="toggleFilter('active')">Active</button>
          <button class="filter-btn" id="f-dormant" onclick="toggleFilter('dormant')">Dormant</button>
        </div>
        <div class="slider-group">
          <label>Weight ≥</label>
          <input type="range" id="weight-slider" min="0" max="1" step="0.05" value="0" oninput="onWeightChange(this.value)">
          <span id="weight-val">0.00</span>
        </div>
        <div style="margin-left:auto;display:flex;gap:4px">
          <button class="icon-btn" id="pause-btn" title="Pause/resume layout" onclick="toggleSimulation()">
            <svg id="pause-icon" width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>
          </button>
          <button class="icon-btn" title="Reset view" onclick="resetView()">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path></svg>
          </button>
        </div>
      </div>

      <div id="graph-container">
        <canvas id="graph-canvas" style="width:100%;height:100%;display:block;"></canvas>
        <div id="graph-stats">nodes: <span id="gs-nodes">0</span> · edges: <span id="gs-edges">0</span></div>
        <div id="graph-overlay">
          <div class="legend-item"><div class="legend-dot" style="background:#22d3ee"></div>Grain (working)</div>
          <div class="legend-item"><div class="legend-dot" style="background:#e0f2fe"></div>Grain (core)</div>
          <div class="legend-item"><div class="legend-dot" style="background:#a78bfa"></div>Entry</div>
          <div class="legend-item"><div class="legend-dot" style="background:#334155"></div>Dormant</div>
        </div>
        <div id="no-data-msg" style="display: none;">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.4"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12" stroke-opacity="0.7"></line><line x1="12" y1="16" x2="12.01" y2="16" stroke-opacity="0.7"></line></svg>
          No graph data
        </div>
      </div>
    </div>

    <!-- RIGHT PANEL -->
    <div id="right-panel">

      <!-- INSPECTOR -->
      <div class="rpanel-section">
        <div class="rpanel-header" onclick="toggleSection('inspector')">
          <div class="rpanel-header-left">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65" stroke-opacity="0.7"></line></svg>
            <span class="rpanel-title">Inspector</span>
          </div>
          <svg class="rpanel-chevron open" id="chev-inspector" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="18 15 12 9 6 15"></polyline></svg>
        </div>
        <div class="rpanel-body" id="body-inspector">
          <div id="inspector-empty">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3"><rect x="3" y="3" width="18" height="18" rx="2"></rect><line x1="9" y1="9" x2="15" y2="15" stroke-opacity="0.7"></line><line x1="15" y1="9" x2="9" y2="15" stroke-opacity="0.7"></line></svg>
            <div style="margin-top:8px">Click a node or edge<br>to inspect its properties</div>
          </div>
          <div id="inspector-content" style="display: none;"></div>
        </div>
      </div>

      <!-- WARNINGS -->
      <div class="rpanel-section">
        <div class="rpanel-header" onclick="toggleSection('warnings')">
          <div class="rpanel-header-left">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13" stroke-opacity="0.7"></line><line x1="12" y1="17" x2="12.01" y2="17" stroke-opacity="0.7"></line></svg>
            <span class="rpanel-title">Warnings</span>
            <span class="rpanel-count warn" id="warn-count">0</span>
          </div>
          <svg class="rpanel-chevron open" id="chev-warnings" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="18 15 12 9 6 15"></polyline></svg>
        </div>
        <div class="rpanel-body" id="body-warnings">
          <div id="warnings-list" style="color:var(--text-dim);font-size:11px">Loading warnings...</div>
        </div>
      </div>

      <!-- HEALTH SIGNALS -->
      <div class="rpanel-section">
        <div class="rpanel-header" onclick="toggleSection('health')">
          <div class="rpanel-header-left">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            <span class="rpanel-title">Health Signals</span>
          </div>
          <svg class="rpanel-chevron open" id="chev-health" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="18 15 12 9 6 15"></polyline></svg>
        </div>
        <div class="rpanel-body" id="body-health">
          <div id="health-table" style="color:var(--text-dim);font-size:11px">Loading health signals...</div>
        </div>
      </div>

    </div>
  </div>
</div>

<script>
// -- STATE ---------------------------------------------------------------------
let healthData = null, graphData = null;
let loadError = null;
let simulation = null, svg = null, g = null, zoom = null;
let animFrame3D = null;
let loop3D = null;
let activeFilters = {grain:true, entry:true, active:true, dormant:false};
let weightThreshold = 0;
let searchQuery = '';
let selectedNode = null, selectedEdge = null;
let simulationPaused = false;
let autoRefreshTimer = null;
let isAutoRefresh = false;

// ── FETCH ─────────────────────────────────────────────────────────────────────
function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}

async function fetchJSON(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
}

async function fetchAll() {
  try {
    const [h, gr] = await Promise.all([
      fetchJSON('/api/health'),
      fetchJSON('/api/graph'),
    ]);
    healthData = h;
    graphData = gr;
    loadError = null;
    renderHealth();
    renderGraph();
    document.getElementById('loading').style.display = 'none';
  } catch (err) {
    loadError = err;
    console.error('Dashboard fetch failed', err);
    document.getElementById('loading').style.display = 'none';
    showLoadError(err.message || String(err));
  }
}

function showLoadError(message) {
  const noData = document.getElementById('no-data-msg');
  noData.style.display = 'flex';
  noData.innerHTML = `<div style="color:var(--rose);font-weight:600">Unable to load Flux data</div><div style="font-size:11px;color:var(--text-dim);margin-top:6px">${esc(message)}</div>`;
  document.getElementById('warnings-list').textContent = 'Flux API is unavailable.';
  document.getElementById('health-table').textContent = 'No health data available.';
}

function toggleAutoRefresh() {
  isAutoRefresh = !isAutoRefresh;
  const btn = document.getElementById('autorefresh-btn');
  const lbl = document.getElementById('refresh-interval');
  btn.classList.toggle('active', isAutoRefresh);
  lbl.style.display = isAutoRefresh ? '' : 'none';
  if (isAutoRefresh) {
    autoRefreshTimer = setInterval(() => { fetchAll(); }, 30000);
  } else {
    clearInterval(autoRefreshTimer);
  }
}

// ── HEALTH RENDER ─────────────────────────────────────────────────────────────
function renderHealth() {
  if (!healthData) return;
  const h = healthData;
  const status = ['healthy','warning','critical'].includes(h.status) ? h.status : 'warning';

  // Status badge
  const badge = document.getElementById('status-badge');
  const pulse = badge.querySelector('.pulse');
  const statusText = document.getElementById('status-text');
  badge.className = 'badge-' + status;
  pulse.className = 'pulse pulse-' + (status==='healthy'?'green':status==='warning'?'amber':'rose');
  statusText.textContent = status.toUpperCase();

  // Computed at
  const ca = h.computed_at ? new Date(h.computed_at) : null;
  document.getElementById('computed-at').textContent = ca && !Number.isNaN(ca.valueOf()) ? ca.toLocaleTimeString() : '-';

  // Metrics strip
  const s = h.signals || {};
  function pct(v) { return (v*100).toFixed(0) + '%'; }
  const rs = s.retrieval_success_rate?.value ?? 0;
  const fb = s.fallback_trigger_rate?.value ?? 0;
  const fc = s.feedback_compliance_rate?.value ?? 0;
  document.getElementById('m-retrieval').textContent = pct(rs);
  document.getElementById('m-retrieval').className = 'm-value ' + (rs>=0.7?'c-green':'c-rose');
  document.getElementById('m-fallback').textContent = pct(fb);
  document.getElementById('m-fallback').className = 'm-value ' + (fb<0.3?'c-green':fb<0.7?'c-amber':'c-rose');
  document.getElementById('m-feedback').textContent = pct(fc);
  document.getElementById('m-feedback').className = 'm-value ' + (fc>=0.8?'c-green':'c-rose');

  // Graph stats from graphData
  if (graphData?.stats) {
    const st = graphData.stats;
    document.getElementById('m-grains').textContent = st.grains ?? '—';
    document.getElementById('m-entries').textContent = st.entries ?? '—';
    document.getElementById('m-conduits').textContent = st.conduits ?? '—';
    document.getElementById('m-embed').textContent = st.embeddings ?? '—';
  }

  // Warnings
  const warns = h.active_warnings || [];
  document.getElementById('warn-count').textContent = warns.length;
  document.getElementById('warn-count').className = 'rpanel-count' + (warns.length ? ' warn' : '');
  const wl = document.getElementById('warnings-list');
  if (!warns.length) {
    wl.innerHTML = '<div style="color:var(--text-dim);font-size:11px">No active warnings</div>';
  } else {
    wl.innerHTML = warns.map(w => `
      <div class="warning-card">
        <div class="warning-signal">${esc(w.signal)}</div>
        <div class="warning-msg">${esc(w.suggestion)}</div>
        <div class="warning-val">Value: ${esc(w.current_value)} · Range: ${esc(w.healthy_range)} · Severity: ${esc(w.severity)}</div>
      </div>`).join('');
  }

  // Health table
  const groups = {
    'Retrieval': ['retrieval_success_rate','avg_hops_per_retrieval','fallback_trigger_rate'],
    'Feedback': ['feedback_compliance_rate','promotion_events'],
    'Graph': ['highway_count','highway_growth_rate','orphan_rate','core_grain_count','avg_conduit_weight'],
    'Decay': ['dormant_grain_rate','conduit_dissolution_rate','avg_weight_drop_on_failure','shortcut_creation_rate'],
  };
  let html = '';
  for (const [group, keys] of Object.entries(groups)) {
    html += `<div class="health-group"><div class="health-group-label">${esc(group)}</div>`;
    for (const k of keys) {
      const sig = s[k]; if (!sig) continue;
      const ok = sig.healthy;
      const v = sig.value;
      const display = (v > 0 && v <= 1 && k.includes('rate')) ? (v*100).toFixed(0)+'%' : v.toFixed ? v.toFixed(2) : v;
      html += `<div class="health-row">
        <div class="health-dot" style="background:${ok?'var(--green)':'var(--rose)'}"></div>
        <div class="health-row-key">${esc(k.replace(/_/g,' '))}</div>
        <div class="health-row-val" style="color:${ok?'var(--text)':'var(--rose)'}">${esc(display)}</div>
      </div>`;
    }
    html += '</div>';
  }
  document.getElementById('health-table').innerHTML = html;
}

// ── GRAPH RENDER ──────────────────────────────────────────────────────────────
function nodeColor(n) {
  if (n.status === 'dormant') return '#1e2d40';
  if (n.node_type === 'entry') return '#7c3aed';
  if (n.decay_class === 'core') return '#e0f2fe';
  if (n.decay_class === 'ephemeral') return '#0e7490';
  return '#22d3ee';
}
function nodeRadius(n) {
  if (n.node_type === 'entry') return 6;
  if (n.decay_class === 'core') return 10;
  return 8;
}
function edgeColor(l) {
  if (l.direction === 'bidirectional') return '#a78bfa';
  if (l.decay_class === 'core') return '#38bdf8';
  if (l.decay_class === 'ephemeral') return '#1e3a4a';
  return '#1e3f5a';
}

function nodePassesFilters(n) {
  const typeOk = (n.node_type === 'grain' && activeFilters.grain) || (n.node_type === 'entry' && activeFilters.entry);
  const statusOk = (n.status === 'active' || !n.status) ? activeFilters.active : (n.status === 'dormant' ? activeFilters.dormant : true);
  return typeOk && statusOk;
}

function nodeMatchesSearch(n) {
  if (!searchQuery) return true;
  const haystack = [
    n.label,
    n.id,
    n.feature,
    n.provenance,
    n.decay_class,
    n.status,
  ].filter(Boolean).join(' ').toLowerCase();
  return haystack.includes(searchQuery);
}

function getVisibleNodes() {
  if (!graphData) return [];
  const baseNodes = graphData.nodes.filter(nodePassesFilters);
  if (!searchQuery) return baseNodes;

  const baseIds = new Set(baseNodes.map(n => n.id));
  const matchedIds = new Set(baseNodes.filter(nodeMatchesSearch).map(n => n.id));
  const visibleIds = new Set(matchedIds);

  for (const link of graphData.links || []) {
    const sid = typeof link.source === 'object' ? link.source.id : link.source;
    const tid = typeof link.target === 'object' ? link.target.id : link.target;
    if (matchedIds.has(sid) && baseIds.has(tid)) visibleIds.add(tid);
    if (matchedIds.has(tid) && baseIds.has(sid)) visibleIds.add(sid);
  }

  return baseNodes.filter(n => visibleIds.has(n.id));
}

function getVisibleLinks(nodeIds) {
  if (!graphData) return [];
  const idSet = new Set(nodeIds);
  return graphData.links.filter(l => {
    const sid = typeof l.source === 'object' ? l.source.id : l.source;
    const tid = typeof l.target === 'object' ? l.target.id : l.target;
    return idSet.has(sid) && idSet.has(tid) && (l.effective_weight ?? l.weight ?? 0) >= weightThreshold;
  });
}


// ── CANVAS STATE ─────────────────────────────────────────────────────────────
let canvas = null, ctx = null;
let canvasNodes = [], canvasLinks = [];
let transform = d3.zoomIdentity;
let hoveredNode = null, hoveredEdge = null;
let dashOffset = 0;
let nudgeTimer = null;

function renderGraph() {
  const container = document.getElementById('graph-container');
  const W = container.clientWidth, H = container.clientHeight;

  if (!graphData || !(graphData.nodes || []).length) {
    document.getElementById('no-data-msg').style.display = 'flex';
    return;
  }
  document.getElementById('no-data-msg').style.display = 'none';

  const visNodes = getVisibleNodes();
  const visLinks = getVisibleLinks(visNodes.map(n => n.id));
  document.getElementById('gs-nodes').textContent = visNodes.length;
  document.getElementById('gs-edges').textContent = visLinks.length;

  // Deep copy so D3 can mutate
  canvasNodes = visNodes.map(n => ({...n}));
  const nodeMap = new Map(canvasNodes.map(n => [n.id, n]));
  assignPhases(canvasNodes);
  canvasLinks = visLinks.map(l => ({
    ...l,
    source: nodeMap.get(typeof l.source==='object'?l.source.id:l.source) || (typeof l.source==='object'?l.source.id:l.source),
    target: nodeMap.get(typeof l.target==='object'?l.target.id:l.target) || (typeof l.target==='object'?l.target.id:l.target),
  })).filter(l => l.source && l.target && typeof l.source==='object' && typeof l.target==='object');

  // Set up canvas
  canvas = document.getElementById('graph-canvas');
  canvas.width = W; canvas.height = H;
  ctx = canvas.getContext('2d');

  // Cancel previous rAF
  if (animFrame3D) { cancelAnimationFrame(animFrame3D); animFrame3D = null; }
  if (nudgeTimer) { clearInterval(nudgeTimer); nudgeTimer = null; }

  // D3 zoom on canvas
  zoom = d3.zoom().scaleExtent([0.1, 5]).on('zoom', e => { transform = e.transform; });
  d3.select(canvas).call(zoom);

  // D3 drag on canvas via pointer events
  canvas.onmousedown = onCanvasMouseDown;
  canvas.onmousemove = onCanvasMouseMove;
  canvas.onmouseup = onCanvasMouseUp;
  canvas.onclick = onCanvasClick;
  canvas.onmouseleave = () => { hideTooltip(); hoveredNode = null; hoveredEdge = null; };

  // Force simulation
  if (simulation) simulation.stop();
  simulation = d3.forceSimulation(canvasNodes)
    .force('link', d3.forceLink(canvasLinks).id(d=>d.id).distance(d => 60 + (1-(d.effective_weight??0.3))*60).strength(0.4))
    .force('charge', d3.forceManyBody().strength(-180).distanceMax(300))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collision', d3.forceCollide(d => nodeRadius(d)+8))
    .alphaDecay(0.015).alphaMin(0.001).alphaTarget(0.004)
    .on('tick', draw);

  // Gentle nudge every 5s
  nudgeTimer = setInterval(() => {
    if (simulation && !simulationPaused) {
      canvasNodes.forEach(n => {
        n.vx = (n.vx||0) + (Math.random()-0.5)*0.18;
        n.vy = (n.vy||0) + (Math.random()-0.5)*0.18;
      });
      simulation.alphaTarget(0.004).restart();
    }
  }, 5000);

  // Dash animation rAF
  loop3D = function() {
    animFrame3D = requestAnimationFrame(loop3D);
    dashOffset -= 0.5;
    draw();
  };
  loop3D();

  if (simulationPaused) { simulation.alphaTarget(0).stop(); }
}

// ── CANVAS DRAW ───────────────────────────────────────────────────────────────
function draw() {
  if (!ctx || !canvas) return;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.save();
  ctx.translate(transform.x, transform.y);
  ctx.scale(transform.k, transform.k);

  const t = performance.now() / 1000;

  // Draw edges
  for (const l of canvasLinks) {
    const sx = l.source.x, sy = l.source.y;
    const tx = l.target.x, ty = l.target.y;
    if (sx==null||sy==null||tx==null||ty==null) continue;

    const isSelected = selectedEdge === l;
    const isHovered = hoveredEdge === l;
    const isDimmed = (selectedNode && l.source !== selectedNode && l.target !== selectedNode);

    const w = Math.max(0.5, (l.effective_weight??0.3)*2.5);
    let alpha = isDimmed ? 0.04 : (isSelected||isHovered) ? 1 : 0.55;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = isSelected ? '#22d3ee' : edgeColor(l);
    ctx.lineWidth = isSelected ? w+1 : w;

    // Dashed flow for core/bidirectional
    if (l.decay_class==='core' || l.direction==='bidirectional') {
      ctx.setLineDash([6, 4]);
      ctx.lineDashOffset = dashOffset;
    } else {
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(tx, ty);
    ctx.stroke();

    // Arrowhead
    const angle = Math.atan2(ty-sy, tx-sx);
    const r = nodeRadius(l.target)+3;
    const ax = tx - r*Math.cos(angle), ay = ty - r*Math.sin(angle);
    ctx.setLineDash([]);
    ctx.fillStyle = isSelected ? '#22d3ee' : edgeColor(l);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - 8*Math.cos(angle-0.4), ay - 8*Math.sin(angle-0.4));
    ctx.lineTo(ax - 8*Math.cos(angle+0.4), ay - 8*Math.sin(angle+0.4));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  // Draw nodes
  for (const n of canvasNodes) {
    if (n.x==null||n.y==null) continue;
    const r = nodeRadius(n);
    const isSelected = selectedNode === n;
    const isHovered = hoveredNode === n;
    const isDimmed = (selectedNode && selectedNode !== n && !isNeighbor(selectedNode, n));
    const alpha = isDimmed ? 0.1 : 1;
    const dormant = n.status === 'dormant';

    ctx.save();
    ctx.globalAlpha = alpha;

    // Glow for core grains
    if (n.decay_class === 'core' && !dormant) {
      const glowR = r + 7 + Math.sin(t * 2.2 + n._phase) * 3;
      const grad = ctx.createRadialGradient(n.x, n.y, r*0.5, n.x, n.y, glowR+4);
      grad.addColorStop(0, 'rgba(34,211,238,0.25)');
      grad.addColorStop(1, 'rgba(34,211,238,0)');
      ctx.beginPath();
      ctx.arc(n.x, n.y, glowR+4, 0, Math.PI*2);
      ctx.fillStyle = grad;
      ctx.fill();
    }

    // Selection ring
    if (isSelected || isHovered) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 4, 0, Math.PI*2);
      ctx.strokeStyle = isSelected ? '#22d3ee' : 'rgba(34,211,238,0.4)';
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.stroke();
    }

    // Breathing radius for working/core grains
    let br = r;
    if (!dormant && n.node_type==='grain') {
      const speed = n.decay_class==='core' ? 2.2 : 1.8;
      br = r + Math.sin(t * speed + (n._phase||0)) * (n.decay_class==='core'?1.5:0.8);
    }

    // Node fill
    ctx.beginPath();
    ctx.arc(n.x, n.y, br, 0, Math.PI*2);
    ctx.fillStyle = nodeColor(n);
    ctx.globalAlpha = alpha * (dormant ? 0.35 : 0.9);
    ctx.fill();

    // Core stroke ring
    if (n.decay_class==='core') {
      ctx.strokeStyle = '#7dd3fc';
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = alpha * 0.7;
      ctx.stroke();
    }

    ctx.globalAlpha = alpha;

    // Label
    const label = n.node_type==='entry' ? n.label : n.label.slice(0,22)+(n.label.length>22?'…':'');
    ctx.fillStyle = isSelected ? '#dde3f0' : '#5a6480';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillText(label, n.x + br + 4, n.y + 3.5);

    ctx.restore();
  }

  ctx.restore();
}

// Assign stable phase offsets to nodes
function assignPhases(nodes) {
  nodes.forEach((n,i) => { n._phase = i * 0.61803 * Math.PI * 2; });
}

function isNeighbor(a, b) {
  for (const l of canvasLinks) {
    if ((l.source===a && l.target===b) || (l.target===a && l.source===b)) return true;
  }
  return false;
}

// ── CANVAS HIT DETECTION ──────────────────────────────────────────────────────
function canvasPoint(e) {
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left - transform.x) / transform.k;
  const my = (e.clientY - rect.top  - transform.y) / transform.k;
  return [mx, my];
}

function hitNode(mx, my) {
  for (let i = canvasNodes.length-1; i >= 0; i--) {
    const n = canvasNodes[i];
    const dx = (n.x||0)-mx, dy = (n.y||0)-my;
    if (Math.sqrt(dx*dx+dy*dy) <= nodeRadius(n)+4) return n;
  }
  return null;
}

function hitEdge(mx, my) {
  const THRESH = 5 / transform.k;
  for (const l of canvasLinks) {
    const sx=l.source.x, sy=l.source.y, tx=l.target.x, ty=l.target.y;
    if (sx==null) continue;
    const dx=tx-sx, dy=ty-sy, len=Math.sqrt(dx*dx+dy*dy);
    if (len<1) continue;
    const t2 = Math.max(0,Math.min(1,((mx-sx)*dx+(my-sy)*dy)/(len*len)));
    const px=sx+t2*dx-mx, py=sy+t2*dy-my;
    if (Math.sqrt(px*px+py*py)<=THRESH) return l;
  }
  return null;
}

// ── CANVAS MOUSE EVENTS ───────────────────────────────────────────────────────
let _dragging = null, _dragMoved = false;

function onCanvasMouseDown(e) {
  if (e.button !== 0) return;
  const [mx, my] = canvasPoint(e);
  const n = hitNode(mx, my);
  if (n) {
    _dragging = n; _dragMoved = false;
    if (!simulation._active) simulation.alphaTarget(0.3).restart();
    n.fx = n.x; n.fy = n.y;
    // Stop zoom from panning while dragging node
    d3.select(canvas).on('.zoom', null);
  }
}

function onCanvasMouseMove(e) {
  const [mx, my] = canvasPoint(e);
  if (_dragging) {
    _dragMoved = true;
    _dragging.fx = mx; _dragging.fy = my;
    simulation.alphaTarget(0.15).restart();
    return;
  }
  const n = hitNode(mx, my);
  if (n) {
    hoveredNode = n; hoveredEdge = null;
    canvas.style.cursor = 'pointer';
    showNodeTooltip(e, n);
  } else {
    const l = hitEdge(mx, my);
    hoveredEdge = l; hoveredNode = null;
    canvas.style.cursor = l ? 'pointer' : 'default';
    if (l) showEdgeTooltip(e, l); else hideTooltip();
  }
}

function onCanvasMouseUp(e) {
  if (_dragging) {
    _dragging.fx = null; _dragging.fy = null;
    simulation.alphaTarget(0.004);
    _dragging = null;
    // Restore zoom
    d3.select(canvas).call(zoom);
  }
}

function onCanvasClick(e) {
  if (_dragMoved) { _dragMoved = false; return; }
  const [mx, my] = canvasPoint(e);
  const n = hitNode(mx, my);
  if (n) { selectNode(n); return; }
  const l = hitEdge(mx, my);
  if (l) { selectEdge(l); return; }
  deselectAll();
}

function dragStart() {}
function dragged() {}
function dragEnd() {}

function toggleSimulation() {
  simulationPaused = !simulationPaused;
  const icon = document.getElementById('pause-icon');
  if (simulationPaused) {
    simulation?.alphaTarget(0).stop();
    cancelAnimationFrame(animFrame3D); animFrame3D = null;
    icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
  } else {
    simulation?.alphaTarget(0.004).restart();
    if (loop3D && !animFrame3D) loop3D();
    icon.innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
  }
}

function resetView() {
  if (!canvas || !zoom) return;
  d3.select(canvas).transition().duration(500).call(zoom.transform, d3.zoomIdentity);
}

// ── TOOLTIP ───────────────────────────────────────────────────────────────────
const tt = document.getElementById('tooltip');
function showTooltip(x, y, title, rows) {
  document.getElementById('tt-title').textContent = title;
  document.getElementById('tt-body').innerHTML = rows.map(([k,v])=>
    `<div class="tt-row"><span class="tt-key">${esc(k)}</span><span class="tt-val">${esc(v)}</span></div>`
  ).join('');
  tt.style.opacity = '1';
  positionTooltip(x, y);
}
function positionTooltip(x, y) {
  const r = tt.getBoundingClientRect();
  tt.style.left = (x + 14 + r.width > window.innerWidth ? x - r.width - 10 : x + 14) + 'px';
  tt.style.top = Math.min(y - 10, window.innerHeight - r.height - 10) + 'px';
}
function moveTooltip(event) { positionTooltip(event.clientX, event.clientY); }
function hideTooltip() { tt.style.opacity = '0'; }
function showNodeTooltip(event, d) {
  showTooltip(event.clientX, event.clientY, String(d.label || d.id || '').slice(0,60), [
    ['type', d.node_type],
    ...(d.decay_class ? [['decay', d.decay_class]] : []),
    ...(d.status ? [['status', d.status]] : []),
    ...(d.provenance ? [['provenance', d.provenance]] : []),
    ...(d.context_spread !== undefined ? [['ctx spread', d.context_spread]] : []),
  ]);
}
function showEdgeTooltip(event, d) {
  const sid = typeof d.source==='object'?d.source.id:d.source;
  const tid = typeof d.target==='object'?d.target.id:d.target;
  showTooltip(event.clientX, event.clientY, `${sid.slice(0,10)}… → ${tid.slice(0,10)}…`, [
    ['weight', (d.weight??0).toFixed(4)],
    ['eff. weight', (d.effective_weight??0).toFixed(4)],
    ['direction', d.direction],
    ['decay', d.decay_class??'—'],
    ['use count', d.use_count??0],
    ['edge type', d.edge_type??'—'],
  ]);
}

// ── SELECTION / INSPECTOR ─────────────────────────────────────────────────────
function deselectAll() {
  selectedNode = null; selectedEdge = null;
  document.getElementById('inspector-empty').style.display = '';
  document.getElementById('inspector-content').style.display = 'none';
}

function selectNode(d) {
  selectedNode = d; selectedEdge = null;
  renderInspector(d, 'node');
}

function selectEdge(d) {
  selectedEdge = d; selectedNode = null;
  renderInspector(d, 'edge');
}

function tagHtml(v, positive) {
  const cls = v==='active'||v==='core'?'cyan':v==='entry'?'violet':v==='dormant'||positive===false?'rose':v==='working'?'cyan':v==='ephemeral'?'dim':'dim';
  return `<span class="tag tag-${cls}">${esc(v)}</span>`;
}

function renderInspector(d, type) {
  document.getElementById('inspector-empty').style.display = 'none';
  const el = document.getElementById('inspector-content');
  el.style.display = '';
  if (type === 'node') {
    const color = nodeColor(d);
    const isEntry = d.node_type==='entry';
    el.innerHTML = `
      <div class="inspector-node-header">
        <div class="inspector-icon" style="background:${color}22;border:1px solid ${color}44">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="${color}"><circle cx="12" cy="12" r="${isEntry?6:8}"/></svg>
        </div>
        <div>
          <div class="inspector-label">${esc(d.label || d.id)}</div>
          <div class="inspector-type">${esc(d.node_type)}${d.decay_class?' · '+esc(d.decay_class):''}</div>
        </div>
      </div>
      <div class="inspector-rows">
        <div class="inspector-row"><span class="ir-key">ID</span><span class="ir-val" style="font-size:10px">${esc(d.id)}</span></div>
        <div class="inspector-row"><span class="ir-key">Type</span><span class="ir-val">${tagHtml(d.node_type)}</span></div>
        ${d.status?`<div class="inspector-row"><span class="ir-key">Status</span><span class="ir-val">${tagHtml(d.status)}</span></div>`:''}
        ${d.decay_class?`<div class="inspector-row"><span class="ir-key">Decay class</span><span class="ir-val">${tagHtml(d.decay_class)}</span></div>`:''}
        ${d.provenance?`<div class="inspector-row"><span class="ir-key">Provenance</span><span class="ir-val"><span class="tag tag-dim">${esc(d.provenance)}</span></span></div>`:''}
        ${d.context_spread!==undefined?`<div class="inspector-row"><span class="ir-key">Context spread</span><span class="ir-val">${esc(d.context_spread)}</span></div>`:''}
        ${d.feature?`<div class="inspector-row"><span class="ir-key">Feature</span><span class="ir-val">${esc(d.feature)}</span></div>`:''}
      </div>`;
  } else {
    const sid = typeof d.source==='object'?d.source.id:d.source;
    const tid = typeof d.target==='object'?d.target.id:d.target;
    const sl = typeof d.source==='object'?d.source.label:sid;
    const tl = typeof d.target==='object'?d.target.label:tid;
    el.innerHTML = `
      <div style="margin-bottom:10px">
        <div class="inspector-label" style="font-size:11px">${esc(String(sl||sid).slice(0,40))}…</div>
        <div style="color:var(--text-dim);font-size:10px;margin:3px 0">↓</div>
        <div class="inspector-label" style="font-size:11px">${esc(String(tl||tid).slice(0,40))}…</div>
      </div>
      <div class="inspector-rows">
        <div class="inspector-row"><span class="ir-key">Weight</span><span class="ir-val">${(d.weight??0).toFixed(4)}</span></div>
        <div class="inspector-row"><span class="ir-key">Eff. weight</span><span class="ir-val">${(d.effective_weight??0).toFixed(4)}</span></div>
        <div class="inspector-row"><span class="ir-key">Direction</span><span class="ir-val">${tagHtml(d.direction)}</span></div>
        <div class="inspector-row"><span class="ir-key">Decay class</span><span class="ir-val">${tagHtml(d.decay_class??'working')}</span></div>
        <div class="inspector-row"><span class="ir-key">Use count</span><span class="ir-val">${d.use_count??0}</span></div>
        <div class="inspector-row"><span class="ir-key">Edge type</span><span class="ir-val"><span class="tag tag-dim">${esc(d.edge_type??'earned')}</span></span></div>
      </div>`;
  }
}

// ── FILTERS / SEARCH ──────────────────────────────────────────────────────────
function toggleFilter(key) {
  activeFilters[key] = !activeFilters[key];
  const btn = document.getElementById('f-'+key);
  if (btn) btn.classList.toggle('active', activeFilters[key]);
  renderGraph();
}
function onSearch(val) {
  searchQuery = val.toLowerCase().trim();
  renderGraph();
}
function onWeightChange(val) {
  weightThreshold = parseFloat(val);
  document.getElementById('weight-val').textContent = parseFloat(val).toFixed(2);
  renderGraph();
}

// ── PANEL COLLAPSE ────────────────────────────────────────────────────────────
function toggleSection(id) {
  const body = document.getElementById('body-'+id);
  const chev = document.getElementById('chev-'+id);
  const visible = body.style.display !== 'none';
  body.style.display = visible ? 'none' : '';
  if (chev) chev.classList.toggle('open', !visible);
}

// ── INIT ──────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  fetchAll();
  window.addEventListener('resize', () => { if (graphData) renderGraph(); });
});
</script>


</body></html>"""


def _recent_events(store: Any, limit: int = 25) -> dict[str, list[dict[str, Any]]]:
    safe_limit = max(1, min(int(limit), 100))
    rows = store.conn.execute(
        """
        SELECT timestamp, category, event, trace_id, data
        FROM events
        WHERE category != 'health'
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (safe_limit,),
    ).fetchall()
    events: list[dict[str, Any]] = []
    for row in rows:
        try:
            data = json.loads(row["data"] or "{}")
        except json.JSONDecodeError:
            data = {}
        events.append({
            "timestamp": row["timestamp"],
            "category": row["category"],
            "event": row["event"],
            "trace_id": row["trace_id"],
            "data": data,
        })
    return {"events": events}


def run_dashboard(
    store: Any,
    host: str = "127.0.0.1",
    port: int = 7462,
    cfg: Any = None,
) -> None:
    """Start the HTTP dashboard server (blocking). Ctrl+C to stop."""
    from .health import flux_health
    from .visualization import cluster_view, export_json

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.debug(fmt, *args)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/" or path == "/index.html":
                self._send(200, "text/html; charset=utf-8", _DASHBOARD_HTML.encode())
            elif path == "/api/health":
                data = flux_health(store, cfg) if cfg else flux_health(store)
                self._send(200, "application/json", json.dumps(data, default=str).encode())
            elif path == "/api/graph":
                data = export_json(store)
                self._send(200, "application/json", json.dumps(data, default=str).encode())
            elif path == "/api/clusters":
                data = cluster_view(store)
                self._send(200, "application/json", json.dumps(data, default=str).encode())
            elif path == "/api/events":
                query = parse_qs(parsed.query)
                try:
                    limit = int(query.get("limit", ["25"])[0])
                except ValueError:
                    limit = 25
                data = _recent_events(store, limit=limit)
                self._send(200, "application/json", json.dumps(data, default=str).encode())
            elif path == "/favicon.ico":
                self._send(204, "image/x-icon", b"")
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
    logger.info("Flux dashboard listening on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
