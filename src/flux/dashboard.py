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

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Flux Memory - Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
  :root {
    --bg: #080a0e;
    --surface: #0f1117;
    --surface-2: #151923;
    --surface-3: #1b2130;
    --border: #222938;
    --border-2: #2d3547;
    --text: #e1e7f3;
    --muted: #7b8498;
    --dim: #4e566b;
    --cyan: #22d3ee;
    --cyan-dim: rgba(34, 211, 238, 0.12);
    --green: #4ade80;
    --green-dim: rgba(74, 222, 128, 0.12);
    --amber: #fbbf24;
    --amber-dim: rgba(251, 191, 36, 0.13);
    --rose: #fb7185;
    --rose-dim: rgba(251, 113, 133, 0.13);
    --violet: #a78bfa;
    --violet-dim: rgba(167, 139, 250, 0.12);
    --radius: 8px;
    --mono: "JetBrains Mono", "Fira Code", Consolas, monospace;
    --ui: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    min-width: 320px;
    background: var(--bg);
    color: var(--text);
    font-family: var(--ui);
    font-size: 13px;
    letter-spacing: 0;
    overflow: hidden;
  }
  button, input { font: inherit; }
  button { color: inherit; }
  #app { display: flex; flex-direction: column; height: 100vh; }

  #topbar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 9px;
    min-width: 170px;
  }
  .brand-mark {
    display: grid;
    place-items: center;
    width: 30px;
    height: 30px;
    border-radius: 7px;
    background: linear-gradient(180deg, #172033, #101723);
    border: 1px solid var(--border-2);
    color: var(--cyan);
    font-weight: 800;
  }
  .brand-name { font-weight: 700; font-size: 14px; line-height: 1.15; }
  .brand-sub { color: var(--muted); font-family: var(--mono); font-size: 10px; margin-top: 2px; }
  .divider { width: 1px; align-self: stretch; background: var(--border); }
  #status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 26px;
    padding: 0 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .badge-healthy { background: var(--green-dim); color: var(--green); border: 1px solid rgba(74, 222, 128, 0.28); }
  .badge-warning { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(251, 191, 36, 0.3); }
  .badge-critical { background: var(--rose-dim); color: var(--rose); border: 1px solid rgba(251, 113, 133, 0.3); }
  .pulse {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.45); opacity: 0.55; }
  }
  #metrics-strip {
    display: grid;
    grid-template-columns: repeat(6, minmax(86px, 1fr));
    gap: 6px;
    flex: 1;
    min-width: 0;
  }
  .metric-pill {
    min-width: 0;
    padding: 5px 9px;
    border-radius: var(--radius);
    background: var(--surface-2);
    border: 1px solid var(--border);
  }
  .metric-pill .label {
    color: var(--muted);
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .metric-pill .value {
    margin-top: 1px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 16px;
    font-weight: 800;
    line-height: 1.2;
  }
  .value-cyan { color: var(--cyan) !important; }
  .value-green { color: var(--green) !important; }
  .value-amber { color: var(--amber) !important; }
  .value-rose { color: var(--rose) !important; }
  #topbar-controls {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  #computed-at {
    color: var(--dim);
    font-family: var(--mono);
    font-size: 10px;
    white-space: nowrap;
  }
  .icon-btn {
    display: inline-grid;
    place-items: center;
    width: 30px;
    height: 30px;
    padding: 0;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--muted);
    cursor: pointer;
  }
  .icon-btn:hover { color: var(--text); border-color: var(--border-2); background: var(--surface-3); }
  .icon-btn.active { color: var(--cyan); border-color: rgba(34, 211, 238, 0.35); background: var(--cyan-dim); }

  #main { display: flex; flex: 1; min-height: 0; overflow: hidden; }
  #graph-panel {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
  }
  #graph-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  #search-box {
    display: flex;
    align-items: center;
    gap: 7px;
    width: 230px;
    min-width: 160px;
    height: 30px;
    padding: 0 9px;
    border-radius: var(--radius);
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--muted);
  }
  #search-input {
    width: 100%;
    min-width: 0;
    outline: none;
    border: 0;
    color: var(--text);
    background: transparent;
    font-size: 12px;
  }
  #search-input::placeholder { color: var(--dim); }
  .filter-group { display: flex; align-items: center; gap: 4px; }
  .filter-btn {
    height: 28px;
    padding: 0 9px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
  }
  .filter-btn.active { color: var(--cyan); border-color: rgba(34, 211, 238, 0.35); background: var(--cyan-dim); }
  .filter-btn.active.violet { color: var(--violet); border-color: rgba(167, 139, 250, 0.35); background: var(--violet-dim); }
  .slider-group {
    display: flex;
    align-items: center;
    gap: 7px;
    color: var(--muted);
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    white-space: nowrap;
  }
  #weight-slider { width: 92px; accent-color: var(--cyan); cursor: pointer; }
  #weight-val { min-width: 30px; color: var(--cyan); font-family: var(--mono); font-size: 11px; }
  #graph-container {
    position: relative;
    flex: 1;
    min-height: 0;
    background:
      linear-gradient(rgba(34, 211, 238, 0.025) 1px, transparent 1px),
      linear-gradient(90deg, rgba(34, 211, 238, 0.025) 1px, transparent 1px),
      #070a0f;
    background-size: 32px 32px;
    overflow: hidden;
  }
  #graph-svg { width: 100%; height: 100%; display: block; }
  #graph-stats {
    position: absolute;
    top: 10px;
    right: 10px;
    display: flex;
    gap: 8px;
    padding: 6px 9px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: rgba(15, 17, 23, 0.82);
    color: var(--muted);
    font-family: var(--mono);
    font-size: 10px;
    backdrop-filter: blur(6px);
  }
  #graph-overlay {
    position: absolute;
    left: 12px;
    bottom: 12px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    padding: 6px 8px;
    border-radius: var(--radius);
    border: 1px solid rgba(34, 41, 56, 0.72);
    background: rgba(15, 17, 23, 0.72);
    backdrop-filter: blur(6px);
  }
  .legend-item { display: inline-flex; align-items: center; gap: 5px; color: var(--muted); font-size: 10px; }
  .legend-dot { width: 8px; height: 8px; border-radius: 50%; }
  #no-data-msg, #error-msg {
    position: absolute;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    color: var(--dim);
    font-size: 13px;
    text-align: center;
    padding: 20px;
  }
  #error-msg { color: var(--rose); }

  #right-panel {
    width: 340px;
    flex-shrink: 0;
    background: var(--surface);
    overflow: auto;
  }
  .r-section { border-bottom: 1px solid var(--border); }
  .r-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 11px 14px;
    cursor: pointer;
    user-select: none;
  }
  .r-title-wrap { display: flex; align-items: center; gap: 7px; min-width: 0; }
  .r-title {
    color: var(--muted);
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .r-count {
    min-width: 20px;
    padding: 1px 6px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--muted);
    font-family: var(--mono);
    font-size: 10px;
    text-align: center;
  }
  .r-count.warn { color: var(--amber); border-color: rgba(251, 191, 36, 0.3); background: var(--amber-dim); }
  .r-body { padding: 0 14px 13px; }
  .chev { color: var(--dim); transition: transform 0.2s; }
  .chev.open { transform: rotate(180deg); }

  #inspector-empty {
    padding: 22px 0;
    color: var(--dim);
    font-size: 12px;
    text-align: center;
  }
  .node-head { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 11px; }
  .node-icon {
    display: grid;
    place-items: center;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .node-label { color: var(--text); font-size: 12px; font-weight: 700; line-height: 1.35; word-break: break-word; }
  .node-type { margin-top: 3px; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; }
  .kv { display: flex; flex-direction: column; gap: 5px; }
  .kv-row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    padding: 5px 0;
    border-bottom: 1px solid rgba(34, 41, 56, 0.72);
  }
  .kv-row:last-child { border-bottom: 0; }
  .kv-key { color: var(--muted); font-size: 11px; flex-shrink: 0; }
  .kv-val {
    color: var(--text);
    font-family: var(--mono);
    font-size: 11px;
    text-align: right;
    word-break: break-all;
  }
  .tag {
    display: inline-block;
    max-width: 160px;
    overflow: hidden;
    text-overflow: ellipsis;
    vertical-align: bottom;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 800;
  }
  .tag-cyan { background: var(--cyan-dim); color: var(--cyan); }
  .tag-violet { background: var(--violet-dim); color: var(--violet); }
  .tag-green { background: var(--green-dim); color: var(--green); }
  .tag-amber { background: var(--amber-dim); color: var(--amber); }
  .tag-rose { background: var(--rose-dim); color: var(--rose); }
  .tag-dim { background: var(--surface-2); color: var(--muted); }

  .warning-card {
    padding: 9px 10px;
    margin-bottom: 7px;
    border-radius: var(--radius);
    border: 1px solid rgba(251, 191, 36, 0.22);
    background: rgba(251, 191, 36, 0.07);
  }
  .warning-card:last-child { margin-bottom: 0; }
  .warning-signal { color: var(--amber); font-family: var(--mono); font-size: 11px; font-weight: 800; }
  .warning-msg { margin-top: 4px; color: #c5cede; font-size: 11px; line-height: 1.45; }
  .warning-val { margin-top: 6px; color: var(--dim); font-family: var(--mono); font-size: 10px; }
  .empty-row { color: var(--dim); font-size: 11px; padding: 8px 0; }

  .health-group { margin-bottom: 11px; }
  .health-group:last-child { margin-bottom: 0; }
  .health-label {
    margin-bottom: 5px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
    color: var(--dim);
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .health-row { display: flex; align-items: center; gap: 7px; padding: 4px 0; }
  .health-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .health-key { flex: 1; min-width: 0; color: var(--muted); font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .health-val { min-width: 52px; color: var(--text); font-family: var(--mono); font-size: 11px; text-align: right; }

  .event-card {
    padding: 7px 0;
    border-bottom: 1px solid rgba(34, 41, 56, 0.72);
  }
  .event-card:last-child { border-bottom: 0; }
  .event-line { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .event-name { color: var(--text); font-family: var(--mono); font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .event-time { color: var(--dim); font-family: var(--mono); font-size: 10px; flex-shrink: 0; }
  .event-data { margin-top: 3px; color: var(--muted); font-size: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  #tooltip {
    position: fixed;
    z-index: 999;
    pointer-events: none;
    max-width: 280px;
    padding: 9px 10px;
    border-radius: var(--radius);
    border: 1px solid var(--border-2);
    background: rgba(15, 17, 23, 0.97);
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
    opacity: 0;
    transition: opacity 0.1s;
  }
  .tt-title { color: var(--text); font-weight: 800; font-size: 12px; line-height: 1.35; margin-bottom: 5px; }
  .tt-row { display: flex; justify-content: space-between; gap: 16px; line-height: 1.6; }
  .tt-key { color: var(--muted); }
  .tt-val { color: var(--text); font-family: var(--mono); }

  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 999px; }
  ::-webkit-scrollbar-track { background: transparent; }

  @media (max-width: 1120px) {
    #metrics-strip { grid-template-columns: repeat(3, minmax(86px, 1fr)); }
    #computed-at { display: none; }
  }
  @media (max-width: 860px) {
    body { overflow: auto; }
    #app { height: auto; min-height: 100vh; }
    #topbar, #graph-toolbar { flex-wrap: wrap; }
    #main { flex-direction: column; overflow: visible; }
    #graph-panel { min-height: 620px; border-right: 0; border-bottom: 1px solid var(--border); }
    #right-panel { width: auto; }
    #metrics-strip { order: 4; flex-basis: 100%; grid-template-columns: repeat(2, minmax(86px, 1fr)); }
    #search-box { flex: 1; width: auto; max-width: none; }
  }
</style>
</head>
<body>
<div id="app">
  <header id="topbar">
    <div class="brand">
      <div class="brand-mark">F</div>
      <div>
        <div class="brand-name">Flux Memory</div>
        <div class="brand-sub">instance dashboard</div>
      </div>
    </div>
    <div id="status-badge" class="badge-warning"><span class="pulse"></span><span id="status-text">loading</span></div>
    <div class="divider"></div>
    <div id="metrics-strip">
      <div class="metric-pill"><div class="label">Grains</div><div class="value value-cyan" id="m-grains">-</div></div>
      <div class="metric-pill"><div class="label">Entries</div><div class="value" id="m-entries">-</div></div>
      <div class="metric-pill"><div class="label">Conduits</div><div class="value value-green" id="m-conduits">-</div></div>
      <div class="metric-pill"><div class="label">Embeddings</div><div class="value value-green" id="m-embeddings">-</div></div>
      <div class="metric-pill"><div class="label">Retrieval</div><div class="value" id="m-retrieval">-</div></div>
      <div class="metric-pill"><div class="label">Fallback</div><div class="value" id="m-fallback">-</div></div>
    </div>
    <div id="topbar-controls">
      <span id="computed-at">not computed</span>
      <button class="icon-btn" id="autorefresh-btn" type="button" title="Auto refresh" aria-label="Auto refresh">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v7h-7"/></svg>
      </button>
      <button class="icon-btn" id="refresh-btn" type="button" title="Refresh" aria-label="Refresh">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 3v6h-6"/></svg>
      </button>
    </div>
  </header>

  <main id="main">
    <section id="graph-panel">
      <div id="graph-toolbar">
        <div id="search-box">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
          <input id="search-input" type="text" placeholder="Search nodes">
        </div>
        <div class="filter-group">
          <button class="filter-btn active" id="f-grain" type="button">Grains</button>
          <button class="filter-btn violet active" id="f-entry" type="button">Entries</button>
          <button class="filter-btn active" id="f-active" type="button">Active</button>
          <button class="filter-btn" id="f-dormant" type="button">Dormant</button>
        </div>
        <div class="slider-group">
          <label for="weight-slider">Min weight</label>
          <input id="weight-slider" type="range" min="0" max="1" step="0.01" value="0">
          <span id="weight-val">0.00</span>
        </div>
        <button class="icon-btn" id="pause-btn" type="button" title="Pause layout" aria-label="Pause layout">
          <svg id="pause-icon" width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
        </button>
        <button class="icon-btn" id="reset-btn" type="button" title="Reset view" aria-label="Reset view">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 3v6h6"/></svg>
        </button>
      </div>
      <div id="graph-container">
        <svg id="graph-svg"></svg>
        <div id="graph-stats"><span>nodes: <b id="gs-nodes">0</b></span><span>edges: <b id="gs-edges">0</b></span></div>
        <div id="graph-overlay">
          <span class="legend-item"><i class="legend-dot" style="background:var(--violet)"></i>Entry</span>
          <span class="legend-item"><i class="legend-dot" style="background:var(--cyan)"></i>Working</span>
          <span class="legend-item"><i class="legend-dot" style="background:#e0f2fe"></i>Core</span>
          <span class="legend-item"><i class="legend-dot" style="background:var(--green)"></i>Strong</span>
        </div>
        <div id="no-data-msg">No graph data</div>
        <div id="error-msg"></div>
      </div>
    </section>

    <aside id="right-panel">
      <section class="r-section">
        <div class="r-header" data-section="inspector">
          <div class="r-title-wrap"><span class="r-title">Inspector</span></div>
          <svg class="chev open" id="chev-inspector" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>
        </div>
        <div class="r-body" id="body-inspector">
          <div id="inspector-empty">No selection</div>
          <div id="inspector-content" style="display:none"></div>
        </div>
      </section>

      <section class="r-section">
        <div class="r-header" data-section="warnings">
          <div class="r-title-wrap"><span class="r-title">Warnings</span><span class="r-count warn" id="warnings-count">0</span></div>
          <svg class="chev open" id="chev-warnings" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>
        </div>
        <div class="r-body" id="body-warnings"><div id="warnings-list"></div></div>
      </section>

      <section class="r-section">
        <div class="r-header" data-section="health">
          <div class="r-title-wrap"><span class="r-title">Health Signals</span></div>
          <svg class="chev open" id="chev-health" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>
        </div>
        <div class="r-body" id="body-health"><div id="health-table"></div></div>
      </section>

      <section class="r-section">
        <div class="r-header" data-section="events">
          <div class="r-title-wrap"><span class="r-title">Recent Activity</span></div>
          <svg class="chev open" id="chev-events" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>
        </div>
        <div class="r-body" id="body-events"><div id="events-list"></div></div>
      </section>
    </aside>
  </main>
</div>

<div id="tooltip">
  <div class="tt-title" id="tt-title"></div>
  <div id="tt-body"></div>
</div>

<script>
let healthData = null;
let graphData = null;
let eventsData = [];
let svg = null;
let graphLayer = null;
let simulation = null;
let zoomBehavior = null;
let nodeSelectionGlobal = null;
let linkSelectionGlobal = null;
let dashFrame = null;
let dashOffset = 0;
let autoRefreshTimer = null;
let simulationPaused = false;
let activeFilters = { grain: true, entry: true, active: true, dormant: false };
let weightThreshold = 0;
let searchQuery = "";
let selectedNodeId = null;
let selectedEdgeKey = null;

const fmt = new Intl.NumberFormat();
const tooltip = document.getElementById("tooltip");

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

function formatNumber(value, digits = 2) {
  const n = Number(value ?? 0);
  if (!Number.isFinite(n)) return "-";
  return Number.isInteger(n) ? fmt.format(n) : n.toFixed(digits);
}

function formatRate(value) {
  const n = Number(value ?? 0);
  if (!Number.isFinite(n)) return "-";
  return `${Math.round(n * 100)}%`;
}

function edgeKey(edge) {
  const sid = typeof edge.source === "object" ? edge.source.id : edge.source;
  const tid = typeof edge.target === "object" ? edge.target.id : edge.target;
  return `${sid}->${tid}:${edge.edge_type || "earned"}:${edge.weight || 0}`;
}

async function fetchJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
}

async function fetchAll() {
  const error = document.getElementById("error-msg");
  error.style.display = "none";
  try {
    const [health, graph, events] = await Promise.all([
      fetchJSON("/api/health"),
      fetchJSON("/api/graph"),
      fetchJSON("/api/events?limit=18").catch(() => ({ events: [] })),
    ]);
    healthData = health;
    graphData = graph;
    eventsData = events.events || [];
    renderHealth();
    renderEvents();
    renderGraph();
  } catch (err) {
    console.error(err);
    error.textContent = `Dashboard API unavailable: ${err.message}`;
    error.style.display = "flex";
  }
}

function renderHealth() {
  if (!healthData) return;
  const status = healthData.status || "warning";
  const badge = document.getElementById("status-badge");
  const badgeStatus = ["healthy", "warning", "critical"].includes(status) ? status : "warning";
  badge.className = `badge-${badgeStatus}`;
  document.getElementById("status-text").textContent = status;

  const computedAt = healthData.computed_at ? new Date(healthData.computed_at) : null;
  document.getElementById("computed-at").textContent =
    computedAt && !Number.isNaN(computedAt.valueOf()) ? computedAt.toLocaleTimeString() : "not computed";

  const stats = graphData?.stats || {};
  const signals = healthData.signals || {};
  setMetric("m-grains", stats.grains, "value-cyan");
  setMetric("m-entries", stats.entries, "");
  setMetric("m-conduits", stats.conduits, stats.conduits > 0 ? "value-green" : "value-amber");
  setMetric("m-embeddings", stats.embeddings, stats.embeddings > 0 ? "value-green" : "value-amber");
  setMetric("m-retrieval", formatRate(signals.retrieval_success_rate?.value), "");
  setMetric("m-fallback", formatRate(signals.fallback_trigger_rate?.value), (signals.fallback_trigger_rate?.value || 0) > 0.2 ? "value-rose" : "value-green");

  const warnings = healthData.active_warnings || [];
  document.getElementById("warnings-count").textContent = warnings.length;
  const warningsList = document.getElementById("warnings-list");
  warningsList.innerHTML = warnings.length ? warnings.map(w => `
    <div class="warning-card">
      <div class="warning-signal">${esc(w.signal)}</div>
      <div class="warning-msg">${esc(w.suggestion)}</div>
      <div class="warning-val">value ${esc(w.current_value)} / range ${esc(w.healthy_range)} / ${esc(w.severity)}</div>
    </div>`).join("") : `<div class="empty-row">None</div>`;

  const groups = {
    "Retrieval": ["retrieval_success_rate", "avg_hops_per_retrieval", "fallback_trigger_rate"],
    "Feedback": ["feedback_compliance_rate", "promotion_events", "shortcut_creation_rate"],
    "Graph": ["highway_count", "highway_growth_rate", "orphan_rate", "core_grain_count", "avg_conduit_weight"],
    "Decay": ["dormant_grain_rate", "conduit_dissolution_rate", "avg_weight_drop_on_failure"],
  };
  let healthHtml = "";
  for (const [group, keys] of Object.entries(groups)) {
    healthHtml += `<div class="health-group"><div class="health-label">${group}</div>`;
    for (const key of keys) {
      const signal = signals[key];
      if (!signal) continue;
      const value = key.includes("rate") ? formatRate(signal.value) : formatNumber(signal.value);
      healthHtml += `
        <div class="health-row">
          <span class="health-dot" style="background:${signal.healthy ? "var(--green)" : "var(--rose)"}"></span>
          <span class="health-key" title="${esc(key)}">${esc(key.replaceAll("_", " "))}</span>
          <span class="health-val" style="color:${signal.healthy ? "var(--text)" : "var(--rose)"}">${value}</span>
        </div>`;
    }
    healthHtml += `</div>`;
  }
  document.getElementById("health-table").innerHTML = healthHtml;
}

function setMetric(id, value, tone) {
  const el = document.getElementById(id);
  el.className = `value ${tone || ""}`;
  el.textContent = typeof value === "string" ? value : formatNumber(value, 0);
}

function renderEvents() {
  const list = document.getElementById("events-list");
  if (!eventsData.length) {
    list.innerHTML = `<div class="empty-row">None</div>`;
    return;
  }
  list.innerHTML = eventsData.map(e => {
    const t = e.timestamp ? new Date(e.timestamp.replace("Z", "+00:00")) : null;
    const time = t && !Number.isNaN(t.valueOf()) ? t.toLocaleTimeString() : "";
    const detail = eventSummary(e.data || {});
    return `<div class="event-card">
      <div class="event-line"><span class="event-name">${esc(e.category)}/${esc(e.event)}</span><span class="event-time">${esc(time)}</span></div>
      <div class="event-data">${esc(detail)}</div>
    </div>`;
  }).join("");
}

function eventSummary(data) {
  const entries = Object.entries(data).filter(([k, v]) => v !== null && v !== undefined);
  if (!entries.length) return "";
  return entries.slice(0, 4).map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`).join(" | ");
}

function nodeColor(node) {
  if (node.status === "dormant") return "#263244";
  if (node.node_type === "entry") return "#a78bfa";
  if (node.decay_class === "core") return "#e0f2fe";
  if (node.decay_class === "ephemeral") return "#0e7490";
  return "#22d3ee";
}

function nodeRadius(node) {
  if (node.node_type === "entry") return 6;
  if (node.decay_class === "core") return 10;
  return 8;
}

function edgeColor(edge) {
  if (edge.direction === "bidirectional") return "#a78bfa";
  if ((edge.effective_weight ?? edge.weight ?? 0) >= 0.55) return "#4ade80";
  if (edge.decay_class === "core") return "#38bdf8";
  return "#1e4a62";
}

function getVisibleNodes() {
  if (!graphData) return [];
  return (graphData.nodes || []).filter(node => {
    const isEntry = node.node_type === "entry";
    const typeOk = isEntry ? activeFilters.entry : activeFilters.grain;
    const status = node.status || "active";
    const statusOk = status === "dormant" ? activeFilters.dormant : activeFilters.active;
    const haystack = `${node.label || ""} ${node.id || ""} ${node.feature || ""}`.toLowerCase();
    const searchOk = !searchQuery || haystack.includes(searchQuery);
    return typeOk && statusOk && searchOk;
  });
}

function getVisibleLinks(nodeIds) {
  if (!graphData) return [];
  const visible = new Set(nodeIds);
  return (graphData.links || []).filter(link => {
    const source = typeof link.source === "object" ? link.source.id : link.source;
    const target = typeof link.target === "object" ? link.target.id : link.target;
    const weight = Number(link.effective_weight ?? link.weight ?? 0);
    return visible.has(source) && visible.has(target) && weight >= weightThreshold;
  });
}

function renderGraph() {
  const container = document.getElementById("graph-container");
  const error = document.getElementById("error-msg");
  if (!window.d3) {
    error.textContent = "Graph renderer unavailable: D3 did not load";
    error.style.display = "flex";
    return;
  }

  if (simulation) simulation.stop();
  if (dashFrame) cancelAnimationFrame(dashFrame);

  const width = Math.max(container.clientWidth, 360);
  const height = Math.max(container.clientHeight, 360);
  d3.select("#graph-svg").selectAll("*").remove();
  svg = d3.select("#graph-svg").attr("width", width).attr("height", height);

  if (!graphData || !(graphData.nodes || []).length) {
    document.getElementById("no-data-msg").style.display = "flex";
    document.getElementById("gs-nodes").textContent = "0";
    document.getElementById("gs-edges").textContent = "0";
    return;
  }
  document.getElementById("no-data-msg").style.display = "none";

  const visibleNodes = getVisibleNodes();
  const visibleLinks = getVisibleLinks(visibleNodes.map(n => n.id));
  document.getElementById("gs-nodes").textContent = visibleNodes.length;
  document.getElementById("gs-edges").textContent = visibleLinks.length;

  const nodes = visibleNodes.map(n => ({ ...n }));
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const links = visibleLinks.map(link => ({
    ...link,
    source: nodeMap.get(typeof link.source === "object" ? link.source.id : link.source),
    target: nodeMap.get(typeof link.target === "object" ? link.target.id : link.target),
  })).filter(link => link.source && link.target);

  const defs = svg.append("defs");
  ["forward", "bidirectional", "selected"].forEach(type => {
    defs.append("marker")
      .attr("id", `arrow-${type}`)
      .attr("viewBox", "0 -4 8 8")
      .attr("refX", 15)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-4L8,0L0,4")
      .attr("fill", type === "selected" ? "#22d3ee" : type === "bidirectional" ? "#a78bfa" : "#1e4a62");
  });

  zoomBehavior = d3.zoom().scaleExtent([0.12, 5]).on("zoom", event => graphLayer.attr("transform", event.transform));
  svg.call(zoomBehavior);
  graphLayer = svg.append("g");

  const linkSelection = graphLayer.append("g")
    .attr("class", "links")
    .selectAll("line")
    .data(links, edgeKey)
    .enter()
    .append("line")
    .attr("stroke", edgeColor)
    .attr("stroke-width", d => Math.max(0.65, Number(d.effective_weight ?? d.weight ?? 0.25) * 3.2))
    .attr("stroke-opacity", 0.72)
    .attr("marker-end", d => `url(#arrow-${d.direction === "bidirectional" ? "bidirectional" : "forward"})`)
    .attr("stroke-dasharray", d => d.direction === "bidirectional" ? "4 5" : d.decay_class === "core" ? "6 4" : null)
    .style("cursor", "pointer")
    .on("mouseenter", showEdgeTooltip)
    .on("mousemove", moveTooltip)
    .on("mouseleave", hideTooltip)
    .on("click", (event, d) => { event.stopPropagation(); selectEdge(d); });

  const nodeSelection = graphLayer.append("g")
    .attr("class", "nodes")
    .selectAll("g")
    .data(nodes, d => d.id)
    .enter()
    .append("g")
    .attr("class", "graph-node")
    .style("cursor", "pointer")
    .call(d3.drag().on("start", dragStart).on("drag", dragged).on("end", dragEnd))
    .on("mouseenter", showNodeTooltip)
    .on("mousemove", moveTooltip)
    .on("mouseleave", hideTooltip)
    .on("click", (event, d) => { event.stopPropagation(); selectNode(d); });

  nodeSelection.filter(d => d.decay_class === "core").append("circle")
    .attr("r", d => nodeRadius(d) + 8)
    .attr("fill", "#22d3ee")
    .attr("fill-opacity", 0.08)
    .attr("stroke", "#22d3ee")
    .attr("stroke-opacity", 0.38)
    .attr("stroke-width", 2);

  nodeSelection.append("circle")
    .attr("class", "node-main")
    .attr("r", nodeRadius)
    .attr("fill", nodeColor)
    .attr("fill-opacity", d => d.status === "dormant" ? 0.38 : 0.92)
    .attr("stroke", d => selectedNodeId === d.id ? "#22d3ee" : d.decay_class === "core" ? "#7dd3fc" : "transparent")
    .attr("stroke-width", d => selectedNodeId === d.id ? 2.5 : 1.5);

  nodeSelection.append("text")
    .text(d => d.node_type === "entry" ? d.label : truncate(d.label || d.id, 24))
    .attr("x", d => nodeRadius(d) + 5)
    .attr("y", 4)
    .attr("fill", "#8d99ad")
    .attr("font-size", "9px")
    .attr("font-family", "JetBrains Mono, Consolas, monospace")
    .attr("pointer-events", "none");

  simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(d => 58 + (1 - Number(d.effective_weight ?? 0.3)) * 74).strength(0.42))
    .force("charge", d3.forceManyBody().strength(-175).distanceMax(320))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide(d => nodeRadius(d) + 10))
    .alphaDecay(0.018)
    .alphaMin(0.001)
    .on("tick", () => {
      linkSelection
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      nodeSelection.attr("transform", d => `translate(${d.x},${d.y})`);
    });

  animateDashes(linkSelection);
  nodeSelectionGlobal = nodeSelection;
  linkSelectionGlobal = linkSelection;
  svg.on("click", () => clearSelection());
  if (simulationPaused) simulation.stop();
}

function animateDashes(linkSelection) {
  function loop() {
    dashOffset -= 0.35;
    linkSelection.attr("stroke-dashoffset", dashOffset);
    dashFrame = requestAnimationFrame(loop);
  }
  loop();
}

function dragStart(event, d) {
  if (!event.active && simulation && !simulationPaused) simulation.alphaTarget(0.25).restart();
  d.fx = d.x;
  d.fy = d.y;
}
function dragged(event, d) {
  d.fx = event.x;
  d.fy = event.y;
}
function dragEnd(event, d) {
  if (!event.active && simulation && !simulationPaused) simulation.alphaTarget(0.015);
  d.fx = null;
  d.fy = null;
}

function showTooltip(event, title, rows) {
  document.getElementById("tt-title").textContent = title;
  document.getElementById("tt-body").innerHTML = rows.map(([key, value]) =>
    `<div class="tt-row"><span class="tt-key">${esc(key)}</span><span class="tt-val">${esc(value)}</span></div>`
  ).join("");
  tooltip.style.opacity = "1";
  positionTooltip(event.clientX, event.clientY);
}
function positionTooltip(x, y) {
  const rect = tooltip.getBoundingClientRect();
  tooltip.style.left = `${x + 14 + rect.width > window.innerWidth ? x - rect.width - 12 : x + 14}px`;
  tooltip.style.top = `${Math.min(y - 10, window.innerHeight - rect.height - 10)}px`;
}
function moveTooltip(event) { positionTooltip(event.clientX, event.clientY); }
function hideTooltip() { tooltip.style.opacity = "0"; }
function showNodeTooltip(event, node) {
  showTooltip(event, truncate(node.label || node.id, 72), [
    ["type", node.node_type],
    ["status", node.status || "-"],
    ["decay", node.decay_class || "-"],
    ["provenance", node.provenance || "-"],
    ["context", node.context_spread ?? "-"],
  ]);
}
function showEdgeTooltip(event, edge) {
  const sid = typeof edge.source === "object" ? edge.source.id : edge.source;
  const tid = typeof edge.target === "object" ? edge.target.id : edge.target;
  showTooltip(event, `${sid.slice(0, 8)} -> ${tid.slice(0, 8)}`, [
    ["weight", formatNumber(edge.weight, 4)],
    ["effective", formatNumber(edge.effective_weight, 4)],
    ["direction", edge.direction || "-"],
    ["decay", edge.decay_class || "-"],
    ["uses", edge.use_count ?? 0],
  ]);
}

function selectNode(node) {
  selectedNodeId = node.id;
  selectedEdgeKey = null;
  const neighbors = new Set([node.id]);
  linkSelectionGlobal?.each(function(edge) {
    if (!edge || !edge.source || !edge.target) return;
    const sid = typeof edge.source === "object" ? edge.source.id : edge.source;
    const tid = typeof edge.target === "object" ? edge.target.id : edge.target;
    if (sid === node.id || tid === node.id) {
      neighbors.add(sid);
      neighbors.add(tid);
    }
  });
  nodeSelectionGlobal?.attr("opacity", d => neighbors.has(d.id) ? 1 : 0.16);
  linkSelectionGlobal?.attr("stroke-opacity", edge => {
    if (!edge || !edge.source || !edge.target) return 0.72;
    const sid = typeof edge.source === "object" ? edge.source.id : edge.source;
    const tid = typeof edge.target === "object" ? edge.target.id : edge.target;
    return sid === node.id || tid === node.id ? 1 : 0.06;
  });
  nodeSelectionGlobal?.select("circle.node-main")
    .attr("stroke", d => d.id === node.id ? "#22d3ee" : d.decay_class === "core" ? "#7dd3fc" : "transparent")
    .attr("stroke-width", d => d.id === node.id ? 2.5 : 1.5);
  renderInspector(node, "node");
}

function selectEdge(edge) {
  selectedEdgeKey = edgeKey(edge);
  selectedNodeId = null;
  nodeSelectionGlobal?.attr("opacity", 0.25);
  linkSelectionGlobal?.attr("stroke-opacity", e => edgeKey(e) === selectedEdgeKey ? 1 : 0.08);
  renderInspector(edge, "edge");
}

function clearSelection() {
  selectedNodeId = null;
  selectedEdgeKey = null;
  nodeSelectionGlobal?.attr("opacity", 1);
  linkSelectionGlobal?.attr("stroke-opacity", 0.72);
  nodeSelectionGlobal?.select("circle.node-main")
    .attr("stroke", d => d.decay_class === "core" ? "#7dd3fc" : "transparent")
    .attr("stroke-width", 1.5);
  document.getElementById("inspector-empty").style.display = "";
  document.getElementById("inspector-content").style.display = "none";
}

function tag(value) {
  const v = String(value ?? "-");
  const cls = v === "entry" ? "violet" : v === "core" || v === "active" ? "cyan" : v === "dormant" ? "rose" : v === "bidirectional" ? "violet" : "dim";
  return `<span class="tag tag-${cls}">${esc(v)}</span>`;
}

function renderInspector(item, type) {
  document.getElementById("inspector-empty").style.display = "none";
  const container = document.getElementById("inspector-content");
  container.style.display = "";
  if (type === "node") {
    const color = nodeColor(item);
    container.innerHTML = `
      <div class="node-head">
        <div class="node-icon" style="background:${color}22;border:1px solid ${color}55">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="${color}"><circle cx="12" cy="12" r="${item.node_type === "entry" ? 6 : 8}"/></svg>
        </div>
        <div>
          <div class="node-label">${esc(item.label || item.id)}</div>
          <div class="node-type">${esc(item.node_type)}${item.decay_class ? " / " + esc(item.decay_class) : ""}</div>
        </div>
      </div>
      <div class="kv">
        ${kv("ID", item.id)}
        ${kv("Type", tag(item.node_type), true)}
        ${item.status ? kv("Status", tag(item.status), true) : ""}
        ${item.decay_class ? kv("Decay", tag(item.decay_class), true) : ""}
        ${item.provenance ? kv("Provenance", tag(item.provenance), true) : ""}
        ${item.context_spread !== undefined ? kv("Context", item.context_spread) : ""}
        ${item.feature ? kv("Feature", item.feature) : ""}
      </div>`;
  } else {
    const sid = typeof item.source === "object" ? item.source.id : item.source;
    const tid = typeof item.target === "object" ? item.target.id : item.target;
    const sourceLabel = typeof item.source === "object" ? item.source.label : sid;
    const targetLabel = typeof item.target === "object" ? item.target.label : tid;
    container.innerHTML = `
      <div class="node-label" style="margin-bottom:10px">${esc(truncate(sourceLabel || sid, 46))}<br><span style="color:var(--dim)">to</span><br>${esc(truncate(targetLabel || tid, 46))}</div>
      <div class="kv">
        ${kv("Weight", formatNumber(item.weight, 4))}
        ${kv("Effective", formatNumber(item.effective_weight, 4))}
        ${kv("Direction", tag(item.direction || "forward"), true)}
        ${kv("Decay", tag(item.decay_class || "working"), true)}
        ${kv("Use count", item.use_count ?? 0)}
        ${kv("Edge type", tag(item.edge_type || "earned"), true)}
      </div>`;
  }
}

function kv(key, value, raw = false) {
  return `<div class="kv-row"><span class="kv-key">${esc(key)}</span><span class="kv-val">${raw ? value : esc(value)}</span></div>`;
}

function truncate(value, max) {
  const text = String(value ?? "");
  return text.length > max ? `${text.slice(0, max - 1)}...` : text;
}

function toggleFilter(key) {
  activeFilters[key] = !activeFilters[key];
  document.getElementById(`f-${key}`).classList.toggle("active", activeFilters[key]);
  clearSelection();
  renderGraph();
}

function onSearch(value) {
  searchQuery = value.toLowerCase().trim();
  renderGraph();
}

function onWeightChange(value) {
  weightThreshold = Number(value);
  document.getElementById("weight-val").textContent = weightThreshold.toFixed(2);
  clearSelection();
  renderGraph();
}

function toggleSimulation() {
  simulationPaused = !simulationPaused;
  document.getElementById("pause-btn").classList.toggle("active", simulationPaused);
  const icon = document.getElementById("pause-icon");
  if (simulationPaused) {
    simulation?.stop();
    if (dashFrame) cancelAnimationFrame(dashFrame);
    dashFrame = null;
    icon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"/>`;
  } else {
    simulation?.alphaTarget(0.025).restart();
    icon.innerHTML = `<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>`;
    renderGraph();
  }
}

function resetView() {
  if (!svg || !zoomBehavior) return;
  svg.transition().duration(450).call(zoomBehavior.transform, d3.zoomIdentity);
}

function toggleAutoRefresh() {
  const btn = document.getElementById("autorefresh-btn");
  if (autoRefreshTimer) {
    clearInterval(autoRefreshTimer);
    autoRefreshTimer = null;
    btn.classList.remove("active");
  } else {
    autoRefreshTimer = setInterval(fetchAll, 30000);
    btn.classList.add("active");
  }
}

function toggleSection(id) {
  const body = document.getElementById(`body-${id}`);
  const chev = document.getElementById(`chev-${id}`);
  const open = body.style.display === "none";
  body.style.display = open ? "" : "none";
  chev?.classList.toggle("open", open);
}

document.getElementById("refresh-btn").addEventListener("click", fetchAll);
document.getElementById("autorefresh-btn").addEventListener("click", toggleAutoRefresh);
document.getElementById("pause-btn").addEventListener("click", toggleSimulation);
document.getElementById("reset-btn").addEventListener("click", resetView);
document.getElementById("search-input").addEventListener("input", event => onSearch(event.target.value));
document.getElementById("weight-slider").addEventListener("input", event => onWeightChange(event.target.value));
["grain", "entry", "active", "dormant"].forEach(key => {
  document.getElementById(`f-${key}`).addEventListener("click", () => toggleFilter(key));
});
document.querySelectorAll(".r-header").forEach(header => {
  header.addEventListener("click", () => toggleSection(header.dataset.section));
});
window.addEventListener("resize", () => { if (graphData) renderGraph(); });
window.addEventListener("load", fetchAll);
</script>
</body>
</html>
"""


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
