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
  #graph-canvas { width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }
  #graph-canvas.dragging { cursor: grabbing; }
  #graph-overlay {
    position: absolute; bottom: 12px; left: 12px;
    display: flex; gap: 8px; flex-wrap: wrap;
  }
  .legend-item { display: flex; align-items: center; gap: 4px; font-size: 10px; color: var(--text-muted); }
  .legend-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  #activity-indicator {
    position: absolute; top: 10px; left: 10px;
    display: flex; align-items: center; gap: 7px;
    background: rgba(15,17,23,0.8); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 5px 9px; font-size: 10px;
    font-family: var(--font-mono); color: var(--text-muted);
    backdrop-filter: blur(4px); pointer-events: none;
  }
  #activity-indicator.live { color: var(--cyan); border-color: rgba(34,211,238,0.35); }
  .activity-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--text-dim); box-shadow: 0 0 0 rgba(34,211,238,0);
  }
  #activity-indicator.live .activity-dot {
    background: var(--cyan); box-shadow: 0 0 14px rgba(34,211,238,0.7);
    animation: activityPulse 900ms ease-out infinite;
  }
  @keyframes activityPulse {
    0% { box-shadow: 0 0 0 0 rgba(34,211,238,0.6); }
    100% { box-shadow: 0 0 0 8px rgba(34,211,238,0); }
  }
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
  .rpanel-actions { display: flex; align-items: center; gap: 6px; }
  .rpanel-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; color: var(--text-muted); }
  .rpanel-count {
    font-size: 10px; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 10px; padding: 1px 6px; font-family: var(--font-mono); color: var(--text-muted);
  }
  .rpanel-count.warn { background: var(--amber-dim); border-color: rgba(251,191,36,0.3); color: var(--amber); }
  .rpanel-body { padding: 0 14px 12px; }
  .rpanel-chevron { color: var(--text-dim); transition: transform 0.2s; }
  .rpanel-chevron.open { transform: rotate(180deg); }
  .inspector-clear {
    width: 22px; height: 22px; opacity: 1;
  }
  .inspector-clear:disabled {
    opacity: 0.28; cursor: default; pointer-events: none;
  }

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

  #mobile-bubble-bar,
  #mobile-sheet-close { display: none; }

  /* RESPONSIVE */
  @media (max-width: 768px) {
    html, body { overflow: hidden; }
    #app { height: 100svh; }
    #topbar {
      position: fixed; top: 0; left: 0; right: 0; z-index: 30;
      padding: calc(8px + env(safe-area-inset-top, 0px)) 10px 8px;
      gap: 8px; background: rgba(8,10,14,0.92); backdrop-filter: blur(16px);
      border-bottom: 1px solid rgba(37,44,58,0.75);
    }
    #topbar .brand svg { width: 20px; height: 20px; }
    #topbar .brand-name { font-size: 13px; }
    #topbar .brand-sub, #computed-at, #refresh-interval, .divider { display: none; }
    #status-badge { padding: 3px 8px; font-size: 10px; }
    #metrics-strip {
      order: 3; width: 100%; flex: none;
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 4px;
    }
    .metric-pill { min-width: 0; padding: 4px 7px; border-radius: 8px; }
    .metric-pill .m-label { font-size: 8px; letter-spacing: 0.35px; }
    .metric-pill .m-value { font-size: 14px; }
    .metric-pill:nth-child(n+5) { display: none; }
    #topbar-controls { margin-left: auto; }
    .icon-btn { width: 32px; height: 32px; border-radius: 10px; }

    #main { height: 100svh; padding-top: 86px; flex-direction: column; }
    #graph-panel { border-right: none; min-height: 0; height: 100%; }
    #graph-toolbar {
      position: fixed; top: calc(86px + env(safe-area-inset-top, 0px)); left: 8px; right: 8px; z-index: 25;
      padding: 6px; gap: 6px; flex-wrap: nowrap; overflow-x: auto;
      border: 1px solid rgba(37,44,58,0.9); border-radius: 14px;
      background: rgba(15,17,23,0.86); backdrop-filter: blur(14px);
    }
    #search-box { min-width: 152px; max-width: none; flex: 1; border-radius: 10px; }
    #search-box input { font-size: 16px; line-height: 20px; }
    .filter-group span, .slider-group label { display: none; }
    .filter-btn { padding: 6px 9px; border-radius: 10px; white-space: nowrap; }
    .slider-group { min-width: 112px; }
    #weight-slider { width: 70px; }
    #graph-container { height: calc(100svh - 86px); }
    #graph-canvas { height: 100%; }
    #graph-overlay { display: none; }
    #graph-stats {
      top: auto; right: auto; left: 10px;
      bottom: calc(76px + env(safe-area-inset-bottom, 0px));
      border-radius: 10px; background: rgba(8,10,14,0.72);
    }
    #activity-indicator {
      top: 58px; left: 10px; right: auto;
      border-radius: 10px; background: rgba(8,10,14,0.72);
    }
    #tooltip { display: none; }

    #right-panel {
      position: fixed; left: 10px; right: 10px;
      bottom: calc(10px + env(safe-area-inset-bottom, 0px)); z-index: 45;
      width: auto; height: min(64svh, 520px); max-height: 64svh;
      border: 1px solid rgba(37,44,58,0.95); border-radius: 18px;
      background: rgba(15,17,23,0.96); box-shadow: 0 22px 70px rgba(0,0,0,0.7);
      transform: translateY(calc(100% + 20px)); transition: transform 180ms ease;
      overflow-y: auto; backdrop-filter: blur(18px);
    }
    body.mobile-sheet-open #right-panel { transform: translateY(0); }
    #mobile-sheet-close {
      display: flex; position: sticky; top: 0; z-index: 3;
      width: 100%; height: 38px; align-items: center; justify-content: center;
      background: rgba(15,17,23,0.98); border: 0; border-bottom: 1px solid var(--border);
      color: var(--text-muted);
    }
    #mobile-sheet-close::before {
      content: ''; width: 46px; height: 4px; border-radius: 4px; background: var(--border2);
    }
    #mobile-bubble-bar {
      display: flex; position: fixed; z-index: 40; left: 50%;
      bottom: calc(12px + env(safe-area-inset-bottom, 0px)); transform: translateX(-50%);
      gap: 8px; padding: 8px; border-radius: 999px;
      background: rgba(15,17,23,0.88); border: 1px solid rgba(37,44,58,0.9);
      box-shadow: 0 16px 45px rgba(0,0,0,0.55); backdrop-filter: blur(18px);
      transition: opacity 150ms ease, transform 150ms ease;
    }
    body.mobile-sheet-open #mobile-bubble-bar {
      opacity: 0; pointer-events: none; transform: translateX(-50%) translateY(12px);
    }
    .mobile-bubble {
      width: 42px; height: 42px; display: flex; align-items: center; justify-content: center;
      border-radius: 999px; border: 1px solid var(--border);
      background: var(--surface2); color: var(--text-muted);
    }
    .mobile-bubble.active {
      color: var(--cyan); border-color: rgba(34,211,238,0.38);
      background: var(--cyan-dim); box-shadow: 0 0 22px rgba(34,211,238,0.14);
    }
    .mobile-bubble.warn.active {
      color: var(--amber); border-color: rgba(251,191,36,0.38);
      background: var(--amber-dim); box-shadow: 0 0 22px rgba(251,191,36,0.12);
    }
  }

  @media (max-width: 430px) and (max-height: 260px) {
    #topbar { padding: 6px 8px; gap: 6px; }
    #topbar .brand-name { font-size: 11px; }
    #status-badge { font-size: 9px; padding: 2px 6px; }
    #metrics-strip {
      display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 3px;
    }
    .metric-pill { padding: 2px 5px; }
    .metric-pill .m-label { font-size: 7px; }
    .metric-pill .m-value { font-size: 11px; }
    .metric-pill:nth-child(n+4), #topbar-controls, #graph-toolbar,
    #graph-overlay, #right-panel, #mobile-bubble-bar { display: none; }
    #main { padding-top: 44px; }
    #graph-container { height: calc(100svh - 44px); }
    #graph-stats { bottom: 6px; left: 6px; font-size: 8px; padding: 3px 6px; }
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
          <input type="text" id="search-input" placeholder="Search nodes…" oninput="onSearch(this.value)" onkeydown="onSearchKey(event)">
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
        <div id="activity-indicator"><span class="activity-dot"></span><span id="activity-text">watching events</span></div>
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
      <button id="mobile-sheet-close" onclick="closeMobileSheet()" aria-label="Close panel"></button>

      <!-- INSPECTOR -->
      <div class="rpanel-section">
        <div class="rpanel-header" onclick="toggleSection('inspector')">
          <div class="rpanel-header-left">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65" stroke-opacity="0.7"></line></svg>
            <span class="rpanel-title">Inspector</span>
          </div>
          <div class="rpanel-actions">
            <button class="icon-btn inspector-clear" id="clear-selection-btn" title="Deselect" aria-label="Deselect" disabled onclick="event.stopPropagation(); deselectAll();">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
            </button>
            <svg class="rpanel-chevron open" id="chev-inspector" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="18 15 12 9 6 15"></polyline></svg>
          </div>
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

  <div id="mobile-bubble-bar" aria-label="Mobile dashboard panels">
    <button class="mobile-bubble active" id="mobile-bubble-inspector" title="Inspector" onclick="openMobileSheet('inspector')" aria-label="Inspector">
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
    </button>
    <button class="mobile-bubble warn" id="mobile-bubble-warnings" title="Warnings" onclick="openMobileSheet('warnings')" aria-label="Warnings">
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
    </button>
    <button class="mobile-bubble" id="mobile-bubble-health" title="Health" onclick="openMobileSheet('health')" aria-label="Health">
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
    </button>
    <button class="mobile-bubble" title="Refresh" onclick="fetchAll()" aria-label="Refresh">
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
    </button>
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
function clamp01(v) { return Math.max(0, Math.min(1, v)); }
function edgeColor(l) {
  if (l.direction === 'bidirectional') return '#a78bfa';
  if (l.decay_class === 'core') return '#38bdf8';
  if (l.decay_class === 'ephemeral') return '#1e3a4a';
  return '#1e3f5a';
}
function edgeWidth(l) {
  return Math.max(0.5, (l.effective_weight??0.3)*2.5);
}
function edgeAlpha(l) {
  return 0.55;
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
  return graphData.nodes.filter(nodePassesFilters);
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
let searchMatchIds = new Set();
let canvasCssWidth = 0, canvasCssHeight = 0, canvasDpr = 1;
let activityNodePulses = new Map();
let activityEdgePulses = new Map();
let activityParticles = [];
let seenEventKeys = new Set();
let eventPollTimer = null;
let lastActivityAt = 0;
let graphRefreshTimer = null;
let activityEventEdgeKeys = null;

function activityColor(ev) {
  if (ev.event === 'conduit_penalized') return '#fb7185';
  if (ev.event === 'shortcut_created') return '#a78bfa';
  if (ev.event === 'highway_formed') return '#22d3ee';
  if (ev.event === 'conduit_reinforced') return '#fbbf24';
  if (ev.category === 'retrieval') return '#22d3ee';
  if (ev.category === 'write') return '#86efac';
  if (ev.category === 'feedback') return ev.data?.useful === false ? '#fb7185' : '#fbbf24';
  if (ev.category === 'system') return '#a78bfa';
  return '#38bdf8';
}

function eventKey(ev) {
  return [
    ev.timestamp,
    ev.category,
    ev.event,
    ev.data?.trace_id,
    ev.data?.grain_id,
    ev.data?.feature,
    ev.data?.conduit_id,
    ev.data?.from_id,
    ev.data?.to_id,
  ]
    .filter(Boolean).join('|');
}

function edgeKey(l) {
  const sid = typeof l.source === 'object' ? l.source.id : l.source;
  const tid = typeof l.target === 'object' ? l.target.id : l.target;
  return `${sid}->${tid}`;
}

function isConduitEvent(ev) {
  const data = ev?.data || {};
  return [
    'conduit_reinforced',
    'conduit_penalized',
    'highway_formed',
    'shortcut_created',
  ].includes(ev?.event) && Boolean(data.conduit_id || (data.from_id && data.to_id));
}

function isGraphRefreshEvent(ev) {
  return ev.event === 'grain_stored'
    || ev.event === 'entry_point_created'
    || ev.event === 'bootstrap_conduits_created'
    || ev.event === 'graph_rebuild_completed';
}

function scheduleGraphRefreshAfterEvent() {
  if (graphRefreshTimer) clearTimeout(graphRefreshTimer);
  graphRefreshTimer = setTimeout(() => {
    graphRefreshTimer = null;
    fetchAll().catch(err => console.warn('Graph refresh after structural event failed', err));
  }, 1200);
}

function getPulse(map, key, now) {
  const pulse = map.get(key);
  if (!pulse) return null;
  const remaining = pulse.until - now;
  if (remaining <= 0) {
    map.delete(key);
    return null;
  }
  return {...pulse, alpha: clamp01(remaining / pulse.duration)};
}

function getNodePulse(n, now) { return getPulse(activityNodePulses, n.id, now); }
function getEdgePulse(l, now) { return getPulse(activityEdgePulses, edgeKey(l), now); }

function setActivityLabel(text, color) {
  const indicator = document.getElementById('activity-indicator');
  const label = document.getElementById('activity-text');
  if (!indicator || !label) return;
  lastActivityAt = performance.now();
  indicator.classList.add('live');
  indicator.style.borderColor = `${color}66`;
  label.textContent = text;
}

function pruneActivity(now) {
  for (const [key, pulse] of activityNodePulses) {
    if (pulse.until <= now) activityNodePulses.delete(key);
  }
  for (const [key, pulse] of activityEdgePulses) {
    if (pulse.until <= now) activityEdgePulses.delete(key);
  }
  activityParticles = activityParticles.filter(p => now - p.start < p.duration);
  const indicator = document.getElementById('activity-indicator');
  const label = document.getElementById('activity-text');
  if (indicator && label && lastActivityAt && now - lastActivityAt > 3500) {
    indicator.classList.remove('live');
    indicator.style.borderColor = '';
    label.textContent = 'watching events';
  }
}

function activateEdge(edge, color, now) {
  if (!edge) return 0;
  const key = edgeKey(edge);
  if (activityEventEdgeKeys?.has(key)) return 0;
  activityEventEdgeKeys?.add(key);
  activityEdgePulses.set(key, { until: now + 2400, duration: 2400, color });
  activityParticles.push({ key, edge, start: now, duration: 1800, color });
  if (activityParticles.length > 700) {
    activityParticles = activityParticles.slice(-700);
  }
  return 1;
}

function pulseNodeOnly(node, color, now) {
  if (!node) return 0;
  activityNodePulses.set(node.id, { until: now + 3200, duration: 3200, color });
  return 1;
}

function activateNode(node, color, now) {
  let count = pulseNodeOnly(node, color, now);
  if (!count) return 0;
  for (const edge of canvasLinks) {
    if (edge.source === node || edge.target === node) {
      activateEdge(edge, color, now);
      count++;
    }
  }
  return count;
}

function activateNodeById(id, color, now) {
  if (!id) return 0;
  const needle = String(id);
  const node = canvasNodes.find(n => n.id === needle || n.feature === needle || n.label === needle);
  return activateNode(node, color, now);
}

function pulseNodeById(id, color, now) {
  if (!id) return 0;
  const needle = String(id);
  const node = canvasNodes.find(n => n.id === needle || n.feature === needle || n.label === needle);
  return pulseNodeOnly(node, color, now);
}

function activateFeature(feature, color, now) {
  if (!feature) return 0;
  const needle = String(feature).toLowerCase();
  let count = 0;
  for (const node of canvasNodes) {
    const haystack = [node.feature, node.label, node.id, node.provenance].filter(Boolean).join(' ').toLowerCase();
    if (haystack.includes(needle)) count += activateNode(node, color, now);
  }
  return count;
}

function pulseFeatureNodes(feature, color, now) {
  if (!feature) return 0;
  const needle = String(feature).toLowerCase();
  let count = 0;
  for (const node of canvasNodes) {
    const haystack = [node.feature, node.label, node.id, node.provenance].filter(Boolean).join(' ').toLowerCase();
    if (haystack.includes(needle)) count += pulseNodeOnly(node, color, now);
  }
  return count;
}

function activateEventTargets(ev, color, now) {
  const data = ev.data || {};
  let activated = 0;
  activityEventEdgeKeys = new Set();
  try {
    activated += activateNodeById(data.grain_id, color, now);
    if (Array.isArray(data.features)) {
      for (const feature of data.features.slice(0, 12)) activated += activateFeature(feature, color, now);
    }
    activated += activateFeature(data.feature, color, now);
  } finally {
    activityEventEdgeKeys = null;
  }
  return activated;
}

function findConduitEdge(data) {
  if (!data) return null;
  const conduitId = data.conduit_id ? String(data.conduit_id) : '';
  const fromId = data.from_id ? String(data.from_id) : '';
  const toId = data.to_id ? String(data.to_id) : '';
  return canvasLinks.find(edge => {
    const sid = edge.source?.id;
    const tid = edge.target?.id;
    if (conduitId && edge.id === conduitId) return true;
    if (fromId && toId && sid === fromId && tid === toId) return true;
    return Boolean(fromId && toId && edge.direction === 'bidirectional' && sid === toId && tid === fromId);
  }) || null;
}

function activateConduitEvent(ev, color, now) {
  const data = ev.data || {};
  let activated = 0;
  activityEventEdgeKeys = new Set();
  try {
    const edge = findConduitEdge(data);
    if (edge) activated += activateEdge(edge, color, now);
    activated += pulseNodeById(data.from_id, color, now);
    activated += pulseNodeById(data.to_id, color, now);
    activated += pulseNodeById(data.grain_id, color, now);
  } finally {
    activityEventEdgeKeys = null;
  }
  return activated;
}

function findTraceEdge(step) {
  return findConduitEdge(step);
}

async function animateTrace(traceId, color) {
  if (!traceId) return 0;
  const payload = await fetchJSON(`/api/trace?trace_id=${encodeURIComponent(traceId)}`);
  const steps = (payload.steps || [])
    .filter(step => step.from_id && step.to_id)
    .sort((a, b) => (a.hop ?? 0) - (b.hop ?? 0) || Math.abs(b.signal ?? 0) - Math.abs(a.signal ?? 0))
    .slice(0, 300);

  const seenEdges = new Set();
  steps.forEach((step, index) => {
    const delay = Math.min(step.hop ?? 0, 6) * 320 + Math.min(index, 90) * 10;
    setTimeout(() => {
      const edge = findTraceEdge(step);
      const now = performance.now();
      pulseNodeById(step.from_id, color, now);
      pulseNodeById(step.to_id, color, now);
      if (edge) {
        const key = edgeKey(edge);
        if (!seenEdges.has(key)) {
          seenEdges.add(key);
          activateEdge(edge, color, now);
        }
      }
      draw();
    }, delay);
  });
  return steps.length;
}

function handleFluxEvent(ev) {
  const now = performance.now();
  const color = activityColor(ev);
  const data = ev.data || {};
  let activated = 0;
  const traceId = data.trace_id || ev.trace_id;

  if (ev.category === 'retrieval' && ev.event === 'grains_returned' && traceId) {
    setActivityLabel(`trace/${String(traceId).slice(0, 8)}`, color);
    animateTrace(traceId, color).then(count => {
      if (!count) {
        const fallbackNow = performance.now();
        const fallback = activateEventTargets(ev, color, fallbackNow);
        if (fallback) draw();
      }
    }).catch(err => {
      console.warn('Trace animation failed', err);
      const fallbackNow = performance.now();
      const fallback = activateEventTargets(ev, color, fallbackNow);
      if (fallback) draw();
    });
  } else if (ev.category === 'retrieval' && ev.event === 'features_extracted') {
    setActivityLabel('retrieval/features', color);
  } else if (ev.event === 'entry_point_created') {
    activated = pulseFeatureNodes(data.feature, color, now);
    setActivityLabel('write/entry_point', color);
    if (activated) draw();
  } else if (isConduitEvent(ev)) {
    activated = activateConduitEvent(ev, color, now);
    const label = ev.category && ev.event ? `${ev.category}/${ev.event}` : 'flux conduit';
    setActivityLabel(label, color);
    if (activated) draw();
  } else {
    activated = activateEventTargets(ev, color, now);
    const label = ev.category && ev.event ? `${ev.category}/${ev.event}` : 'flux activity';
    setActivityLabel(label, color);
    if (activated || ev.category !== 'system') draw();
  }

  if (isGraphRefreshEvent(ev)) {
    scheduleGraphRefreshAfterEvent();
  }
}

async function fetchEvents() {
  const payload = await fetchJSON('/api/events?limit=50');
  const events = payload.events || [];
  const firstLoad = seenEventKeys.size === 0;
  for (const ev of events.slice().reverse()) {
    const key = eventKey(ev);
    if (!key || seenEventKeys.has(key)) continue;
    seenEventKeys.add(key);
    const when = Date.parse(ev.timestamp || '');
    const fresh = Number.isFinite(when) && Date.now() - when < 12000;
    if (!firstLoad || fresh) handleFluxEvent(ev);
  }
  if (seenEventKeys.size > 300) {
    seenEventKeys = new Set(Array.from(seenEventKeys).slice(-200));
  }
}

function startEventPolling() {
  if (eventPollTimer) return;
  fetchEvents().catch(err => console.warn('Initial event poll failed', err));
  eventPollTimer = setInterval(() => {
    fetchEvents().catch(err => console.warn('Event poll failed', err));
  }, 1500);
}

function drawActivityParticles(now) {
  for (const particle of activityParticles) {
    const edge = particle.edge;
    if (!edge?.source || !edge?.target) continue;
    const sx = edge.source.x, sy = edge.source.y, tx = edge.target.x, ty = edge.target.y;
    if (sx==null||sy==null||tx==null||ty==null) continue;
    const raw = clamp01((now - particle.start) / particle.duration);
    const eased = raw < 0.5 ? 2 * raw * raw : 1 - Math.pow(-2 * raw + 2, 2) / 2;
    const x = sx + (tx - sx) * eased;
    const y = sy + (ty - sy) * eased;
    const tail = Math.max(0, eased - 0.08);
    const tx2 = sx + (tx - sx) * tail;
    const ty2 = sy + (ty - sy) * tail;
    const alpha = Math.sin(raw * Math.PI);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = particle.color;
    ctx.lineWidth = 2.2 / Math.max(0.65, transform.k);
    ctx.beginPath();
    ctx.moveTo(tx2, ty2);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.fillStyle = particle.color;
    ctx.arc(x, y, 3.5 / Math.max(0.75, transform.k), 0, Math.PI*2);
    ctx.fill();
    ctx.restore();
  }
}

function updateSearchFocus() {
  searchMatchIds = new Set();
  for (const node of canvasNodes) {
    node._searchMatch = Boolean(searchQuery && nodeMatchesSearch(node));
    if (node._searchMatch) searchMatchIds.add(node.id);
  }
}

function edgeMatchesSearch(edge) {
  return Boolean(searchQuery && (
    searchMatchIds.has(edge.source?.id) || searchMatchIds.has(edge.target?.id)
  ));
}

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
  updateSearchFocus();
  canvasLinks = visLinks.map(l => ({
    ...l,
    source: nodeMap.get(typeof l.source==='object'?l.source.id:l.source) || (typeof l.source==='object'?l.source.id:l.source),
    target: nodeMap.get(typeof l.target==='object'?l.target.id:l.target) || (typeof l.target==='object'?l.target.id:l.target),
  })).filter(l => l.source && l.target && typeof l.source==='object' && typeof l.target==='object');

  // Set up canvas
  canvas = document.getElementById('graph-canvas');
  canvasCssWidth = W;
  canvasCssHeight = H;
  canvasDpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 3));
  canvas.width = Math.max(1, Math.floor(W * canvasDpr));
  canvas.height = Math.max(1, Math.floor(H * canvasDpr));
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx = canvas.getContext('2d');

  // Cancel previous rAF
  if (animFrame3D) { cancelAnimationFrame(animFrame3D); animFrame3D = null; }
  if (nudgeTimer) { clearInterval(nudgeTimer); nudgeTimer = null; }

  // D3 zoom on canvas
  zoom = d3.zoom().scaleExtent([0.1, 5]).on('zoom', e => { transform = e.transform; });
  d3.select(canvas).call(zoom);

  // Pointer events support mouse and touch node dragging while D3 handles pan/zoom.
  if (window.PointerEvent) {
    canvas.onpointerdown = onCanvasPointerDown;
    canvas.onpointermove = onCanvasPointerMove;
    canvas.onpointerup = onCanvasPointerUp;
    canvas.onpointercancel = onCanvasPointerUp;
    canvas.onmousedown = null;
    canvas.onmousemove = null;
    canvas.onmouseup = null;
  } else {
    canvas.onmousedown = onCanvasPointerDown;
    canvas.onmousemove = onCanvasPointerMove;
    canvas.onmouseup = onCanvasPointerUp;
  }
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
  const W = canvasCssWidth || canvas.clientWidth;
  const H = canvasCssHeight || canvas.clientHeight;
  const now = performance.now();
  pruneActivity(now);
  ctx.setTransform(canvasDpr, 0, 0, canvasDpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.save();
  ctx.translate(transform.x, transform.y);
  ctx.scale(transform.k, transform.k);

  const t = now / 1000;

  // Draw edges
  for (const l of canvasLinks) {
    const sx = l.source.x, sy = l.source.y;
    const tx = l.target.x, ty = l.target.y;
    if (sx==null||sy==null||tx==null||ty==null) continue;

    const isSelected = selectedEdge === l;
    const isHovered = hoveredEdge === l;
    const isSearchMatch = edgeMatchesSearch(l);
    const isSearchDimmed = searchQuery && !isSearchMatch;
    const isDimmed = !isSelected && !isHovered && (
      (selectedNode && l.source !== selectedNode && l.target !== selectedNode) || isSearchDimmed
    );

    const pulse = getEdgePulse(l, now);
    const w = edgeWidth(l);
    let alpha = isDimmed ? 0.035 : (isSelected||isHovered||isSearchMatch||pulse) ? 1 : edgeAlpha(l);
    const stroke = pulse ? pulse.color : (isSelected || isSearchMatch) ? '#22d3ee' : edgeColor(l);

    ctx.save();
    if (pulse) {
      ctx.globalAlpha = 0.35 * pulse.alpha;
      ctx.strokeStyle = pulse.color;
      ctx.lineWidth = w + 7;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(tx, ty);
      ctx.stroke();
    }

    ctx.globalAlpha = alpha;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = isSelected ? w+1.3 : isSearchMatch ? w+1 : w;

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
    ctx.fillStyle = stroke;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - 8*Math.cos(angle-0.4), ay - 8*Math.sin(angle-0.4));
    ctx.lineTo(ax - 8*Math.cos(angle+0.4), ay - 8*Math.sin(angle+0.4));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  drawActivityParticles(now);

  // Draw nodes
  for (const n of canvasNodes) {
    if (n.x==null||n.y==null) continue;
    const r = nodeRadius(n);
    const isSelected = selectedNode === n;
    const isHovered = hoveredNode === n;
    const isSearchMatch = Boolean(searchQuery && n._searchMatch);
    const isSearchDimmed = searchQuery && !isSearchMatch;
    const isDimmed = !isSelected && !isHovered && (
      (selectedNode && selectedNode !== n && !isNeighbor(selectedNode, n)) || isSearchDimmed
    );
    const pulse = getNodePulse(n, now);
    const alpha = isDimmed && !pulse ? 0.14 : 1;
    const dormant = n.status === 'dormant';

    ctx.save();
    ctx.globalAlpha = alpha;

    // Search focus ring
    if (isSearchMatch) {
      const glowR = r + 12 + Math.sin(t * 3 + n._phase) * 2;
      const grad = ctx.createRadialGradient(n.x, n.y, r, n.x, n.y, glowR);
      grad.addColorStop(0, 'rgba(251,191,36,0.34)');
      grad.addColorStop(1, 'rgba(251,191,36,0)');
      ctx.beginPath();
      ctx.arc(n.x, n.y, glowR, 0, Math.PI*2);
      ctx.fillStyle = grad;
      ctx.fill();
    }

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

    // Live Flux activity pulse
    if (pulse) {
      const pulseR = r + 10 + (1 - pulse.alpha) * 14;
      const grad = ctx.createRadialGradient(n.x, n.y, r, n.x, n.y, pulseR);
      grad.addColorStop(0, `${pulse.color}66`);
      grad.addColorStop(1, `${pulse.color}00`);
      ctx.globalAlpha = Math.max(0.18, pulse.alpha);
      ctx.beginPath();
      ctx.arc(n.x, n.y, pulseR, 0, Math.PI*2);
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.globalAlpha = pulse.alpha;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 5, 0, Math.PI*2);
      ctx.strokeStyle = pulse.color;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.globalAlpha = alpha;
    }

    // Selection ring
    if (isSelected || isHovered || isSearchMatch) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + (isSearchMatch ? 6 : 4), 0, Math.PI*2);
      ctx.strokeStyle = isSelected ? '#22d3ee' : isSearchMatch ? '#fbbf24' : 'rgba(34,211,238,0.4)';
      ctx.lineWidth = isSelected || isSearchMatch ? 2 : 1;
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
    ctx.globalAlpha = isSearchMatch ? 1 : alpha * (dormant ? 0.35 : 0.9);
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
    ctx.fillStyle = isSearchMatch ? '#fbbf24' : isSelected ? '#dde3f0' : '#5a6480';
    ctx.font = `${isSearchMatch ? '600 ' : ''}${window.innerWidth <= 768 ? 10 : 9}px JetBrains Mono, monospace`;
    ctx.textBaseline = 'middle';
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
let _dragging = null, _dragMoved = false, _activePointerId = null;

function onCanvasPointerDown(e) {
  if (e.button !== undefined && e.button !== 0) return;
  const [mx, my] = canvasPoint(e);
  const n = hitNode(mx, my);
  if (n) {
    _dragging = n; _dragMoved = false; _activePointerId = e.pointerId ?? null;
    canvas.setPointerCapture?.(e.pointerId);
    if (!simulation._active) simulation.alphaTarget(0.3).restart();
    n.fx = n.x; n.fy = n.y;
    d3.select(canvas).on('.zoom', null);
    e.preventDefault();
  }
}

function onCanvasPointerMove(e) {
  if (_activePointerId !== null && e.pointerId !== undefined && e.pointerId !== _activePointerId) return;
  const [mx, my] = canvasPoint(e);
  if (_dragging) {
    _dragMoved = true;
    _dragging.fx = mx; _dragging.fy = my;
    simulation.alphaTarget(0.15).restart();
    e.preventDefault();
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

function onCanvasPointerUp(e) {
  if (_activePointerId !== null && e.pointerId !== undefined && e.pointerId !== _activePointerId) return;
  if (_dragging) {
    _dragging.fx = null; _dragging.fy = null;
    simulation.alphaTarget(0.004);
    canvas.releasePointerCapture?.(e.pointerId);
    _dragging = null;
    _activePointerId = null;
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
  document.getElementById('mobile-bubble-inspector')?.classList.remove('active');
  updateSelectionControls();
  draw();
}

function selectNode(d) {
  selectedNode = d; selectedEdge = null;
  renderInspector(d, 'node');
  updateSelectionControls();
  markInspectorReady();
}

function selectEdge(d) {
  selectedEdge = d; selectedNode = null;
  renderInspector(d, 'edge');
  updateSelectionControls();
  markInspectorReady();
}

function updateSelectionControls() {
  const clearBtn = document.getElementById('clear-selection-btn');
  if (clearBtn) clearBtn.disabled = !(selectedNode || selectedEdge);
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
  updateSearchFocus();
  draw();
}
function onSearchKey(event) {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  event.currentTarget.blur();
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

function openMobileSheet(section) {
  document.body.classList.add('mobile-sheet-open');
  for (const id of ['inspector', 'warnings', 'health']) {
    const body = document.getElementById('body-' + id);
    const chev = document.getElementById('chev-' + id);
    const open = id === section;
    if (body) body.style.display = open ? '' : 'none';
    if (chev) chev.classList.toggle('open', open);
    document.getElementById('mobile-bubble-' + id)?.classList.toggle('active', open);
  }
}

function closeMobileSheet() {
  document.body.classList.remove('mobile-sheet-open');
}

function markInspectorReady() {
  document.getElementById('mobile-bubble-inspector')?.classList.add('active');
}

// ── INIT ──────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  fetchAll().then(() => startEventPolling());
  window.addEventListener('resize', () => { if (graphData) renderGraph(); });
  window.addEventListener('keydown', event => {
    if (event.key === 'Escape' && (selectedNode || selectedEdge)) deselectAll();
  });
});
</script>


</body></html>"""


_MOBILE_PREVIEW_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mobile Preview - Flux Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    min-height: 100vh; padding: 32px 24px 48px; overflow-x: auto;
    background: #060810; color: #dde3f0; font-family: system-ui, sans-serif;
  }
  h1 {
    margin-bottom: 32px; color: #3a4255; text-align: center;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px;
  }
  .devices-row {
    display: flex; align-items: flex-start; justify-content: center;
    gap: 40px; flex-wrap: wrap;
  }
  .device { display: flex; flex-direction: column; align-items: center; gap: 12px; }
  .device-label { font-size: 10px; color: #3a4255; text-transform: uppercase; letter-spacing: 0.6px; }
  .device-sub { font-size: 10px; color: #22253a; }
  .iphone-air {
    width: 280px; background: #1c1c1e; border-radius: 50px;
    border: 1.5px solid #2c2c2e; position: relative; padding: 14px;
    box-shadow: 0 0 0 1px #0a0a0c, inset 0 0 0 1px #3a3a3e,
      0 40px 80px rgba(0,0,0,0.9), 0 0 40px rgba(34,211,238,0.04);
  }
  .iphone-air .dynamic-island {
    position: absolute; top: 14px; left: 50%; transform: translateX(-50%);
    width: 90px; height: 30px; background: #000; border-radius: 20px; z-index: 10;
  }
  .iphone-air .screen {
    width: 100%; height: 580px; border-radius: 38px; overflow: hidden;
    background: #080a0e; position: relative;
  }
  .iphone-air .screen iframe {
    width: 390px; height: 844px; border: none; display: block;
    transform: scale(0.642); transform-origin: top left;
  }
  .iphone-air .home-bar, .razr-open .home-bar {
    height: 4px; border-radius: 2px; background: rgba(255,255,255,0.12); margin: 10px auto 0;
  }
  .iphone-air .home-bar { width: 80px; }
  .iphone-air::before {
    content: ''; position: absolute; right: -3px; top: 120px;
    width: 3px; height: 60px; background: #2a2a2c; border-radius: 0 3px 3px 0;
  }
  .iphone-air::after {
    content: ''; position: absolute; left: -3px; top: 100px;
    width: 3px; height: 100px; background: #2a2a2c; border-radius: 3px 0 0 3px;
    box-shadow: 0 50px 0 #2a2a2c, 0 -50px 0 #2a2a2c;
  }
  .razr-open {
    width: 258px; background: #18181a; border-radius: 30px;
    border: 1.5px solid #2c2c2e; position: relative; padding: 12px;
    box-shadow: 0 0 0 1px #0a0a0c, inset 0 0 0 1px #303032, 0 40px 80px rgba(0,0,0,0.9);
  }
  .razr-open .hinge-seam {
    position: absolute; left: 12px; right: 12px; top: 50%; transform: translateY(-50%);
    height: 6px; background: #0f0f11; border-top: 1px solid #222224;
    border-bottom: 1px solid #222224; z-index: 20; pointer-events: none;
  }
  .razr-open .punch-hole, .razr-folded .punch-hole {
    position: absolute; left: 50%; transform: translateX(-50%);
    background: #000; border-radius: 50%; z-index: 10;
  }
  .razr-open .punch-hole { top: 18px; width: 10px; height: 10px; }
  .razr-open .screen {
    width: 100%; height: 540px; border-radius: 20px; overflow: hidden; background: #080a0e;
  }
  .razr-open .screen iframe {
    width: 360px; height: 780px; border: none; display: block;
    transform: scale(0.653); transform-origin: top left;
  }
  .razr-open .home-bar { width: 70px; }
  .razr-folded {
    width: 200px; background: #18181a; border-radius: 24px 24px 8px 8px;
    border: 1.5px solid #2c2c2e; position: relative; padding: 10px;
    box-shadow: 0 0 0 1px #0a0a0c, inset 0 0 0 1px #303032, 0 20px 50px rgba(0,0,0,0.9);
  }
  .razr-folded .cover-screen {
    width: 100%; height: 120px; border-radius: 16px 16px 4px 4px;
    overflow: hidden; background: #080a0e; position: relative;
  }
  .razr-folded .cover-screen iframe {
    width: 390px; height: 162px; border: none; display: block;
    transform: scale(0.462); transform-origin: top left;
  }
  .razr-folded .hinge {
    height: 14px; background: linear-gradient(180deg, #111113 0%, #1a1a1c 50%, #111113 100%);
    border-top: 1px solid #2a2a2c; border-bottom: 1px solid #2a2a2c;
    display: flex; align-items: center; justify-content: center;
  }
  .razr-folded .hinge::after { content: ''; width: 40px; height: 3px; border-radius: 2px; background: #222224; }
  .razr-folded .body-bottom { height: 14px; background: #18181a; border-radius: 0 0 6px 6px; }
  .razr-folded .punch-hole { top: 10px; width: 9px; height: 9px; }
  .note { font-size: 10px; color: #2a3040; text-align: center; line-height: 1.7; max-width: 200px; }
  .note span { color: #22d3ee; }
</style>
</head>
<body>
<h1>Flux Memory Dashboard - Mobile Layouts</h1>
<div class="devices-row">
  <div class="device">
    <div class="device-label">iPhone Air</div>
    <div class="device-sub">390 x 844</div>
    <div class="iphone-air">
      <div class="dynamic-island"></div>
      <div class="screen"><iframe src="/" scrolling="no"></iframe></div>
      <div class="home-bar"></div>
    </div>
    <div class="note">Single finger: drag node<br>Two fingers: pan + pinch zoom<br>Tap node -> <span>inspector bubble glows</span><br>Tap bubble -> sheet slides up</div>
  </div>
  <div class="device">
    <div class="device-label">Motorola Razr+</div>
    <div class="device-sub">360 x 780 unfolded</div>
    <div class="razr-open">
      <div class="punch-hole"></div>
      <div class="hinge-seam"></div>
      <div class="screen"><iframe src="/" scrolling="no"></iframe></div>
      <div class="home-bar"></div>
    </div>
    <div class="note">Full screen graph<br>Bubble bar at bottom<br>Hinge seam is cosmetic only</div>
  </div>
  <div class="device">
    <div class="device-label">Motorola Razr+</div>
    <div class="device-sub">390 x 162 folded outer screen</div>
    <div class="razr-folded">
      <div class="punch-hole"></div>
      <div class="cover-screen"><iframe src="/" scrolling="no"></iframe></div>
      <div class="hinge"></div>
      <div class="body-bottom"></div>
    </div>
    <div class="note">Ultra-compact mode<br>Status + graph only<br><span>No bubbles, no sheets</span><br>Quick glance view</div>
  </div>
</div>
</body>
</html>"""


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


def _trace_details(store: Any, trace_id: str) -> dict[str, Any]:
    if not trace_id:
        return {"trace_id": "", "steps": []}
    trace = store.get_trace(trace_id)
    if trace is None:
        return {"trace_id": trace_id, "steps": []}
    try:
        steps = json.loads(trace.trace_data or "[]")
    except json.JSONDecodeError:
        steps = []
    return {
        "trace_id": trace.id,
        "query": trace.query_text,
        "hop_count": trace.hop_count,
        "activated_grain_count": trace.activated_grain_count,
        "steps": steps if isinstance(steps, list) else [],
    }


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
            elif path == "/mobile-preview":
                self._send(200, "text/html; charset=utf-8", _MOBILE_PREVIEW_HTML.encode())
            elif path == "/api/health":
                data = flux_health(store, cfg) if cfg else flux_health(store)
                self._send(200, "application/json", json.dumps(data, default=str).encode())
            elif path == "/api/graph":
                data = export_json(store)
                self._send(200, "application/json", json.dumps(data, default=str).encode())
            elif path == "/api/clusters":
                data = cluster_view(store)
                self._send(200, "application/json", json.dumps(data, default=str).encode())
            elif path == "/api/trace":
                query = parse_qs(parsed.query)
                trace_id = query.get("trace_id", [""])[0]
                data = _trace_details(store, trace_id=trace_id)
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
