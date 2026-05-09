// ── MOCK DATA ─────────────────────────────────────────────────────────────────
const MOCK_HEALTH = {"status":"warning","signals":{"highway_count":{"value":0,"healthy":false},"core_grain_count":{"value":12,"healthy":true},"orphan_rate":{"value":0.03,"healthy":true},"avg_conduit_weight":{"value":0.385,"healthy":true},"dormant_grain_rate":{"value":0.02,"healthy":true},"highway_growth_rate":{"value":0,"healthy":false},"shortcut_creation_rate":{"value":0.05,"healthy":true},"conduit_dissolution_rate":{"value":0,"healthy":false},"avg_weight_drop_on_failure":{"value":0,"healthy":false},"promotion_events":{"value":3,"healthy":true},"avg_hops_per_retrieval":{"value":2.1,"healthy":true},"retrieval_success_rate":{"value":0.74,"healthy":true},"fallback_trigger_rate":{"value":1.0,"healthy":false},"feedback_compliance_rate":{"value":0.0,"healthy":false}},"active_warnings":[{"signal":"feedback_compliance_rate","severity":"WARNING","current_value":0.0,"healthy_range":">= 0.8","first_seen":"2026-04-23T13:43:13Z","last_seen":"2026-04-24T03:15:47Z","suggestion":"Main AI is not calling flux_feedback reliably. Prompt engineering issue."},{"signal":"fallback_trigger_rate","severity":"WARNING","current_value":1.0,"healthy_range":"< 0.3","first_seen":"2026-04-23T14:00:00Z","last_seen":"2026-04-24T03:15:47Z","suggestion":"All retrievals are using fallback. Check retrieval routing logic."}],"computed_at":"2026-04-24T03:15:47.162197+05:30"};
const MOCK_GRAPH = {directed:true,multigraph:false,stats:{grains:0,active_grains:0,dormant_grains:0,entries:0,conduits:0,embeddings:0},nodes:[],links:[]};

// ── STATE ─────────────────────────────────────────────────────────────────────
let healthData = null, graphData = null;
let simulation = null;
let canvas, ctx, W, H, dpr;
let transform = {x: 0, y: 0, k: 1};
let simNodes = [], simLinks = [];
let hoveredNode = null, selectedNode = null, selectedEdge = null;
let isDragging = false, dragNode = null;
let activeFilters = {grain:true, entry:true, active:true, dormant:false};
let weightThreshold = 0;
let searchQuery = '';
let simulationPaused = false;
let autoRefreshTimer = null;
let isAutoRefresh = false;
let animId = null;
let dashOffset = 0;

// ── FETCH ─────────────────────────────────────────────────────────────────────
async function fetchAll() {
  try {
    const [h, gr] = await Promise.all([
      fetch('/api/health').then(r => r.json()).catch(() => MOCK_HEALTH),
      fetch('/api/graph').then(r => r.json()).catch(() => MOCK_GRAPH),
    ]);
    healthData = h; graphData = gr;
  } catch(e) {
    healthData = MOCK_HEALTH; graphData = MOCK_GRAPH;
  }
  renderHealth();
  renderGraph();
  document.getElementById('loading').style.display = 'none';
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
  const badge = document.getElementById('status-badge');
  const pulse = badge.querySelector('.pulse');
  const statusText = document.getElementById('status-text');
  badge.className = 'badge-' + h.status;
  pulse.className = 'pulse pulse-' + (h.status==='healthy'?'green':h.status==='warning'?'amber':'rose');
  statusText.textContent = h.status.toUpperCase();
  const ca = new Date(h.computed_at);
  document.getElementById('computed-at').textContent = ca.toLocaleTimeString();
  const s = h.signals;
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
  if (graphData?.stats) {
    const st = graphData.stats;
    document.getElementById('m-grains').textContent = st.grains ?? '—';
    document.getElementById('m-entries').textContent = st.entries ?? '—';
    document.getElementById('m-conduits').textContent = st.conduits ?? '—';
    document.getElementById('m-embed').textContent = st.embeddings ?? '—';
  }
  const warns = h.active_warnings || [];
  document.getElementById('warn-count').textContent = warns.length;
  document.getElementById('warn-count').className = 'rpanel-count' + (warns.length ? ' warn' : '');
  const wl = document.getElementById('warnings-list');
  if (!warns.length) {
    wl.innerHTML = '<div style="color:var(--text-dim);font-size:11px">No active warnings</div>';
  } else {
    wl.innerHTML = warns.map(w => '<div class="warning-card"><div class="warning-signal">' + w.signal + '</div><div class="warning-msg">' + w.suggestion + '</div><div class="warning-val">Value: ' + w.current_value + ' · Range: ' + w.healthy_range + ' · Severity: ' + w.severity + '</div></div>').join('');
  }
  const groups = {
    'Retrieval': ['retrieval_success_rate','avg_hops_per_retrieval','fallback_trigger_rate'],
    'Feedback': ['feedback_compliance_rate','promotion_events'],
    'Graph': ['highway_count','highway_growth_rate','orphan_rate','core_grain_count','avg_conduit_weight'],
    'Decay': ['dormant_grain_rate','conduit_dissolution_rate','avg_weight_drop_on_failure','shortcut_creation_rate'],
  };
  let html = '';
  for (const [group, keys] of Object.entries(groups)) {
    html += '<div class="health-group"><div class="health-group-label">' + group + '</div>';
    for (const k of keys) {
      const sig = s[k]; if (!sig) continue;
      const ok = sig.healthy;
      const v = sig.value;
      const display = (v > 0 && v <= 1 && k.includes('rate')) ? (v*100).toFixed(0)+'%' : v.toFixed ? v.toFixed(2) : v;
      html += '<div class="health-row"><div class="health-dot" style="background:' + (ok?'var(--green)':'var(--rose)') + '"></div><div class="health-row-key">' + k.replace(/_/g,' ') + '</div><div class="health-row-val" style="color:' + (ok?'var(--text)':'var(--rose)') + '">' + display + '</div></div>';
    }
    html += '</div>';
  }
  document.getElementById('health-table').innerHTML = html;
}

// ── COLORS / SIZES ────────────────────────────────────────────────────────────
function nodeColor(n) {
  if (n.status === 'dormant') return '#1e2d40';
  if (n.node_type === 'entry') return '#7c3aed';
  if (n.decay_class === 'core') return '#e0f2fe';
  if (n.decay_class === 'ephemeral') return '#0e7490';
  return '#22d3ee';
}
function nodeFillColor(n) {
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

// ── FILTER HELPERS ────────────────────────────────────────────────────────────
function getVisibleNodes() {
  if (!graphData) return [];
  return graphData.nodes.filter(n => {
    const typeOk = (n.node_type === 'grain' && activeFilters.grain) || (n.node_type === 'entry' && activeFilters.entry);
    const statusOk = (n.status === 'active' || !n.status) ? activeFilters.active : (n.status === 'dormant' ? activeFilters.dormant : true);
    const searchOk = !searchQuery || n.label.toLowerCase().includes(searchQuery) || n.id.includes(searchQuery);
    return typeOk && statusOk && searchOk;
  });
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

// ── GRAPH RENDER ──────────────────────────────────────────────────────────────
function renderGraph() {
  const container = document.getElementById('graph-container');
  canvas = document.getElementById('graph-canvas');
  W = container.clientWidth;
  H = container.clientHeight;
  dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  if (!graphData || !graphData.nodes || !graphData.nodes.length) {
    document.getElementById('no-data-msg').style.display = 'flex';
    return;
  }
  document.getElementById('no-data-msg').style.display = 'none';

  const visNodes = getVisibleNodes();
  const visLinks = getVisibleLinks(visNodes.map(n => n.id));

  document.getElementById('gs-nodes').textContent = visNodes.length;
  document.getElementById('gs-edges').textContent = visLinks.length;

  // Deep copy for D3 simulation
  simNodes = visNodes.map(n => ({...n}));
  const nodeMap = new Map(simNodes.map(n => [n.id, n]));
  simLinks = visLinks.map(l => ({
    ...l,
    source: nodeMap.get(typeof l.source === 'object' ? l.source.id : l.source) || l.source,
    target: nodeMap.get(typeof l.target === 'object' ? l.target.id : l.target) || l.target,
  })).filter(l => l.source && l.target);

  // Reset transform
  transform = {x: 0, y: 0, k: 1};

  // D3 force simulation
  if (simulation) simulation.stop();
  simulation = d3.forceSimulation(simNodes)
    .force('link', d3.forceLink(simLinks).id(d => d.id).distance(d => {
      const w = d.effective_weight ?? 0.3;
      return 60 + (1 - w) * 60;
    }).strength(0.4))
    .force('charge', d3.forceManyBody().strength(-180).distanceMax(300))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collision', d3.forceCollide(d => nodeRadius(d) + 8))
    .alphaDecay(0.015)
    .alphaMin(0.001)
    .alphaTarget(0.004)
    .on('tick', draw);

  // Periodic gentle nudge
  if (window._nudgeTimer) clearInterval(window._nudgeTimer);
  window._nudgeTimer = setInterval(() => {
    if (simulation && !simulationPaused) {
      simNodes.forEach(n => {
        n.vx = (n.vx || 0) + (Math.random() - 0.5) * 0.18;
        n.vy = (n.vy || 0) + (Math.random() - 0.5) * 0.18;
      });
      simulation.alphaTarget(0.004).restart();
    }
  }, 5000);

  // Start render loop
  if (animId) cancelAnimationFrame(animId);
  renderLoop();

  // Click away to deselect
  canvas.onclick = function(e) {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const n = findNodeAt(cx, cy);
    if (n) { selectNode(n); }
    else {
      const l = findEdgeAt(cx, cy);
      if (l) { selectEdge(l); }
      else { deselectAll(); }
    }
  };

  canvas.onmousemove = function(e) {
    if (isDragging) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const n = findNodeAt(cx, cy);
    const l = !n ? findEdgeAt(cx, cy) : null;
    if (n) {
      canvas.style.cursor = 'pointer';
      showNodeTooltip(e, n);
    } else if (l) {
      canvas.style.cursor = 'pointer';
      showEdgeTooltip(e, l);
    } else {
      canvas.style.cursor = 'default';
      hideTooltip();
    }
    hoveredNode = n;
    if (!simulationPaused) draw();
  };

  canvas.onmouseleave = function() {
    hideTooltip();
    hoveredNode = null;
    if (!simulationPaused) draw();
  };

  // Pan and zoom via d3-zoom on canvas
  d3.select(canvas).call(
    d3.zoom()
      .scaleExtent([0.15, 4])
      .on('zoom', (event) => {
        transform = {x: event.transform.x, y: event.transform.y, k: event.transform.k};
        draw();
      })
  );

  // Drag via d3-drag on canvas
  d3.select(canvas).call(
    d3.drag()
      .on('start', (event) => {
        const rect = canvas.getBoundingClientRect();
        const cx = event.x - rect.left;
        const cy = event.y - rect.top;
        const n = findNodeAt(cx, cy);
        if (n) {
          isDragging = true;
          dragNode = n;
          n.fx = n.x;
          n.fy = n.y;
          simulation.alphaTarget(0.3).restart();
        }
      })
      .on('drag', (event) => {
        if (dragNode) {
          dragNode.fx = event.x;
          dragNode.fy = event.y;
          dragNode.x = event.x;
          dragNode.y = event.y;
        }
      })
      .on('end', (event) => {
        if (dragNode) {
          simulation.alphaTarget(0.012);
          dragNode.fx = null;
          dragNode.fy = null;
          dragNode = null;
          isDragging = false;
        }
      })
  );
}

// ── DRAW ───────────────────────────────────────────────────────────────────────
function draw() {
  if (!ctx || !canvas) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#080a0e';
  ctx.fillRect(0, 0, W, H);
  if (!simNodes.length) return;

  ctx.save();
  ctx.translate(transform.x, transform.y);
  ctx.scale(transform.k, transform.k);

  const now = Date.now();
  dashOffset -= 0.45;

  // Get neighbor IDs for selection highlighting
  const neighborIds = new Set();
  if (selectedNode) {
    neighborIds.add(selectedNode.id);
    for (const l of simLinks) {
      const sid = typeof l.source === 'object' ? l.source.id : l.source;
      const tid = typeof l.target === 'object' ? l.target.id : l.target;
      if (sid === selectedNode.id || tid === selectedNode.id) {
        neighborIds.add(sid);
        neighborIds.add(tid);
      }
    }
  }

  // Draw links
  for (const l of simLinks) {
    const sx = l.source.x, sy = l.source.y;
    const tx = l.target.x, ty = l.target.y;
    if (sx == null || sy == null || tx == null || ty == null) continue;

    const w = Math.max(0.5, (l.effective_weight ?? l.weight ?? 0.3) * 2.5);
    let alpha = 0.7;
    if (selectedNode) {
      const sid = typeof l.source === 'object' ? l.source.id : l.source;
      const tid = typeof l.target === 'object' ? l.target.id : l.target;
      alpha = (sid === selectedNode.id || tid === selectedNode.id) ? 1 : 0.05;
    } else if (searchQuery) {
      alpha = 0.1;
    }

    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(tx, ty);
    ctx.strokeStyle = edgeColor(l);
    ctx.globalAlpha = alpha;
    ctx.lineWidth = w;
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Glow for core/nearby links when selected
    if (selectedNode && alpha === 1 && (l.decay_class === 'core' || l.direction === 'bidirectional')) {
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(tx, ty);
      ctx.strokeStyle = l.direction === 'bidirectional' ? '#a78bfa' : '#38bdf8';
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = w + 2;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  // Draw nodes
  for (const n of simNodes) {
    if (n.x == null || n.y == null) continue;
    const x = n.x, y = n.y;
    const r = nodeRadius(n);
    let alpha = 1;
    if (selectedNode && !neighborIds.has(n.id)) alpha = 0.15;
    if (searchQuery && !n.label.toLowerCase().includes(searchQuery) && !n.id.includes(searchQuery)) alpha = 0.1;

    const fill = nodeFillColor(n);
    const isCore = n.decay_class === 'core' && n.node_type === 'grain';
    const isDormant = n.status === 'dormant';

    // Glow for core grains
    if (isCore) {
      const pulse = 0.5 + 0.3 * Math.sin(now / 600 + n.id.length);
      const glowR = Math.max(1, r + 7);
      ctx.beginPath();
      ctx.arc(x, y, glowR, 0, Math.PI * 2);
      ctx.fillStyle = '#22d3ee';
      ctx.globalAlpha = pulse * 0.15 * alpha;
      ctx.fill();
      ctx.globalAlpha = 1;

      // Glow stroke
      ctx.beginPath();
      ctx.arc(x, y, glowR, 0, Math.PI * 2);
      ctx.strokeStyle = '#22d3ee';
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.55 * pulse * alpha;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Breathing animation for working/core nodes
    let drawR = r;
    if (!isDormant && n.node_type !== 'entry') {
      const period = isCore ? 2800 : 3500;
      const amp = isCore ? 1.5 : 1;
      drawR = r + amp * Math.sin(now / period + n.id.length * 0.37);
    }

    // Main circle
    ctx.beginPath();
    ctx.arc(x, y, Math.max(1, drawR), 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.globalAlpha = (isDormant ? 0.4 : 0.9) * alpha;
    ctx.fill();
    ctx.globalAlpha = 1;

    // Stroke for core nodes
    if (isCore) {
      ctx.beginPath();
      ctx.arc(x, y, Math.max(1, drawR), 0, Math.PI * 2);
      ctx.strokeStyle = '#7dd3fc';
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = alpha;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Selected highlight
    if (selectedNode && n.id === selectedNode.id) {
      ctx.beginPath();
      ctx.arc(x, y, Math.max(1, drawR + 3), 0, Math.PI * 2);
      ctx.strokeStyle = '#22d3ee';
      ctx.lineWidth = 2.5;
      ctx.globalAlpha = alpha;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Label
    if (alpha > 0.3) {
      const label = n.node_type === 'entry' ? n.label : n.label.slice(0, 22) + (n.label.length > 22 ? '...' : '');
      ctx.fillStyle = '#8899b0';
      ctx.globalAlpha = alpha * 0.9;
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x + r + 4, y);
      ctx.globalAlpha = 1;
    }
  }

  ctx.restore();
}

function renderLoop() {
  draw();
  animId = requestAnimationFrame(renderLoop);
}

// ── HIT DETECTION ─────────────────────────────────────────────────────────────
function screenToWorld(cx, cy) {
  return [(cx - transform.x) / transform.k, (cy - transform.y) / transform.k];
}

function findNodeAt(cx, cy) {
  const [wx, wy] = screenToWorld(cx, cy);
  let closest = null, minDist = 20 / transform.k;
  for (const n of simNodes) {
    if (n.x == null) continue;
    const dx = wx - n.x, dy = wy - n.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const r = nodeRadius(n);
    if (dist < r + minDist) { minDist = dist; closest = n; }
  }
  return closest;
}

function findEdgeAt(cx, cy) {
  const [wx, wy] = screenToWorld(cx, cy);
  let closest = null, minDist = 8 / transform.k;
  for (const l of simLinks) {
    if (l.source.x == null || l.target.x == null) continue;
    const dist = pointToLineDistance(wx, wy, l.source.x, l.source.y, l.target.x, l.target.y);
    if (dist < minDist) { minDist = dist; closest = l; }
  }
  return closest;
}

function pointToLineDistance(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
  let t = ((px - x1) * dx + (py - y1) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  const projX = x1 + t * dx, projY = y1 + t * dy;
  return Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);
}

// ── TOOLTIP ───────────────────────────────────────────────────────────────────
const tt = document.getElementById('tooltip');
function showTooltip(x, y, title, rows) {
  document.getElementById('tt-title').textContent = title;
  document.getElementById('tt-body').innerHTML = rows.map(([k,v]) =>
    '<div class="tt-row"><span class="tt-key">' + k + '</span><span class="tt-val">' + v + '</span></div>'
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
  showTooltip(event.clientX, event.clientY, d.label.slice(0,60), [
    ['type', d.node_type],
    ...(d.decay_class ? [['decay', d.decay_class]] : []),
    ...(d.status ? [['status', d.status]] : []),
    ...(d.provenance ? [['provenance', d.provenance]] : []),
    ...(d.context_spread !== undefined ? [['ctx spread', d.context_spread]] : []),
  ]);
}
function showEdgeTooltip(event, d) {
  const sid = typeof d.source === 'object' ? d.source.id : d.source;
  const tid = typeof d.target === 'object' ? d.target.id : d.target;
  showTooltip(event.clientX, event.clientY, sid.slice(0,10) + '...' + ' to ' + tid.slice(0,10) + '...', [
    ['weight', (d.weight ?? 0).toFixed(4)],
    ['eff. weight', (d.effective_weight ?? 0).toFixed(4)],
    ['direction', d.direction],
    ['decay', d.decay_class ?? '-'],
    ['use count', d.use_count ?? 0],
    ['edge type', d.edge_type ?? '-'],
  ]);
}

// ── SELECTION / INSPECTOR ─────────────────────────────────────────────────────
function deselectAll() {
  selectedNode = null; selectedEdge = null;
  document.getElementById('inspector-empty').style.display = '';
  document.getElementById('inspector-content').style.display = 'none';
  if (simulationPaused) draw();
}

function selectNode(d) {
  selectedNode = d; selectedEdge = null;
  renderInspector(d, 'node');
  draw();
}

function selectEdge(d) {
  selectedEdge = d; selectedNode = null;
  renderInspector(d, 'edge');
  draw();
}

function tagHtml(v, positive) {
  const cls = v==='active'||v==='core'?'cyan':v==='entry'?'violet':v==='dormant'||positive===false?'rose':v==='working'?'cyan':v==='ephemeral'?'dim':'dim';
  return '<span class="tag tag-' + cls + '">' + v + '</span>';
}

function renderInspector(d, type) {
  document.getElementById('inspector-empty').style.display = 'none';
  const el = document.getElementById('inspector-content');
  el.style.display = '';
  if (type === 'node') {
    const color = nodeColor(d);
    const isEntry = d.node_type === 'entry';
    el.innerHTML = '<div class="inspector-node-header"><div class="inspector-icon" style="background:' + color + '22;border:1px solid ' + color + '44"><svg width="14" height="14" viewBox="0 0 24 24" fill="' + color + '"><circle cx="12" cy="12" r="' + (isEntry?6:8) + '"/></svg></div><div><div class="inspector-label">' + d.label + '</div><div class="inspector-type">' + d.node_type + (d.decay_class ? ' . ' + d.decay_class : '') + '</div></div></div><div class="inspector-rows"><div class="inspector-row"><span class="ir-key">ID</span><span class="ir-val" style="font-size:10px">' + d.id + '</span></div><div class="inspector-row"><span class="ir-key">Type</span><span class="ir-val">' + tagHtml(d.node_type) + '</span></div>' + (d.status ? '<div class="inspector-row"><span class="ir-key">Status</span><span class="ir-val">' + tagHtml(d.status) + '</span></div>' : '') + (d.decay_class ? '<div class="inspector-row"><span class="ir-key">Decay class</span><span class="ir-val">' + tagHtml(d.decay_class) + '</span></div>' : '') + (d.provenance ? '<div class="inspector-row"><span class="ir-key">Provenance</span><span class="ir-val"><span class="tag tag-dim">' + d.provenance + '</span></span></div>' : '') + (d.context_spread !== undefined ? '<div class="inspector-row"><span class="ir-key">Context spread</span><span class="ir-val">' + d.context_spread + '</span></div>' : '') + (d.feature ? '<div class="inspector-row"><span class="ir-key">Feature</span><span class="ir-val">' + d.feature + '</span></div>' : '') + '</div>';
  } else {
    const sid = typeof d.source === 'object' ? d.source.id : d.source;
    const tid = typeof d.target === 'object' ? d.target.id : d.target;
    const sl = typeof d.source === 'object' ? d.source.label : sid;
    const tl = typeof d.target === 'object' ? d.target.label : tid;
    el.innerHTML = '<div style="margin-bottom:10px"><div class="inspector-label" style="font-size:11px">' + (sl||sid).slice(0,40) + '...</div><div style="color:var(--text-dim);font-size:10px;margin:3px 0">down</div><div class="inspector-label" style="font-size:11px">' + (tl||tid).slice(0,40) + '...</div></div><div class="inspector-rows"><div class="inspector-row"><span class="ir-key">Weight</span><span class="ir-val">' + (d.weight??0).toFixed(4) + '</span></div><div class="inspector-row"><span class="ir-key">Eff. weight</span><span class="ir-val">' + (d.effective_weight??0).toFixed(4) + '</span></div><div class="inspector-row"><span class="ir-key">Direction</span><span class="ir-val">' + tagHtml(d.direction) + '</span></div><div class="inspector-row"><span class="ir-key">Decay class</span><span class="ir-val">' + tagHtml(d.decay_class??'working') + '</span></div><div class="inspector-row"><span class="ir-key">Use count</span><span class="ir-val">' + (d.use_count??0) + '</span></div><div class="inspector-row"><span class="ir-key">Edge type</span><span class="ir-val"><span class="tag tag-dim">' + (d.edge_type??'earned') + '</span></span></div></div>';
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
  if (simulationPaused) draw();
}
function onWeightChange(val) {
  weightThreshold = parseFloat(val);
  document.getElementById('weight-val').textContent = parseFloat(val).toFixed(2);
  renderGraph();
}

// ── SIMULATION CONTROLS ───────────────────────────────────────────────────────
function toggleSimulation() {
  simulationPaused = !simulationPaused;
  const icon = document.getElementById('pause-icon');
  if (simulationPaused) {
    simulation.alphaTarget(0).stop();
    icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
  } else {
    simulation.alphaTarget(0.012).restart();
    icon.innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
  }
}

function resetView() {
  transform = {x: 0, y: 0, k: 1};
  if (simulation) draw();
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