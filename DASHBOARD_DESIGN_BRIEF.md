# Flux Memory Dashboard Design Brief

This is the context to hand to a UI/design assistant before redesigning the Flux Memory dashboard.

## Goal

Build a professional, modern, interactive operational dashboard for Flux Memory. The current dashboard works but is visually flat and the graph is static. The desired result should feel like a dense observability/control surface, not a marketing page.

Primary improvements:

- Interactive graph: pan, zoom, drag nodes, hover tooltips, click-to-inspect node/edge details, highlight connected neighborhood.
- Useful controls: search, node-type filters, edge-weight threshold slider, health-state filter, reset view, pause/resume layout.
- Clear operational hierarchy: top summary metrics, health/warnings, graph explorer, recent activity/events, selected object inspector.
- Responsive desktop-first layout that still works on mobile.

## Repository Context

Important files:

- `src/flux/dashboard.py`
  - Owns the current dashboard.
  - Serves one inline HTML document from `_DASHBOARD_HTML`.
  - Uses `ThreadingHTTPServer`, not a frontend build system.
  - Current JS fetches same-origin `/api/health` and `/api/graph`.
- `src/flux/visualization.py`
  - Builds graph JSON via `export_json(store)`.
  - Also supports GraphML/DOT export.
- `src/flux/health.py`
  - Computes health signals and active warnings.
- `src/flux/rest_api.py`
  - Separate FastAPI service on port `7465`.
  - Useful for store/retrieve/feedback/grains, but the dashboard should normally use same-origin dashboard endpoints.

Current local instance:

- Dashboard: `http://localhost:7462`
- REST API: `http://localhost:7465`
- Instance DB: `~/.flux/<name>/flux.db`

## Serving Model

The dashboard is intentionally simple:

```python
_DASHBOARD_HTML = r"""<!DOCTYPE html>
...
</html>
"""
```

`run_dashboard(store, host="127.0.0.1", port=7462, cfg=None)` serves:

- `GET /` or `/index.html`: inline HTML
- `GET /api/health`: `flux_health(store, cfg)`
- `GET /api/graph`: `export_json(store)`
- `GET /api/clusters`: `cluster_view(store)`

Implementation constraints:

- Prefer plain HTML/CSS/JS inside `src/flux/dashboard.py`.
- There is no React/Vite/Webpack build step.
- If adding a graph library, either:
  - use vanilla JS/SVG/canvas, or
  - vendor a single JS file into the repo and serve it, or
  - use a CDN only if a no-network fallback is acceptable.
- Keep the dashboard same-origin: fetch `/api/health`, `/api/graph`, `/api/clusters`.
- Avoid depending on the REST service at `7465` for the dashboard's core view.

## Data Contracts

### `GET /api/health`

Shape:

```json
{
  "status": "healthy|warning|critical",
  "signals": {
    "highway_count": {"value": 0.0, "healthy": false},
    "core_grain_count": {"value": 0.0, "healthy": true},
    "orphan_rate": {"value": 0.0, "healthy": true},
    "avg_conduit_weight": {"value": 0.385, "healthy": true},
    "dormant_grain_rate": {"value": 0.0, "healthy": false},
    "highway_growth_rate": {"value": 0.0, "healthy": false},
    "shortcut_creation_rate": {"value": 0.0, "healthy": false},
    "conduit_dissolution_rate": {"value": 0.0, "healthy": false},
    "avg_weight_drop_on_failure": {"value": 0.0, "healthy": false},
    "promotion_events": {"value": 0.0, "healthy": false},
    "avg_hops_per_retrieval": {"value": 0.0, "healthy": true},
    "retrieval_success_rate": {"value": 0.0, "healthy": false},
    "fallback_trigger_rate": {"value": 1.0, "healthy": false},
    "feedback_compliance_rate": {"value": 0.0, "healthy": false}
  },
  "active_warnings": [
    {
      "signal": "feedback_compliance_rate",
      "severity": "WARNING",
      "current_value": 0.0,
      "healthy_range": ">= 0.8",
      "first_seen": "2026-04-23T13:43:13Z",
      "last_seen": "2026-04-23T21:45:47Z",
      "suggestion": "Main AI is not calling flux_feedback reliably. Prompt engineering issue."
    }
  ],
  "computed_at": "2026-04-24T03:15:47.162197+05:30"
}
```

Health signal notes:

- `value` is numeric.
- Rate values are stored as fractions, e.g. `fallback_trigger_rate = 1.0` means `100%`.
- `healthy` is already computed by the backend.
- `status` should drive the global status badge.

### `GET /api/graph`

Shape:

```json
{
  "directed": true,
  "multigraph": false,
  "stats": {
    "grains": 29,
    "active_grains": 29,
    "dormant_grains": 0,
    "entries": 135,
    "conduits": 260,
    "embeddings": 29
  },
  "nodes": [
    {
      "id": "04e6faffdeda4921a46d775bedbf4570",
      "label": "Codex should use Flux Memory MCP server...",
      "node_type": "grain",
      "decay_class": "working",
      "status": "active",
      "provenance": "ai_stated",
      "context_spread": 0
    },
    {
      "id": "bdbd071d8455432fbb1c774deaf5d3ba",
      "label": "user",
      "node_type": "entry",
      "feature": "user"
    }
  ],
  "links": [
    {
      "source": "entry_or_grain_id",
      "target": "grain_id",
      "weight": 0.5,
      "effective_weight": 0.4998,
      "direction": "forward|bidirectional",
      "decay_class": "working|core|ephemeral",
      "use_count": 0,
      "edge_type": "earned|shortcut|bootstrap"
    }
  ]
}
```

Graph semantics:

- `node_type="entry"` means a feature anchor or keyword.
- `node_type="grain"` means a memory fact.
- Entry nodes usually connect to grain nodes.
- Grain-to-grain links can exist, especially shortcuts.
- `effective_weight` is the best visual edge strength because it includes lazy decay.
- `weight` is the stored DB weight.
- `direction="bidirectional"` should be visually distinct from normal forward edges.
- `decay_class="core"` should visually stand out from `working`.
- `status="dormant"` should be muted.

### `GET /api/clusters`

Shape:

```json
{
  "clusters": [
    {"cluster_id": "cluster-id", "members": ["entry-id-1", "entry-id-2"]}
  ]
}
```

This is optional for the first redesign but can drive cluster coloring or a clusters panel.

## Current Dashboard Limitations

Current graph renderer:

- Static SVG.
- No force simulation.
- No pan/zoom.
- No node dragging.
- No selected-node inspector.
- No graph controls.
- Layout is a ring/radial approximation, so dense graphs become visually noisy.

Current UI:

- Dark but flat.
- Metric cards are useful but not visually refined.
- Warnings and health table lack context and trend/history.
- It does not clearly distinguish operational health from graph content health.

## Recommended Redesign

Use a two-column operations layout on desktop:

- Top band:
  - Instance name and status.
  - Last computed time.
  - Refresh/auto-refresh controls.
  - Core metrics: grains, entries, conduits, embeddings, retrieval success, fallback rate, feedback compliance.
- Main left:
  - Interactive graph explorer.
  - Graph toolbar: search, type filters, weight threshold, reset view, pause layout.
- Main right:
  - Selected node/edge inspector.
  - Active warnings.
  - Health signal table grouped by graph, retrieval, feedback, decay.
- Bottom or secondary tab:
  - Recent events/activity if backend endpoint is added later.
  - Cluster summary if `/api/clusters` has data.

Graph interaction expectations:

- Hover node: show label, type, status, provenance, degree.
- Click node: lock selection and show details in inspector.
- Hover/click edge: show source, target, stored weight, effective weight, use count, decay class.
- Search should zoom/select matching nodes.
- Weight slider should hide weak edges and isolated nodes optionally.
- Neighborhood mode should dim unrelated nodes/edges.
- Auto-refresh should preserve zoom, selected node, and filter state.

Visual direction:

- Quiet professional dark UI, high contrast text.
- Avoid a one-hue palette. Use restrained neutrals plus cyan/green/amber/rose for state.
- Use compact typography. This is an operational tool, not a landing page.
- Cards should be individual panels only; avoid nested cards.
- Keep graph full-width within its panel, not a small decorative preview.

## Backend Changes Recently Made

The health instrumentation gap has been remediated in the current branch:

- `reinforcement.py` now emits:
  - `feedback/conduit_reinforced`
  - `feedback/highway_formed`
  - `feedback/shortcut_created`
  - `feedback/conduit_penalized`
- `promotion.py` now emits:
  - `feedback/promotion_triggered`
- `decay.py` now emits:
  - `decay/cleanup_pass_completed`
  - `decay/expiry_pass_completed`
- `retrieval.py` now calls promotion checks after successful feedback.

So the dashboard can trust these health signals to become nonzero once the system receives retrievals, feedback, decay cleanup, and promotion activity.

## Testing Expectations

After modifying `src/flux/dashboard.py`:

1. Run Python tests:

   ```powershell
   python -m pytest tests/test_dashboard.py tests/test_visualization.py tests/test_health.py -q
   ```

2. Start local service:

   ```powershell
   flux start --name my-memory
   ```

3. Open:

   ```text
   http://localhost:7462
   ```

4. Verify:

- No JS console errors.
- `/api/health` and `/api/graph` fetch successfully.
- Graph renders with nonzero links.
- Pan/zoom/hover/click interactions work.
- Text does not overlap in 1366x768 desktop or narrow mobile widths.
- Auto-refresh does not reset selection/filter state unexpectedly.

