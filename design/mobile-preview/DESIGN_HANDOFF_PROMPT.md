# Flux Mobile Dashboard Handoff Prompt

Use this prompt when asking another AI, designer, or frontend engineer to recreate the same mobile dashboard layout.

## Objective

Recreate the Flux Memory mobile dashboard experience exactly in spirit: a professional, modern, dense, operational graph dashboard optimized for phone viewing. This is not a marketing page and not a generic card dashboard. The graph is the primary surface.

Use these files as source references:

- `Mobile Preview.html`
- `Mobile Preview - live.html`
- `Flux Dashboard.html`

## Required Mobile Layout

The mobile dashboard must behave like this:

- First screen is the graph, not metrics cards.
- The graph fills the available mobile viewport.
- Controls are compact and overlaid or docked.
- Secondary panels are hidden until opened.
- Inspector, warnings, health, and refresh are accessed from a bottom bubble bar.
- Tapping a node highlights it and updates inspector state.
- Tapping the inspector bubble opens a bottom sheet.
- The bottom sheet must slide up over the graph and be dismissible.
- Search must dim non-matching nodes and highlight matching nodes; it must not hide the rest of the graph.
- Search must not permanently zoom the graph; if zoom is used, it must reset cleanly.
- Node deselection must be available by tapping empty graph space or using a clear selected state.

## Device Preview Frames

The preview page should include these device frames:

1. iPhone Air
   - Viewport: `390 x 844`
   - Frame width: `280px`
   - Screen height: `580px`
   - Iframe scale: `0.642`
   - Dynamic island at top

2. Motorola Razr+ unfolded
   - Viewport: `360 x 780`
   - Frame width: `258px`
   - Screen height: `540px`
   - Iframe scale: `0.653`
   - Cosmetic hinge seam across center

3. Motorola Razr+ folded outer screen
   - Viewport: `390 x 162`
   - Frame width: `200px`
   - Cover screen height: `120px`
   - Iframe scale: `0.462`
   - Ultra-compact glance mode

## Visual Direction

Keep the visual style close to the reference:

- Dark operational interface.
- Background near `#060810` / `#080a0e`.
- Device shell near `#18181a` / `#1c1c1e`.
- Accent cyan near `#22d3ee`.
- Muted slate text for labels.
- Compact typography.
- Minimal decorative elements.
- No large hero sections.
- No oversized cards.
- No bright marketing gradients.
- No thick edge rendering for graph strength.

## Graph Behavior

The graph must feel alive and useful:

- Pan and zoom on touch.
- Drag nodes with one finger.
- Pinch zoom with two fingers.
- Highlight selected node and its neighborhood.
- Dim unrelated nodes instead of removing them.
- Show real-time or recent signal propagation when trace/activity data exists.
- Keep static edges thin; activity can pulse along paths instead of making all edges thick.
- Text labels must be sharp and readable; avoid blurred canvas text where possible.

## Data Contracts

Use same-origin dashboard APIs:

- `GET /api/graph`
- `GET /api/health`
- `GET /api/events`
- `GET /api/trace?trace_id=<id>`

Do not require the REST API port directly from the mobile frontend.

## Implementation Constraints

The current Flux dashboard is plain HTML/CSS/JS embedded in:

- `src/flux/dashboard.py`

There is no React/Vite/Webpack build step in this Flux system. If implementing in the existing repo, keep it compatible with inline HTML/CSS/JS unless the owner explicitly approves a frontend build system.

For another Flux system, either:

- copy the static HTML/CSS/JS structure directly, or
- port the same layout to that system's frontend stack while preserving behavior.

## Acceptance Checklist

The result is acceptable only if:

- iPhone viewport opens directly into a usable graph.
- Bottom bubble bar is visible and usable.
- Inspector opens as a bottom sheet.
- Search highlights matches and dims non-matches.
- Tap empty graph deselects selected node.
- Graph labels are readable on mobile.
- Touch pan, pinch, and node drag work.
- Health/warnings are reachable but do not dominate the first screen.
- The preview page shows all three device frames.
