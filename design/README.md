# Flux Mobile Dashboard Design Pack

This folder contains the mobile dashboard design files that can be reused on another Flux system.

## Files

- `Mobile Preview.html`
  - Static mobile preview.
  - Depends on `Flux Dashboard.html` in the same folder.
  - Good for design review or handing to another UI assistant.

- `Flux Dashboard.html`
  - Standalone dashboard mock/design used by the preview iframe.
  - Includes the visual language, graph panel, inspector, health panels, and fallback mock data.

- `Mobile Preview - live.html`
  - Same device-frame preview, but the iframe source is `/`.
  - Use this when embedding into a running Flux dashboard server route such as `/mobile-preview`.

## Mobile Layout Targets

- iPhone Air: `390 x 844`
- Motorola Razr+ unfolded: `360 x 780`
- Motorola Razr+ folded outer screen: `390 x 162`

## Key Interaction Pattern

- Full-screen graph first.
- Bottom bubble bar for inspector, warnings, health, and refresh.
- Tap node to select/highlight.
- Tap inspector bubble to open the mobile sheet.
- Two-finger pan/pinch for graph navigation.

## Current Flux Integration Location

The live implementation is embedded in:

- `src/flux/dashboard.py`

Relevant sections:

- `_DASHBOARD_HTML`: main dashboard HTML/CSS/JS.
- `_MOBILE_PREVIEW_HTML`: mobile preview route.
- `/api/graph`: graph data.
- `/api/health`: health data.
- `/api/events`: recent activity.
- `/api/trace`: trace animation data.
