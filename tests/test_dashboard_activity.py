"""Dashboard activity animation coverage."""
from __future__ import annotations

import re

from flux import dashboard


def test_dashboard_animates_conduit_specific_events() -> None:
    html = dashboard._DASHBOARD_HTML

    assert "function isConduitEvent(ev)" in html
    assert "function activateConduitEvent(ev, color, now)" in html
    assert "function findConduitEdge(data)" in html
    assert "data.conduit_id" in html
    assert "pulseNodeById(data.from_id" in html
    assert "pulseNodeById(data.to_id" in html
    assert "activateEdge(edge, color, now)" in html

    for event_name in (
        "conduit_reinforced",
        "conduit_penalized",
        "highway_formed",
        "shortcut_created",
    ):
        assert event_name in html


def test_dashboard_refreshes_after_structural_graph_changes_only() -> None:
    html = dashboard._DASHBOARD_HTML

    assert "function isGraphRefreshEvent(ev)" in html
    refresh_match = re.search(r"function isGraphRefreshEvent\(ev\) \{(?P<body>.*?)\n\}", html, re.S)
    assert refresh_match is not None
    refresh_body = refresh_match.group("body")

    assert "ev.category === 'write'" not in refresh_body
    assert "isConduitEvent(ev)" not in refresh_body
    assert "ev.event === 'grain_stored'" in refresh_body
    assert "ev.event === 'entry_point_created'" in refresh_body
    assert "ev.event === 'bootstrap_conduits_created'" in refresh_body
    assert "ev.event === 'graph_rebuild_completed'" in refresh_body
    assert "function scheduleGraphRefreshAfterEvent()" in html
    assert "}, 1200);" in html
