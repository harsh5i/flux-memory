"""Dashboard activity animation coverage."""
from __future__ import annotations

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


def test_dashboard_refreshes_after_conduit_graph_changes() -> None:
    html = dashboard._DASHBOARD_HTML

    assert "function isGraphRefreshEvent(ev)" in html
    assert "function scheduleGraphRefreshAfterEvent(ev, color)" in html
    assert "isConduitEvent(ev)" in html
