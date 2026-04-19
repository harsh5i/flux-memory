"""Schema-level checks: tables, indexes, WAL mode, pragmas."""
from __future__ import annotations


def test_schema_creates_all_tables(store):
    rows = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {r["name"] for r in rows}
    expected = {
        "grains",
        "conduits",
        "entries",
        "entry_cluster_membership",
        "entry_cooccurrence",
        "clusters",
        "grain_cluster_touch",
        "traces",
        "co_retrieval_counts",
    }
    assert expected.issubset(names), f"Missing tables: {expected - names}"


def test_schema_creates_expected_indexes(store):
    rows = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    names = {r["name"] for r in rows}
    for required in (
        "idx_conduits_from",
        "idx_conduits_to",
        "idx_grains_status",
        "idx_traces_created",
        "idx_conduits_last_used",
    ):
        assert required in names, f"missing index {required}"


def test_wal_mode_enabled(store):
    mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_schema_is_idempotent(tmp_path):
    """Re-opening the same DB file must not fail on CREATE TABLE IF NOT EXISTS."""
    from flux.storage import FluxStore

    path = tmp_path / "flux.db"
    FluxStore(path).close()
    FluxStore(path).close()  # would raise if DDL is not idempotent
