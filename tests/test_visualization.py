"""Tests for Track 4 Step 3 — Visualization export."""
from __future__ import annotations

import json
import pytest

from flux import FluxStore, Grain, Conduit, Entry
from flux.embedding import MockEmbeddingBackend, store_embedding
from flux.visualization import export_graphml, export_json, export_dot, subgraph, cluster_view


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    with FluxStore(db) as s:
        yield s


@pytest.fixture
def populated_store(store):
    g1 = Grain(content="Python is great", provenance="user_stated")
    g2 = Grain(content="ML frameworks", provenance="ai_stated", decay_class="core")
    store.insert_grain(g1)
    store.insert_grain(g2)
    c = Conduit(from_id=g1.id, to_id=g2.id, weight=0.6)
    store.insert_conduit(c)
    e = Entry(feature="python")
    store.insert_entry(e)
    return store, g1, g2, c, e


class TestExportJson:
    def test_returns_dict_with_nodes_and_links(self, populated_store):
        store, *_ = populated_store
        data = export_json(store)
        assert "nodes" in data
        assert "links" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["links"], list)

    def test_node_count_matches(self, populated_store):
        store, g1, g2, c, e = populated_store
        data = export_json(store)
        node_ids = {n["id"] for n in data["nodes"]}
        assert g1.id in node_ids
        assert g2.id in node_ids
        assert e.id in node_ids

    def test_link_present(self, populated_store):
        store, g1, g2, c, _ = populated_store
        data = export_json(store)
        sources = {l["source"] for l in data["links"]}
        assert g1.id in sources

    def test_effective_weight_present(self, populated_store):
        store, *_ = populated_store
        data = export_json(store)
        for link in data["links"]:
            assert "effective_weight" in link

    def test_empty_store_returns_empty_lists(self, store):
        data = export_json(store)
        assert data["nodes"] == []
        assert data["links"] == []


class TestExportGraphml:
    def test_returns_string(self, populated_store):
        store, *_ = populated_store
        result = export_graphml(store)
        assert isinstance(result, str)
        assert "graphml" in result.lower()

    def test_contains_node_ids(self, populated_store):
        store, g1, g2, *_ = populated_store
        result = export_graphml(store)
        assert g1.id in result
        assert g2.id in result

    def test_valid_xml(self, populated_store):
        import xml.etree.ElementTree as ET
        store, *_ = populated_store
        result = export_graphml(store)
        ET.fromstring(result)  # raises if invalid


class TestExportDot:
    def test_returns_string(self, populated_store):
        store, *_ = populated_store
        result = export_dot(store)
        assert isinstance(result, str)
        assert "digraph" in result

    def test_contains_edge_arrow(self, populated_store):
        store, *_ = populated_store
        result = export_dot(store)
        assert "->" in result


class TestSubgraph:
    def test_returns_reachable_nodes(self, populated_store):
        store, g1, g2, c, e = populated_store
        data = subgraph(store, [e.feature])
        node_ids = {n["id"] for n in data["nodes"]}
        assert e.id in node_ids

    def test_empty_features_returns_empty(self, populated_store):
        store, *_ = populated_store
        data = subgraph(store, [])
        assert data["nodes"] == []


class TestClusterView:
    def test_returns_clusters_key(self, store):
        data = cluster_view(store)
        assert "clusters" in data
        assert isinstance(data["clusters"], list)
