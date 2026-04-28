"""Tests for Flux Memory REST API (§1A.4)."""
from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

from flux.config import Config
from flux.rest_api import build_app
from flux.service import FluxService
from flux.storage import FluxStore

from mocks import MockEmbeddingBackend, MockLLMBackend


@pytest.fixture
def cfg():
    return dataclasses.replace(
        Config(),
        OPERATING_MODE="caller_extracts",
        MAX_GRAINS_PER_CALL=5,
        MAX_WRITE_QUEUE_DEPTH=100,
        MAX_GRAINS_PER_MINUTE=200,
        READ_WORKERS=2,
    )


@pytest.fixture
def svc(tmp_path, cfg):
    store = FluxStore(tmp_path / "flux.db")
    s = FluxService(store, MockLLMBackend(), MockEmbeddingBackend(), cfg)
    s.start()
    yield s
    s.stop()
    store.close()


@pytest.fixture
def client(svc, cfg):
    app = build_app(svc, cfg)
    return TestClient(app)


# ---------------------------------------------------------------- /health

class TestRootEndpoint:
    def test_root_returns_service_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Flux Memory"
        assert body["health"] == "/health"


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status(self, client):
        body = client.get("/health").json()
        assert "status" in body


# ---------------------------------------------------------------- /store

class TestStoreEndpoint:
    def test_store_success(self, client):
        resp = client.post("/store", json={"content": "Rome is in Italy"})
        assert resp.status_code == 200
        body = resp.json()
        assert "grain_id" in body
        assert body["status"] == "stored"

    def test_store_with_provenance(self, client):
        resp = client.post("/store", json={
            "content": "user said hello",
            "provenance": "user_stated",
        })
        assert resp.status_code == 200

    def test_store_with_caller_id_header(self, client):
        resp = client.post(
            "/store",
            json={"content": "caller tagged grain"},
            headers={"X-Caller-Id": "agent-007"},
        )
        assert resp.status_code == 200

    def test_store_missing_content_422(self, client):
        resp = client.post("/store", json={})
        assert resp.status_code == 422

    def test_store_empty_content_422(self, client):
        resp = client.post("/store", json={"content": "   "})
        assert resp.status_code == 422


# ---------------------------------------------------------------- /store/batch

class TestStoreBatchEndpoint:
    def test_batch_store_success(self, client):
        items = [{"content": f"batch fact {i}"} for i in range(3)]
        resp = client.post("/store/batch", json={"items": items})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert len(body["grain_ids"]) == 3

    def test_batch_exceeds_cap_422(self, client, cfg):
        items = [{"content": f"f{i}"} for i in range(cfg.MAX_GRAINS_PER_CALL + 1)]
        resp = client.post("/store/batch", json={"items": items})
        assert resp.status_code == 422

    def test_batch_at_cap_succeeds(self, client, cfg):
        items = [{"content": f"f{i}"} for i in range(cfg.MAX_GRAINS_PER_CALL)]
        resp = client.post("/store/batch", json={"items": items})
        assert resp.status_code == 200


# ---------------------------------------------------------------- /retrieve

class TestRetrieveEndpoint:
    def test_retrieve_empty_store(self, client):
        resp = client.post("/retrieve", json={"query": "anything"})
        assert resp.status_code == 200
        body = resp.json()
        assert "grains" in body
        assert "trace_id" in body
        assert "confidence" in body

    def test_retrieve_after_store(self, client):
        client.post("/store", json={"content": "Python is a programming language"})
        resp = client.post("/retrieve", json={"query": "Python programming"})
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["grains"], list)

    def test_retrieve_missing_query_422(self, client):
        resp = client.post("/retrieve", json={})
        assert resp.status_code == 422

    def test_retrieve_with_caller_id(self, client):
        resp = client.post(
            "/retrieve",
            json={"query": "test"},
            headers={"X-Caller-Id": "claude"},
        )
        assert resp.status_code == 200

    def test_retrieve_with_caller_client_and_role_headers(self, client):
        client.post("/store", json={"content": "portable REST caller"})
        resp = client.post(
            "/retrieve",
            json={"query": "portable REST caller"},
            headers={
                "X-Flux-Client": "local-agent-1",
                "X-Flux-Role": "background_lookup",
            },
        )
        assert resp.status_code == 200

        health = client.get("/health").json()
        callers = {c["caller_id"] for c in health["caller_feedback"]}
        assert "local-agent-1:background_lookup" in callers


# ---------------------------------------------------------------- /feedback

class TestFeedbackEndpoint:
    def test_feedback_success(self, client):
        client.post("/store", json={"content": "testable fact"})
        ret = client.post("/retrieve", json={"query": "testable"}).json()
        if ret["grains"]:
            resp = client.post("/feedback", json={
                "trace_id": ret["trace_id"],
                "grain_id": ret["grains"][0]["id"],
                "useful": True,
            })
            assert resp.status_code == 200
            assert resp.json()["status"] == "queued"

    def test_feedback_missing_fields_422(self, client):
        resp = client.post("/feedback", json={"trace_id": "x"})
        assert resp.status_code == 422


# ---------------------------------------------------------------- /grains

class TestGrainsEndpoint:
    def test_list_grains_empty(self, client):
        resp = client.get("/grains")
        assert resp.status_code == 200
        body = resp.json()
        assert "grains" in body

    def test_list_grains_after_store(self, client):
        client.post("/store", json={"content": "visible grain"})
        resp = client.get("/grains")
        assert resp.status_code == 200
        assert len(resp.json()["grains"]) >= 1

    def test_list_grains_status_filter(self, client):
        client.post("/store", json={"content": "active grain"})
        resp = client.get("/grains?status=active")
        assert resp.status_code == 200
        grains = resp.json()["grains"]
        assert all(g["status"] == "active" for g in grains)

    def test_list_grains_invalid_status_422(self, client):
        resp = client.get("/grains?status=bogus")
        assert resp.status_code == 422

    def test_list_grains_limit(self, client):
        for i in range(5):
            client.post("/store", json={"content": f"limited grain {i}"})
        resp = client.get("/grains?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()["grains"]) <= 2


# ---------------------------------------------------------------- caller ID propagation

class TestCallerIdPropagation:
    def test_default_caller_id_anonymous(self, client):
        resp = client.post("/store", json={"content": "anonymous fact"})
        assert resp.status_code == 200

    def test_multiple_callers_independent_rate_limits(self, client, cfg):
        for i in range(cfg.MAX_GRAINS_PER_MINUTE // 2):
            client.post(
                "/store",
                json={"content": f"alice {i}"},
                headers={"X-Caller-Id": "alice"},
            )
        resp = client.post(
            "/store",
            json={"content": "bob independent"},
            headers={"X-Caller-Id": "bob"},
        )
        assert resp.status_code == 200
