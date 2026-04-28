"""Tests for MCP server v0.6 features: onboarding, list_grains, caller_id (§1A.4–1A.6)."""
from __future__ import annotations

import dataclasses

import pytest

from flux.config import Config
from flux.mcp_server import _dispatch
from flux.service import FluxService
from flux.storage import FluxStore

from mocks import MockEmbeddingBackend, MockLLMBackend


@pytest.fixture
def cfg():
    return dataclasses.replace(
        Config(),
        OPERATING_MODE="caller_extracts",
        MCP_SERVER_NAME="test-flux",
        READ_WORKERS=2,
        MAX_GRAINS_PER_CALL=10,
        MAX_WRITE_QUEUE_DEPTH=100,
        MAX_GRAINS_PER_MINUTE=200,
    )


@pytest.fixture
def store(tmp_path):
    s = FluxStore(tmp_path / "flux.db")
    yield s
    s.close()


@pytest.fixture
def svc(store, cfg):
    s = FluxService(store, MockLLMBackend(), MockEmbeddingBackend(), cfg)
    s.start()
    yield s
    s.stop()


def dispatch(name, args, store, cfg, svc=None):
    return _dispatch(name, args, store, MockLLMBackend(), MockEmbeddingBackend(), cfg, service=svc)


# ---------------------------------------------------------------- flux_onboard

class TestFluxOnboard:
    def test_onboard_returns_instructions(self, store, cfg):
        result = dispatch("flux_onboard", {}, store, cfg)
        assert "instructions" in result
        assert "operating_mode" in result
        assert "server_name" in result

    def test_onboard_server_name_matches_config(self, store, cfg):
        result = dispatch("flux_onboard", {}, store, cfg)
        assert result["server_name"] == "test-flux"

    def test_onboard_mode_in_instructions(self, store, cfg):
        result = dispatch("flux_onboard", {}, store, cfg)
        assert cfg.OPERATING_MODE in result["instructions"]

    def test_onboard_caller_id_accepted(self, store, cfg):
        result = dispatch("flux_onboard", {"caller_id": "agent-x"}, store, cfg)
        assert "instructions" in result

    def test_onboard_instructions_contain_workflow(self, store, cfg):
        result = dispatch("flux_onboard", {}, store, cfg)
        instructions = result["instructions"]
        assert "flux_retrieve" in instructions
        assert "flux_store" in instructions
        assert "flux_feedback" in instructions

    def test_onboard_instructions_define_generic_caller_identity(self, store, cfg):
        result = dispatch("flux_onboard", {}, store, cfg)
        instructions = result["instructions"]

        assert "client" in instructions
        assert "role" in instructions
        assert "client:role" in instructions
        assert "memory_writer" in instructions
        assert "Save these instructions" in instructions


# ---------------------------------------------------------------- flux_list_grains

class TestFluxListGrains:
    def test_list_grains_empty(self, store, cfg):
        result = dispatch("flux_list_grains", {}, store, cfg)
        assert "grains" in result
        assert result["count"] == 0

    def test_list_grains_after_store(self, store, cfg):
        dispatch("flux_store", {"content": "listed grain", "caller_id": "t"}, store, cfg)
        result = dispatch("flux_list_grains", {}, store, cfg)
        assert result["count"] >= 1
        grain = result["grains"][0]
        assert "id" in grain
        assert "content_snippet" in grain
        assert "status" in grain
        assert "provenance" in grain

    def test_list_grains_status_filter_active(self, store, cfg):
        dispatch("flux_store", {"content": "active one"}, store, cfg)
        result = dispatch("flux_list_grains", {"status": "active"}, store, cfg)
        assert all(g["status"] == "active" for g in result["grains"])

    def test_list_grains_invalid_status_raises(self, store, cfg):
        with pytest.raises(ValueError, match="Invalid status"):
            dispatch("flux_list_grains", {"status": "unknown"}, store, cfg)

    def test_list_grains_limit(self, store, cfg):
        for i in range(5):
            dispatch("flux_store", {"content": f"grain {i}"}, store, cfg)
        result = dispatch("flux_list_grains", {"limit": 2}, store, cfg)
        assert len(result["grains"]) <= 2

    def test_list_grains_via_service(self, store, cfg, svc):
        svc.store("service grain", caller_id="test")
        result = dispatch("flux_list_grains", {}, store, cfg, svc=svc)
        assert result["count"] >= 1


# ---------------------------------------------------------------- flux_store with caller_id

class TestFluxStoreCaller:
    def test_store_with_caller_id(self, store, cfg):
        result = dispatch("flux_store", {
            "content": "tagged grain",
            "caller_id": "agent-a",
        }, store, cfg)
        assert "grain_id" in result
        assert result["status"] == "stored"

    def test_store_default_caller(self, store, cfg):
        result = dispatch("flux_store", {"content": "no caller"}, store, cfg)
        assert result["status"] == "stored"


# ---------------------------------------------------------------- flux_retrieve with caller_id

class TestFluxRetrieveCaller:
    def test_retrieve_with_caller_id(self, store, cfg):
        dispatch("flux_store", {"content": "retrievable"}, store, cfg)
        result = dispatch("flux_retrieve", {"query": "retrievable", "caller_id": "agent-b"}, store, cfg)
        assert "grains" in result
        assert "trace_id" in result

    def test_retrieve_accepts_client_and_role_fields(self, store, cfg):
        dispatch("flux_store", {"content": "portable caller identity"}, store, cfg)
        result = dispatch(
            "flux_retrieve",
            {
                "query": "portable caller identity",
                "client": "my-custom-bot",
                "role": "background_lookup",
            },
            store,
            cfg,
        )

        assert "trace_id" in result

        health = dispatch("flux_health", {}, store, cfg)
        callers = {c["caller_id"] for c in health["caller_feedback"]}
        assert "my-custom-bot:background_lookup" in callers


# ---------------------------------------------------------------- unknown tool

class TestUnknownTool:
    def test_unknown_tool_raises(self, store, cfg):
        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch("flux_nonexistent", {}, store, cfg)


# ---------------------------------------------------------------- via service (booth-aware)

class TestMCPViaService:
    def test_store_via_service(self, store, cfg, svc):
        result = dispatch("flux_store", {"content": "booth-stored"}, store, cfg, svc=svc)
        assert result["status"] == "stored"

    def test_retrieve_via_service(self, store, cfg, svc):
        svc.store("booth grain", caller_id="setup")
        result = dispatch("flux_retrieve", {"query": "booth"}, store, cfg, svc=svc)
        assert "grains" in result

    def test_feedback_via_service(self, store, cfg, svc):
        svc.store("feedback target", caller_id="setup")
        ret = dispatch("flux_retrieve", {"query": "feedback"}, store, cfg, svc=svc)
        if ret["grains"]:
            result = dispatch("flux_feedback", {
                "trace_id": ret["trace_id"],
                "grain_id": ret["grains"][0]["id"],
                "useful": True,
            }, store, cfg, svc=svc)
            assert result["status"] == "queued"
