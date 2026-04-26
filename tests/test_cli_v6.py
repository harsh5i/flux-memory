"""Tests for CLI integration helpers."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

import flux.cli as flux_cli


def _make_instance(tmp_path, monkeypatch, name="test1"):
    monkeypatch.setattr(flux_cli, "_FLUX_HOME", tmp_path)
    idir = tmp_path / name
    idir.mkdir(parents=True)
    (idir / "config.yaml").write_text(
        "MCP_SERVER_NAME: test1\nOPERATING_MODE: caller_extracts\n",
        encoding="utf-8",
    )
    return idir


class TestMCPClientConfig:
    def test_write_mcp_client_configs(self, tmp_path, monkeypatch):
        _make_instance(tmp_path, monkeypatch)
        paths = flux_cli._write_mcp_client_configs("test1")

        assert paths["codex"].exists()
        codex = paths["codex"].read_text(encoding="utf-8")
        assert '[mcp_servers."flux-test1"]' in codex
        assert '"-m", "flux.cli", "mcp", "--name", "test1"' in codex

        claude = json.loads(paths["claude"].read_text(encoding="utf-8"))
        assert claude["mcpServers"]["flux-test1"]["args"][-2:] == ["--name", "test1"]

    def test_mcp_config_command_writes_snippets(self, tmp_path, monkeypatch):
        _make_instance(tmp_path, monkeypatch)
        result = CliRunner().invoke(flux_cli.cli, ["mcp-config", "--name", "test1"])

        assert result.exit_code == 0
        assert "MCP client snippets written" in result.output
        assert (tmp_path / "test1" / "integrations" / "codex.toml").exists()


class TestStartHelp:
    def test_start_help_does_not_claim_to_start_mcp_stdio(self):
        result = CliRunner().invoke(flux_cli.cli, ["start", "--help"])

        assert result.exit_code == 0
        assert "Launch REST API and dashboard" in result.output
        assert "--broadcast" in result.output

    def test_dashboard_probe_host_uses_loopback_for_broadcast_bind(self):
        assert flux_cli._dashboard_probe_host("0.0.0.0") == "127.0.0.1"
        assert flux_cli._dashboard_probe_host("::") == "127.0.0.1"
        assert flux_cli._dashboard_probe_host("localhost") == "localhost"


class TestWarmupCommand:
    def test_warmup_command_loads_configured_embedding_model(self, tmp_path, monkeypatch):
        _make_instance(tmp_path, monkeypatch)
        calls = []

        def fake_warmup(cfg):
            calls.append(cfg.EMBEDDING_MODEL_NAME)
            return cfg.EMBEDDING_MODEL_NAME, 384, 0.01

        monkeypatch.setattr(flux_cli, "_warmup_embedding_model", fake_warmup)

        result = CliRunner().invoke(flux_cli.cli, ["warmup", "--name", "test1"])

        assert result.exit_code == 0
        assert calls == ["all-MiniLM-L6-v2"]
        assert "Embedding model warmed: all-MiniLM-L6-v2 (384 dims, 0.01s)" in result.output

    def test_warmup_command_requires_initialized_instance(self, tmp_path, monkeypatch):
        monkeypatch.setattr(flux_cli, "_FLUX_HOME", tmp_path)

        result = CliRunner().invoke(flux_cli.cli, ["warmup", "--name", "missing"])

        assert result.exit_code == 1
        assert "Instance 'missing' not initialized" in result.output


class TestRebuildGraphCommand:
    def test_rebuild_graph_command_reports_stats(self, tmp_path, monkeypatch):
        _make_instance(tmp_path, monkeypatch)

        def fake_rebuild(name, limit=None):
            assert name == "test1"
            assert limit == 12
            return {
                "grains_scanned": 15,
                "grains_rebuilt": 10,
                "embeddings_created": 9,
                "conduits_created": 42,
            }

        monkeypatch.setattr(flux_cli, "_rebuild_instance_graph", fake_rebuild)

        result = CliRunner().invoke(
            flux_cli.cli,
            ["rebuild-graph", "--name", "test1", "--limit", "12"],
        )

        assert result.exit_code == 0
        assert "Graph rebuild complete: 10 rebuilt, 9 embeddings, 42 conduits" in result.output

    def test_rebuild_graph_requires_initialized_instance(self, tmp_path, monkeypatch):
        monkeypatch.setattr(flux_cli, "_FLUX_HOME", tmp_path)

        result = CliRunner().invoke(flux_cli.cli, ["rebuild-graph", "--name", "missing"])

        assert result.exit_code == 1
        assert "Instance 'missing' not initialized" in result.output

    def test_rebuild_graph_reports_locked_database(self, tmp_path, monkeypatch):
        _make_instance(tmp_path, monkeypatch)

        def fake_rebuild(name, limit=None):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(flux_cli, "_rebuild_instance_graph", fake_rebuild)

        result = CliRunner().invoke(flux_cli.cli, ["rebuild-graph", "--name", "test1"])

        assert result.exit_code == 1
        assert "Database is locked" in result.output


class TestMCPMessaging:
    def test_mcp_client_config_hint_points_to_codex_snippet(self, monkeypatch):
        monkeypatch.setattr(flux_cli, "_FLUX_HOME", Path("flux-home"))

        assert str(flux_cli._mcp_client_config_hint("test1")).endswith(
            "test1\\integrations\\codex.toml"
        ) or str(flux_cli._mcp_client_config_hint("test1")).endswith(
            "test1/integrations/codex.toml"
        )
