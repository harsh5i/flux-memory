"""Tests for CLI integration helpers."""
from __future__ import annotations

import json
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


class TestMCPMessaging:
    def test_mcp_client_config_hint_points_to_codex_snippet(self, monkeypatch):
        monkeypatch.setattr(flux_cli, "_FLUX_HOME", Path("flux-home"))

        assert str(flux_cli._mcp_client_config_hint("test1")).endswith(
            "test1\\integrations\\codex.toml"
        ) or str(flux_cli._mcp_client_config_hint("test1")).endswith(
            "test1/integrations/codex.toml"
        )
