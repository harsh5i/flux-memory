"""Tests for Config.from_yaml (Track 5 Step 1)."""
from __future__ import annotations

import pytest
from pathlib import Path

from flux import Config


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "flux.yaml"
    p.write_text(content)
    return p


class TestConfigFromYaml:
    def test_loads_single_override(self, tmp_path):
        p = _write_yaml(tmp_path, "ATTENUATION: 0.70\n")
        cfg = Config.from_yaml(p)
        assert cfg.ATTENUATION == pytest.approx(0.70)
        assert cfg.MAX_HOPS == 5  # default unchanged

    def test_loads_multiple_overrides(self, tmp_path):
        p = _write_yaml(tmp_path, "ATTENUATION: 0.70\nMAX_HOPS: 3\nTOP_K: 10\n")
        cfg = Config.from_yaml(p)
        assert cfg.ATTENUATION == pytest.approx(0.70)
        assert cfg.MAX_HOPS == 3
        assert cfg.TOP_K == 10

    def test_unknown_keys_ignored(self, tmp_path):
        p = _write_yaml(tmp_path, "FUTURE_PARAM: 42\nATTENUATION: 0.80\n")
        cfg = Config.from_yaml(p)  # should not raise
        assert cfg.ATTENUATION == pytest.approx(0.80)

    def test_empty_file_returns_defaults(self, tmp_path):
        p = _write_yaml(tmp_path, "")
        cfg = Config.from_yaml(p)
        default = Config()
        assert cfg.ATTENUATION == default.ATTENUATION
        assert cfg.MAX_HOPS == default.MAX_HOPS

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(tmp_path / "nonexistent.yaml")

    def test_non_mapping_yaml_raises(self, tmp_path):
        p = _write_yaml(tmp_path, "- item1\n- item2\n")
        with pytest.raises(ValueError):
            Config.from_yaml(p)

    def test_string_param_override(self, tmp_path):
        p = _write_yaml(tmp_path, "LLM_MODEL: phi3:mini\n")
        cfg = Config.from_yaml(p)
        assert cfg.LLM_MODEL == "phi3:mini"

    def test_full_defaults_yaml_loads(self, tmp_path):
        """The bundled flux.yaml must load without error and produce defaults."""
        yaml_path = Path(__file__).parent.parent / "config" / "flux.yaml"
        if not yaml_path.exists():
            pytest.skip("config/flux.yaml not found")
        cfg = Config.from_yaml(yaml_path)
        default = Config()
        # Core values from the YAML should match spec defaults.
        assert cfg.ATTENUATION == pytest.approx(default.ATTENUATION)
        assert cfg.WEIGHT_FLOOR == pytest.approx(default.WEIGHT_FLOOR)
        assert cfg.PROMOTION_THRESHOLD == default.PROMOTION_THRESHOLD
