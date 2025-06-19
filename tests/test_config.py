from __future__ import annotations

from pathlib import Path

import pytest

from causal_consistency_nn.config import Settings


def test_settings_from_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
model:
  hidden_dim: 128
loss:
  x_yz: 0.5
train:
  batch_size: 16
"""
    )
    s = Settings.from_yaml(cfg)
    assert s.model.hidden_dim == 128
    assert s.loss.x_yz == 0.5
    assert s.train.batch_size == 16


def test_settings_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  num_layers: 1\n")
    monkeypatch.setenv("MODEL__NUM_LAYERS", "4")
    s = Settings.from_yaml(cfg)
    assert s.model.num_layers == 4


def test_example_config_loads() -> None:
    cfg = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "scripts"
        / "train_config.yaml"
    )
    s = Settings.from_yaml(cfg)
    assert s.train.epochs == 10
    assert s.model.hidden_dim == 32
