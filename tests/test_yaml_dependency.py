from __future__ import annotations

from pathlib import Path

from causal_consistency_nn.config import Settings


def test_yaml_dependency(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("train:\n  epochs: 7\n")
    s = Settings.from_yaml(cfg)
    assert s.train.epochs == 7
