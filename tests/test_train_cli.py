from __future__ import annotations

from pathlib import Path

from causal_consistency_nn import train


def test_cli_override(tmp_path: Path, capsys) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("loss:\n  unsup: 0.1\n")
    train.main(["--config", str(cfg), "--loss-unsup", "0.5"])
    out = capsys.readouterr().out
    assert "unsup=0.5" in out
