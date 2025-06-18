from pathlib import Path
from causal_consistency_nn import train


def test_cli_all_overrides(tmp_path: Path, capsys) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  hidden_dim: 8\n  num_layers: 1\n")
    train.main(
        [
            "--config",
            str(cfg),
            "--model-hidden-dim",
            "16",
            "--model-num-layers",
            "2",
            "--loss-z-yx",
            "0.5",
            "--loss-y-xz",
            "0.5",
            "--loss-x-yz",
            "0.5",
            "--loss-unsup",
            "0.2",
        ]
    )
    out = capsys.readouterr().out
    assert "hidden_dim=16" in out
    assert "num_layers=2" in out
    assert "unsup=0.2" in out
