from __future__ import annotations

from causal_consistency_nn import __main__


def test_package_help(capsys) -> None:
    __main__.main([])
    out = capsys.readouterr().out
    assert "train" in out
    assert "eval" in out
    assert "serve" in out
