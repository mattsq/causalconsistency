from causal_consistency_nn.utils import logging as clog


def test_log(capsys) -> None:
    clog.log("hello")
    captured = capsys.readouterr().out
    assert "hello" in captured
