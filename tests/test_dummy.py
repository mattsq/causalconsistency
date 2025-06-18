from causal_consistency_nn.data.dummy import load_dummy


def test_load_dummy() -> None:
    assert load_dummy() == [1, 2, 3]
