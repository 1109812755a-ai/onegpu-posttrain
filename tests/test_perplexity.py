import math

from onegpu_posttrain.evaluation.perplexity import aggregate_nll


def test_aggregate_nll_is_token_weighted():
    total, tokens, mean_nll, ppl = aggregate_nll([(2.0, 2), (9.0, 3)])
    assert total == 11.0
    assert tokens == 5
    assert mean_nll == 2.2
    assert math.isclose(ppl, math.exp(2.2))


def test_aggregate_nll_empty():
    total, tokens, mean_nll, ppl = aggregate_nll([])
    assert total == 0.0
    assert tokens == 0
    assert mean_nll is None
    assert ppl is None
