from onegpu_posttrain.evaluation.preference import aggregate_pair_scores


def test_aggregate_pair_scores_common_metric():
    result = aggregate_pair_scores([
        {"chosen_mean_logprob": -1.0, "rejected_mean_logprob": -2.0, "response_tokens": 10},
        {"chosen_mean_logprob": -3.0, "rejected_mean_logprob": -2.5, "response_tokens": 12},
        {"chosen_mean_logprob": -1.5, "rejected_mean_logprob": -1.5, "response_tokens": 8},
    ])
    assert result.pairs == 3
    assert result.policy_pair_accuracy == 1 / 3
    assert result.ties == 1
    assert abs(result.policy_pair_margin - ((1.0 - 0.5 + 0.0) / 3)) < 1e-9
    assert result.max_response_tokens_observed == 12


def test_score_candidate_only_counts_response_tokens():
    from types import SimpleNamespace

    import torch

    from onegpu_posttrain.evaluation.preference import score_candidate

    class FakeTokenizer:
        chat_template = None
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 64 for c in text]

    class FakeModel:
        def __call__(self, input_ids, attention_mask=None):
            b, n = input_ids.shape
            return SimpleNamespace(logits=torch.zeros((b, n, 64), dtype=torch.float32))

    score = score_candidate(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        prompt=[{"role": "user", "content": "2+2?"}],
        candidate="four",
        device="cpu",
        max_length=128,
    )
    assert score.token_count == 4
