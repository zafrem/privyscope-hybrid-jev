import math
import types

import pytest

torch = pytest.importorskip("torch")

from privyscope_hybrid_jev.artifact import JevMeta  # noqa: E402
from privyscope_hybrid_jev.head import LabelHead  # noqa: E402
from privyscope_hybrid_jev.scorer import JevScorer  # noqa: E402

LABELS = ["PER", "LOC"]  # classes: PER, LOC, NONE


class Tok:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) % 97 + 1 for c in text]}


class OneHot:
    """Hidden = one-hot of the last token id mod 3 (x10)."""

    def __call__(self, input_ids, attention_mask, position_ids):
        h = torch.nn.functional.one_hot(input_ids % 3, 3).float() * 10
        return types.SimpleNamespace(last_hidden_state=h)


def scorer(temperature=1.0, bias=None, batch_size=8, max_length=64):
    head = LabelHead(3, 3)
    with torch.no_grad():
        head.linear.weight.copy_(torch.eye(3))
        head.linear.bias.copy_(torch.tensor(bias or [0.0, 0.0, 0.0]))
    return JevScorer(OneHot(), Tok(), head, JevMeta("m", "rev", LABELS, temperature, max_length),
                     batch_size=batch_size)


def test_probabilities_are_named_and_sum_to_one():
    out = scorer().score_prompts(["xa", "xb", "xc"])  # last char id%3 -> classes 1, 2, 0
    assert [max(o, key=o.get) for o in out] == ["LOC", "NONE", "PER"]
    assert all(set(o) == {"PER", "LOC", "NONE"} for o in out)
    assert sum(out[0].values()) == pytest.approx(1.0)


def test_temperature_flattens_distribution():
    assert max(scorer(5.0).score_prompts(["xa"])[0].values()) < max(scorer(1.0).score_prompts(["xa"])[0].values())


def test_batch_size_does_not_change_scores():
    prompts = ["xa", "xbb", "xcccc", "xa"]
    for a, b in zip(scorer(batch_size=1).score_prompts(prompts), scorer(batch_size=8).score_prompts(prompts)):
        assert a == pytest.approx(b)


def test_non_finite_logits_raise():
    s = scorer()
    with torch.no_grad():
        s.head.linear.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        s.score_prompts(["xa"])


def test_probabilities_builds_the_shared_prompt_and_follows_bias():
    s = scorer(bias=[0, 0, 50])
    out = s.probabilities([("some context", "Alice")])
    assert max(out[0], key=out[0].get) == "NONE" and math.isclose(sum(out[0].values()), 1.0)
