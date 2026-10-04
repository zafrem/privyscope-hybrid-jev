import types

import pytest

torch = pytest.importorskip("torch")

from privyscope_hybrid_jev.encoding import clip_ids, encode_prompts, last_token_hidden  # noqa: E402
from privyscope_hybrid_jev.head import LabelHead, load_head, save_head  # noqa: E402


class Tok:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) % 97 + 1 for c in text]}


class PositionSensitive:
    """Hidden state depends on position ids, so wrong padding/positions change the output."""

    def __call__(self, input_ids, attention_mask, position_ids):
        h = (input_ids.float() + 0.5 * position_ids.float()).unsqueeze(-1).expand(-1, -1, 4)
        return types.SimpleNamespace(last_hidden_state=h)


def test_clip_keeps_head_and_tail():
    assert clip_ids(list(range(10)), 4) == [0, 1, 8, 9]
    assert clip_ids(list(range(10)), 5) == [0, 1, 7, 8, 9]
    assert clip_ids(list(range(3)), 4) == [0, 1, 2]


def test_over_long_prompt_keeps_candidate_at_the_end():
    ids, _ = encode_prompts(Tok(), ["a" * 50 + "Z"], 8)
    assert ids.shape == (1, 8)
    assert ids[0, -1].item() == ord("Z") % 97 + 1


def test_left_padding_and_mask():
    ids, mask = encode_prompts(Tok(), ["ab", "abcd"], 16)
    assert mask.tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    assert ids[0, -1].item() == ord("b") % 97 + 1


def test_batched_equals_single_for_different_lengths():
    single = last_token_hidden(PositionSensitive(), *encode_prompts(Tok(), ["ab"], 16))
    batched = last_token_hidden(PositionSensitive(), *encode_prompts(Tok(), ["ab", "abcdefg"], 16))
    assert torch.allclose(single[0], batched[0])


def test_head_roundtrip(tmp_path):
    head = LabelHead(4, 3)
    save_head(head, tmp_path / "head.safetensors")
    again = load_head(tmp_path / "head.safetensors")
    x = torch.randn(2, 4)
    assert torch.allclose(head(x), again(x))
