"""Shared by inference AND private training so both see identical tensors."""
from __future__ import annotations

from typing import List, Sequence


def clip_ids(ids: List[int], n: int) -> List[int]:
    """Keep the head and the tail (the candidate); drop the middle (context)."""
    if len(ids) <= n:
        return ids
    return ids[: n // 2] + ids[-(n - n // 2):]


def encode_prompts(tokenizer, prompts: Sequence[str], max_length: int, device="cpu"):
    """Left-padded ``(input_ids, attention_mask)`` so the last position is always a real token."""
    import torch

    seqs = [clip_ids(list(tokenizer(p, add_special_tokens=True)["input_ids"]), max_length) for p in prompts]
    width = max(len(s) for s in seqs)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, width - len(s):] = 1
    return ids.to(device), mask.to(device)


def last_token_hidden(backbone, ids, mask):
    """Final-position hidden state; explicit position ids make left padding harmless."""
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    out = backbone(input_ids=ids, attention_mask=mask, position_ids=position_ids)
    return out.last_hidden_state[:, -1, :]
