from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class LabelHead(torch.nn.Module):
    """Linear decision head over the last-token hidden state: classes = labels + NONE."""

    def __init__(self, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


def save_head(head: LabelHead, path) -> None:
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(path))


def load_head(path) -> LabelHead:
    state = load_file(str(Path(path)))
    n_classes, hidden = state["linear.weight"].shape
    head = LabelHead(hidden, n_classes)
    head.load_state_dict(state)
    return head
