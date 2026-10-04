from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .artifact import NONE, JevMeta
from .encoding import encode_prompts, last_token_hidden
from .prompt import build_prompt


class JevScorer:
    """One forward pass per batch; trained head -> per-label probabilities. No generation."""

    def __init__(self, backbone, tokenizer, head, meta: JevMeta, device="cpu", batch_size: int = 8) -> None:
        self.backbone, self.tokenizer, self.head = backbone, tokenizer, head
        self.meta, self.device, self.batch_size = meta.validate(), device, max(1, batch_size)
        self.names = list(meta.labels) + [NONE]

    def score_prompts(self, prompts: Sequence[str]) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for i in range(0, len(prompts), self.batch_size):
            out.extend(self._batch(list(prompts[i : i + self.batch_size])))
        return out

    def probabilities(self, items: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
        """``items`` are ``(context, candidate)``."""
        return self.score_prompts([build_prompt(c, s) for c, s in items])

    def _batch(self, prompts: List[str]) -> List[Dict[str, float]]:
        import torch

        with torch.no_grad():
            ids, mask = encode_prompts(self.tokenizer, prompts, self.meta.max_length, self.device)
            logits = self.head(last_token_hidden(self.backbone, ids, mask).float())
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite logits from jev head")
            probs = torch.softmax(logits / self.meta.temperature, dim=-1).cpu().tolist()
        return [dict(zip(self.names, row)) for row in probs]
