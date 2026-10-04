"""Context windows and candidate generation."""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from .types import Span

_SENT_END = re.compile(r"[.!?。！？\n]")
_TOKEN = re.compile(r"\S+")
_EDGE = ".,;:!?()[]{}<>\"'“”‘’「」『』、，。"


def context_for(text: str, span: Span, chars: int) -> str:
    """The sentence around ``span``, clipped to ``chars`` on each side."""
    lo = max(0, span.start - chars)
    hi = min(len(text), span.end + chars)
    left = max((m.end() for m in _SENT_END.finditer(text, lo, span.start)), default=lo)
    nxt = _SENT_END.search(text, span.end, hi)
    right = nxt.end() if nxt else hi
    return text[left:right].strip()


def _overlaps(a: Tuple[int, int], b: Span) -> bool:
    return not (a[1] <= b.start or a[0] >= b.end)


def candidate_windows(text: str, taken: Sequence[Span], max_ngram: int, max_candidates: int) -> List[Tuple[int, int]]:
    """Whitespace-token n-gram windows that avoid already-detected spans.

    Space-delimited scripts only: a CJK run without spaces is one token, so ``extend`` recall
    for ja/zh/ko is limited to what whitespace tokenisation exposes.
    """
    tokens: List[Tuple[int, int]] = []
    for m in _TOKEN.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s] in _EDGE:
            s += 1
        while e > s and text[e - 1] in _EDGE:
            e -= 1
        if e > s and e - s <= 64:
            tokens.append((s, e))
    out: List[Tuple[int, int]] = []
    for n in range(1, max_ngram + 1):
        for i in range(len(tokens) - n + 1):
            win = (tokens[i][0], tokens[i + n - 1][1])
            if any(_overlaps(win, t) for t in taken):
                continue
            out.append(win)
            if len(out) >= max_candidates:
                return out
    return out
