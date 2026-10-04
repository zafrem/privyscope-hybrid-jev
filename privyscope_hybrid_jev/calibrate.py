"""Threshold sweep. The temperature is already fitted at training time; this picks
none_threshold / skip_verified (+ accept_threshold for extend) on labelled data.
The pipeline runs once per document upstream; each grid point is replayed offline.
Tuning and reporting use disjoint splits."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import JevConfig
from .stage import JevStage
from .types import Span

# (text, gold spans, merged spans from the pipeline, regex spans from the pipeline)
Record = Tuple[str, Sequence[Span], Sequence[Span], Sequence[Span]]


class CachingScorer:
    """Memoise probabilities per (context, candidate); misses go out in one batch."""

    def __init__(self, inner) -> None:
        self.inner, self.cache, self.misses = inner, {}, 0

    @property
    def meta(self):
        return self.inner.meta

    def probabilities(self, items):
        missing = list(dict.fromkeys(i for i in items if i not in self.cache))
        if missing:
            self.misses += len(missing)
            for k, v in zip(missing, self.inner.probabilities(missing)):
                self.cache[k] = v
        return [self.cache[i] for i in items]


def micro_f1(pairs: Iterable[Tuple[Sequence[Span], Sequence[Span]]]) -> Dict[str, float]:
    """Strict (label, start, end) micro P/R/F1 - the same rule as ``privyscope eval``."""
    tp = fp = fn = 0
    for gold, pred in pairs:
        g = {(s.label, s.start, s.end) for s in gold}
        p = {(s.label, s.start, s.end) for s in pred}
        tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


@dataclass
class Result:
    config: JevConfig
    metrics: Dict[str, float]
    failures: int = 0


def grid(base: JevConfig, modes: Sequence[str] = ("verify",),
         none_thresholds: Sequence[float] = (0.3, 0.5, 0.7, 0.9),
         skip_verified: Sequence[bool] = (True, False),
         accept_thresholds: Sequence[float] = (0.5, 0.7, 0.9)) -> List[JevConfig]:
    out: List[JevConfig] = []
    for mode in modes:
        accepts = accept_thresholds if mode == "extend" else (base.accept_threshold,)
        for n, sv, a in itertools.product(none_thresholds, skip_verified, accepts):
            out.append(replace(base, mode=mode, none_threshold=n, skip_verified=sv, accept_threshold=a))
    return out


def evaluate(cfg: JevConfig, scorer, records: Sequence[Record]) -> Result:
    """A failing document keeps its incoming spans, exactly as the core does at runtime."""
    stage = JevStage(cfg, scorer)
    failures, pairs = 0, []
    for text, gold, merged, regex_spans in records:
        try:
            pred = stage.refine(text, merged, regex_spans)
        except Exception:  # noqa: BLE001 - mirror the core fallback
            failures, pred = failures + 1, merged
        pairs.append((gold, pred))
    return Result(cfg, micro_f1(pairs), failures)


def split(records: Sequence[Record], tune_fraction: float = 0.5) -> Tuple[List[Record], List[Record]]:
    """Interleaved split (not head/tail), so ordered datasets do not bias either side."""
    if not 0 < tune_fraction < 1:
        raise ValueError("tune_fraction must be in (0, 1)")
    tune, held, acc = [], [], 0.0
    for r in records:
        acc += tune_fraction
        if acc >= 1.0:
            acc -= 1.0
            tune.append(r)
        else:
            held.append(r)
    return tune, held


@dataclass
class Report:
    baseline_tune: Dict[str, float]
    baseline_held: Dict[str, float]
    best: Optional[Result]
    best_held: Optional[Result]
    results: List[Result]
    server_calls: int

    @property
    def helps(self) -> bool:
        return bool(self.best_held and self.best_held.metrics["f1"] > self.baseline_held["f1"])


def calibrate(records: Sequence[Record], scorer, configs: Sequence[JevConfig], tune_fraction: float = 0.5) -> Report:
    cache = scorer if isinstance(scorer, CachingScorer) else CachingScorer(scorer)
    tune, held = split(records, tune_fraction)
    base_tune = micro_f1((g, m) for _t, g, m, _r in tune)
    base_held = micro_f1((g, m) for _t, g, m, _r in held)
    results = [evaluate(c, cache, tune) for c in configs]
    best = max(results, key=lambda r: r.metrics["f1"], default=None)  # first max = cheapest config
    best_held = evaluate(best.config, cache, held) if best else None
    return Report(base_tune, base_held, best, best_held, results, cache.misses)


def config_to_yaml(cfg: JevConfig) -> str:
    import yaml

    return yaml.safe_dump(dict(vars(cfg)), allow_unicode=True, sort_keys=False)
